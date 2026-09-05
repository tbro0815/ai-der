/* P6.1: Qwen3-VL vision tower on CUDA.  Mirrors q38v_encode step for step:
 * patch GEMM + interpolated position embedding, 27 pre-norm blocks (LayerNorm,
 * fused qkv, 2-D rotary, full attention, o-proj, LayerNorm, up, GELU(tanh),
 * down), post LayerNorm, the 2x2 merger.  Weights F16 on the device (the
 * mmproj precision), activations f32; one generic tiled GEMM serves the
 * weight products (B = W[N][K] as half) and the attention products (B in
 * f32, transposed or not).  The position table and the rotary cos/sin table
 * are computed by the caller exactly as the CPU path does and uploaded. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vision_qwen3vl_cuda.h"

#define VCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[vis-gpu] %s: %s\n", #x, cudaGetErrorString(e_)); return -1; } } while (0)

typedef struct {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *qkv_b, *o_b, *up_b, *down_b;
    __half *qkv_w, *o_w, *up_w, *down_w;
} DevBlk;

static struct {
    int loaded;
    Q38VisCfg c;
    __half *patch_w, *mm0_w, *mm2_w;
    float *patch_b, *post_w, *post_b, *mm0_b, *mm2_b;
    DevBlk *blk;
    size_t bytes;
} G;

static void *dmalloc(size_t n) { void *p = NULL; if (cudaMalloc(&p, n) != cudaSuccess) return NULL; G.bytes += n; return p; }

static int up_f32(float **dst, const float *src, size_t n) {
    *dst = (float *)dmalloc(n * sizeof(float));
    if (!*dst) return -1;
    VCK(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

static int up_f16(__half **dst, const float *src, size_t n) {
    __half *tmp = (__half *)malloc(n * sizeof(__half));
    if (!tmp) return -1;
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(src[i]);
    *dst = (__half *)dmalloc(n * sizeof(__half));
    if (!*dst) { free(tmp); return -1; }
    cudaError_t e = cudaMemcpy(*dst, tmp, n * sizeof(__half), cudaMemcpyHostToDevice);
    free(tmp);
    if (e != cudaSuccess) { fprintf(stderr, "[vis-gpu] upload: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;
}

extern "C" size_t q38vg_vram_bytes(void) { return G.loaded ? G.bytes : 0; }

extern "C" void q38vg_free(void) {
    if (!G.loaded) return;
    cudaFree(G.patch_w); cudaFree(G.mm0_w); cudaFree(G.mm2_w);
    cudaFree(G.patch_b); cudaFree(G.post_w); cudaFree(G.post_b); cudaFree(G.mm0_b); cudaFree(G.mm2_b);
    for (int l = 0; l < G.c.depth; l++) {
        DevBlk *b = &G.blk[l];
        cudaFree(b->ln1_w); cudaFree(b->ln1_b); cudaFree(b->ln2_w); cudaFree(b->ln2_b);
        cudaFree(b->qkv_b); cudaFree(b->o_b); cudaFree(b->up_b); cudaFree(b->down_b);
        cudaFree(b->qkv_w); cudaFree(b->o_w); cudaFree(b->up_w); cudaFree(b->down_w);
    }
    free(G.blk);
    memset(&G, 0, sizeof G);
}

extern "C" int q38vg_load(const Q38VisTower *t) {
    if (G.loaded) return 0;
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    memset(&G, 0, sizeof G);
    G.c = t->c;
    const Q38VisCfg *c = &t->c;
    const size_t H = c->hidden, I = c->inter, PW = (size_t)3 * c->patch * c->patch, P = c->proj_in;
    G.blk = (DevBlk *)calloc(c->depth, sizeof(DevBlk));
    if (!G.blk) return -1;
    int rc = 0;
    rc |= up_f16(&G.patch_w, t->patch_w, H * PW);
    rc |= up_f32(&G.patch_b, t->patch_b, H);
    rc |= up_f32(&G.post_w, t->post_w, H);
    rc |= up_f32(&G.post_b, t->post_b, H);
    rc |= up_f16(&G.mm0_w, t->mm0_w, P * P);
    rc |= up_f32(&G.mm0_b, t->mm0_b, P);
    rc |= up_f16(&G.mm2_w, t->mm2_w, (size_t)c->out_dim * P);
    rc |= up_f32(&G.mm2_b, t->mm2_b, c->out_dim);
    for (int l = 0; l < c->depth && !rc; l++) {
        const Q38VisBlk *s = &t->blk[l]; DevBlk *d = &G.blk[l];
        rc |= up_f32(&d->ln1_w, s->ln1_w, H); rc |= up_f32(&d->ln1_b, s->ln1_b, H);
        rc |= up_f32(&d->ln2_w, s->ln2_w, H); rc |= up_f32(&d->ln2_b, s->ln2_b, H);
        rc |= up_f16(&d->qkv_w, s->qkv_w, 3 * H * H); rc |= up_f32(&d->qkv_b, s->qkv_b, 3 * H);
        rc |= up_f16(&d->o_w, s->o_w, H * H);          rc |= up_f32(&d->o_b, s->o_b, H);
        rc |= up_f16(&d->up_w, s->up_w, I * H);        rc |= up_f32(&d->up_b, s->up_b, I);
        rc |= up_f16(&d->down_w, s->down_w, H * I);    rc |= up_f32(&d->down_b, s->down_b, H);
    }
    if (rc) { G.loaded = 1; q38vg_free(); return -1; }
    G.loaded = 1;
    return 0;
}

/* ---------------------------------------------------------------- kernels */

/* C[m][n] = sum_k A[m][k] * B(k, n) + bias[n], batched over blockIdx.z with
 * element strides sA/sB/sC.  TRANSB: B stored [N][K] (weights, K rows);
 * else B stored [K][N] (the V rows in P.V).  64x64 tiles, 16-deep K steps,
 * each thread a 4x4 micro tile; every bound is guarded. */
#define TM 64
#define TN 64
#define TK 16
template <typename TB, bool TRANSB>
__global__ void k_gemm(int M, int N, int K, const float *__restrict__ A, int lda,
                       const TB *__restrict__ B, int ldb, const float *__restrict__ bias,
                       float *__restrict__ C, int ldc, long sA, long sB, long sC, float scale) {
    __shared__ float As[TK][TM + 1];
    __shared__ float Bs[TK][TN + 1];
    A += (long)blockIdx.z * sA; B += (long)blockIdx.z * sB; C += (long)blockIdx.z * sC;
    const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;     /* 16x16 threads */
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        /* A tile: 64 rows x 16 k; 1024 elements, 4 per thread; k fastest */
        for (int e = threadIdx.x; e < TM * TK; e += 256) {
            const int k = e % TK, m = e / TK;
            As[k][m] = (m0 + m < M && k0 + k < K) ? A[(long)(m0 + m) * lda + k0 + k] : 0.f;
        }
        for (int e = threadIdx.x; e < TN * TK; e += 256) {
            float v = 0.f;
            if (TRANSB) {
                const int k = e % TK, n = e / TK;
                if (n0 + n < N && k0 + k < K) v = (float)B[(long)(n0 + n) * ldb + k0 + k];
                Bs[k][n] = v;
            } else {
                const int n = e % TN, k = e / TN;
                if (n0 + n < N && k0 + k < K) v = (float)B[(long)(k0 + k) * ldb + n0 + n];
                Bs[k][n] = v;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[k][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            float v = acc[i][j] * scale;
            if (bias) v += bias[n];
            C[(long)m * ldc + n] = v;
        }
    }
}

template <typename TB, bool TRANSB>
static void gemm(int M, int N, int K, const float *A, int lda, const TB *B, int ldb,
                 const float *bias, float *C, int ldc, int batch = 1,
                 long sA = 0, long sB = 0, long sC = 0, float scale = 1.f) {
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM, batch);
    k_gemm<TB, TRANSB><<<grid, 256>>>(M, N, K, A, lda, B, ldb, bias, C, ldc, sA, sB, sC, scale);
}

/* LayerNorm per row, f32 two-pass mean/variance like the CPU */
__global__ void k_layernorm(const float *__restrict__ in, float *__restrict__ out,
                            const float *__restrict__ w, const float *__restrict__ b, int n, float eps) {
    const float *x = in + (long)blockIdx.x * n;
    float *y = out + (long)blockIdx.x * n;
    __shared__ float red[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float mean = red[0] / n;
    __syncthreads();
    float v2 = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float d = x[i] - mean; v2 += d * d; }
    for (int o = 16; o > 0; o >>= 1) v2 += __shfl_xor_sync(0xffffffffu, v2, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v2;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / sqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

__global__ void k_add(float *__restrict__ a, const float *__restrict__ b, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_gelu(float *__restrict__ a, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = a[i]; a[i] = 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * (x + 0.044715f * x * x * x))); }
}

/* rotary on q and k: pair (j, j+half) of every head; cs[token] = cos[half] | sin[half] */
__global__ void k_rope(float *__restrict__ qkv, const float *__restrict__ cs, int tokens, int hidden,
                       int heads, int hd) {
    const int half = hd / 2;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long per_tok = (long)2 * heads * half;
    if (i >= (long)tokens * per_tok) return;
    const int tk = (int)(i / per_tok);
    int r = (int)(i % per_tok);
    const int pass = r / (heads * half); r %= heads * half;
    const int h = r / half, j = r % half;
    float *v = qkv + (long)tk * 3 * hidden + (long)pass * hidden + (long)h * hd;
    const float *c = cs + (long)tk * half * 2;
    const float x0 = v[j], x1 = v[j + half];
    v[j]        = x0 * c[j] - x1 * c[half + j];
    v[j + half] = x1 * c[j] + x0 * c[half + j];
}

/* softmax over rows of length n (scores already scaled) */
__global__ void k_softmax(float *__restrict__ s, int n) {
    float *row = s + (long)blockIdx.x * n;
    __shared__ float red[32];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float tot = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float e = expf(row[i] - mx); row[i] = e; tot += e; }
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = tot;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= inv;
}

/* ---------------------------------------------------------------- encode */

/* device scratch one encode allocates (the VA list below), so the caller can
 * make room in the expert tier first (P6.6) */
extern "C" size_t q38vg_encode_bytes(const Q38VisTower *t, int grid_h, int grid_w) {
    const Q38VisCfg *c = &t->c;
    const int T = grid_h * grid_w, H = c->hidden, I = c->inter, hd = c->head_dim, heads = c->heads;
    const int half = hd / 2, PW = 3 * c->patch * c->patch, m = c->merge;
    const int n_out = (grid_h / (m ? m : 1)) * (grid_w / (m ? m : 1)), P = c->proj_in;
    int grp = heads;
    while (grp > 1 && (size_t)grp * T * T * sizeof(float) > (size_t)256 << 20) grp = (grp + 1) / 2;
    size_t n = (size_t)T * PW + (size_t)T * H + (size_t)T * half * 2 + 3 * (size_t)T * H
             + (size_t)T * 3 * H + (size_t)T * I + (size_t)grp * T * T
             + (size_t)n_out * P + (size_t)n_out * c->out_dim;
    return n * sizeof(float);
}

extern "C" int q38vg_encode(const Q38VisTower *t, const float *patches, int grid_h, int grid_w,
                            const float *pos, const float *cs, float *out) {
    if (!G.loaded) return -1;
    const Q38VisCfg *c = &t->c;
    const int T = grid_h * grid_w, H = c->hidden, I = c->inter, hd = c->head_dim, heads = c->heads;
    const int half = hd / 2, PW = 3 * c->patch * c->patch, m = c->merge;
    if (grid_h < m || grid_w < m || grid_h % m || grid_w % m) return -1;
    const int n_out = (grid_h / m) * (grid_w / m), P = c->proj_in;
    /* head groups for the score buffer: <= 256 MB */
    int grp = heads;
    while (grp > 1 && (size_t)grp * T * T * sizeof(float) > (size_t)256 << 20) grp = (grp + 1) / 2;

    float *d_patch = NULL, *d_pos = NULL, *d_cs = NULL, *state = NULL, *normed = NULL, *branch = NULL,
          *qkv = NULL, *mlp = NULL, *score = NULL, *mid = NULL, *d_out = NULL;
    int rc = -1;
    #define VA(p, n) do { if (!(p = (float *)dmalloc((size_t)(n) * sizeof(float)))) goto done; } while (0)
    size_t before = G.bytes;
    VA(d_patch, (size_t)T * PW); VA(d_pos, (size_t)T * H); VA(d_cs, (size_t)T * half * 2);
    VA(state, (size_t)T * H); VA(normed, (size_t)T * H); VA(branch, (size_t)T * H);
    VA(qkv, (size_t)T * 3 * H); VA(mlp, (size_t)T * I); VA(score, (size_t)grp * T * T);
    VA(mid, (size_t)n_out * P); VA(d_out, (size_t)n_out * c->out_dim);
    #undef VA
    if (cudaMemcpy(d_patch, patches, (size_t)T * PW * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_pos, pos, (size_t)T * H * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (cudaMemcpy(d_cs, cs, (size_t)T * half * 2 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) goto done;

    {
        const long nH = (long)T * H, nI = (long)T * I;
        const int bs = 256;
        /* patch embed + position embedding */
        gemm<__half, true>(T, H, PW, d_patch, PW, G.patch_w, PW, G.patch_b, state, H);
        k_add<<<(unsigned)((nH + bs - 1) / bs), bs>>>(state, d_pos, nH);
        const float scale = 1.0f / sqrtf((float)hd);
        for (int l = 0; l < c->depth; l++) {
            const DevBlk *b = &G.blk[l];
            k_layernorm<<<T, 256>>>(state, normed, b->ln1_w, b->ln1_b, H, c->eps);
            gemm<__half, true>(T, 3 * H, H, normed, H, b->qkv_w, H, b->qkv_b, qkv, 3 * H);
            {
                const long nr = (long)T * 2 * heads * half;
                k_rope<<<(unsigned)((nr + bs - 1) / bs), bs>>>(qkv, d_cs, T, H, heads, hd);
            }
            for (int h0 = 0; h0 < heads; h0 += grp) {
                const int g = heads - h0 < grp ? heads - h0 : grp;
                /* scores[g][T][T] = scale * q_h . k_h */
                gemm<float, true>(T, T, hd, qkv + (long)h0 * hd, 3 * H, qkv + H + (long)h0 * hd, 3 * H,
                                  NULL, score, T, g, hd, hd, (long)T * T, scale);
                k_softmax<<<(unsigned)((long)g * T), 256>>>(score, T);
                /* branch[:, h*hd..] = P . v_h */
                gemm<float, false>(T, hd, T, score, T, qkv + 2 * H + (long)h0 * hd, 3 * H,
                                   NULL, branch + (long)h0 * hd, H, g, (long)T * T, hd, hd);
            }
            gemm<__half, true>(T, H, H, branch, H, b->o_w, H, b->o_b, normed, H);
            k_add<<<(unsigned)((nH + bs - 1) / bs), bs>>>(state, normed, nH);
            k_layernorm<<<T, 256>>>(state, normed, b->ln2_w, b->ln2_b, H, c->eps);
            gemm<__half, true>(T, I, H, normed, H, b->up_w, H, b->up_b, mlp, I);
            k_gelu<<<(unsigned)((nI + bs - 1) / bs), bs>>>(mlp, nI);
            gemm<__half, true>(T, H, I, mlp, I, b->down_w, I, b->down_b, normed, H);
            k_add<<<(unsigned)((nH + bs - 1) / bs), bs>>>(state, normed, nH);
        }
        k_layernorm<<<T, 256>>>(state, normed, G.post_w, G.post_b, H, c->eps);
        /* merger: the 4 tokens of a block are adjacent, so normed IS [n_out][P] */
        gemm<__half, true>(n_out, P, P, normed, P, G.mm0_w, P, G.mm0_b, mid, P);
        {
            const long nP = (long)n_out * P;
            k_gelu<<<(unsigned)((nP + bs - 1) / bs), bs>>>(mid, nP);
        }
        gemm<__half, true>(n_out, c->out_dim, P, mid, P, G.mm2_w, P, G.mm2_b, d_out, c->out_dim);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr, "[vis-gpu] encode: %s\n", cudaGetErrorString(e)); goto done; }
        if (cudaMemcpy(out, d_out, (size_t)n_out * c->out_dim * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
    }
    rc = 0;
done:
    cudaFree(d_patch); cudaFree(d_pos); cudaFree(d_cs); cudaFree(state); cudaFree(normed); cudaFree(branch);
    cudaFree(qkv); cudaFree(mlp); cudaFree(score); cudaFree(mid); cudaFree(d_out);
    G.bytes = before;
    if (rc) cudaGetLastError();
    return rc;
}
