/* qwen38_cuda.cu -- P4: dense per-token GPU path for the qwen38 engine.
 * See qwen38_cuda.h for the contract.
 *
 * Weight layout: raw GGML blocks are re-laid out ONCE at upload into a planar
 * form -- int8 quants [O][I] plus one f32 scale per quant group ([O][I/32]
 * for Q8_0, [O][I/16] for Q6_K with the fp16 super-scale folded in).  The
 * per-element dequantized VALUES are bit-identical to the CPU decoders
 * (products formed in the same f32 order); the planar form exists purely so
 * the GEMV kernel can issue aligned char4/float4 loads and stream the row
 * bytes at near-peak bandwidth (the generic ggml_blocks_cuda kernel decodes
 * per element and tops out around ~175 GB/s -- too slow for ~3.5 GB of dense
 * weights per token).
 *
 * Kernel inventory:
 *   k_gemv_i8<GSHIFT>   planar int8 GEMV, warp per row, char4/float4 loads
 *   k_gemv_f32          f32 GEMV, warp per row (inject mats, GDN b/alpha)
 *   k_bcast_res         res[s] = x for the 4 hyper-connection streams
 *   k_hc_norm           per-stream RMS norm (double accum) * folded gamma
 *   k_silu_div          lo = silu(lo/4)
 *   k_hc_out            mixed[j] = mean_s xn[s,j]*sigmoid(gate[s,j])
 *   k_combine           res[s,j] += blk[j] * 2*sigmoid(inject[s]/4)
 *   k_gdn_gate          beta = sigmoid(b), eg = exp(a*softplus(alpha+dt))
 *   k_gdn_conv          depthwise causal conv (kernel 4) + silu, ring update
 *   k_gdn_splitnorm     head split (tiled k-head broadcast) + L2 norm q/k
 *   k_gdn_delta         gated delta rule, 2 fused state sweeps (state resident)
 *   k_gdn_headnorm      per-head RMS norm * gamma * sigmoid(z)
 *
 * All math mirrors the CPU functions in qwen38.c expression-for-expression
 * (sigmoid/silu/softplus formulas included); reductions that the CPU runs in
 * double (RMS/L2 norms) accumulate in double here too.  The delta-rule sweep
 * order over kd is the CPU's exact sequential order per (head, v-dim). */
#ifdef COLI_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml_blocks.h"
#include "qwen38_cuda.h"

#define G_HC 4
#define G_MAXMAT 1024
#define G_BMAX 1024
#define G_BTILE 16
#define G_SPEC_BMAX 8

typedef struct {
    int kind;          /* 0 = planar int8 gs=32 (Q8_0), 1 = planar int8 gs=16 (Q6_K), 2 = f32 */
    int O, I;
    int8_t *q;         /* device, kind 0/1 */
    /* kind 0: fp16 group scales -- the fp16 `d` exactly as stored on disk, so
     * __half2float reproduces the CPU decoder's scale bit-for-bit at half the
     * bytes (scales are re-read every token, so this is bandwidth too).
     * kind 1: f32 scales (Q6_K's d*sub-scale product is not an fp16 value).
     * kind 2: the f32 tensor itself. */
    __half *h;
    float  *s;
} GMat;

static struct {
    int on, dead;
    int nl, D, W, R, vh, vk, kd, vd, convk, cdim, vtot, keytot;
    int H, KV, hd, vocab;
    float eps;
    cudaStream_t st;
    /* side streams + events for fork/join of mutually independent GEMVs
     * inside a graph capture (single-stream capture would serialize them) */
    cudaStream_t s1, s2;
    cudaEvent_t ef, e1, e2;
    GMat mat[G_MAXMAT];
    int nmat;
    size_t vram;
    /* activation + scratch buffers */
    float *d_res, *d_xn, *d_lo, *d_gt, *d_inj, *d_mixed, *d_blk;
    float *d_qkv, *d_z, *d_ba, *d_conv, *d_outr;
    float *d_o, *d_b_o;        /* P10 P8: pre-norm GDN output between the split sweep and the norm */
    int gdn_split_off;         /* Q38_GPU_GDN_SPLIT=0: single block per head */
    float *d_qsa, *d_ctx, *d_logits;
    /* fixed-capacity prefill scratch (batch-major, B <= G_BMAX) */
    float *d_b_res, *d_b_xn, *d_b_lo, *d_b_gt, *d_b_inj, *d_b_mixed;
    float *d_b_qf, *d_b_kk, *d_b_vv;
    float *d_b_qkv, *d_b_z, *d_b_ba, *d_b_conv, *d_b_outr;
    float *d_b_in, *d_b_out;
    float *d_b_sg, *d_b_su, *d_b_sh;   /* shared expert over a chunk: gate, up, silu*mul */
    float *h_b_sh;                     /* pinned [G_BMAX*D]: async shared-expert output */
    cudaEvent_t ev_shb;
    int b_in_max;
    int tiled_off;     /* Q38_GPU_TILED=0: keep the untiled batch GEMMs (A/B) */
    int gdn_chunk_off; /* Q38_GPU_GDN_CHUNK=0: per-token GDN launches (A/B) */
    /* P9 QSA on the device for batched prefill: q8 KV mirror per QSA layer
     * (host cache stays the source of truth; rows [0, kv_valid) are equal on
     * both sides), normalized/roped queries, selection lists, context */
    int8_t *d_k8[129], *d_v8[129];
    float *d_k8s[129], *d_v8s[129];
    int kv_valid[129];
    int kv_max_t, kv_heads;
    float *d_b_q, *d_b_ctx;       /* [G_BMAX * H * hd] */
    int *d_nsel, *d_sel;          /* [G_BMAX], [G_BMAX * sel_cap] */
    int *d_mp;                    /* [G_BMAX * 3] IMRoPE positions of a chunk (P6.3) */
    int sel_cap;
    /* indexer block scoring: pooled block keys per layer (mirror of IBK),
     * chunk queries, per-token block counts, scores (pinned copy on host) */
    float *d_ibk[129];
    int ibk_valid[129], ibk_cap;
    float *d_b_qi, *d_b_score, *h_b_score;
    int *d_b_nfb;
    int idx_nh, idx_dim;
    size_t b_out_cap;
    /* CUDA graphs: one per captured launch span (GDN layer body, QSA hc
     * mixes, head); Q38_GPU_GRAPHS=0 disables. */
    cudaGraphExec_t gexec[3 * 128 + 1];
    int graphs_off, capturing;
    /* Q38_GPU_KPROF=1: sync-and-time each kernel group (forces graphs off) */
    int kprof; double kp_ms[12]; long kp_n[12];
    /* per-layer GDN state (NULL on QSA layers) */
    float **d_rec, **d_ring;
    int gdn_slot[128], n_gdn, spec_n;
    float *d_spec_rec, *d_spec_ring;
    /* pinned staging: h_pin for downloads (always followed by a sync), an
     * up-ring for uploads so the host never has to sync just to reuse the
     * staging bytes (each layer's two uploads are separated by the mixed_get
     * sync, and the ring depth covers a whole layer). */
    float *h_pin; size_t pin_sz;
    float *h_up; size_t up_stride; int up_slot;
} g;
#define G_UPRING 4

static int gck(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "[q38g] CUDA error in %s: %s -> dense GPU path dead\n",
            what, cudaGetErrorString(e));
    g.dead = 1;
    return 0;
}
#define GCK(call) gck((call), #call)

/* kprof slots: 0 hc_norm 1 hc_down 2 hc_up 3 hc_out 4 hc_inj
 *              5 gdn_qkv 6 gdn_z 7 gdn_ba 8 gdn_conv 9 gdn_core 10 gdn_out
 *              11 lm_head */
static const char *kp_name[12] = { "hc_norm", "hc_down", "hc_up", "hc_out", "hc_inj",
                                   "gdn_qkv", "gdn_z", "gdn_ba", "gdn_conv", "gdn_core",
                                   "gdn_out", "lm_head" };
static double kp_t0;
static double kp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                             return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static void kp_begin(void) { if (g.kprof) { cudaStreamSynchronize(g.st); kp_t0 = kp_now(); } }
static void kp_end(int slot) {
    if (!g.kprof) return;
    cudaStreamSynchronize(g.st);
    g.kp_ms[slot] += kp_now() - kp_t0;
    g.kp_n[slot]++;
}

static void *gmalloc(size_t n, const char *what) {
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) {
        fprintf(stderr, "[q38g] cudaMalloc %zu B failed (%s)\n", n, what);
        return NULL;
    }
    g.vram += n;
    return p;
}

/* ==================== device math helpers ==================== */

static __device__ __forceinline__ float d_sigmoid(float v) {
    return v >= 0.f ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
}
static __device__ __forceinline__ float d_silu(float v) { return v * d_sigmoid(v); }
static __device__ __forceinline__ float d_softplus(float z) {
    return z > 20.f ? z : log1pf(expf(z));
}

static __device__ __forceinline__ float warp_red_f(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
static __device__ __forceinline__ double warp_red_d(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
/* block-wide double sum, valid in thread 0; blockDim.x <= 1024 */
static __device__ double block_red_d(double v) {
    __shared__ double sh[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_red_d(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < (unsigned)nw) ? sh[threadIdx.x] : 0.0;
    if (warp == 0) v = warp_red_d(v);
    return v;
}

/* ==================== kernels ==================== */

/* planar int8 GEMV: y[r] = sum_i scale[r][i>>GSHIFT] * q[r][i] * x[i].
 * One warp per row; lane L covers elements 4L + 128t as an aligned char4
 * against an aligned float4 of x.  I is a multiple of 4 for every matrix in
 * this model (320..10240); the 4-element chunk never crosses a scale group. */
template <int GSHIFT, typename ST>
static __device__ __forceinline__ float gsc(const ST *s, int i);
template <> __device__ __forceinline__ float gsc<5, __half>(const __half *s, int i) { return __half2float(s[i]); }
template <> __device__ __forceinline__ float gsc<4, float>(const float *s, int i) { return s[i]; }

/* one lane's 16-element chunk: an int4 of quants against four float4 of x.
 * A 16-elem chunk at a 16-aligned offset never crosses a scale group for
 * either group size (32 or 16), so one scale covers GSHIFT==4 and two cover
 * GSHIFT==5 halves -- handled by scaling per 4-dot with the chunk's group. */
template <int GSHIFT, typename ST>
static __device__ __forceinline__ float dot16(const int8_t *w, const ST *sr,
                                              const float *x, int i) {
    const int4 wq = *(const int4 *)(w + i);
    const float4 x0 = *(const float4 *)(x + i);
    const float4 x1 = *(const float4 *)(x + i + 4);
    const float4 x2 = *(const float4 *)(x + i + 8);
    const float4 x3 = *(const float4 *)(x + i + 12);
    const char4 a = *(const char4 *)&wq.x, b = *(const char4 *)&wq.y;
    const char4 c = *(const char4 *)&wq.z, d = *(const char4 *)&wq.w;
    float dot = (float)a.x * x0.x + (float)a.y * x0.y + (float)a.z * x0.z + (float)a.w * x0.w;
    dot += (float)b.x * x1.x + (float)b.y * x1.y + (float)b.z * x1.z + (float)b.w * x1.w;
    dot += (float)c.x * x2.x + (float)c.y * x2.y + (float)c.z * x2.z + (float)c.w * x2.w;
    dot += (float)d.x * x3.x + (float)d.y * x3.y + (float)d.z * x3.z + (float)d.w * x3.w;
    return gsc<GSHIFT, ST>(sr, i >> GSHIFT) * dot;
}

template <int GSHIFT, typename ST>
static __global__ void k_gemv_i8(int O, int I, const int8_t *__restrict__ q,
                                 const ST *__restrict__ s,
                                 const float *__restrict__ x, float *__restrict__ y) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 4 + warp;
    if (row >= O) return;
    const int8_t *w = q + (size_t)row * I;
    const ST *sr = s + (size_t)row * (I >> GSHIFT);
    float acc = 0.f;
    for (int i = lane * 16; i < I; i += 512)
        acc += dot16<GSHIFT, ST>(w, sr, x, i);
    acc = warp_red_f(acc);
    if (lane == 0) y[row] = acc;
}

/* skinny variant: one 256-thread BLOCK per row (a warp per row starves the
 * SMs when O is 320/512/48 -- the hyper-connection down mats, QSA k/v,
 * GDN b/alpha). */
template <int GSHIFT, typename ST>
static __global__ void k_gemv_i8_wide(int I, const int8_t *__restrict__ q,
                                      const ST *__restrict__ s,
                                      const float *__restrict__ x, float *__restrict__ y) {
    const int row = blockIdx.x;
    const int8_t *w = q + (size_t)row * I;
    const ST *sr = s + (size_t)row * (I >> GSHIFT);
    float acc = 0.f;
    for (int i = threadIdx.x * 16; i < I; i += 16 * blockDim.x)
        acc += dot16<GSHIFT, ST>(w, sr, x, i);
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_red_f(acc);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        acc = warp_red_f(acc);
        if (lane == 0) y[row] = acc;
    }
}

/* f32 mats are all skinny here (inject [4,10240], GDN b/alpha [48,2560]):
 * one 256-thread block per row. */
static __global__ void k_gemv_f32(int I, const float *__restrict__ W,
                                  const float *__restrict__ x, float *__restrict__ y) {
    const int row = blockIdx.x;
    const float *w = W + (size_t)row * I;
    float acc = 0.f;
    for (int i = threadIdx.x; i < I; i += blockDim.x) acc += w[i] * x[i];
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_red_f(acc);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        acc = warp_red_f(acc);
        if (lane == 0) y[row] = acc;
    }
}

/* Prefill GEMM: one block owns one output row and up to G_BTILE tokens, so
 * each quant/scalar weight load feeds several independent activations. */
template <int GSHIFT, typename ST, bool ACT>
static __global__ void k_gemm_i8(int O, int I, const int8_t *__restrict__ q,
                                 const ST *__restrict__ s,
                                 const float *__restrict__ X, float *__restrict__ Y,
                                 int B, int ldy) {
    const int row = blockIdx.x, t0 = blockIdx.y * G_BTILE;
    if (row >= O || t0 >= B) return;
    const int8_t *w = q + (size_t)row * I;
    const ST *sr = s + (size_t)row * (I >> GSHIFT);
    float acc[G_BTILE] = {0.f};
    if constexpr (ACT) {
        /* Match k_gemv_i8_act exactly: four values per lane and scale before
         * accumulation.  Reassociating four dots under one scale changes the
         * target logits used by speculative verification. */
        for (int i = threadIdx.x * 4; i < I; i += blockDim.x * 4) {
            const char4 wv = *(const char4 *)(w + i);
            const float sc = gsc<GSHIFT, ST>(sr, i >> GSHIFT);
#pragma unroll
            for (int b = 0; b < G_BTILE; b++) {
                if (t0 + b >= B) break;
                const float4 xv = *(const float4 *)(X + (size_t)(t0 + b) * I + i);
                float dot = (float)wv.x * d_silu(xv.x / (float)G_HC);
                dot += (float)wv.y * d_silu(xv.y / (float)G_HC);
                dot += (float)wv.z * d_silu(xv.z / (float)G_HC);
                dot += (float)wv.w * d_silu(xv.w / (float)G_HC);
                acc[b] += sc * dot;
            }
        }
    } else {
        for (int i = threadIdx.x * 16; i < I; i += blockDim.x * 16) {
            const int4 wq = *(const int4 *)(w + i);
            const char4 wa = *(const char4 *)&wq.x, wb = *(const char4 *)&wq.y;
            const char4 wc = *(const char4 *)&wq.z, wd = *(const char4 *)&wq.w;
            const float sc = gsc<GSHIFT, ST>(sr, i >> GSHIFT);
#pragma unroll
            for (int b = 0; b < G_BTILE; b++) {
                if (t0 + b >= B) break;
                const float *xp = X + (size_t)(t0 + b) * I + i;
                const float4 xa = *(const float4 *)(xp), xb = *(const float4 *)(xp + 4);
                const float4 xc = *(const float4 *)(xp + 8), xd = *(const float4 *)(xp + 12);
                float dot = (float)wa.x*xa.x + (float)wa.y*xa.y +
                            (float)wa.z*xa.z + (float)wa.w*xa.w;
                dot += (float)wb.x*xb.x + (float)wb.y*xb.y +
                       (float)wb.z*xb.z + (float)wb.w*xb.w;
                dot += (float)wc.x*xc.x + (float)wc.y*xc.y +
                       (float)wc.z*xc.z + (float)wc.w*xc.w;
                dot += (float)wd.x*xd.x + (float)wd.y*xd.y +
                       (float)wd.z*xd.z + (float)wd.w*xd.w;
                acc[b] += sc * dot;
            }
        }
    }
    __shared__ float sh[G_BTILE][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) {
        acc[b] = warp_red_f(acc[b]);
        if (lane == 0) sh[b][warp] = acc[b];
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        for (int b = 0; b < G_BTILE && t0 + b < B; b++) {
            float sum = lane < nw ? sh[b][lane] : 0.f;
            sum = warp_red_f(sum);
            if (lane == 0) Y[(size_t)(t0 + b) * ldy + row] = sum;
        }
    }
}

/* ---- P9: shared-memory tiled twins of k_gemm_i8 for large prefill chunks.
 * Same lane -> element mapping, same per-lane expression, same reduction
 * order as k_gemm_i8 (so the bits are identical); the difference is that a
 * block computes several rows against one staged activation tile instead of
 * every row-block streaming the whole chunk from L2/DRAM. ---- */

/* wide rows (k_gemm_i8 with 32 threads/row): GT_ROWS warps, one row each,
 * against a 16-token x 512-element tile in shared memory. */
#define GT_ROWS 8
#define GT_KW   512
template <int GSHIFT, typename ST>
static __global__ void k_gemm_i8_tiled(int O, int I, const int8_t *__restrict__ q,
                                       const ST *__restrict__ s,
                                       const float *__restrict__ X, float *__restrict__ Y,
                                       int B, int ldy) {
    __shared__ __align__(16) float xs[G_BTILE * GT_KW];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * GT_ROWS + warp, t0 = blockIdx.y * G_BTILE;
    const int nb = B - t0 < G_BTILE ? B - t0 : G_BTILE;
    const bool live = row < O;
    const int8_t *w = live ? q + (size_t)row * I : q;
    const ST *sr = live ? s + (size_t)row * (I >> GSHIFT) : s;
    float acc[G_BTILE];
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) acc[b] = 0.f;
    for (int k0 = 0; k0 < I; k0 += GT_KW) {
        const int kw = I - k0 < GT_KW ? I - k0 : GT_KW;
        __syncthreads();
        for (int e = threadIdx.x * 4; e < nb * GT_KW; e += blockDim.x * 4) {
            const int b = e / GT_KW, j = e % GT_KW;
            if (j < kw) *(float4 *)(xs + e) = *(const float4 *)(X + (size_t)(t0 + b) * I + k0 + j);
        }
        __syncthreads();
        const int i = k0 + lane * 16;
        if (live && lane * 16 < kw) {
            {
                const int4 wq = *(const int4 *)(w + i);
                const char4 wa = *(const char4 *)&wq.x, wb = *(const char4 *)&wq.y;
                const char4 wc = *(const char4 *)&wq.z, wd = *(const char4 *)&wq.w;
                const float sc = gsc<GSHIFT, ST>(sr, i >> GSHIFT);
#pragma unroll
                for (int b = 0; b < G_BTILE; b++) {
                    if (b >= nb) break;
                    const float *xp = xs + b * GT_KW + lane * 16;
                    const float4 xa = *(const float4 *)(xp), xb = *(const float4 *)(xp + 4);
                    const float4 xc = *(const float4 *)(xp + 8), xd = *(const float4 *)(xp + 12);
                    float dot = (float)wa.x*xa.x + (float)wa.y*xa.y +
                                (float)wa.z*xa.z + (float)wa.w*xa.w;
                    dot += (float)wb.x*xb.x + (float)wb.y*xb.y +
                           (float)wb.z*xb.z + (float)wb.w*xb.w;
                    dot += (float)wc.x*xc.x + (float)wc.y*xc.y +
                           (float)wc.z*xc.z + (float)wc.w*xc.w;
                    dot += (float)wd.x*xd.x + (float)wd.y*xd.y +
                           (float)wd.z*xd.z + (float)wd.w*xd.w;
                    acc[b] += sc * dot;
                }
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) {
        const float v = warp_red_f(acc[b]);
        if (lane == 0 && b < nb) Y[(size_t)(t0 + b) * ldy + row] = v;
    }
}

/* ACT variant of the wide-row tile: k_gemm_i8<ACT> walks 4 elements per lane
 * per step (i = tid*4 + step*128, 32 threads) with silu(x/4) on the load. */
template <int GSHIFT, typename ST>
static __global__ void k_gemm_i8_tiled_act(int O, int I, const int8_t *__restrict__ q,
                                           const ST *__restrict__ s,
                                           const float *__restrict__ X, float *__restrict__ Y,
                                           int B, int ldy) {
    __shared__ __align__(16) float xs[G_BTILE * GT_KW];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * GT_ROWS + warp, t0 = blockIdx.y * G_BTILE;
    const int nb = B - t0 < G_BTILE ? B - t0 : G_BTILE;
    const bool live = row < O;
    const int8_t *w = live ? q + (size_t)row * I : q;
    const ST *sr = live ? s + (size_t)row * (I >> GSHIFT) : s;
    float acc[G_BTILE];
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) acc[b] = 0.f;
    for (int k0 = 0; k0 < I; k0 += GT_KW) {
        const int kw = I - k0 < GT_KW ? I - k0 : GT_KW;
        __syncthreads();
        for (int e = threadIdx.x * 4; e < nb * GT_KW; e += blockDim.x * 4) {
            const int b = e / GT_KW, j = e % GT_KW;
            if (j < kw) *(float4 *)(xs + e) = *(const float4 *)(X + (size_t)(t0 + b) * I + k0 + j);
        }
        __syncthreads();
        if (!live) continue;
        for (int j = lane * 4; j < kw; j += 128) {
            const int i = k0 + j;
            const char4 wv = *(const char4 *)(w + i);
            const float sc = gsc<GSHIFT, ST>(sr, i >> GSHIFT);
#pragma unroll
            for (int b = 0; b < G_BTILE; b++) {
                if (b >= nb) break;
                const float4 xv = *(const float4 *)(xs + b * GT_KW + j);
                float dot = (float)wv.x * d_silu(xv.x / (float)G_HC);
                dot += (float)wv.y * d_silu(xv.y / (float)G_HC);
                dot += (float)wv.z * d_silu(xv.z / (float)G_HC);
                dot += (float)wv.w * d_silu(xv.w / (float)G_HC);
                acc[b] += sc * dot;
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) {
        const float v = warp_red_f(acc[b]);
        if (lane == 0 && b < nb) Y[(size_t)(t0 + b) * ldy + row] = v;
    }
}

/* skinny rows (k_gemm_i8 with 256 threads/row, i = tid*16 + step*4096):
 * one 256-thread block, GTS_ROWS rows in sequence, GTS_T tokens against a
 * GTS_T x 4096 tile (64 KB dynamic shared memory). */
#define GTS_ROWS 8
#define GTS_T    4
#define GTS_KW   4096
template <int GSHIFT, typename ST>
static __global__ void k_gemm_i8_tiled_skinny(int O, int I, const int8_t *__restrict__ q,
                                              const ST *__restrict__ s,
                                              const float *__restrict__ X, float *__restrict__ Y,
                                              int B, int ldy) {
    extern __shared__ __align__(16) float xss[];
    const int r0 = blockIdx.x * GTS_ROWS, t0 = blockIdx.y * GTS_T;
    const int nb = B - t0 < GTS_T ? B - t0 : GTS_T;
    const int nr = O - r0 < GTS_ROWS ? O - r0 : GTS_ROWS;
    float acc[GTS_ROWS][GTS_T];
#pragma unroll
    for (int r = 0; r < GTS_ROWS; r++)
#pragma unroll
        for (int b = 0; b < GTS_T; b++) acc[r][b] = 0.f;
    for (int k0 = 0; k0 < I; k0 += GTS_KW) {
        const int kw = I - k0 < GTS_KW ? I - k0 : GTS_KW;
        __syncthreads();
        for (int e = threadIdx.x * 4; e < nb * GTS_KW; e += blockDim.x * 4) {
            const int b = e / GTS_KW, j = e % GTS_KW;
            if (j < kw) *(float4 *)(xss + e) = *(const float4 *)(X + (size_t)(t0 + b) * I + k0 + j);
        }
        __syncthreads();
        const int j = threadIdx.x * 16;
        if (j < kw) {
            const int i = k0 + j;
            for (int r = 0; r < nr; r++) {
                const int8_t *w = q + (size_t)(r0 + r) * I;
                const ST *sr = s + (size_t)(r0 + r) * (I >> GSHIFT);
                const int4 wq = *(const int4 *)(w + i);
                const char4 wa = *(const char4 *)&wq.x, wb = *(const char4 *)&wq.y;
                const char4 wc = *(const char4 *)&wq.z, wd = *(const char4 *)&wq.w;
                const float sc = gsc<GSHIFT, ST>(sr, i >> GSHIFT);
#pragma unroll
                for (int b = 0; b < GTS_T; b++) {
                    if (b >= nb) break;
                    const float *xp = xss + b * GTS_KW + j;
                    const float4 xa = *(const float4 *)(xp), xb = *(const float4 *)(xp + 4);
                    const float4 xc = *(const float4 *)(xp + 8), xd = *(const float4 *)(xp + 12);
                    float dot = (float)wa.x*xa.x + (float)wa.y*xa.y +
                                (float)wa.z*xa.z + (float)wa.w*xa.w;
                    dot += (float)wb.x*xb.x + (float)wb.y*xb.y +
                           (float)wb.z*xb.z + (float)wb.w*xb.w;
                    dot += (float)wc.x*xc.x + (float)wc.y*xc.y +
                           (float)wc.z*xc.z + (float)wc.w*xc.w;
                    dot += (float)wd.x*xd.x + (float)wd.y*xd.y +
                           (float)wd.z*xd.z + (float)wd.w*xd.w;
                    acc[r][b] += sc * dot;
                }
            }
        }
    }
    /* k_gemm_i8's block reduction: warp sums, then warp 0 adds the 8
     * partials in lane order */
    __shared__ float sh[GTS_ROWS][GTS_T][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = 0; r < GTS_ROWS; r++)
#pragma unroll
        for (int b = 0; b < GTS_T; b++) {
            const float v = warp_red_f(acc[r][b]);
            if (lane == 0) sh[r][b][warp] = v;
        }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        for (int r = 0; r < nr; r++)
            for (int b = 0; b < nb; b++) {
                float sum = lane < nw ? sh[r][b][lane] : 0.f;
                sum = warp_red_f(sum);
                if (lane == 0) Y[(size_t)(t0 + b) * ldy + r0 + r] = sum;
            }
    }
}

/* ---- P9 QSA attention for a prefill chunk ----
 * Mirrors qsa_core_x: per head rms_norm(q) * qn, rope on the first n_rot
 * dims; per kv head the same for k, then q8 rows into the cache; then for
 * every (token, head) the walk over the selected cells: scores in the CPU's
 * serial per-cell order, softmax, weighted V in the CPU's per-element cell
 * order, times sigmoid(gate).  Transcendentals and the parallel reductions
 * are the accepted bit-close class of the GPU dense path. */
static __device__ __forceinline__ float d_sigmoid_qsa(float v) {
    return v >= 0.f ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v));
}
/* block = (token, head); 256 threads == hd.  in: [B][H][2*hd] interleaved
 * (q | gate) rows when stride2 != 0, or [B][KV][hd] plain rows. */
/* mp3: P6.3 IMRoPE positions [B][3] (t,y,x) of the chunk's cells, or NULL for
 * the sequential position pos0 + t.  With mp3 the CPU's interleaved section
 * rule picks the position per pair: y if d%3==1 && d<s1, x if d%3==2 &&
 * d<s2, else t (rope_head_m in qwen38.c). */
static __global__ void k_qsa_norm_rope(const float *__restrict__ in, int in_stride_head,
                                       int heads, const float *__restrict__ wn,
                                       float *__restrict__ out, int hd, int n_rot,
                                       float theta, float eps, int pos0,
                                       const int *__restrict__ mp3, int s1, int s2) {
    extern __shared__ float ys[];
    const int t = blockIdx.x / heads, h = blockIdx.x % heads, d = threadIdx.x;
    const float *x = in + ((size_t)t * heads + h) * in_stride_head;
    const float v = d < hd ? x[d] : 0.f;
    const double ms = block_red_d((double)v * v);
    __shared__ float rr;
    if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / hd) + eps);
    __syncthreads();
    if (d < hd) ys[d] = v * rr * wn[d];
    __syncthreads();
    if (d >= hd) return;
    float *o = out + ((size_t)t * heads + h) * hd;
    const int hh = n_rot / 2;
    int pos = pos0 + t;
    if (mp3) {
        pos = mp3[3 * t];
        if (d % 3 == 1 && d < s1) pos = mp3[3 * t + 1];
        else if (d % 3 == 2 && d < s2) pos = mp3[3 * t + 2];
    }
    if (d < hh) {
        const float inv = powf(theta, -2.0f * d / n_rot);
        const float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        const float a = ys[d], b2 = ys[d + hh];
        o[d] = a * cs - b2 * sn;
        o[d + hh] = b2 * cs + a * sn;
    } else if (d >= n_rot) {
        o[d] = ys[d];
    }
}
/* q8 rows: block = (token, kvh), 256 threads == hd, warps == 32-groups */
static __global__ void k_qsa_kv_q8(const float *__restrict__ k, const float *__restrict__ v,
                                   int KV, int hd, int max_t, int pos0,
                                   int8_t *__restrict__ K8, float *__restrict__ K8s,
                                   int8_t *__restrict__ V8, float *__restrict__ V8s) {
    const int t = blockIdx.x / KV, kvh = blockIdx.x % KV, d = threadIdx.x;
    if (d >= hd) return;
    const int ns = hd / 32, g = d >> 5, lane = d & 31;
    const size_t row = (size_t)kvh * max_t + pos0 + t;
    const float *src[2] = { k + ((size_t)t * KV + kvh) * hd, v + ((size_t)t * KV + kvh) * hd };
    int8_t *dq[2] = { K8 + row * hd, V8 + row * hd };
    float *ds[2] = { K8s + row * ns, V8s + row * ns };
#pragma unroll
    for (int w = 0; w < 2; w++) {
        const float xv = src[w][d];
        float mx = fabsf(xv);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        const float sc = mx > 0.f ? mx / 127.f : 0.f, inv = mx > 0.f ? 127.f / mx : 0.f;
        dq[w][d] = (int8_t)rintf(xv * inv);
        if (lane == 0) ds[w][g] = sc;
    }
}
#define QSA_WALK_MAX 4096
/* block = (token, head), 256 threads == hd; smem: q[hd] + scores[nwalk] */
static __global__ void k_qsa_attn(const float *__restrict__ q, const float *__restrict__ qf,
                                  const int8_t *__restrict__ K8, const float *__restrict__ K8s,
                                  const int8_t *__restrict__ V8, const float *__restrict__ V8s,
                                  int H, int KV, int hd, int max_t, int pos0,
                                  const int *__restrict__ nsel, const int *__restrict__ sel,
                                  int sel_cap, float scale, float *__restrict__ ctx) {
    extern __shared__ float sm[];
    float *qs = sm, *sc = sm + hd;
    const int t = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x;
    const int kvh = h / (H / KV), ns = hd / 32, pos = pos0 + t;
    const int ns_t = nsel[t];
    const int nwalk = ns_t < 0 ? pos + 1 : ns_t;
    const int *sl = sel + (size_t)t * sel_cap;
    if (d < hd) qs[d] = q[((size_t)t * H + h) * hd + d];
    __syncthreads();
    /* scores: one thread per cell, the CPU's serial group order */
    float mloc = -1e30f;
    for (int j = d; j < nwalk; j += blockDim.x) {
        const int cell = ns_t < 0 ? j : sl[j];
        const size_t row = (size_t)kvh * max_t + cell;
        const int8_t *kr = K8 + row * hd;
        const float *ks = K8s + row * ns;
        float acc = 0.f;
        for (int g = 0; g < ns; g++) {
            float p = 0.f;
            for (int e = 0; e < 32; e++) p += qs[g * 32 + e] * (float)kr[g * 32 + e];
            acc += p * ks[g];
        }
        const float s = acc * scale;
        sc[j] = s;
        mloc = fmaxf(mloc, s);
    }
    /* block max */
    __shared__ float red[32];
    const int lane = d & 31, warp = d >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, off));
    if (lane == 0) red[warp] = mloc;
    __syncthreads();
    float m = red[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); w++) m = fmaxf(m, red[w]);
    __syncthreads();
    float sloc = 0.f;
    for (int j = d; j < nwalk; j += blockDim.x) { const float e = expf(sc[j] - m); sc[j] = e; sloc += e; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sloc += __shfl_xor_sync(0xffffffffu, sloc, off);
    if (lane == 0) red[warp] = sloc;
    __syncthreads();
    float ssum = 0.f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) ssum += red[w];
    __syncthreads();
    for (int j = d; j < nwalk; j += blockDim.x) sc[j] /= ssum;
    __syncthreads();
    if (d >= hd) return;
    /* weighted V, the CPU's per-element cell order */
    const int g = d >> 5;
    float cx = 0.f;
    for (int j = 0; j < nwalk; j++) {
        const int cell = ns_t < 0 ? j : sl[j];
        const size_t row = (size_t)kvh * max_t + cell;
        const float as = sc[j] * V8s[row * ns + g];
        cx += as * (float)V8[row * hd + d];
    }
    const float gate = qf[((size_t)t * H + h) * 2 * hd + hd + d];
    ctx[((size_t)t * H + h) * hd + d] = cx * d_sigmoid_qsa(gate);
}

/* indexer block scores for a chunk: score[t][b] = sum_h relu(q_th . k_b),
 * one thread per (token, block) with the CPU's serial j loop per head and
 * the head order preserved (identical bits).  A block of threads owns a
 * 64-block key tile in shared memory and walks every token against it. */
#define IDX_TILE 64
static __global__ void k_idx_score(const float *__restrict__ qi, const float *__restrict__ ibk,
                                   const int *__restrict__ nfb, int nfb_max, int B,
                                   int nh, int id, float *__restrict__ score) {
    extern __shared__ float kt[];               /* [IDX_TILE][id] */
    const int b0 = blockIdx.x * IDX_TILE;
    for (int e = threadIdx.x; e < IDX_TILE * id; e += blockDim.x) {
        const int bb = b0 + e / id;
        kt[e] = bb < nfb_max ? ibk[(size_t)bb * id + e % id] : 0.f;
    }
    __syncthreads();
    const int bl = threadIdx.x % IDX_TILE, b = b0 + bl;
    for (int t = threadIdx.x / IDX_TILE; t < B; t += blockDim.x / IDX_TILE) {
        if (b >= nfb[t]) continue;
        const float *kb = kt + bl * id;
        float sacc = 0.f;
        for (int h = 0; h < nh; h++) {
            const float *qh = qi + ((size_t)t * nh + h) * id;
            float d = 0.f;
            for (int j = 0; j < id; j++) d += qh[j] * kb[j];
            if (d > 0.f) sacc += d;
        }
        score[(size_t)t * nfb_max + b] = sacc;
    }
}

static __global__ void k_gemm_f32(int O, int I, const float *__restrict__ W,
                                  const float *__restrict__ X, float *__restrict__ Y,
                                  int B, int ldy) {
    const int row = blockIdx.x, t0 = blockIdx.y * G_BTILE;
    if (row >= O || t0 >= B) return;
    const float *w = W + (size_t)row * I;
    float acc[G_BTILE] = {0.f};
    for (int i = threadIdx.x; i < I; i += blockDim.x) {
        const float wi = w[i];
#pragma unroll
        for (int b = 0; b < G_BTILE; b++) {
            if (t0 + b >= B) break;
            acc[b] += wi * X[(size_t)(t0 + b) * I + i];
        }
    }
    __shared__ float sh[G_BTILE][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int b = 0; b < G_BTILE; b++) {
        acc[b] = warp_red_f(acc[b]);
        if (lane == 0) sh[b][warp] = acc[b];
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        for (int b = 0; b < G_BTILE && t0 + b < B; b++) {
            float sum = lane < nw ? sh[b][lane] : 0.f;
            sum = warp_red_f(sum);
            if (lane == 0) Y[(size_t)(t0 + b) * ldy + row] = sum;
        }
    }
}

static __global__ void k_bcast_res(float *__restrict__ res, const float *__restrict__ x, int D) {
    const int s = blockIdx.y;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < D; j += gridDim.x * blockDim.x)
        res[s * D + j] = x[j];
}

/* xn[s,:] = rmsnorm(res[s,:]) * wnorm[s,:]  (double mean-square, like the CPU) */
static __global__ void k_hc_norm(const float *__restrict__ res, const float *__restrict__ wn,
                                 float *__restrict__ xn, int D, float eps) {
    const int s = blockIdx.x;
    const float *r = res + (size_t)s * D;
    double ms = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { double v = r[j]; ms += v * v; }
    ms = block_red_d(ms);
    __shared__ float rr;
    if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / D) + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        xn[s * D + j] = r[j] * rr * wn[s * D + j];
}

static __global__ void k_hc_norm_batch(const float *__restrict__ res,
                                       const float *__restrict__ wn,
                                       float *__restrict__ xn, int D, float eps) {
    const int t = blockIdx.x / G_HC, s = blockIdx.x % G_HC;
    const float *r = res + ((size_t)t * G_HC + s) * D;
    double ms = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { double v = r[j]; ms += v * v; }
    ms = block_red_d(ms);
    __shared__ float rr;
    if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / D) + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        xn[((size_t)t * G_HC + s) * D + j] = r[j] * rr * wn[(size_t)s * D + j];
}

static __global__ void k_silu_div(float *__restrict__ lo, int n, float div) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lo[i] = d_silu(lo[i] / div);
}

/* hyper-connection up-projection with the silu(lo/4) fused into the x load:
 * the activation value is computed with the identical formula, so the result
 * is bit-identical to k_silu_div + plain GEMV, minus one launch.  I is 320
 * (the hc rank), so a warp covers a whole row in one strided pass. */
template <int GSHIFT, typename ST>
static __global__ void k_gemv_i8_act(int O, int I, const int8_t *__restrict__ q,
                                     const ST *__restrict__ s,
                                     const float *__restrict__ x, float *__restrict__ y) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 4 + warp;
    if (row >= O) return;
    const int8_t *w = q + (size_t)row * I;
    const ST *sr = s + (size_t)row * (I >> GSHIFT);
    float acc = 0.f;
    for (int i = lane * 4; i < I; i += 128) {
        const char4 wv = *(const char4 *)(w + i);
        const float4 xv = *(const float4 *)(x + i);
        float dot = (float)wv.x * d_silu(xv.x / (float)G_HC);
        dot += (float)wv.y * d_silu(xv.y / (float)G_HC);
        dot += (float)wv.z * d_silu(xv.z / (float)G_HC);
        dot += (float)wv.w * d_silu(xv.w / (float)G_HC);
        acc += gsc<GSHIFT, ST>(sr, i >> GSHIFT) * dot;
    }
    acc = warp_red_f(acc);
    if (lane == 0) y[row] = acc;
}

static __global__ void k_hc_out(const float *__restrict__ xn, const float *__restrict__ gate,
                                float *__restrict__ mixed, int D) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < G_HC; s++) acc += xn[s * D + j] * d_sigmoid(gate[s * D + j]);
    mixed[j] = acc / (float)G_HC;
}

/* P10 D3a: k_hc_out plus the inject GEMV ([G_HC, W] f32 over xn) in the
 * same launch -- blocks past the D range each run one k_gemv_f32 row with
 * the identical 256-thread mapping and reduction, so the values are
 * bit-identical to the separate launch (96 x ~17 us per token). */
static __global__ void k_hc_out_inj(const float *__restrict__ xn, const float *__restrict__ gate,
                                    float *__restrict__ mixed, int D, int nb,
                                    const float *__restrict__ Winj, float *__restrict__ inj, int W) {
    if ((int)blockIdx.x < nb) {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= D) return;
        float acc = 0.f;
#pragma unroll
        for (int s = 0; s < G_HC; s++) acc += xn[s * D + j] * d_sigmoid(gate[s * D + j]);
        mixed[j] = acc / (float)G_HC;
        return;
    }
    const int row = blockIdx.x - nb;
    const float *w = Winj + (size_t)row * W;
    float acc = 0.f;
    for (int i = threadIdx.x; i < W; i += blockDim.x) acc += w[i] * xn[i];
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    acc = warp_red_f(acc);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
        acc = warp_red_f(acc);
        if (lane == 0) inj[row] = acc;
    }
}

static __global__ void k_hc_out_batch(const float *__restrict__ xn,
                                      const float *__restrict__ gate,
                                      float *__restrict__ mixed, int D, int B) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B * D) return;
    const int t = p / D, j = p % D;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < G_HC; s++) {
        const size_t q = ((size_t)t * G_HC + s) * D + j;
        acc += xn[q] * d_sigmoid(gate[q]);
    }
    mixed[p] = acc / (float)G_HC;
}

/* h = silu(g) * u, full-precision expf (matches the tier's chain kernel) */
static __global__ void k_silu_mul(const float *__restrict__ g, const float *__restrict__ u,
                                  float *__restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    h[i] = gv / (1.0f + expf(-gv)) * u[i];
}

static __global__ void k_combine(float *__restrict__ res, const float *__restrict__ blk,
                                 const float *__restrict__ inj, int D) {
    const int s = blockIdx.y;
    const float w = 2.f * d_sigmoid(inj[s] / (float)G_HC);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < D; j += gridDim.x * blockDim.x)
        res[s * D + j] += blk[j] * w;
}

static __global__ void k_combine_batch(float *__restrict__ res,
                                       const float *__restrict__ blk,
                                       const float *__restrict__ inj, int D, int B) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B * G_HC * D) return;
    const int t = p / (G_HC * D), s = p / D % G_HC, j = p % D;
    const float w = 2.f * d_sigmoid(inj[(size_t)t * G_HC + s] / (float)G_HC);
    res[p] += blk[(size_t)t * D + j] * w;
}

/* causal depthwise conv over cdim channels, kernel convk, ring state resident */
static __global__ void k_gdn_conv(const float *__restrict__ qkv, const float *__restrict__ w,
                                  float *__restrict__ ring, float *__restrict__ conv,
                                  int cdim, int convk) {
    const int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= cdim) return;
    const float *wc = w + (size_t)cc * convk;
    float *rg = ring + (size_t)cc * (convk - 1);
    float acc = 0.f;
    for (int k = 0; k < convk - 1; k++) acc += wc[k] * rg[k];
    acc += wc[convk - 1] * qkv[cc];
    conv[cc] = d_silu(acc);
    for (int k = 0; k < convk - 2; k++) rg[k] = rg[k + 1];
    rg[convk - 2] = qkv[cc];
}

/* Fused GDN core, one block per v-head (thread = v-dim):
 *   1. split (tiled k-head h%vk) + L2-normalize q/k into shared memory
 *   2. beta = sigmoid(b[h]), exp(g) = exp(a[h]*softplus(alpha[h]+dt[h]))
 *   3. gated delta rule, both state sweeps against a SHARED-MEMORY copy of
 *      the 128x128 state (one global read + one write instead of two of
 *      each), per-(head,v2) accumulation order identical to the CPU loops
 *   4. per-head RMS norm (double accum) * gamma * sigmoid(z)
 * Dynamic shared: kd*vd state + 2*kd normalized q/k floats (~66 KB, opt-in
 * above the 48 KB default via cudaFuncSetAttribute at init). */
static __global__ void k_gdn_core(float *__restrict__ rec, const float *__restrict__ conv,
                                  const float *__restrict__ ba, const float *__restrict__ a,
                                  const float *__restrict__ dt, const float *__restrict__ z,
                                  const float *__restrict__ gamma, float *__restrict__ outr,
                                  int vh, int vk, int kd, int vd, int keytot,
                                  float eps, float scale) {
    const int h = blockIdx.x;
    const int kh = h % vk;
    const float *qi = conv + (size_t)kh * kd;
    const float *ki = conv + keytot + (size_t)kh * kd;
    extern __shared__ float sh[];              /* S[kd*vd] | k[kd] | q[kd] */
    float *S = sh, *ks = sh + kd * vd, *qs = ks + kd;
    /* L2 norms of the head's raw q/k (double, splitnorm order) */
    double sq = 0.0, sk = 0.0;
    for (int d = threadIdx.x; d < kd; d += blockDim.x) {
        double av = qi[d], bv = ki[d];
        sq += av * av; sk += bv * bv;
    }
    __shared__ double shq[32], shk[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    sq = warp_red_d(sq); sk = warp_red_d(sk);
    if (lane == 0) { shq[warp] = sq; shk[warp] = sk; }
    __syncthreads();
    __shared__ float fq, fk, beta_s, eg_s;
    if (threadIdx.x == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        double tq = 0.0, tk = 0.0;
        for (int wi = 0; wi < nw; wi++) { tq += shq[wi]; tk += shk[wi]; }
        float nq = fmaxf(sqrtf((float)tq), eps), nk = fmaxf(sqrtf((float)tk), eps);
        fq = scale / nq; fk = 1.f / nk;
        beta_s = d_sigmoid(ba[h]);
        eg_s = expf(a[h] * d_softplus(ba[vh + h] + dt[h]));
    }
    __syncthreads();
    for (int d = threadIdx.x; d < kd; d += blockDim.x) {
        qs[d] = qi[d] * fq;
        ks[d] = ki[d] * fk;
    }
    /* state to shared */
    float *Sg = rec + (size_t)h * kd * vd;
    for (int i = threadIdx.x; i < kd * vd; i += blockDim.x) S[i] = Sg[i];
    __syncthreads();
    const int v2 = threadIdx.x;
    float o = 0.f;
    if (v2 < vd) {
        const float egh = eg_s, beta = beta_s;
        float kvl = 0.f;
        for (int kk = 0; kk < kd; kk++) {
            float sv = S[kk * vd + v2] * egh;
            S[kk * vd + v2] = sv;
            kvl += ks[kk] * sv;
        }
        const float dl = (conv[(size_t)2 * keytot + (size_t)h * vd + v2] - kvl) * beta;
        for (int kk = 0; kk < kd; kk++) {
            float sv = S[kk * vd + v2] + ks[kk] * dl;
            S[kk * vd + v2] = sv;
            o += qs[kk] * sv;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kd * vd; i += blockDim.x) Sg[i] = S[i];
    /* head RMS norm * gamma * sigmoid(z) */
    double ms = (v2 < vd) ? (double)o * o : 0.0;
    ms = block_red_d(ms);
    __shared__ float rr;
    if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / vd) + eps);
    __syncthreads();
    if (v2 < vd)
        outr[(size_t)h * vd + v2] = o * rr * gamma[v2] * d_sigmoid(z[(size_t)h * vd + v2]);
}

/* ---- P9: chunk-looped twins of k_gdn_conv / k_gdn_core.  The recurrence is
 * sequential by nature, so the loop over the chunk's tokens moves inside the
 * kernel: the conv ring lives in registers and the 128x128 state stays in
 * shared memory for all B tokens instead of round-tripping global memory
 * and paying a launch per token.  Per-token arithmetic is the same code. */
static __global__ void k_gdn_conv_chunk(const float *__restrict__ qkv, const float *__restrict__ w,
                                        float *__restrict__ ring, float *__restrict__ conv,
                                        int cdim, int convk, int B) {
    const int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= cdim) return;
    const float *wc = w + (size_t)cc * convk;
    float *rg = ring + (size_t)cc * (convk - 1);
    float r[8];
    for (int k = 0; k < convk - 1; k++) r[k] = rg[k];
    for (int t = 0; t < B; t++) {
        const float x = qkv[(size_t)t * cdim + cc];
        float acc = 0.f;
        for (int k = 0; k < convk - 1; k++) acc += wc[k] * r[k];
        acc += wc[convk - 1] * x;
        conv[(size_t)t * cdim + cc] = d_silu(acc);
        for (int k = 0; k < convk - 2; k++) r[k] = r[k + 1];
        r[convk - 2] = x;
    }
    for (int k = 0; k < convk - 1; k++) rg[k] = r[k];
}

static __global__ void k_gdn_core_chunk(float *__restrict__ rec, const float *__restrict__ conv,
                                        const float *__restrict__ ba, const float *__restrict__ a,
                                        const float *__restrict__ dt, const float *__restrict__ z,
                                        const float *__restrict__ gamma, float *__restrict__ outr,
                                        int vh, int vk, int kd, int vd, int keytot, int cdim,
                                        float eps, float scale, int B) {
    const int h = blockIdx.x;
    const int kh = h % vk;
    extern __shared__ float sh[];              /* S[kd*vd] | k[kd] | q[kd] */
    float *S = sh, *ks = sh + kd * vd, *qs = ks + kd;
    __shared__ double shq[32], shk[32];
    __shared__ float fq, fk, beta_s, eg_s, rr;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int v2 = threadIdx.x;
    float *Sg = rec + (size_t)h * kd * vd;
    for (int i = threadIdx.x; i < kd * vd; i += blockDim.x) S[i] = Sg[i];
    __syncthreads();
    for (int t = 0; t < B; t++) {
        const float *cv = conv + (size_t)t * cdim;
        const float *qi = cv + (size_t)kh * kd;
        const float *ki = cv + keytot + (size_t)kh * kd;
        const float *bat = ba + (size_t)t * 2 * vh;
        double sq = 0.0, sk = 0.0;
        for (int d = threadIdx.x; d < kd; d += blockDim.x) {
            double av = qi[d], bv = ki[d];
            sq += av * av; sk += bv * bv;
        }
        sq = warp_red_d(sq); sk = warp_red_d(sk);
        if (lane == 0) { shq[warp] = sq; shk[warp] = sk; }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int nw = (blockDim.x + 31) >> 5;
            double tq = 0.0, tk = 0.0;
            for (int wi = 0; wi < nw; wi++) { tq += shq[wi]; tk += shk[wi]; }
            float nq = fmaxf(sqrtf((float)tq), eps), nk = fmaxf(sqrtf((float)tk), eps);
            fq = scale / nq; fk = 1.f / nk;
            beta_s = d_sigmoid(bat[h]);
            eg_s = expf(a[h] * d_softplus(bat[vh + h] + dt[h]));
        }
        __syncthreads();
        for (int d = threadIdx.x; d < kd; d += blockDim.x) {
            qs[d] = qi[d] * fq;
            ks[d] = ki[d] * fk;
        }
        __syncthreads();
        float o = 0.f;
        if (v2 < vd) {
            const float egh = eg_s, beta = beta_s;
            float kvl = 0.f;
            for (int kk = 0; kk < kd; kk++) {
                float sv = S[kk * vd + v2] * egh;
                S[kk * vd + v2] = sv;
                kvl += ks[kk] * sv;
            }
            const float dl = (cv[(size_t)2 * keytot + (size_t)h * vd + v2] - kvl) * beta;
            for (int kk = 0; kk < kd; kk++) {
                float sv = S[kk * vd + v2] + ks[kk] * dl;
                S[kk * vd + v2] = sv;
                o += qs[kk] * sv;
            }
        }
        double ms = (v2 < vd) ? (double)o * o : 0.0;
        ms = block_red_d(ms);
        if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / vd) + eps);
        __syncthreads();
        if (v2 < vd)
            outr[(size_t)t * (size_t)vh * vd + (size_t)h * vd + v2] =
                o * rr * gamma[v2] * d_sigmoid(z[(size_t)t * (size_t)vh * vd + (size_t)h * vd + v2]);
        __syncthreads();
    }
    for (int i = threadIdx.x; i < kd * vd; i += blockDim.x) Sg[i] = S[i];
}

/* ---- P10 P8: the GDN core with GDN_SPLIT one-warp blocks per head.
 * k_gdn_core keeps the 128x128 state of a head in one 128-thread block, so
 * only vh=48 blocks exist on 84 SMs.  Here each block owns vd/GDN_SPLIT
 * state columns (one v-dim per lane) and the per-head RMS norm moves to a
 * second kernel.  Arithmetic is the original's: the q/k L2 norms are summed
 * as four warp partials added in order (that is exactly what block_red_d
 * did with 128 threads), the sweeps are per (head, v-dim), and k_gdn_norm
 * reduces 128 pre-norm values with block_red_d again -- bit-identical. */
#define GDN_SPLIT 4
static __global__ void k_gdn_sweep_chunk(float *__restrict__ rec, const float *__restrict__ conv,
                                         const float *__restrict__ ba, const float *__restrict__ a,
                                         const float *__restrict__ dt, float *__restrict__ obuf,
                                         int vh, int vk, int kd, int vd, int keytot, int cdim,
                                         float eps, float scale, int B) {
    const int h = blockIdx.x / GDN_SPLIT, part = blockIdx.x % GDN_SPLIT;
    const int vs = vd / GDN_SPLIT, v0 = part * vs;
    const int kh = h % vk, lane = threadIdx.x;
    extern __shared__ float sh[];              /* S[kd*vs] | k[kd] | q[kd] */
    float *S = sh, *ks = sh + kd * vs, *qs = ks + kd;
    __shared__ double shq[4], shk[4];
    __shared__ float fq, fk, beta_s, eg_s;
    float *Sg = rec + (size_t)h * kd * vd;
    for (int i = lane; i < kd * vs; i += 32) S[i] = Sg[(size_t)(i / vs) * vd + v0 + (i % vs)];
    __syncthreads();
    for (int t = 0; t < B; t++) {
        const float *cv = conv + (size_t)t * cdim;
        const float *qi = cv + (size_t)kh * kd;
        const float *ki = cv + keytot + (size_t)kh * kd;
        const float *bat = ba + (size_t)t * 2 * vh;
        for (int w = 0; w < kd / 32; w++) {
            double av = qi[w * 32 + lane], bv = ki[w * 32 + lane];
            double pq = warp_red_d(av * av), pk = warp_red_d(bv * bv);
            if (lane == 0) { shq[w] = pq; shk[w] = pk; }
        }
        __syncthreads();
        if (lane == 0) {
            double tq = 0.0, tk = 0.0;
            for (int wi = 0; wi < kd / 32; wi++) { tq += shq[wi]; tk += shk[wi]; }
            float nq = fmaxf(sqrtf((float)tq), eps), nk = fmaxf(sqrtf((float)tk), eps);
            fq = scale / nq; fk = 1.f / nk;
            beta_s = d_sigmoid(bat[h]);
            eg_s = expf(a[h] * d_softplus(bat[vh + h] + dt[h]));
        }
        __syncthreads();
        for (int d = lane; d < kd; d += 32) { qs[d] = qi[d] * fq; ks[d] = ki[d] * fk; }
        __syncthreads();
        const int v2 = v0 + lane;
        float o = 0.f;
        {
            const float egh = eg_s, beta = beta_s;
            float kvl = 0.f;
            for (int kk = 0; kk < kd; kk++) {
                float sv = S[kk * vs + lane] * egh;
                S[kk * vs + lane] = sv;
                kvl += ks[kk] * sv;
            }
            const float dl = (cv[(size_t)2 * keytot + (size_t)h * vd + v2] - kvl) * beta;
            for (int kk = 0; kk < kd; kk++) {
                float sv = S[kk * vs + lane] + ks[kk] * dl;
                S[kk * vs + lane] = sv;
                o += qs[kk] * sv;
            }
        }
        obuf[(size_t)t * (size_t)vh * vd + (size_t)h * vd + v2] = o;
        __syncthreads();
    }
    for (int i = lane; i < kd * vs; i += 32) Sg[(size_t)(i / vs) * vd + v0 + (i % vs)] = S[i];
}
/* block = (token, head), 128 threads == vd */
static __global__ void k_gdn_norm(const float *__restrict__ obuf, const float *__restrict__ z,
                                  const float *__restrict__ gamma, float *__restrict__ outr,
                                  int vh, int vd, float eps) {
    const int t = blockIdx.x / vh, h = blockIdx.x % vh, v2 = threadIdx.x;
    const size_t at = (size_t)t * (size_t)vh * vd + (size_t)h * vd + v2;
    const float o = v2 < vd ? obuf[at] : 0.f;
    double ms = (v2 < vd) ? (double)o * o : 0.0;
    ms = block_red_d(ms);
    __shared__ float rr;
    if (threadIdx.x == 0) rr = 1.f / sqrtf((float)(ms / vd) + eps);
    __syncthreads();
    if (v2 < vd) outr[at] = o * rr * gamma[v2] * d_sigmoid(z[at]);
}

/* ==================== host-side weight transform ==================== */

/* Q8_0 -> planar: q[O][I], h[O][I/32] = the raw fp16 `d` bits (verbatim) */
static int up_q8_0(GMat *m, int O, int I, const uint8_t *raw) {
    const int nb = I / GGML_BK_QK8_0;
    size_t qn = (size_t)O * I, sn = (size_t)O * nb;
    int8_t *hq = (int8_t *)malloc(qn);
    uint16_t *hs = (uint16_t *)malloc(sn * sizeof(uint16_t));
    if (!hq || !hs) { free(hq); free(hs); return -1; }
    const size_t rb = ggml_bk_row_bytes(GGML_BK_Q8_0, (size_t)I);
    for (int r = 0; r < O; r++) {
        const ggml_bk_block_q8_0 *b = (const ggml_bk_block_q8_0 *)(raw + (size_t)r * rb);
        for (int i = 0; i < nb; i++) {
            hs[(size_t)r * nb + i] = (uint16_t)b[i].d;
            memcpy(hq + (size_t)r * I + (size_t)i * 32, b[i].qs, 32);
        }
    }
    m->kind = 0; m->O = O; m->I = I;
    m->q = (int8_t *)gmalloc(qn, "q8 quants");
    m->h = (__half *)gmalloc(sn * sizeof(uint16_t), "q8 scales");
    int ok = m->q && m->h &&
             cudaMemcpy(m->q, hq, qn, cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(m->h, hs, sn * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess;
    free(hq); free(hs);
    return ok ? 0 : -1;
}

/* Q6_K -> planar: q[O][I] = q-32, s[O][I/16] = fp16(d) * sub-scale (the f32
 * product the CPU forms; values then match d*sc*(q-32) exactly). */
static int up_q6_K(GMat *m, int O, int I, const uint8_t *raw) {
    const int nb = I / GGML_BK_QK_K;
    const int ng = I / 16;
    size_t qn = (size_t)O * I, sn = (size_t)O * ng;
    int8_t *hq = (int8_t *)malloc(qn);
    float *hs = (float *)malloc(sn * sizeof(float));
    if (!hq || !hs) { free(hq); free(hs); return -1; }
    const size_t rb = ggml_bk_row_bytes(GGML_BK_Q6_K, (size_t)I);
    for (int r = 0; r < O; r++) {
        const ggml_bk_block_q6_K *bs = (const ggml_bk_block_q6_K *)(raw + (size_t)r * rb);
        for (int i = 0; i < nb; i++) {
            const ggml_bk_block_q6_K *b = &bs[i];
            const float d = ggml_bk_fp16_to_fp32(b->d);
            int8_t *q = hq + (size_t)r * I + (size_t)i * GGML_BK_QK_K;
            float *s = hs + (size_t)r * ng + (size_t)i * 16;
            for (int t = 0; t < 16; t++) s[t] = d * (float)b->scales[t];
            const uint8_t *ql = b->ql, *qh = b->qh;
            for (int half = 0; half < 2; half++) {
                for (int l = 0; l < 32; l++) {
                    q[l +  0] = (int8_t)((int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    q[l + 32] = (int8_t)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    q[l + 64] = (int8_t)((int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    q[l + 96] = (int8_t)((int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
                q += 128; ql += 64; qh += 32;
            }
        }
    }
    m->kind = 1; m->O = O; m->I = I;
    m->q = (int8_t *)gmalloc(qn, "q6 quants");
    m->s = (float *)gmalloc(sn * sizeof(float), "q6 scales");
    int ok = m->q && m->s &&
             cudaMemcpy(m->q, hq, qn, cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(m->s, hs, sn * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    free(hq); free(hs);
    return ok ? 0 : -1;
}

/* ==================== public API ==================== */

extern "C" int q38g_init(int n_layers, const uint8_t *is_attn, int D,
                         int hc_rank, int vh, int vk, int kd, int vd, int convk, int cdim,
                         int H, int KV, int hd, int vocab, float eps) {
    memset(&g, 0, sizeof g);
    g.nl = n_layers; g.D = D; g.W = G_HC * D; g.R = hc_rank;
    g.vh = vh; g.vk = vk; g.kd = kd; g.vd = vd; g.convk = convk; g.cdim = cdim;
    g.vtot = vh * vd; g.keytot = vk * kd;
    g.H = H; g.KV = KV; g.hd = hd; g.vocab = vocab; g.eps = eps;
    g.b_in_max = H * hd > g.vtot ? H * hd : g.vtot;
    if (g.b_in_max < D) g.b_in_max = D;
    g.b_out_cap = (size_t)G_BMAX * D;
    if (g.b_out_cap < (size_t)G_SPEC_BMAX * vocab)
        g.b_out_cap = (size_t)G_SPEC_BMAX * vocab;
    {   /* same device selection as the expert tier */
        const char *gl = getenv("COLI_GPUS");
        int dev = (gl && *gl) ? atoi(gl) : 0, ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= dev ||
            cudaSetDevice(dev) != cudaSuccess) {
            fprintf(stderr, "[q38g] no CUDA device %d -> dense stays on the CPU\n", dev);
            return 0;
        }
    }
    if (!GCK(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking)) ||
        !GCK(cudaStreamCreateWithFlags(&g.s1, cudaStreamNonBlocking)) ||
        !GCK(cudaStreamCreateWithFlags(&g.s2, cudaStreamNonBlocking)) ||
        !GCK(cudaEventCreateWithFlags(&g.ef, cudaEventDisableTiming)) ||
        !GCK(cudaEventCreateWithFlags(&g.e1, cudaEventDisableTiming)) ||
        !GCK(cudaEventCreateWithFlags(&g.e2, cudaEventDisableTiming))) return 0;

    const int qsa_n = H * 2 * hd + 2 * KV * hd;
    struct { float **p; size_t n; const char *nm; } bufs[] = {
        { &g.d_res,   (size_t)g.W, "res" },   { &g.d_xn,   (size_t)g.W, "xn" },
        { &g.d_lo,    (size_t)g.R, "lo" },    { &g.d_gt,   (size_t)g.W, "gt" },
        { &g.d_inj,   (size_t)G_HC, "inj" },  { &g.d_mixed,(size_t)D, "mixed" },
        { &g.d_blk,   (size_t)D, "blk" },
        { &g.d_qkv,   (size_t)cdim, "qkv" },  { &g.d_z,    (size_t)g.vtot, "z" },
        { &g.d_ba,    (size_t)2 * vh, "ba" },
        { &g.d_conv,  (size_t)cdim, "conv" }, { &g.d_outr, (size_t)g.vtot, "outr" },
        { &g.d_o,     (size_t)g.vtot, "gdn o" }, { &g.d_b_o, (size_t)G_BMAX * g.vtot, "batch gdn o" },
        { &g.d_qsa,   (size_t)qsa_n, "qsa" }, { &g.d_ctx,  (size_t)H * hd, "ctx" },
        { &g.d_logits,(size_t)vocab, "logits" },
        { &g.d_b_res, (size_t)G_BMAX * g.W, "batch res" },
        { &g.d_b_xn,  (size_t)G_BMAX * g.W, "batch xn" },
        { &g.d_b_lo,  (size_t)G_BMAX * g.R, "batch lo" },
        { &g.d_b_gt,  (size_t)G_BMAX * g.W, "batch gate" },
        { &g.d_b_inj, (size_t)G_BMAX * G_HC, "batch inject" },
        { &g.d_b_mixed,(size_t)G_BMAX * D, "batch mixed" },
        { &g.d_b_qf,  (size_t)G_BMAX * H * 2 * hd, "batch q" },
        { &g.d_b_kk,  (size_t)G_BMAX * KV * hd, "batch k" },
        { &g.d_b_vv,  (size_t)G_BMAX * KV * hd, "batch v" },
        { &g.d_b_qkv, (size_t)G_BMAX * cdim, "batch qkv" },
        { &g.d_b_z,   (size_t)G_BMAX * g.vtot, "batch z" },
        { &g.d_b_ba,  (size_t)G_BMAX * 2 * vh, "batch beta-alpha" },
        { &g.d_b_conv,(size_t)G_BMAX * cdim, "batch conv" },
        { &g.d_b_outr,(size_t)G_BMAX * g.vtot, "batch recurrent out" },
        { &g.d_b_in,  (size_t)G_BMAX * g.b_in_max, "batch generic input" },
        { &g.d_b_sg,  (size_t)G_BMAX * D, "batch shared gate" },
        { &g.d_b_su,  (size_t)G_BMAX * D, "batch shared up" },
        { &g.d_b_sh,  (size_t)G_BMAX * D, "batch shared act" },
        { &g.d_b_out, g.b_out_cap, "batch generic output" },
    };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        *bufs[i].p = (float *)gmalloc(bufs[i].n * sizeof(float), bufs[i].nm);
        if (!*bufs[i].p) { q38g_disable(); return 0; }
    }
    g.d_rec = (float **)calloc(n_layers, sizeof(float *));
    g.d_ring = (float **)calloc(n_layers, sizeof(float *));
    if (!g.d_rec || !g.d_ring) { q38g_disable(); return 0; }
    for (int i = 0; i < 128; i++) g.gdn_slot[i] = -1;
    for (int i = 0; i < n_layers; i++) {
        if (is_attn[i]) continue;
        g.gdn_slot[i] = g.n_gdn++;
        g.d_rec[i] = (float *)gmalloc((size_t)vh * kd * vd * sizeof(float), "rec");
        g.d_ring[i] = (float *)gmalloc((size_t)cdim * (convk - 1) * sizeof(float), "ring");
        if (!g.d_rec[i] || !g.d_ring[i]) { q38g_disable(); return 0; }
    }
    g.pin_sz = (size_t)(vocab > qsa_n ? vocab : qsa_n) * sizeof(float);
    if (g.pin_sz < (size_t)g.W * sizeof(float)) g.pin_sz = (size_t)g.W * sizeof(float);
    if (!GCK(cudaHostAlloc((void **)&g.h_pin, g.pin_sz, cudaHostAllocDefault))) {
        q38g_disable(); return 0;
    }
    if (!GCK(cudaHostAlloc((void **)&g.h_b_sh, (size_t)G_BMAX * D * sizeof(float), cudaHostAllocDefault)) ||
        !GCK(cudaEventCreateWithFlags(&g.ev_shb, cudaEventDisableTiming))) {
        q38g_disable(); return 0;
    }
    /* upload ring: the widest single upload is res (W) or the QSA context */
    g.up_stride = (size_t)(g.W > H * hd ? g.W : H * hd);
    if (!GCK(cudaHostAlloc((void **)&g.h_up, G_UPRING * g.up_stride * sizeof(float),
                           cudaHostAllocDefault))) { q38g_disable(); return 0; }
    /* the fused GDN core keeps the 128x128 state in shared memory (~66 KB) */
    {
        const size_t shb = ((size_t)kd * vd + 2 * kd) * sizeof(float);
        if (!GCK(cudaFuncSetAttribute(k_gdn_core,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb)) ||
            !GCK(cudaFuncSetAttribute(k_gdn_core_chunk,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb))) {
            q38g_disable(); return 0;
        }
    }
    if (getenv("Q38_GPU_GRAPHS") && atoi(getenv("Q38_GPU_GRAPHS")) == 0) g.graphs_off = 1;
    if (getenv("Q38_GPU_TILED") && atoi(getenv("Q38_GPU_TILED")) == 0) g.tiled_off = 1;
    if (getenv("Q38_GPU_GDN_CHUNK") && atoi(getenv("Q38_GPU_GDN_CHUNK")) == 0) g.gdn_chunk_off = 1;
    if (getenv("Q38_GPU_GDN_SPLIT") && atoi(getenv("Q38_GPU_GDN_SPLIT")) == 0) g.gdn_split_off = 1;
    {   /* the skinny tiled GEMM stages a 4 x 4096 activation tile (64 KB) */
        const int shb = GTS_T * GTS_KW * (int)sizeof(float);
        if (!GCK(cudaFuncSetAttribute(k_gemm_i8_tiled_skinny<5, __half>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, shb)) ||
            !GCK(cudaFuncSetAttribute(k_gemm_i8_tiled_skinny<4, float>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, shb))) {
            q38g_disable(); return 0;
        }
    }
    if (getenv("Q38_GPU_KPROF") && atoi(getenv("Q38_GPU_KPROF"))) { g.kprof = 1; g.graphs_off = 1; }
    g.on = 1;
    return 1;
}

extern "C" int q38g_active(void) { return g.on && !g.dead; }

extern "C" void q38g_disable(void) {
    /* device memory is freed wholesale at process exit; just go inactive */
    g.on = 0;
}

extern "C" void q38g_shutdown(void) {
    if (g.on && g.kprof) {
        double tot = 0;
        for (int i = 0; i < 12; i++) tot += g.kp_ms[i];
        fprintf(stderr, "[q38g-kprof] total %.1f ms over the run:\n", tot);
        for (int i = 0; i < 12; i++)
            if (g.kp_n[i])
                fprintf(stderr, "  %-9s %8.1f ms  n=%-7ld  %6.1f us/call\n",
                        kp_name[i], g.kp_ms[i], g.kp_n[i], 1e3 * g.kp_ms[i] / g.kp_n[i]);
    }
    g.on = 0;
}

extern "C" size_t q38g_vram_used(void) { return g.vram; }

extern "C" int q38g_up_raw(int type, int O, int I, const void *raw) {
    if (!g.on || g.nmat >= G_MAXMAT) return -1;
    GMat *m = &g.mat[g.nmat];
    int rc;
    if (type == GGML_BK_Q8_0)      rc = up_q8_0(m, O, I, (const uint8_t *)raw);
    else if (type == GGML_BK_Q6_K) rc = up_q6_K(m, O, I, (const uint8_t *)raw);
    else {
        fprintf(stderr, "[q38g] unsupported dense type %d ([%d,%d])\n", type, O, I);
        return -1;
    }
    if (rc != 0) return -1;
    return g.nmat++;
}

extern "C" int q38g_up_f32(const float *p, int64_t n) {
    if (!g.on || g.nmat >= G_MAXMAT) return -1;
    GMat *m = &g.mat[g.nmat];
    m->kind = 2; m->O = 0; m->I = (int)n; m->q = NULL;
    m->s = (float *)gmalloc((size_t)n * sizeof(float), "f32 tensor");
    if (!m->s || cudaMemcpy(m->s, p, (size_t)n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;
    return g.nmat++;
}

/* GEMV dispatch on stream `st`: y = mat[h] @ x */
static void gemv_s(int h, const float *x, float *y, int O, int I, cudaStream_t st) {
    GMat *m = &g.mat[h];
    if (m->kind == 2) { k_gemv_f32<<<O, 256, 0, st>>>(I, m->s, x, y); return; }
    if (O <= 512) {    /* skinny: block per row */
        if (m->kind == 0) k_gemv_i8_wide<5, __half><<<O, 256, 0, st>>>(I, m->q, m->h, x, y);
        else              k_gemv_i8_wide<4, float><<<O, 256, 0, st>>>(I, m->q, m->s, x, y);
        return;
    }
    const dim3 grid((O + 3) / 4), block(128);
    if (m->kind == 0) k_gemv_i8<5, __half><<<grid, block, 0, st>>>(O, I, m->q, m->h, x, y);
    else              k_gemv_i8<4, float><<<grid, block, 0, st>>>(O, I, m->q, m->s, x, y);
}
static void gemv(int h, const float *x, float *y, int O, int I) {
    gemv_s(h, x, y, O, I, g.st);
}

/* device batch-major Y[B,ldy] = mat[h][O,I] @ X[B,I] */
static void gemm_batch(int h, const float *X, float *Y, int O, int I,
                       int B, int ldy, int act) {
    GMat *m = &g.mat[h];
    const dim3 grid(O, (B + G_BTILE - 1) / G_BTILE);
    /* Match gemv_s's per-row reduction order.  Speculative verification uses
     * intermediate batch rows as target logits, not just the final row. */
    const int threads = m->kind == 2 || O <= 512 ? 256 : 32;
    if (m->kind != 2 && B >= 16 && !g.tiled_off && (I & 15) == 0) {
        /* P9 tiled twins: identical bits, activation tile shared across rows */
        if (O <= 512) {
            const dim3 gr((O + GTS_ROWS - 1) / GTS_ROWS, (B + GTS_T - 1) / GTS_T);
            const size_t shb = (size_t)GTS_T * GTS_KW * sizeof(float);
            if (act) { g.dead = 1; return; }      /* no skinny ACT matrix in this model */
            if (m->kind == 0) k_gemm_i8_tiled_skinny<5, __half><<<gr, 256, shb, g.st>>>(O, I, m->q, m->h, X, Y, B, ldy);
            else              k_gemm_i8_tiled_skinny<4, float><<<gr, 256, shb, g.st>>>(O, I, m->q, m->s, X, Y, B, ldy);
        } else {
            const dim3 gr((O + GT_ROWS - 1) / GT_ROWS, (B + G_BTILE - 1) / G_BTILE);
            if (act) {
                if (m->kind == 0) k_gemm_i8_tiled_act<5, __half><<<gr, 32 * GT_ROWS, 0, g.st>>>(O, I, m->q, m->h, X, Y, B, ldy);
                else              k_gemm_i8_tiled_act<4, float><<<gr, 32 * GT_ROWS, 0, g.st>>>(O, I, m->q, m->s, X, Y, B, ldy);
            } else {
                if (m->kind == 0) k_gemm_i8_tiled<5, __half><<<gr, 32 * GT_ROWS, 0, g.st>>>(O, I, m->q, m->h, X, Y, B, ldy);
                else              k_gemm_i8_tiled<4, float><<<gr, 32 * GT_ROWS, 0, g.st>>>(O, I, m->q, m->s, X, Y, B, ldy);
            }
        }
        return;
    }
    if (m->kind == 2) {
        k_gemm_f32<<<grid, threads, 0, g.st>>>(O, I, m->s, X, Y, B, ldy);
    } else if (m->kind == 0) {
        if (act) k_gemm_i8<5, __half, true><<<grid, threads, 0, g.st>>>(O, I, m->q, m->h, X, Y, B, ldy);
        else     k_gemm_i8<5, __half, false><<<grid, threads, 0, g.st>>>(O, I, m->q, m->h, X, Y, B, ldy);
    } else {
        if (act) k_gemm_i8<4, float, true><<<grid, threads, 0, g.st>>>(O, I, m->q, m->s, X, Y, B, ldy);
        else     k_gemm_i8<4, float, false><<<grid, threads, 0, g.st>>>(O, I, m->q, m->s, X, Y, B, ldy);
    }
}

/* fork the side streams off the main one (capture-safe), and join them back.
 * Currently unused -- see the note in q38g_gdn. */
static void fork2(void) __attribute__((unused));
static void join2(void) __attribute__((unused));
static void fork2(void) {
    cudaEventRecord(g.ef, g.st);
    cudaStreamWaitEvent(g.s1, g.ef, 0);
    cudaStreamWaitEvent(g.s2, g.ef, 0);
}
static void join2(void) {
    cudaEventRecord(g.e1, g.s1);
    cudaEventRecord(g.e2, g.s2);
    cudaStreamWaitEvent(g.st, g.e1, 0);
    cudaStreamWaitEvent(g.st, g.e2, 0);
}

/* stage n floats into the next upload-ring slot and enqueue the H2D */
static int up_stage(const float *src, size_t n, void *dst) {
    float *slot = g.h_up + (size_t)g.up_slot * g.up_stride;
    g.up_slot = (g.up_slot + 1) % G_UPRING;
    memcpy(slot, src, n * sizeof(float));
    return GCK(cudaMemcpyAsync(dst, slot, n * sizeof(float), cudaMemcpyHostToDevice, g.st));
}

extern "C" void q38g_tok_begin(const float *x) {
    if (!q38g_active()) return;
    if (!up_stage(x, (size_t)g.D, g.d_mixed)) return;
    dim3 grid((g.D + 255) / 256, G_HC);
    k_bcast_res<<<grid, 256, 0, g.st>>>(g.d_res, g.d_mixed, g.D);
    GCK(cudaGetLastError());
}

extern "C" void q38g_res_get(float *res) {
    if (!q38g_active()) return;
    if (!GCK(cudaMemcpyAsync(g.h_pin, g.d_res, (size_t)g.W * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    if (!GCK(cudaStreamSynchronize(g.st))) return;
    memcpy(res, g.h_pin, (size_t)g.W * sizeof(float));
}

extern "C" void q38g_res_set(const float *res) {
    if (!q38g_active()) return;
    up_stage(res, (size_t)g.W, g.d_res);
}

extern "C" void q38g_hc_mix(int hdown, int hup, int hnorm, int hinj) {
    if (!q38g_active()) return;
    kp_begin();
    k_hc_norm<<<G_HC, 256, 0, g.st>>>(g.d_res, g.mat[hnorm].s, g.d_xn, g.D, g.eps);
    kp_end(0); kp_begin();
    gemv(hdown, g.d_xn, g.d_lo, g.R, g.W);
    kp_end(1); kp_begin();
    /* up-projection with silu(lo/4) fused into the activation load */
    {
        GMat *m = &g.mat[hup];
        const dim3 grid((g.W + 3) / 4), block(128);
        if (m->kind == 0) k_gemv_i8_act<5, __half><<<grid, block, 0, g.st>>>(g.W, g.R, m->q, m->h, g.d_lo, g.d_gt);
        else              k_gemv_i8_act<4, float><<<grid, block, 0, g.st>>>(g.W, g.R, m->q, m->s, g.d_lo, g.d_gt);
    }
    kp_end(2); kp_begin();
    if (hinj >= 0 && g.mat[hinj].kind == 2) {
        const int nb = (g.D + 255) / 256;
        k_hc_out_inj<<<nb + G_HC, 256, 0, g.st>>>(g.d_xn, g.d_gt, g.d_mixed, g.D, nb,
                                                   g.mat[hinj].s, g.d_inj, g.W);
        kp_end(3); kp_begin();
    } else {
        k_hc_out<<<(g.D + 255) / 256, 256, 0, g.st>>>(g.d_xn, g.d_gt, g.d_mixed, g.D);
        kp_end(3); kp_begin();
        if (hinj >= 0) gemv(hinj, g.d_xn, g.d_inj, G_HC, g.W);
    }
    kp_end(4);
    GCK(cudaGetLastError());
}

/* ---- CUDA graphs over fixed per-layer launch spans ---- */
extern "C" int q38g_graph_open(int id) {
    if (!q38g_active() || g.graphs_off) return 0;
    if (g.gexec[id]) {
        if (!GCK(cudaGraphLaunch(g.gexec[id], g.st))) return 0;
        return 1;
    }
    if (cudaStreamBeginCapture(g.st, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        g.graphs_off = 1;           /* run the span directly from now on */
        cudaGetLastError();
        return 0;
    }
    g.capturing = 1;
    return 0;
}

extern "C" void q38g_graph_close(int id) {
    if (!q38g_active() || !g.capturing) return;
    g.capturing = 0;
    cudaGraph_t graph = NULL;
    if (!GCK(cudaStreamEndCapture(g.st, &graph))) return;
    cudaGraphExec_t exec = NULL;
    if (!GCK(cudaGraphInstantiate(&exec, graph, 0))) { cudaGraphDestroy(graph); return; }
    cudaGraphDestroy(graph);
    g.gexec[id] = exec;
    /* the captured span did not execute -- run it now */
    GCK(cudaGraphLaunch(exec, g.st));
}

extern "C" void q38g_mixed_get(float *mixed) {
    if (!q38g_active()) return;
    if (!GCK(cudaMemcpyAsync(g.h_pin, g.d_mixed, (size_t)g.D * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    if (!GCK(cudaStreamSynchronize(g.st))) return;
    memcpy(mixed, g.h_pin, (size_t)g.D * sizeof(float));
}

extern "C" void q38g_gdn(int layer, int hqkv, int hz, int hba,
                         int ha, int hdt, int hnorm, int hconv, int hout) {
    if (!q38g_active()) return;
    /* NOTE: qkv/z/b‖alpha are mutually independent and were tried on three
     * streams with capture-safe fork/join -- no measurable gain (gdn 9.3 ->
     * 9.2 ms/token).  Q38_GPU_KPROF shows why: these three already run at
     * 785/671/103 GB/s, i.e. the two big ones are at the bandwidth roof, so
     * there is no latency left to hide.  Kept serial; the fork/join helpers
     * are retained for the small-kernel work described in the status doc. */
    kp_begin(); gemv(hqkv, g.d_mixed, g.d_qkv, g.cdim, g.D);       kp_end(5);
    kp_begin(); gemv(hz, g.d_mixed, g.d_z, g.vtot, g.D);           kp_end(6);
    kp_begin(); gemv(hba, g.d_mixed, g.d_ba, 2 * g.vh, g.D);       kp_end(7);
    kp_begin();
    k_gdn_conv<<<(g.cdim + 255) / 256, 256, 0, g.st>>>(g.d_qkv, g.mat[hconv].s,
                                                       g.d_ring[layer], g.d_conv, g.cdim, g.convk);
    kp_end(8); kp_begin();
    const float scale = 1.f / sqrtf((float)g.kd);
    const size_t shb = ((size_t)g.kd * g.vd + 2 * g.kd) * sizeof(float);
    if (!g.gdn_split_off && g.kd == 128 && g.vd == 128) {
        const size_t shs = ((size_t)g.kd * (g.vd / GDN_SPLIT) + 2 * g.kd) * sizeof(float);
        k_gdn_sweep_chunk<<<g.vh * GDN_SPLIT, 32, shs, g.st>>>(g.d_rec[layer], g.d_conv, g.d_ba,
            g.mat[ha].s, g.mat[hdt].s, g.d_o, g.vh, g.vk, g.kd, g.vd, g.keytot, g.cdim, g.eps, scale, 1);
        k_gdn_norm<<<g.vh, g.vd, 0, g.st>>>(g.d_o, g.d_z, g.mat[hnorm].s, g.d_outr, g.vh, g.vd, g.eps);
    } else
    k_gdn_core<<<g.vh, g.vd, shb, g.st>>>(g.d_rec[layer], g.d_conv, g.d_ba,
                                          g.mat[ha].s, g.mat[hdt].s, g.d_z,
                                          g.mat[hnorm].s, g.d_outr,
                                          g.vh, g.vk, g.kd, g.vd, g.keytot, g.eps, scale);
    kp_end(9); kp_begin();
    gemv(hout, g.d_outr, g.d_blk, g.D, g.vtot);
    kp_end(10);
    GCK(cudaGetLastError());
}

extern "C" void q38g_qsa_proj(int hq, int hk, int hv, int nq, int nkv,
                              float *qfull, float *kk, float *vv) {
    if (!q38g_active()) return;
    gemv(hq, g.d_mixed, g.d_qsa, nq, g.D);
    gemv(hk, g.d_mixed, g.d_qsa + nq, nkv, g.D);
    gemv(hv, g.d_mixed, g.d_qsa + nq + nkv, nkv, g.D);
    const size_t tot = ((size_t)nq + 2 * nkv) * sizeof(float);
    if (!GCK(cudaMemcpyAsync(g.h_pin, g.d_qsa, tot, cudaMemcpyDeviceToHost, g.st))) return;
    if (!GCK(cudaStreamSynchronize(g.st))) return;
    memcpy(qfull, g.h_pin, (size_t)nq * sizeof(float));
    memcpy(kk, g.h_pin + nq, (size_t)nkv * sizeof(float));
    memcpy(vv, g.h_pin + nq + nkv, (size_t)nkv * sizeof(float));
}

extern "C" void q38g_qsa_out(int ho, const float *ctx, int nctx) {
    if (!q38g_active()) return;
    if (!up_stage(ctx, (size_t)nctx, g.d_ctx)) return;
    gemv(ho, g.d_ctx, g.d_blk, g.D, nctx);
    GCK(cudaGetLastError());
}

/* P10 D1: decode o-projection straight from the device context written by
 * q38g_qsa_batch (B=1) into d_blk, same GEMV as q38g_qsa_out */
extern "C" void q38g_qsa_out_dev(int ho) {
    if (!q38g_active()) return;
    gemv(ho, g.d_b_ctx, g.d_blk, g.D, g.H * g.hd);
    GCK(cudaGetLastError());
}

/* P10 D4 (opt-in): router logits on the device.  Decode: gate[E,D] f32 @
 * d_mixed -> E floats to the host.  Prefill: the chunk GEMM from d_b_mixed.
 * Softmax/top-k stay on the host; the GEMV reduction order differs from the
 * host's serial matf (bit-close, and it decides expert identity on ties). */
extern "C" int q38g_router(int hgate, int E, float *pr) {
    if (!q38g_active()) return 0;
    gemv(hgate, g.d_mixed, g.d_b_out, E, g.D);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(pr, g.d_b_out, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaStreamSynchronize(g.st))) return 0;
    return 1;
}
extern "C" int q38g_router_batch(int hgate, int E, int B, float *pr) {
    if (!q38g_active() || B < 1 || B > G_BMAX || (size_t)B * E > g.b_out_cap) return 0;
    gemm_batch(hgate, g.d_b_mixed, g.d_b_out, E, g.D, B, E, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(pr, g.d_b_out, (size_t)B * E * sizeof(float), cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaStreamSynchronize(g.st))) return 0;
    return 1;
}

extern "C" void q38g_combine(void) {
    if (!q38g_active()) return;
    dim3 grid((g.D + 255) / 256, G_HC);
    k_combine<<<grid, 256, 0, g.st>>>(g.d_res, g.d_blk, g.d_inj, g.D);
    GCK(cudaGetLastError());
}

extern "C" void q38g_blk_combine(const float *blk) {
    if (!q38g_active()) return;
    if (!up_stage(blk, (size_t)g.D, g.d_blk)) return;
    q38g_combine();
}

extern "C" void q38g_head(int hdown, int hup, int hnorm, int hlm, float *logits) {
    if (!q38g_active()) return;
    if (!q38g_graph_open(3 * 128)) {
        q38g_hc_mix(hdown, hup, hnorm, -1);
        kp_begin(); gemv(hlm, g.d_mixed, g.d_logits, g.vocab, g.D); kp_end(11);
        if (!GCK(cudaMemcpyAsync(g.h_pin, g.d_logits, (size_t)g.vocab * sizeof(float),
                                 cudaMemcpyDeviceToHost, g.st))) return;
        q38g_graph_close(3 * 128);
    }
    if (!GCK(cudaStreamSynchronize(g.st))) return;
    memcpy(logits, g.h_pin, (size_t)g.vocab * sizeof(float));
}

extern "C" int q38g_batch_max(void) { return q38g_active() ? G_BMAX : 0; }

/* Chunk residual resident on the device across the layer loop: put once
 * before layer 0, get once after the last layer (and around the host-side
 * PLE layer).  The batch helpers then take res == NULL / mixed == NULL to
 * mean "already in d_b_res / d_b_mixed". */
extern "C" void q38g_res_batch_put(const float *res, int B) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    GCK(cudaMemcpyAsync(g.d_b_res, res, (size_t)B * g.W * sizeof(float),
                        cudaMemcpyHostToDevice, g.st));
}
extern "C" void q38g_res_batch_get(float *res, int B) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    if (!GCK(cudaMemcpyAsync(res, g.d_b_res, (size_t)B * g.W * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

extern "C" void q38g_hc_mix_batch(int hdown, int hup, int hnorm, int hinj,
                                    const float *res, int B,
                                    float *mixed, float *inject) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    if (res && !GCK(cudaMemcpyAsync(g.d_b_res, res, (size_t)B * g.W * sizeof(float),
                                    cudaMemcpyHostToDevice, g.st))) return;
    k_hc_norm_batch<<<B * G_HC, 256, 0, g.st>>>(g.d_b_res, g.mat[hnorm].s,
                                                g.d_b_xn, g.D, g.eps);
    gemm_batch(hdown, g.d_b_xn, g.d_b_lo, g.R, g.W, B, g.R, 0);
    gemm_batch(hup, g.d_b_lo, g.d_b_gt, g.W, g.R, B, g.W, 1);
    k_hc_out_batch<<<(B * g.D + 255) / 256, 256, 0, g.st>>>(g.d_b_xn, g.d_b_gt,
                                                             g.d_b_mixed, g.D, B);
    if (hinj >= 0) gemm_batch(hinj, g.d_b_xn, g.d_b_inj, G_HC, g.W, B, G_HC, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(mixed, g.d_b_mixed, (size_t)B * g.D * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    if (hinj >= 0 && inject &&
        !GCK(cudaMemcpyAsync(inject, g.d_b_inj, (size_t)B * G_HC * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

extern "C" void q38g_qsa_proj_batch(int hq, int hk, int hv,
                                      const float *mixed, int B,
                                      float *qfull, float *kk, float *vv) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    const int nq = g.H * 2 * g.hd, nkv = g.KV * g.hd;
    if (mixed && !GCK(cudaMemcpyAsync(g.d_b_mixed, mixed, (size_t)B * g.D * sizeof(float),
                                      cudaMemcpyHostToDevice, g.st))) return;
    gemm_batch(hq, g.d_b_mixed, g.d_b_qf, nq, g.D, B, nq, 0);
    gemm_batch(hk, g.d_b_mixed, g.d_b_kk, nkv, g.D, B, nkv, 0);
    gemm_batch(hv, g.d_b_mixed, g.d_b_vv, nkv, g.D, B, nkv, 0);
    if (!GCK(cudaGetLastError())) return;
    if (!qfull) return;                    /* projections stay on the device */
    if (!GCK(cudaMemcpyAsync(qfull, g.d_b_qf, (size_t)B * nq * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaMemcpyAsync(kk, g.d_b_kk, (size_t)B * nkv * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaMemcpyAsync(vv, g.d_b_vv, (size_t)B * nkv * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

extern "C" void q38g_gdn_batch(int layer, int hqkv, int hz, int hba,
                                 int ha, int hdt, int hnorm, int hconv, int hout,
                                 const float *mixed, int B, float *blk, int snapshot) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    if (mixed && !GCK(cudaMemcpyAsync(g.d_b_mixed, mixed, (size_t)B * g.D * sizeof(float),
                                      cudaMemcpyHostToDevice, g.st))) return;
    gemm_batch(hqkv, g.d_b_mixed, g.d_b_qkv, g.cdim, g.D, B, g.cdim, 0);
    gemm_batch(hz,   g.d_b_mixed, g.d_b_z,   g.vtot, g.D, B, g.vtot, 0);
    gemm_batch(hba,  g.d_b_mixed, g.d_b_ba,  2 * g.vh, g.D, B, 2 * g.vh, 0);
    const float scale = 1.f / sqrtf((float)g.kd);
    const size_t shb = ((size_t)g.kd * g.vd + 2 * g.kd) * sizeof(float);
    if (!snapshot && B > 1 && !g.gdn_chunk_off) {
        /* P9: whole chunk in two launches, state resident in shared memory */
        k_gdn_conv_chunk<<<(g.cdim + 255) / 256, 256, 0, g.st>>>(
            g.d_b_qkv, g.mat[hconv].s, g.d_ring[layer], g.d_b_conv, g.cdim, g.convk, B);
        if (!g.gdn_split_off && g.kd == 128 && g.vd == 128) {
            /* P10 P8: GDN_SPLIT one-warp blocks per head + the norm kernel */
            const size_t shs = ((size_t)g.kd * (g.vd / GDN_SPLIT) + 2 * g.kd) * sizeof(float);
            k_gdn_sweep_chunk<<<g.vh * GDN_SPLIT, 32, shs, g.st>>>(
                g.d_rec[layer], g.d_b_conv, g.d_b_ba, g.mat[ha].s, g.mat[hdt].s, g.d_b_o,
                g.vh, g.vk, g.kd, g.vd, g.keytot, g.cdim, g.eps, scale, B);
            k_gdn_norm<<<B * g.vh, g.vd, 0, g.st>>>(g.d_b_o, g.d_b_z, g.mat[hnorm].s, g.d_b_outr, g.vh, g.vd, g.eps);
        } else
        k_gdn_core_chunk<<<g.vh, g.vd, shb, g.st>>>(
            g.d_rec[layer], g.d_b_conv, g.d_b_ba, g.mat[ha].s, g.mat[hdt].s,
            g.d_b_z, g.mat[hnorm].s, g.d_b_outr,
            g.vh, g.vk, g.kd, g.vd, g.keytot, g.cdim, g.eps, scale, B);
    } else
    for (int t = 0; t < B; t++) {
        k_gdn_conv<<<(g.cdim + 255) / 256, 256, 0, g.st>>>(
            g.d_b_qkv + (size_t)t * g.cdim, g.mat[hconv].s, g.d_ring[layer],
            g.d_b_conv + (size_t)t * g.cdim, g.cdim, g.convk);
        k_gdn_core<<<g.vh, g.vd, shb, g.st>>>(
            g.d_rec[layer], g.d_b_conv + (size_t)t * g.cdim,
            g.d_b_ba + (size_t)t * 2 * g.vh, g.mat[ha].s, g.mat[hdt].s,
            g.d_b_z + (size_t)t * g.vtot, g.mat[hnorm].s,
            g.d_b_outr + (size_t)t * g.vtot,
            g.vh, g.vk, g.kd, g.vd, g.keytot, g.eps, scale);
        if (snapshot && t < B - 1 && t < g.spec_n) {
            const int gi = layer < 128 ? g.gdn_slot[layer] : -1;
            const size_t rn = (size_t)g.vh * g.kd * g.vd;
            const size_t cn = (size_t)g.cdim * (g.convk - 1);
            if (gi >= 0) {
                GCK(cudaMemcpyAsync(g.d_spec_rec + ((size_t)t*g.n_gdn + gi)*rn,
                                    g.d_rec[layer], rn*sizeof(float),
                                    cudaMemcpyDeviceToDevice, g.st));
                GCK(cudaMemcpyAsync(g.d_spec_ring + ((size_t)t*g.n_gdn + gi)*cn,
                                    g.d_ring[layer], cn*sizeof(float),
                                    cudaMemcpyDeviceToDevice, g.st));
            }
        }
    }
    gemm_batch(hout, g.d_b_outr, g.d_b_out, g.D, g.vtot, B, g.D, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(blk, g.d_b_out, (size_t)B * g.D * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

extern "C" void q38g_out_batch(int h, const float *x, int B, float *y) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    GMat *m = &g.mat[h];
    if (m->I > g.b_in_max || (size_t)B * m->O > g.b_out_cap) { g.dead = 1; return; }
    if (!GCK(cudaMemcpyAsync(g.d_b_in, x, (size_t)B * m->I * sizeof(float),
                             cudaMemcpyHostToDevice, g.st))) return;
    gemm_batch(h, g.d_b_in, g.d_b_out, m->O, m->I, B, m->O, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(y, g.d_b_out, (size_t)B * m->O * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

/* Shared expert over a prefill chunk on the dense GEMM path: y = down(silu(gate x)
 * * (up x)) for B tokens, int8 weight tiles reused across 16 tokens.  The
 * caller scales by the shared gate and accumulates on the host.  Shapes:
 * gate/up [Is, D], down [D, Is] with Is <= D (scratch is sized G_BMAX x D). */
extern "C" void q38g_shared_batch(int hg, int hu, int hd, const float *x, int B, float *y) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    GMat *mg = &g.mat[hg], *md = &g.mat[hd];
    const int Is = mg->O;
    if (mg->I != g.D || Is > g.D || md->O != g.D || md->I != Is) { g.dead = 1; return; }
    if (x && !GCK(cudaMemcpyAsync(g.d_b_mixed, x, (size_t)B * g.D * sizeof(float),
                                  cudaMemcpyHostToDevice, g.st))) return;
    gemm_batch(hg, g.d_b_mixed, g.d_b_sg, Is, g.D, B, Is, 0);
    gemm_batch(hu, g.d_b_mixed, g.d_b_su, Is, g.D, B, Is, 0);
    k_silu_mul<<<(B * Is + 255) / 256, 256, 0, g.st>>>(g.d_b_sg, g.d_b_su, g.d_b_sh, B * Is);
    gemm_batch(hd, g.d_b_sh, g.d_b_out, g.D, Is, B, g.D, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(y, g.d_b_out, (size_t)B * g.D * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}
/* async twin: x == NULL means d_b_mixed already holds the chunk; the result
 * lands in a pinned buffer and _take copies it out after the event */
extern "C" const float *q38g_b_mixed_dev(void) {
    return q38g_active() ? g.d_b_mixed : NULL;
}

extern "C" int q38g_shared_batch_issue(int hg, int hu, int hd, const float *x, int B) {
    if (!q38g_active() || B < 1 || B > G_BMAX || !g.h_b_sh) return 0;
    GMat *mg = &g.mat[hg], *md = &g.mat[hd];
    const int Is = mg->O;
    if (mg->I != g.D || Is > g.D || md->O != g.D || md->I != Is) return 0;
    if (x && !GCK(cudaMemcpyAsync(g.d_b_mixed, x, (size_t)B * g.D * sizeof(float),
                                  cudaMemcpyHostToDevice, g.st))) return 0;
    gemm_batch(hg, g.d_b_mixed, g.d_b_sg, Is, g.D, B, Is, 0);
    gemm_batch(hu, g.d_b_mixed, g.d_b_su, Is, g.D, B, Is, 0);
    k_silu_mul<<<(B * Is + 255) / 256, 256, 0, g.st>>>(g.d_b_sg, g.d_b_su, g.d_b_sh, B * Is);
    gemm_batch(hd, g.d_b_sh, g.d_b_out, g.D, Is, B, g.D, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(g.h_b_sh, g.d_b_out, (size_t)B * g.D * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaEventRecord(g.ev_shb, g.st))) return 0;
    return 1;
}
extern "C" int q38g_shared_batch_take(int B, float *y) {
    if (!q38g_active() || !GCK(cudaEventSynchronize(g.ev_shb))) return 0;
    memcpy(y, g.h_b_sh, (size_t)B * g.D * sizeof(float));
    return 1;
}

/* ---- P9 QSA batch entry points ---- */
extern "C" int q38g_qsa_kv_setup(int n_layers, const uint8_t *is_attn, int kv_heads,
                                 int hd, int max_t, int H, int sel_cap) {
    if (!g.on) return 0;
    const int ns = hd / 32;
    g.kv_max_t = max_t; g.kv_heads = kv_heads; g.sel_cap = sel_cap;
    for (int i = 0; i < n_layers && i < 128; i++) {
        if (!is_attn[i]) continue;
        const size_t rows = (size_t)kv_heads * max_t;
        g.d_k8[i]  = (int8_t *)gmalloc(rows * hd, "kv k8");
        g.d_v8[i]  = (int8_t *)gmalloc(rows * hd, "kv v8");
        g.d_k8s[i] = (float *)gmalloc(rows * ns * sizeof(float), "kv k8s");
        g.d_v8s[i] = (float *)gmalloc(rows * ns * sizeof(float), "kv v8s");
        g.kv_valid[i] = 0;
        if (!g.d_k8[i] || !g.d_v8[i] || !g.d_k8s[i] || !g.d_v8s[i]) return 0;
    }
    g.d_b_q   = (float *)gmalloc((size_t)G_BMAX * H * hd * sizeof(float), "qsa q");
    g.d_b_ctx = (float *)gmalloc((size_t)G_BMAX * H * hd * sizeof(float), "qsa ctx");
    g.d_nsel  = (int *)gmalloc((size_t)G_BMAX * sizeof(int), "qsa nsel");
    g.d_sel   = (int *)gmalloc((size_t)G_BMAX * sel_cap * sizeof(int), "qsa sel");
    g.d_mp    = (int *)gmalloc((size_t)G_BMAX * 3 * sizeof(int), "qsa mp");
    if (!g.d_b_q || !g.d_b_ctx || !g.d_nsel || !g.d_sel || !g.d_mp) return 0;
    const int shb = (256 + QSA_WALK_MAX) * (int)sizeof(float);
    if (!GCK(cudaFuncSetAttribute(k_qsa_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, shb)))
        return 0;
    return 1;
}
extern "C" int q38g_idx_setup(int n_layers, const uint8_t *is_attn, const int *ratio,
                              int max_t, int nh, int id) {
    if (!g.on) return 0;
    int cap = 0;
    for (int i = 0; i < n_layers && i < 128; i++) {
        if (!is_attn[i] || ratio[i] <= 0) continue;
        const int nb = max_t / ratio[i] + 1;
        if (nb > cap) cap = nb;
        g.d_ibk[i] = (float *)gmalloc((size_t)nb * id * sizeof(float), "idx block keys");
        g.ibk_valid[i] = 0;
        if (!g.d_ibk[i]) return 0;
    }
    g.ibk_cap = cap; g.idx_nh = nh; g.idx_dim = id;
    g.d_b_qi = (float *)gmalloc((size_t)G_BMAX * nh * id * sizeof(float), "idx q");
    g.d_b_nfb = (int *)gmalloc((size_t)G_BMAX * sizeof(int), "idx nfb");
    g.d_b_score = (float *)gmalloc((size_t)G_BMAX * cap * sizeof(float), "idx scores");
    if (!g.d_b_qi || !g.d_b_nfb || !g.d_b_score) return 0;
    if (!GCK(cudaHostAlloc((void **)&g.h_b_score, (size_t)G_BMAX * cap * sizeof(float), cudaHostAllocDefault)))
        return 0;
    const int shb = IDX_TILE * id * (int)sizeof(float);
    if (!GCK(cudaFuncSetAttribute(k_idx_score, cudaFuncAttributeMaxDynamicSharedMemorySize, shb)))
        return 0;
    return 1;
}
/* scores for a chunk: qi [B][nh][id] (normalized+roped), host IBK rows
 * [ibk_valid, nblk) are uploaded first; nfb[t] blocks per token; the result
 * (row stride nfb_max) is returned through a pinned host buffer. */
extern "C" const float *q38g_idx_score(int layer, const float *ibk_host, int nblk,
                                       const float *qi, const int *nfb, int nfb_max, int B) {
    if (!q38g_active() || layer >= 128 || !g.d_ibk[layer] || nblk > g.ibk_cap ||
        nfb_max > g.ibk_cap || B < 1 || B > G_BMAX) return NULL;
    const int id = g.idx_dim, nh = g.idx_nh;
    const int a = g.ibk_valid[layer];
    if (nblk > a &&
        !GCK(cudaMemcpyAsync(g.d_ibk[layer] + (size_t)a * id, ibk_host + (size_t)a * id,
                             (size_t)(nblk - a) * id * sizeof(float), cudaMemcpyHostToDevice, g.st)))
        return NULL;
    if (nblk > a) g.ibk_valid[layer] = nblk;
    if (!GCK(cudaMemcpyAsync(g.d_b_qi, qi, (size_t)B * nh * id * sizeof(float), cudaMemcpyHostToDevice, g.st)) ||
        !GCK(cudaMemcpyAsync(g.d_b_nfb, nfb, (size_t)B * sizeof(int), cudaMemcpyHostToDevice, g.st)))
        return NULL;
    const int nblocks = (nfb_max + IDX_TILE - 1) / IDX_TILE;
    if (nblocks > 0)
        k_idx_score<<<nblocks, 256, (size_t)IDX_TILE * id * sizeof(float), g.st>>>(
            g.d_b_qi, g.d_ibk[layer], g.d_b_nfb, nfb_max, B, nh, id, g.d_b_score);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(g.h_b_score, g.d_b_score, (size_t)B * nfb_max * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaStreamSynchronize(g.st))) return NULL;
    return g.h_b_score;
}
extern "C" void q38g_idx_invalidate(int nblk) {
    for (int i = 0; i < 128; i++) if (g.ibk_valid[i] > nblk) g.ibk_valid[i] = nblk;
}
extern "C" int q38g_qsa_ready(int layer, int max_t) {
    return q38g_active() && layer < 128 && g.d_k8[layer] && max_t <= g.kv_max_t;
}
extern "C" int q38g_qsa_alloc_t(void) { return g.kv_max_t; }
extern "C" size_t q38g_free_bytes(void) { size_t fb = 0, tb = 0; return cudaMemGetInfo(&fb, &tb) == cudaSuccess ? fb : 0; }
/* bytes the KV mirror + indexer mirror take at context t */
extern "C" size_t q38g_mirror_bytes(int n_layers, const uint8_t *is_attn, const int *ratio, int t) {
    size_t b = 0;
    const int ns = g.hd / 32;
    for (int i = 0; i < n_layers && i < 128; i++) {
        if (!is_attn[i]) continue;
        b += (size_t)g.kv_heads * t * (2 * g.hd + 2 * ns * sizeof(float));
        if (ratio && ratio[i] > 0) b += (size_t)(t / ratio[i] + 1) * g.idx_dim * sizeof(float);
    }
    return b;
}
/* P12: grow the q8 KV mirror to new_t rows per kv head, keeping the valid
 * rows (per-head row stride changes).  0 = failed, the old mirror stays. */
extern "C" int q38g_qsa_kv_grow(int n_layers, const uint8_t *is_attn, int new_t) {
    if (!g.on || new_t <= g.kv_max_t) return new_t <= g.kv_max_t;
    const int hd = g.hd, KV = g.kv_heads, ns = hd / 32, old_t = g.kv_max_t;
    for (int i = 0; i < n_layers && i < 128; i++) {
        if (!is_attn[i] || !g.d_k8[i]) continue;
        const size_t rows = (size_t)KV * new_t;
        int8_t *k8 = (int8_t *)gmalloc(rows * hd, "kv k8 grow");
        int8_t *v8 = (int8_t *)gmalloc(rows * hd, "kv v8 grow");
        float *k8s = (float *)gmalloc(rows * ns * sizeof(float), "kv k8s grow");
        float *v8s = (float *)gmalloc(rows * ns * sizeof(float), "kv v8s grow");
        if (!k8 || !v8 || !k8s || !v8s) {
            cudaFree(k8); cudaFree(v8); cudaFree(k8s); cudaFree(v8s);
            return 0;
        }
        const int valid = g.kv_valid[i];
        int ok = 1;
        for (int kvh = 0; kvh < KV && valid > 0 && ok; kvh++) {
            ok = GCK(cudaMemcpyAsync(k8 + (size_t)kvh * new_t * hd, g.d_k8[i] + (size_t)kvh * old_t * hd, (size_t)valid * hd, cudaMemcpyDeviceToDevice, g.st)) &&
                 GCK(cudaMemcpyAsync(v8 + (size_t)kvh * new_t * hd, g.d_v8[i] + (size_t)kvh * old_t * hd, (size_t)valid * hd, cudaMemcpyDeviceToDevice, g.st)) &&
                 GCK(cudaMemcpyAsync(k8s + (size_t)kvh * new_t * ns, g.d_k8s[i] + (size_t)kvh * old_t * ns, (size_t)valid * ns * sizeof(float), cudaMemcpyDeviceToDevice, g.st)) &&
                 GCK(cudaMemcpyAsync(v8s + (size_t)kvh * new_t * ns, g.d_v8s[i] + (size_t)kvh * old_t * ns, (size_t)valid * ns * sizeof(float), cudaMemcpyDeviceToDevice, g.st));
        }
        if (!ok || !GCK(cudaStreamSynchronize(g.st))) {
            cudaFree(k8); cudaFree(v8); cudaFree(k8s); cudaFree(v8s);
            return 0;
        }
        cudaFree(g.d_k8[i]); cudaFree(g.d_v8[i]); cudaFree(g.d_k8s[i]); cudaFree(g.d_v8s[i]);
        g.d_k8[i] = k8; g.d_v8[i] = v8; g.d_k8s[i] = k8s; g.d_v8s[i] = v8s;
        /* the layers already grown use new_t as their stride; kv_max_t is
         * updated at the end, so a failure mid-way would leave a mixed state:
         * grow every layer before touching the shared stride */
    }
    g.kv_max_t = new_t;
    return 1;
}
extern "C" int q38g_idx_grow(int n_layers, const uint8_t *is_attn, const int *ratio, int new_t) {
    if (!g.on || !g.d_b_qi) return 1;
    int cap = 0;
    for (int i = 0; i < n_layers && i < 128; i++) {
        if (!is_attn[i] || ratio[i] <= 0 || !g.d_ibk[i]) continue;
        const int nb = new_t / ratio[i] + 1;
        if (nb > cap) cap = nb;
        float *ibk = (float *)gmalloc((size_t)nb * g.idx_dim * sizeof(float), "idx block keys grow");
        if (!ibk) return 0;
        const int valid = g.ibk_valid[i] < nb ? g.ibk_valid[i] : nb;
        if (valid > 0 && !GCK(cudaMemcpy(ibk, g.d_ibk[i], (size_t)valid * g.idx_dim * sizeof(float), cudaMemcpyDeviceToDevice))) {
            cudaFree(ibk); return 0;
        }
        cudaFree(g.d_ibk[i]); g.d_ibk[i] = ibk;
    }
    if (cap > g.ibk_cap) {
        float *sc = (float *)gmalloc((size_t)G_BMAX * cap * sizeof(float), "idx scores grow");
        if (!sc) return 0;
        float *hs = NULL;
        if (!GCK(cudaHostAlloc((void **)&hs, (size_t)G_BMAX * cap * sizeof(float), cudaHostAllocDefault))) {
            cudaFree(sc); return 0;
        }
        cudaFree(g.d_b_score); g.d_b_score = sc;
        cudaFreeHost(g.h_b_score); g.h_b_score = hs;
        g.ibk_cap = cap;
    }
    return 1;
}
extern "C" void q38g_qsa_kv_invalidate(int pos) {
    for (int i = 0; i < 128; i++) if (g.kv_valid[i] > pos) g.kv_valid[i] = pos;
}
/* attention for chunk [pos0, pos0+B) of QSA layer `layer`: the projections
 * are already in d_b_qf / d_b_kk / d_b_vv (q38g_qsa_proj_batch); host rows
 * [kv_valid, pos0) are uploaded first, the chunk's new rows come back to the
 * host afterwards.  Returns 0 on failure (nothing written on the host). */
extern "C" int q38g_qsa_batch(int layer, int pos0, int B, int hqn, int hkn,
                              int8_t *hK8, float *hK8s, int8_t *hV8, float *hV8s, int host_max_t,
                              const int *nsel, const int *sel, int n_rot, float theta,
                              float eps, int walk_max, const int *mp3, int s1, int s2) {
    if (!q38g_qsa_ready(layer, pos0 + B) || B < 1 || B > G_BMAX || walk_max > QSA_WALK_MAX) return 0;
    if (host_max_t < pos0 + B) return 0;
    const int hd = g.hd, H = g.H, KV = g.KV, ns = hd / 32, max_t = g.kv_max_t;
    /* 1. sync the host rows the walk may touch (host and device caches have
     * their own row strides: per-request vs configured context) */
    for (int kvh = 0; kvh < KV; kvh++) {
        const int a = g.kv_valid[layer], n = pos0 - a;
        if (n <= 0) break;
        const size_t r0 = (size_t)kvh * max_t + a, h0 = (size_t)kvh * host_max_t + a;
        if (!GCK(cudaMemcpyAsync(g.d_k8[layer] + r0 * hd, hK8 + h0 * hd, (size_t)n * hd, cudaMemcpyHostToDevice, g.st)) ||
            !GCK(cudaMemcpyAsync(g.d_v8[layer] + r0 * hd, hV8 + h0 * hd, (size_t)n * hd, cudaMemcpyHostToDevice, g.st)) ||
            !GCK(cudaMemcpyAsync(g.d_k8s[layer] + r0 * ns, hK8s + h0 * ns, (size_t)n * ns * sizeof(float), cudaMemcpyHostToDevice, g.st)) ||
            !GCK(cudaMemcpyAsync(g.d_v8s[layer] + r0 * ns, hV8s + h0 * ns, (size_t)n * ns * sizeof(float), cudaMemcpyHostToDevice, g.st)))
            return 0;
    }
    g.kv_valid[layer] = pos0 > g.kv_valid[layer] ? pos0 : g.kv_valid[layer];
    /* 2. q norm+rope, k norm+rope, q8 rows */
    const size_t shn = (size_t)hd * sizeof(float);
    const int *dmp = NULL;
    if (mp3) {
        if (!GCK(cudaMemcpyAsync(g.d_mp, mp3, (size_t)B * 3 * sizeof(int), cudaMemcpyHostToDevice, g.st))) return 0;
        dmp = g.d_mp;
    }
    k_qsa_norm_rope<<<B * H, 256, shn, g.st>>>(g.d_b_qf, 2 * hd, H, g.mat[hqn].s, g.d_b_q, hd, n_rot, theta, eps, pos0, dmp, s1, s2);
    k_qsa_norm_rope<<<B * KV, 256, shn, g.st>>>(g.d_b_kk, hd, KV, g.mat[hkn].s, g.d_b_kk, hd, n_rot, theta, eps, pos0, dmp, s1, s2);
    k_qsa_kv_q8<<<B * KV, 256, 0, g.st>>>(g.d_b_kk, g.d_b_vv, KV, hd, max_t, pos0,
                                          g.d_k8[layer], g.d_k8s[layer], g.d_v8[layer], g.d_v8s[layer]);
    /* 3. selection lists + the walk */
    if (!GCK(cudaMemcpyAsync(g.d_nsel, nsel, (size_t)B * sizeof(int), cudaMemcpyHostToDevice, g.st))) return 0;
    int any = 0; for (int t = 0; t < B; t++) if (nsel[t] >= 0) any = 1;
    if (any && !GCK(cudaMemcpyAsync(g.d_sel, sel, (size_t)B * g.sel_cap * sizeof(int), cudaMemcpyHostToDevice, g.st))) return 0;
    const size_t sha = (size_t)(hd + walk_max) * sizeof(float);
    k_qsa_attn<<<B * H, 256, sha, g.st>>>(g.d_b_q, g.d_b_qf, g.d_k8[layer], g.d_k8s[layer], g.d_v8[layer], g.d_v8s[layer],
                                          H, KV, hd, max_t, pos0, g.d_nsel, g.d_sel, g.sel_cap,
                                          1.f / sqrtf((float)hd), g.d_b_ctx);
    if (!GCK(cudaGetLastError())) return 0;
    /* 4. the chunk's rows back to the host cache */
    for (int kvh = 0; kvh < KV; kvh++) {
        const size_t r0 = (size_t)kvh * max_t + pos0, h0 = (size_t)kvh * host_max_t + pos0;
        if (!GCK(cudaMemcpyAsync(hK8 + h0 * hd, g.d_k8[layer] + r0 * hd, (size_t)B * hd, cudaMemcpyDeviceToHost, g.st)) ||
            !GCK(cudaMemcpyAsync(hV8 + h0 * hd, g.d_v8[layer] + r0 * hd, (size_t)B * hd, cudaMemcpyDeviceToHost, g.st)) ||
            !GCK(cudaMemcpyAsync(hK8s + h0 * ns, g.d_k8s[layer] + r0 * ns, (size_t)B * ns * sizeof(float), cudaMemcpyDeviceToHost, g.st)) ||
            !GCK(cudaMemcpyAsync(hV8s + h0 * ns, g.d_v8s[layer] + r0 * ns, (size_t)B * ns * sizeof(float), cudaMemcpyDeviceToHost, g.st)))
            return 0;
    }
    if (!GCK(cudaStreamSynchronize(g.st))) return 0;
    g.kv_valid[layer] = pos0 + B;
    return 1;
}
/* o-projection from the device-resident context */
extern "C" void q38g_qsa_out_batch(int ho, int B, float *blk) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    GMat *m = &g.mat[ho];
    gemm_batch(ho, g.d_b_ctx, g.d_b_out, m->O, m->I, B, m->O, 0);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(blk, g.d_b_out, (size_t)B * m->O * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

/* P10 D5: argmax of one row of lm_head logits on the device (first maximum,
 * as the host scan's strict `>`), so an MTP draft returns 4 bytes instead of
 * the 994 KB logits vector it only needed the argmax of. */
static __global__ void k_argmax_first(const float *__restrict__ v, int n, int *__restrict__ out) {
    float best = -3.4028235e38f; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > best || (x == best && i < bi)) { best = x; bi = i; }
    }
    __shared__ float sv[1024]; __shared__ int si[1024];
    sv[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float ov = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov; si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = si[0];
}
extern "C" int q38g_out_argmax(int h, const float *x) {
    if (!q38g_active()) return -1;
    GMat *m = &g.mat[h];
    if (m->I > g.b_in_max || (size_t)m->O > g.b_out_cap) { g.dead = 1; return -1; }
    if (!GCK(cudaMemcpyAsync(g.d_b_in, x, (size_t)m->I * sizeof(float),
                             cudaMemcpyHostToDevice, g.st))) return -1;
    gemm_batch(h, g.d_b_in, g.d_b_out, m->O, m->I, 1, m->O, 0);
    k_argmax_first<<<1, 1024, 0, g.st>>>(g.d_b_out, m->O, (int *)g.d_b_inj);
    if (!GCK(cudaGetLastError()) ||
        !GCK(cudaMemcpyAsync(g.h_pin, g.d_b_inj, sizeof(int), cudaMemcpyDeviceToHost, g.st)) ||
        !GCK(cudaStreamSynchronize(g.st))) return -1;
    return *(const int *)g.h_pin;
}

extern "C" void q38g_combine_batch(const float *blk, int B, float *res) {
    if (!q38g_active() || B < 1 || B > G_BMAX) return;
    if (blk && !GCK(cudaMemcpyAsync(g.d_b_out, blk, (size_t)B * g.D * sizeof(float),
                                    cudaMemcpyHostToDevice, g.st))) return;
    k_combine_batch<<<(B * g.W + 255) / 256, 256, 0, g.st>>>(
        g.d_b_res, g.d_b_out, g.d_b_inj, g.D, B);
    if (!GCK(cudaGetLastError())) return;
    if (!res) return;                       /* residual stays on the device */
    if (!GCK(cudaMemcpyAsync(res, g.d_b_res, (size_t)B * g.W * sizeof(float),
                             cudaMemcpyDeviceToHost, g.st))) return;
    GCK(cudaStreamSynchronize(g.st));
}

extern "C" int q38g_spec_init(int boundaries) {
    if (!q38g_active() || boundaries < 1 || boundaries >= G_SPEC_BMAX) return 0;
    if (g.d_spec_rec || g.d_spec_ring) return boundaries <= g.spec_n;
    const size_t rn = (size_t)g.vh * g.kd * g.vd;
    const size_t cn = (size_t)g.cdim * (g.convk - 1);
    g.d_spec_rec = (float *)gmalloc((size_t)boundaries * g.n_gdn * rn * sizeof(float),
                                    "spec rec snapshots");
    g.d_spec_ring = (float *)gmalloc((size_t)boundaries * g.n_gdn * cn * sizeof(float),
                                     "spec ring snapshots");
    if (!g.d_spec_rec || !g.d_spec_ring) return 0;
    g.spec_n = boundaries;
    return 1;
}

extern "C" void q38g_spec_restore(int boundary) {
    if (!q38g_active() || boundary < 0 || boundary >= g.spec_n) return;
    const size_t rn = (size_t)g.vh * g.kd * g.vd;
    const size_t cn = (size_t)g.cdim * (g.convk - 1);
    for (int layer = 0; layer < g.nl; layer++) {
        const int gi = layer < 128 ? g.gdn_slot[layer] : -1;
        if (gi < 0) continue;
        GCK(cudaMemcpyAsync(g.d_rec[layer],
                            g.d_spec_rec + ((size_t)boundary*g.n_gdn + gi)*rn,
                            rn*sizeof(float), cudaMemcpyDeviceToDevice, g.st));
        GCK(cudaMemcpyAsync(g.d_ring[layer],
                            g.d_spec_ring + ((size_t)boundary*g.n_gdn + gi)*cn,
                            cn*sizeof(float), cudaMemcpyDeviceToDevice, g.st));
    }
}

extern "C" void q38g_state_put(int layer, const float *rec, const float *ring) {
    if (!q38g_active() || !g.d_rec[layer]) return;
    GCK(cudaMemcpy(g.d_rec[layer], rec, (size_t)g.vh * g.kd * g.vd * sizeof(float),
                   cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(g.d_ring[layer], ring, (size_t)g.cdim * (g.convk - 1) * sizeof(float),
                   cudaMemcpyHostToDevice));
}

extern "C" void q38g_state_get(int layer, float *rec, float *ring) {
    if (!q38g_active() || !g.d_rec[layer]) return;
    GCK(cudaMemcpy(rec, g.d_rec[layer], (size_t)g.vh * g.kd * g.vd * sizeof(float),
                   cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(ring, g.d_ring[layer], (size_t)g.cdim * (g.convk - 1) * sizeof(float),
                   cudaMemcpyDeviceToHost));
}

extern "C" void q38g_sync(void) {
    if (!q38g_active()) return;
    GCK(cudaStreamSynchronize(g.st));
}

#endif /* COLI_CUDA */
