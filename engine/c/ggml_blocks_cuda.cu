/* ggml_blocks_cuda.cu - CUDA dequant-GEMV over raw GGML block rows.
 *
 * Mirror of the CPU decoders in ggml_blocks.h for the five formats the
 * Qwen3.8-Flash-Next GGUF uses (Q8_0, Q6_K, IQ4_NL, IQ3_S, IQ4_XS).  Self-contained:
 * no ggml headers; block layouts and tables come from ggml_blocks.h /
 * ggml_blocks_tables.h (the same bytes llama.cpp writes).
 *
 * Kernel shape: one warp per output row, dequant-on-the-fly against an f32
 * activation, f32 accumulate, __shfl_down_sync reduction — the structure of
 * llama.cpp's mmvq kernels, minus the q8_1 activation quantization (x stays
 * f32 for v1 correctness; revisit if the batch path gets hot).  Lane L of a
 * warp handles elements e = L + 32*t of each block, so every per-element
 * decode below is written index-first and shared with the dequant kernel.
 *
 * Built for CUDA 12.x / sm_86; only __shfl_down_sync and __half2float beyond
 * baseline C++.  Errors: 0 ok, -1 bad type/shape, else cudaError_t as int.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "ggml_blocks.h"
#include "ggml_blocks_cuda.h"

/* Device copies of the codebooks (host copies live in ggml_blocks_tables.h).
 * P10: ordinary globals read with __ldg, not __constant__ -- constant memory
 * broadcasts only uniform addresses and serializes the per-lane codebook
 * lookups these decoders make (up to 8 distinct IQ3_S entries per warp per
 * element, 16 for IQ4_NL). Same values, same arithmetic. */
static __device__ int8_t   gk_iq4nl[16]  = GGML_BK_KVALUES_IQ4NL_INIT;
static __device__ uint32_t gk_iq3s[512]  = GGML_BK_IQ3S_GRID_INIT;

static __device__ __forceinline__ float gk_fp16(uint16_t bits) {
    __half_raw hr;
    hr.x = bits;
    return __half2float(hr);
}

/* ---- per-element decode: value of element e of the block at `blk` ---- */

static __device__ __forceinline__ float gk_dec_q8_0(const uint8_t *blk, int e) {
    const ggml_bk_block_q8_0 *b = (const ggml_bk_block_q8_0 *)blk;
    return gk_fp16(b->d) * (float)b->qs[e];
}

static __device__ __forceinline__ float gk_dec_q6_K(const uint8_t *blk, int e) {
    const ggml_bk_block_q6_K *b = (const ggml_bk_block_q6_K *)blk;
    const int half = e >> 7;          /* two 128-elem halves */
    const int r    = e & 127;
    const int l    = r & 31;
    const int quad = r >> 5;          /* which 32-elem quadrant of the half */
    const uint8_t *ql = b->ql + half * 64;
    const uint8_t *qh = b->qh + half * 32;
    const int8_t  *sc = b->scales + half * 8;
    int q;
    switch (quad) {
    case 0:  q = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
    case 1:  q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
    case 2:  q = (ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4); break;
    default: q = (ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4); break;
    }
    return gk_fp16(b->d) * (float)sc[quad * 2 + (l >> 4)] * (float)(q - 32);
}

static __device__ __forceinline__ float gk_dec_iq4_nl(const uint8_t *blk, int e) {
    const ggml_bk_block_iq4_nl *b = (const ggml_bk_block_iq4_nl *)blk;
    const uint8_t byte = b->qs[e & 15];
    const int nib = (e < 16) ? (byte & 0xF) : (byte >> 4);
    return gk_fp16(b->d) * (float)__ldg(&gk_iq4nl[nib]);
}

static __device__ __forceinline__ float gk_dec_iq3_s(const uint8_t *blk, int e) {
    const ggml_bk_block_iq3_s *b = (const ggml_bk_block_iq3_s *)blk;
    const int ib32 = e >> 5;          /* 32-elem group */
    const int j    = e & 31;
    const int l    = j >> 3;          /* 8-elem cell pair inside the group */
    const int w8   = j & 7;
    const int pair = w8 >> 2;         /* 0: qs[2l], 1: qs[2l+1] */
    const uint8_t sb = b->scales[ib32 >> 1];
    const int sn = (ib32 & 1) ? (sb >> 4) : (sb & 0xF);
    const uint32_t hi = ((uint32_t)b->qh[ib32] >> (2 * l + pair)) & 1;
    const uint32_t g  = __ldg(&gk_iq3s[b->qs[ib32 * 8 + 2 * l + pair] | (hi << 8)]);
    const float mag   = (float)((g >> (8 * (w8 & 3))) & 0xFF);
    const float sgn   = ((b->signs[ib32 * 4 + l] >> w8) & 1) ? -1.0f : 1.0f;
    return gk_fp16(b->d) * (float)(1 + 2 * sn) * mag * sgn;
}

static __device__ __forceinline__ float gk_dec_iq4_xs(const uint8_t *blk, int e) {
    const ggml_bk_block_iq4_xs *b = (const ggml_bk_block_iq4_xs *)blk;
    const int ib = e >> 5;            /* 32-elem sub-block */
    const int j  = e & 31;
    const int ls = ((b->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) |
                   (((b->scales_h >> (2 * ib)) & 3) << 4);
    const uint8_t byte = b->qs[ib * 16 + (j & 15)];
    const int nib = (j < 16) ? (byte & 0xF) : (byte >> 4);
    return gk_fp16(b->d) * (float)(ls - 32) * (float)__ldg(&gk_iq4nl[nib]);
}

/* Compile-time per-type dispatch (avoids __device__ function pointers as
 * template arguments, which nvcc handles inconsistently across versions). */
template <int TYPE>
static __device__ __forceinline__ float gk_dec(const uint8_t *blk, int e) {
    if (TYPE == GGML_BK_Q8_0)   return gk_dec_q8_0(blk, e);
    if (TYPE == GGML_BK_Q6_K)   return gk_dec_q6_K(blk, e);
    if (TYPE == GGML_BK_IQ4_NL) return gk_dec_iq4_nl(blk, e);
    if (TYPE == GGML_BK_IQ4_XS) return gk_dec_iq4_xs(blk, e);
    return gk_dec_iq3_s(blk, e);
}

template <int TYPE>
static __host__ __device__ __forceinline__ int gk_qk(void) {
    return (TYPE == GGML_BK_Q8_0 || TYPE == GGML_BK_IQ4_NL) ? GGML_BK_QK8_0
                                                            : GGML_BK_QK_K;
}

template <int TYPE>
static __host__ __device__ __forceinline__ size_t gk_bb(void) {
    if (TYPE == GGML_BK_Q8_0)   return sizeof(ggml_bk_block_q8_0);
    if (TYPE == GGML_BK_Q6_K)   return sizeof(ggml_bk_block_q6_K);
    if (TYPE == GGML_BK_IQ4_NL) return sizeof(ggml_bk_block_iq4_nl);
    if (TYPE == GGML_BK_IQ4_XS) return sizeof(ggml_bk_block_iq4_xs);
    return sizeof(ggml_bk_block_iq3_s);
}

/* ---- kernels ---- */

#define GK_WARPS_PER_BLOCK 4

template <int TYPE>
static __global__ void gk_dequant_kernel(const uint8_t *x, float *y, size_t n) {
    const int    qk = gk_qk<TYPE>();
    const size_t bb = gk_bb<TYPE>();
    const size_t e0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t e = e0; e < n; e += stride) {
        y[e] = gk_dec<TYPE>(x + (e / qk) * bb, (int)(e % qk));
    }
}

/* One warp per output row.  Wlist == NULL: single matrix at W0.  Otherwise
 * Wlist is a device array of matrix base pointers and blockIdx.y selects the
 * matrix.  xstride == 0: all matrices share the activation x (the gate/up
 * expert-group case); xstride > 0: matrix i reads x + i*xstride (the down
 * projection, where every expert has its own hidden vector). */
template <int TYPE>
static __global__ void gk_gemv_kernel(int m, int k, const void *W0,
                                      const void *const *Wlist,
                                      const float *x0, size_t xstride, float *y) {
    const float *x = x0 + (size_t)blockIdx.y * xstride;
    const int    qk = gk_qk<TYPE>();
    const size_t bb = gk_bb<TYPE>();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * GK_WARPS_PER_BLOCK + warp;
    if (row >= m) return;
    const uint8_t *base = (const uint8_t *)(Wlist ? Wlist[blockIdx.y] : W0);
    const uint8_t *wrow = base + (size_t)row * ((size_t)(k / qk) * bb);
    const int nb = k / qk;
    float acc = 0.0f;
    if (TYPE == GGML_BK_IQ3_S) {
        /* same 8-element-cell-per-lane order as the grouped kernel, so the
         * per-pair and grouped paths stay bit-identical to each other */
        const int cell = lane, ib32 = cell >> 2, l = cell & 3;
        for (int i = 0; i < nb; ++i) {
            const ggml_bk_block_iq3_s *b = (const ggml_bk_block_iq3_s *)(wrow + (size_t)i * bb);
            const float d = gk_fp16(b->d);
            const uint8_t sb = b->scales[ib32 >> 1];
            const int sn = (ib32 & 1) ? (sb >> 4) : (sb & 0xF);
            const float sc = d * (float)(1 + 2 * sn);
            const uint32_t qh = b->qh[ib32];
            const uint32_t q0 = b->qs[ib32 * 8 + 2 * l], q1 = b->qs[ib32 * 8 + 2 * l + 1];
            const uint32_t g0 = __ldg(&gk_iq3s[q0 | (((qh >> (2 * l)) & 1) << 8)]);
            const uint32_t g1 = __ldg(&gk_iq3s[q1 | (((qh >> (2 * l + 1)) & 1) << 8)]);
            const uint32_t sg = b->signs[ib32 * 4 + l];
            const float *xb = x + (size_t)i * qk + (size_t)cell * 8;
            const float4 xa = *(const float4 *)(xb), xc = *(const float4 *)(xb + 4);
            const float xv[8] = { xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w };
#pragma unroll
            for (int w8 = 0; w8 < 8; ++w8) {
                const uint32_t g = w8 < 4 ? g0 : g1;
                const float mag = (float)((g >> (8 * (w8 & 3))) & 0xFF);
                const float v = sc * mag;
                acc += (((sg >> w8) & 1) ? -v : v) * xv[w8];
            }
        }
    } else
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = wrow + (size_t)i * bb;
        const float *xb = x + i * qk;
        for (int e = lane; e < qk; e += 32) {
            acc += gk_dec<TYPE>(blk, e) * xb[e];
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[(size_t)blockIdx.y * m + row] = acc;
    }
}

/* One warp owns one output row and reuses its decoded weights across up to
 * four token/expert pairs. Pair-indexed output preserves the caller's stable
 * token-major accumulation order. */
#define GK_GROUP_TILE 4
template <int TYPE>
static __global__ void gk_gemv_grouped_kernel(int m, int k,
                                              const void *const *W,
                                              const int *tile_group,
                                              const int *tile_offsets,
                                              const int *pair_indices,
                                              const int *pair_x,
                                              const float *x, size_t xstride,
                                              float *y) {
    const int group = tile_group[blockIdx.y];
    const int begin = tile_offsets[blockIdx.y];
    const int end = tile_offsets[blockIdx.y + 1];
    const int qk = gk_qk<TYPE>();
    const size_t bb = gk_bb<TYPE>();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * GK_WARPS_PER_BLOCK + warp;
    if (row >= m) return;
    const uint8_t *wrow = (const uint8_t *)W[group] +
                          (size_t)row * ((size_t)(k / qk) * bb);
    float acc[GK_GROUP_TILE];
#pragma unroll
    for (int j = 0; j < GK_GROUP_TILE; ++j) acc[j] = 0.f;
    const int count = end - begin;
    /* x row per pair: the pair itself, or its token when pair_x maps pairs
     * to a per-token activation block (no per-pair replication) */
    const float *xr[GK_GROUP_TILE];
#pragma unroll
    for (int j = 0; j < GK_GROUP_TILE; ++j) {
        const int p = j < count ? pair_indices[begin + j] : pair_indices[begin];
        xr[j] = x + (size_t)(pair_x ? pair_x[p] : p) * xstride;
    }
    if (TYPE == GGML_BK_IQ3_S) {
        /* P10 P3: one 8-element cell per lane (32 cells = one 256-block per
         * warp step): five small loads + two grid lookups decode eight
         * values, and the eight activations come as two float4.  Per-element
         * values equal gk_dec_iq3_s bit for bit (same d*(1+2s), *mag, sign);
         * the lane->element mapping is contiguous instead of strided, so the
         * warp sum order differs from the per-element kernel (bit-close
         * class, same as GPU experts vs the CPU decoders).  The ncu profile
         * of the per-element kernel: SM issue and L1/TEX both at 89 %. */
        const int cell = lane, ib32 = cell >> 2, l = cell & 3;
        for (int i = 0; i < k / qk; ++i) {
            const ggml_bk_block_iq3_s *b = (const ggml_bk_block_iq3_s *)(wrow + (size_t)i * bb);
            const float d = gk_fp16(b->d);
            const uint8_t sb = b->scales[ib32 >> 1];
            const int sn = (ib32 & 1) ? (sb >> 4) : (sb & 0xF);
            const float sc = d * (float)(1 + 2 * sn);
            const uint32_t qh = b->qh[ib32];
            const uint32_t q0 = b->qs[ib32 * 8 + 2 * l], q1 = b->qs[ib32 * 8 + 2 * l + 1];
            const uint32_t g0 = __ldg(&gk_iq3s[q0 | (((qh >> (2 * l)) & 1) << 8)]);
            const uint32_t g1 = __ldg(&gk_iq3s[q1 | (((qh >> (2 * l + 1)) & 1) << 8)]);
            const uint32_t sg = b->signs[ib32 * 4 + l];
            float wv[8];
#pragma unroll
            for (int w8 = 0; w8 < 8; ++w8) {
                const uint32_t g = w8 < 4 ? g0 : g1;
                const float mag = (float)((g >> (8 * (w8 & 3))) & 0xFF);
                const float v = sc * mag;
                wv[w8] = ((sg >> w8) & 1) ? -v : v;
            }
            const size_t e0 = (size_t)i * qk + (size_t)cell * 8;
#pragma unroll
            for (int j = 0; j < GK_GROUP_TILE; ++j) {
                if (j < count) {
                    const float4 xa = *(const float4 *)(xr[j] + e0);
                    const float4 xb = *(const float4 *)(xr[j] + e0 + 4);
                    acc[j] += wv[0] * xa.x; acc[j] += wv[1] * xa.y;
                    acc[j] += wv[2] * xa.z; acc[j] += wv[3] * xa.w;
                    acc[j] += wv[4] * xb.x; acc[j] += wv[5] * xb.y;
                    acc[j] += wv[6] * xb.z; acc[j] += wv[7] * xb.w;
                }
            }
        }
    } else
    for (int i = 0; i < k / qk; ++i) {
        const uint8_t *blk = wrow + (size_t)i * bb;
        for (int e = lane; e < qk; e += 32) {
            const float wv = gk_dec<TYPE>(blk, e);
#pragma unroll
            for (int j = 0; j < GK_GROUP_TILE; ++j)
                if (j < count)
                    acc[j] += wv * xr[j][(size_t)i * qk + e];
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int j = 0; j < GK_GROUP_TILE; ++j)
            acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
    if (lane == 0)
        for (int j = 0; j < count; ++j)
            y[(size_t)pair_indices[begin + j] * m + row] = acc[j];
}

/* SwiGLU glue for the expert chain: h[i] = silu(g[i]) * u[i].  Lets the MoE
 * tier run gate|up -> silu*mul -> down as one async chain with no host hop. */
static __global__ void gk_silu_mul_kernel(const float *g, const float *u,
                                          float *h, size_t n) {
    const size_t i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = i0; i < n; i += stride) {
        const float gv = g[i];
        h[i] = gv / (1.0f + expf(-gv)) * u[i];   /* full-precision expf: keep
                                                    GPU/CPU silu drift at ulps */
    }
}

/* ---- launch helpers ---- */

static int gk_launch_dequant(int type, const void *x, float *y, size_t n,
                             cudaStream_t st) {
    const int threads = 256;
    size_t want = (n + threads - 1) / threads;
    const int blocks = (int)(want > 4096 ? 4096 : want);
    const uint8_t *xb = (const uint8_t *)x;
    switch (type) {
    case GGML_BK_Q8_0:
        gk_dequant_kernel<GGML_BK_Q8_0><<<blocks, threads, 0, st>>>(xb, y, n);
        break;
    case GGML_BK_Q6_K:
        gk_dequant_kernel<GGML_BK_Q6_K><<<blocks, threads, 0, st>>>(xb, y, n);
        break;
    case GGML_BK_IQ4_NL:
        gk_dequant_kernel<GGML_BK_IQ4_NL><<<blocks, threads, 0, st>>>(xb, y, n);
        break;
    case GGML_BK_IQ3_S:
        gk_dequant_kernel<GGML_BK_IQ3_S><<<blocks, threads, 0, st>>>(xb, y, n);
        break;
    case GGML_BK_IQ4_XS:
        gk_dequant_kernel<GGML_BK_IQ4_XS><<<blocks, threads, 0, st>>>(xb, y, n);
        break;
    default:
        return -1;
    }
    return (int)cudaGetLastError();
}

static int gk_launch_gemv(int type, int m, int k, int n_mat, const void *W0,
                          const void *const *Wlist, const float *x,
                          size_t xstride, float *y, cudaStream_t st) {
    const dim3 grid((m + GK_WARPS_PER_BLOCK - 1) / GK_WARPS_PER_BLOCK, n_mat);
    const dim3 block(32 * GK_WARPS_PER_BLOCK);
    switch (type) {
    case GGML_BK_Q8_0:
        gk_gemv_kernel<GGML_BK_Q8_0><<<grid, block, 0, st>>>(m, k, W0, Wlist, x, xstride, y);
        break;
    case GGML_BK_Q6_K:
        gk_gemv_kernel<GGML_BK_Q6_K><<<grid, block, 0, st>>>(m, k, W0, Wlist, x, xstride, y);
        break;
    case GGML_BK_IQ4_NL:
        gk_gemv_kernel<GGML_BK_IQ4_NL><<<grid, block, 0, st>>>(m, k, W0, Wlist, x, xstride, y);
        break;
    case GGML_BK_IQ3_S:
        gk_gemv_kernel<GGML_BK_IQ3_S><<<grid, block, 0, st>>>(m, k, W0, Wlist, x, xstride, y);
        break;
    case GGML_BK_IQ4_XS:
        gk_gemv_kernel<GGML_BK_IQ4_XS><<<grid, block, 0, st>>>(m, k, W0, Wlist, x, xstride, y);
        break;
    default:
        return -1;
    }
    return (int)cudaGetLastError();
}

/* out[t][d] = ((out[t][d] + w[p0]*y[p0][d]) + w[p1]*y[p1][d]) + ... over the
 * token's pairs in list order: the host accumulation loop, one FMA per pair
 * (the x86-64-v3 host build contracts `out += w*row` the same way). */
static __global__ void gk_scatter_add_kernel(const float *y, int D,
                                             const int *tok_off, const int *tok_list,
                                             const float *w, float *out) {
    const int t = blockIdx.x;
    const int b = tok_off[t], e = tok_off[t + 1];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = out[(size_t)t * D + d];
        for (int i = b; i < e; ++i) {
            const int p = tok_list[i];
            acc = fmaf(w[p], y[(size_t)p * D + d], acc);
        }
        out[(size_t)t * D + d] = acc;
    }
}

static int gk_launch_grouped(int type, int m, int k, int n_tile,
                             const void *const *W, const int *tile_group,
                             const int *tile_offsets, const int *pair_indices,
                             const int *pair_x,
                             const float *x, size_t xstride, float *y,
                             cudaStream_t st) {
    const dim3 grid((m + GK_WARPS_PER_BLOCK - 1) / GK_WARPS_PER_BLOCK, n_tile);
    const dim3 block(32 * GK_WARPS_PER_BLOCK);
#define GK_GROUP_CASE(T) \
    case T: gk_gemv_grouped_kernel<T><<<grid, block, 0, st>>>( \
        m, k, W, tile_group, tile_offsets, pair_indices, pair_x, x, xstride, y); break
    switch (type) {
    GK_GROUP_CASE(GGML_BK_Q8_0);
    GK_GROUP_CASE(GGML_BK_Q6_K);
    GK_GROUP_CASE(GGML_BK_IQ4_NL);
    GK_GROUP_CASE(GGML_BK_IQ3_S);
    GK_GROUP_CASE(GGML_BK_IQ4_XS);
    default: return -1;
    }
#undef GK_GROUP_CASE
    return (int)cudaGetLastError();
}

/* ---- public C entries (see ggml_blocks_cuda.h) ---- */

extern "C" int ggml_blocks_dequant_row_cuda(int type, const void *x, float *y,
                                            size_t n, void *stream) {
    if (ggml_bk_row_bytes(type, n) == 0) return -1;
    return gk_launch_dequant(type, x, y, n, (cudaStream_t)stream);
}

extern "C" int ggml_blocks_gemv_cuda(int type, int m, int k, const void *W,
                                     const float *x, float *y, void *stream) {
    if (m <= 0 || ggml_bk_row_bytes(type, (size_t)k) == 0) return -1;
    return gk_launch_gemv(type, m, k, 1, W, NULL, x, 0, y, (cudaStream_t)stream);
}

extern "C" int ggml_blocks_gemv_batch(int type, int m, int k, int n_mat,
                                      const void **W, const float *x, float *y,
                                      void *stream) {
    if (m <= 0 || n_mat <= 0 || ggml_bk_row_bytes(type, (size_t)k) == 0) return -1;
    return gk_launch_gemv(type, m, k, n_mat, NULL, (const void *const *)W, x, 0,
                          y, (cudaStream_t)stream);
}

extern "C" int ggml_blocks_silu_mul_cuda(const float *g, const float *u,
                                         float *h, size_t n, void *stream) {
    const int threads = 256;
    size_t want = (n + threads - 1) / threads;
    const int blocks = (int)(want > 1024 ? 1024 : want);
    gk_silu_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(g, u, h, n);
    return (int)cudaGetLastError();
}

extern "C" int ggml_blocks_gemv_batch_sx(int type, int m, int k, int n_mat,
                                         const void **W, const float *x,
                                         size_t xstride, float *y,
                                         void *stream) {
    if (m <= 0 || n_mat <= 0 || ggml_bk_row_bytes(type, (size_t)k) == 0) return -1;
    return gk_launch_gemv(type, m, k, n_mat, NULL, (const void *const *)W, x,
                          xstride, y, (cudaStream_t)stream);
}

extern "C" int ggml_blocks_gemv_grouped_sx(
        int type, int m, int k, int n_tile, const void **W,
        const int *tile_group, const int *tile_offsets,
        const int *pair_indices, const float *x,
        size_t xstride, float *y, void *stream) {
    if (m <= 0 || n_tile <= 0 ||
        ggml_bk_row_bytes(type, (size_t)k) == 0) return -1;
    return gk_launch_grouped(type, m, k, n_tile, (const void *const *)W,
                             tile_group, tile_offsets, pair_indices, NULL, x, xstride, y,
                             (cudaStream_t)stream);
}

extern "C" int ggml_blocks_gemv_grouped_x(
        int type, int m, int k, int n_tile, const void **W,
        const int *tile_group, const int *tile_offsets,
        const int *pair_indices, const int *pair_x, const float *x,
        size_t xstride, float *y, void *stream) {
    if (m <= 0 || n_tile <= 0 ||
        ggml_bk_row_bytes(type, (size_t)k) == 0) return -1;
    return gk_launch_grouped(type, m, k, n_tile, (const void *const *)W,
                             tile_group, tile_offsets, pair_indices, pair_x, x, xstride, y,
                             (cudaStream_t)stream);
}

extern "C" int ggml_blocks_scatter_add_cuda(const float *y, int D, int n_tok,
                                            const int *tok_off, const int *tok_list,
                                            const float *w, float *out, void *stream) {
    if (D <= 0 || n_tok <= 0) return -1;
    gk_scatter_add_kernel<<<n_tok, 256, 0, (cudaStream_t)stream>>>(y, D, tok_off, tok_list, w, out);
    return (int)cudaGetLastError();
}
