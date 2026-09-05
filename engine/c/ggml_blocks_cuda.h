/* ggml_blocks_cuda.h - CUDA GEMV over raw GGML block rows (ggml_blocks.h).
 *
 * C-callable interface; `stream` is a cudaStream_t passed as void* so this
 * header needs no CUDA headers on the C side (same trick the engine uses for
 * its loader-facing GPU entries).  All pointers are DEVICE pointers; the
 * `W` array of the batch entry is a device array of device pointers.
 *
 * Compiled from ggml_blocks_cuda.cu only; every entry returns 0 on success,
 * -1 on bad type/shape, or a positive cudaError_t from the launch.
 */
#ifndef COLIBRI_GGML_BLOCKS_CUDA_H
#define COLIBRI_GGML_BLOCKS_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dequantize one row of n elements to f32 (n a multiple of the block size).
 * x: device pointer to raw blocks; y: device pointer to n floats. */
int ggml_blocks_dequant_row_cuda(int type, const void *x, float *y, size_t n,
                                 void *stream);

/* y[m] = W[m,k] * x[k].  W: raw block rows, contiguous; f32 in/out. */
int ggml_blocks_gemv_cuda(int type, int m, int k, const void *W,
                          const float *x, float *y, void *stream);

/* Batched GEMV for expert groups: n_mat matrices of identical shape [m,k]
 * share one activation x[k] (the MoE case: several experts, one token).
 * W is a DEVICE array of n_mat device pointers to raw block rows;
 * y is n_mat * m floats, matrix-major (y + i*m for matrix i). */
int ggml_blocks_gemv_batch(int type, int m, int k, int n_mat,
                           const void **W, const float *x, float *y,
                           void *stream);

/* h[i] = silu(g[i]) * u[i], elementwise on device (the MoE SwiGLU glue). */
int ggml_blocks_silu_mul_cuda(const float *g, const float *u, float *h,
                              size_t n, void *stream);

/* Batched GEMV with a PER-MATRIX activation: matrix i reads x + i*xstride
 * (floats).  xstride == 0 degenerates to ggml_blocks_gemv_batch.  The MoE
 * down projection uses this: every expert has its own hidden vector. */
int ggml_blocks_gemv_batch_sx(int type, int m, int k, int n_mat,
                              const void **W, const float *x, size_t xstride,
                              float *y, void *stream);

/* Grouped batched GEMV. Tile t owns W[tile_group[t]] and pair_indices in
 * [tile_offsets[t], tile_offsets[t+1]); tiles contain at most four pairs.
 * Each pair p reads x+p*xstride and writes y+p*m. The kernel decodes each
 * weight row once per tile, reusing it across repeated routes to one expert.
 * All metadata lives on device. */
int ggml_blocks_gemv_grouped_sx(int type, int m, int k, int n_tile,
                                const void **W, const int *tile_group,
                                const int *tile_offsets,
                                const int *pair_indices, const float *x,
                                size_t xstride, float *y, void *stream);
/* As above, but x row for pair p is pair_x[p] (a per-token activation block)
 * instead of p itself; pair_x == NULL restores the per-pair layout. */
int ggml_blocks_gemv_grouped_x(int type, int m, int k, int n_tile,
                               const void **W, const int *tile_group,
                               const int *tile_offsets,
                               const int *pair_indices, const int *pair_x,
                               const float *x, size_t xstride, float *y,
                               void *stream);
/* Weighted per-token accumulation of pair outputs, host order preserved:
 * for token t, for i in tok_off[t]..tok_off[t+1]-1: out[t] += w[p]*y[p]
 * with p = tok_list[i]; one FMA per pair per element. */
int ggml_blocks_scatter_add_cuda(const float *y, int D, int n_tok,
                                 const int *tok_off, const int *tok_list,
                                 const float *w, float *out, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_GGML_BLOCKS_CUDA_H */
