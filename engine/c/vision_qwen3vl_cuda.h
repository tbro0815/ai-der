/* P6.1: the Qwen3-VL vision tower on the GPU.  Same contract as q38v_encode
 * (vision_qwen3vl.h); the weights are uploaded once as F16 (the mmproj's own
 * precision, ~0.9 GB), activations stay f32.  Bit-close to the CPU tower (the
 * GEMMs reorder the sums); `Q38_GPU_VIT=0` keeps the CPU path.  Every entry
 * returns -1 when the port is unavailable, and the caller falls back. */
#ifndef COLIBRI_VISION_QWEN3VL_CUDA_H
#define COLIBRI_VISION_QWEN3VL_CUDA_H
#include <stddef.h>
#include "vision_qwen3vl_types.h"

#ifdef COLI_CUDA
#ifdef __cplusplus
extern "C" {
#endif
int    q38vg_load(const Q38VisTower *t);      /* upload the weights; 0 ok */
int    q38vg_encode(const Q38VisTower *t, const float *patches, int grid_h, int grid_w,
                    const float *pos /*[tokens*hidden]*/, const float *cs /*[tokens*head_dim]*/,
                    float *out);                                  /* 0 ok, -1 fall back */
size_t q38vg_vram_bytes(void);
size_t q38vg_encode_bytes(const Q38VisTower *t, int grid_h, int grid_w); /* scratch of one encode */
void   q38vg_free(void);
#ifdef __cplusplus
}
#endif
#else
static inline int    q38vg_load(const Q38VisTower *t) { (void)t; return -1; }
static inline int    q38vg_encode(const Q38VisTower *t, const float *p, int h, int w,
                                  const float *pos, const float *cs, float *o) {
    (void)t; (void)p; (void)h; (void)w; (void)pos; (void)cs; (void)o; return -1; }
static inline size_t q38vg_vram_bytes(void) { return 0; }
static inline size_t q38vg_encode_bytes(const Q38VisTower *t, int h, int w) { (void)t; (void)h; (void)w; return 0; }
static inline void   q38vg_free(void) {}
#endif
#endif
