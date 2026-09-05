/* Qwen3-VL vision tower: the parsed geometry and the f32 host weights, shared
 * by the CPU implementation (vision_qwen3vl.h) and the CUDA port
 * (vision_qwen3vl_cuda.cu).  Plain C, no includes beyond stdint. */
#ifndef COLIBRI_VISION_QWEN3VL_TYPES_H
#define COLIBRI_VISION_QWEN3VL_TYPES_H

typedef struct {
    int depth, hidden, heads, head_dim, inter;
    int patch, merge, out_dim, pos_side, proj_in;
    float eps, theta;
} Q38VisCfg;

typedef struct {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    float *qkv_w, *qkv_b;        /* [3*hidden, hidden], [3*hidden] */
    float *o_w, *o_b;            /* [hidden, hidden] */
    float *up_w, *up_b;          /* [inter, hidden] */
    float *down_w, *down_b;      /* [hidden, inter] */
} Q38VisBlk;

typedef struct {
    Q38VisCfg c;
    float *patch_w;              /* folded w0+w1: [hidden, 3*patch*patch] */
    float *patch_b;              /* [hidden] */
    float *pos_embd;             /* [pos_side*pos_side, hidden] row-major */
    float *post_w, *post_b;      /* [hidden] */
    float *mm0_w, *mm0_b;        /* [proj_in, proj_in] */
    float *mm2_w, *mm2_b;        /* [out_dim, proj_in] */
    Q38VisBlk *blk;
} Q38VisTower;

#endif
