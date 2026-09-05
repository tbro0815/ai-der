/* ggml_blocks.h - GGML block-quant decode for raw GGUF tensor bytes.
 *
 * Self-contained, header-only C99.  Decodes the four block formats the
 * Qwen3.8-Flash-Next GGUF actually uses (docs/research/gguf-structure.md):
 *
 *   Q8_0    32 elems / 34 B   fp16 scale + 32 int8
 *   Q6_K   256 elems / 210 B  fp16 super-scale, 16 int8 sub-scales, 6-bit q
 *   IQ4_NL  32 elems / 18 B   fp16 scale + 16 nibbles into a 16-entry LUT
 *   IQ3_S  256 elems / 110 B  fp16 scale, 4-bit sub-scales, 512-entry grid
 *   IQ4_XS 256 elems / 136 B  fp16 scale, 6-bit sub-scales, IQ4_NL's LUT
 *
 * Block layouts are byte-identical to llama.cpp's ggml-common.h; the structs
 * below carry static asserts on sizeof.  Rows are plain concatenations of
 * blocks (k must be a multiple of the block size; e.g. an IQ4_NL row of 160
 * elements is 5 blocks = 90 bytes).
 *
 * v1 is correctness-first: straightforward scalar loops with per-block scale
 * hoisting, OpenMP across output rows in the GEMVs, no allocations.  The CUDA
 * mirror lives in ggml_blocks_cuda.{h,cu}.
 */
#ifndef COLIBRI_GGML_BLOCKS_H
#define COLIBRI_GGML_BLOCKS_H

#include <stddef.h>
#include <stdint.h>

#include "ggml_blocks_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__cplusplus)
#define GGML_BK_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define GGML_BK_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define GGML_BK_SA_PASTE2(a, b) a##b
#define GGML_BK_SA_PASTE(a, b) GGML_BK_SA_PASTE2(a, b)
#define GGML_BK_STATIC_ASSERT(cond, msg) \
    typedef char GGML_BK_SA_PASTE(ggml_bk_static_assert_, __LINE__)[(cond) ? 1 : -1]
#endif

/* fp16 bits as stored on disk; decoded in software, no F16C dependence. */
typedef uint16_t ggml_bk_half;

enum ggml_bk_type {
    GGML_BK_Q8_0   = 0,
    GGML_BK_Q6_K   = 1,
    GGML_BK_IQ4_NL = 2,
    GGML_BK_IQ3_S  = 3,
    GGML_BK_IQ4_XS = 4,
    GGML_BK_TYPE_COUNT = 5
};

#define GGML_BK_QK8_0  32
#define GGML_BK_QK4_NL 32
#define GGML_BK_QK_K   256

typedef struct {
    ggml_bk_half d;            /* scale */
    int8_t qs[GGML_BK_QK8_0];  /* quants */
} ggml_bk_block_q8_0;
GGML_BK_STATIC_ASSERT(sizeof(ggml_bk_block_q8_0) == 34, "q8_0 block is 34 B");

typedef struct {
    uint8_t ql[GGML_BK_QK_K / 2];      /* quants, lower 4 bits */
    uint8_t qh[GGML_BK_QK_K / 4];      /* quants, upper 2 bits */
    int8_t  scales[GGML_BK_QK_K / 16]; /* per-16 sub-scales, int8 */
    ggml_bk_half d;                    /* super-block scale */
} ggml_bk_block_q6_K;
GGML_BK_STATIC_ASSERT(sizeof(ggml_bk_block_q6_K) == 210, "q6_K block is 210 B");

typedef struct {
    ggml_bk_half d;                 /* scale */
    uint8_t qs[GGML_BK_QK4_NL / 2]; /* nibble indices into kvalues_iq4nl */
} ggml_bk_block_iq4_nl;
GGML_BK_STATIC_ASSERT(sizeof(ggml_bk_block_iq4_nl) == 18, "iq4_nl block is 18 B");

typedef struct {
    ggml_bk_half d;                   /* scale */
    uint8_t qs[GGML_BK_QK_K / 4];     /* grid indices, low 8 bits */
    uint8_t qh[GGML_BK_QK_K / 32];    /* grid indices, 9th bit (8 per byte) */
    uint8_t signs[GGML_BK_QK_K / 8];  /* one sign bit per element */
    uint8_t scales[GGML_BK_QK_K / 64];/* 4-bit sub-scales, 2 per byte */
} ggml_bk_block_iq3_s;
GGML_BK_STATIC_ASSERT(sizeof(ggml_bk_block_iq3_s) == 110, "iq3_s block is 110 B");

typedef struct {
    ggml_bk_half d;                    /* super-block scale */
    uint16_t scales_h;                 /* sub-scale bits 4..5, 2 per sub-block */
    uint8_t  scales_l[GGML_BK_QK_K / 64]; /* sub-scale bits 0..3, 2 per byte */
    uint8_t  qs[GGML_BK_QK_K / 2];     /* nibble indices into kvalues_iq4nl */
} ggml_bk_block_iq4_xs;
GGML_BK_STATIC_ASSERT(sizeof(ggml_bk_block_iq4_xs) == 136, "iq4_xs block is 136 B");

/* ---- fp16 -> fp32 (scalar, round-trip exact for all finite inputs) ---- */

static inline float ggml_bk_fp16_to_fp32(ggml_bk_half h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t em   = (uint32_t)(h & 0x7fffu);
    uint32_t bits;
    float out;
    if (em >= 0x7c00u) {                       /* inf / NaN */
        bits = sign | 0x7f800000u | ((em & 0x03ffu) << 13);
    } else if (em >= 0x0400u) {                /* normal */
        bits = sign | ((em + 0x1c000u) << 13); /* rebias 15 -> 127 */
    } else if (em != 0) {                      /* subnormal: em/2^24 */
        uint32_t m = em, e = 0;
        while (!(m & 0x0400u)) { m <<= 1; ++e; }
        bits = sign | ((uint32_t)(113 - e) << 23) | ((m & 0x03ffu) << 13);
    } else {                                   /* +-0 */
        bits = sign;
    }
    { union { uint32_t u; float f; } cvt; cvt.u = bits; out = cvt.f; }
    return out;
}

/* ---- row-size helpers ---- */

static inline int ggml_bk_block_elems(int type) {
    switch (type) {
    case GGML_BK_Q8_0:   return GGML_BK_QK8_0;
    case GGML_BK_Q6_K:   return GGML_BK_QK_K;
    case GGML_BK_IQ4_NL: return GGML_BK_QK4_NL;
    case GGML_BK_IQ3_S:  return GGML_BK_QK_K;
    case GGML_BK_IQ4_XS: return GGML_BK_QK_K;
    default:             return 0;
    }
}

static inline size_t ggml_bk_block_bytes(int type) {
    switch (type) {
    case GGML_BK_Q8_0:   return sizeof(ggml_bk_block_q8_0);
    case GGML_BK_Q6_K:   return sizeof(ggml_bk_block_q6_K);
    case GGML_BK_IQ4_NL: return sizeof(ggml_bk_block_iq4_nl);
    case GGML_BK_IQ3_S:  return sizeof(ggml_bk_block_iq3_s);
    case GGML_BK_IQ4_XS: return sizeof(ggml_bk_block_iq4_xs);
    default:             return 0;
    }
}

/* Bytes for n elements (n must be a multiple of the block size; returns 0
 * otherwise so a bad shape fails loudly at the caller's size check).
 * IQ4_NL n=160 -> 5 blocks -> 90 bytes. */
static inline size_t ggml_bk_row_bytes(int type, size_t n) {
    const int qk = ggml_bk_block_elems(type);
    if (qk == 0 || n % (size_t)qk != 0) return 0;
    return (n / (size_t)qk) * ggml_bk_block_bytes(type);
}

/* ---- dequant: one row of n elements -> f32 ---- */

static inline void ggml_bk_dequant_row_q8_0(const void *vx, float *y, size_t n) {
    const ggml_bk_block_q8_0 *x = (const ggml_bk_block_q8_0 *)vx;
    const size_t nb = n / GGML_BK_QK8_0;
    for (size_t i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(x[i].d);
        for (int j = 0; j < GGML_BK_QK8_0; ++j) {
            y[i * GGML_BK_QK8_0 + j] = d * (float)x[i].qs[j];
        }
    }
}

static inline void ggml_bk_dequant_row_q6_K(const void *vx, float *y, size_t n) {
    const ggml_bk_block_q6_K *x = (const ggml_bk_block_q6_K *)vx;
    const size_t nb = n / GGML_BK_QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int half = 0; half < 2; ++half) {   /* two 128-elem halves */
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static inline void ggml_bk_dequant_row_iq4_nl(const void *vx, float *y, size_t n) {
    const ggml_bk_block_iq4_nl *x = (const ggml_bk_block_iq4_nl *)vx;
    const size_t nb = n / GGML_BK_QK4_NL;
    for (size_t i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        for (int j = 0; j < GGML_BK_QK4_NL / 2; ++j) {
            y[j +  0] = d * (float)ggml_bk_kvalues_iq4nl[qs[j] & 0xF];
            y[j + 16] = d * (float)ggml_bk_kvalues_iq4nl[qs[j] >>  4];
        }
        y += GGML_BK_QK4_NL;
    }
}

static inline void ggml_bk_dequant_row_iq3_s(const void *vx, float *y, size_t n) {
    const ggml_bk_block_iq3_s *x = (const ggml_bk_block_iq3_s *)vx;
    const size_t nb = n / GGML_BK_QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(x[i].d);
        const uint8_t *qs    = x[i].qs;
        const uint8_t *qh    = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (int ib32 = 0; ib32 < GGML_BK_QK_K / 32; ++ib32) {
            const uint8_t sn = (ib32 & 1) ? (x[i].scales[ib32 / 2] >> 4)
                                          : (x[i].scales[ib32 / 2] & 0xF);
            const float db = d * (float)(1 + 2 * (int)sn);
            for (int l = 0; l < 4; ++l) {  /* 4 pairs of 4-elem grid cells */
                const uint32_t g1 = ggml_bk_iq3s_grid[qs[2 * l + 0] |
                    (((uint32_t)qh[ib32] << (8 - 2 * l)) & 256)];
                const uint32_t g2 = ggml_bk_iq3s_grid[qs[2 * l + 1] |
                    (((uint32_t)qh[ib32] << (7 - 2 * l)) & 256)];
                const uint8_t sg = signs[l];
                for (int j = 0; j < 4; ++j) {
                    const float m1 = (float)((g1 >> (8 * j)) & 0xFF);
                    const float m2 = (float)((g2 >> (8 * j)) & 0xFF);
                    y[j + 0] = db * m1 * ((sg >> (j + 0)) & 1 ? -1.0f : 1.0f);
                    y[j + 4] = db * m2 * ((sg >> (j + 4)) & 1 ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs    += 8;
            signs += 4;
        }
    }
}

static inline void ggml_bk_dequant_row_iq4_xs(const void *vx, float *y, size_t n) {
    const ggml_bk_block_iq4_xs *x = (const ggml_bk_block_iq4_xs *)vx;
    const size_t nb = n / GGML_BK_QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        for (int ib = 0; ib < GGML_BK_QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) |
                           (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * (float)ggml_bk_kvalues_iq4nl[qs[j] & 0xF];
                y[j + 16] = dl * (float)ggml_bk_kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

/* Generic dispatch; returns 0 on success, -1 on unknown type / bad n. */
static inline int ggml_bk_dequant_row(int type, const void *x, float *y, size_t n) {
    if (ggml_bk_row_bytes(type, n) == 0) return -1;
    switch (type) {
    case GGML_BK_Q8_0:   ggml_bk_dequant_row_q8_0(x, y, n);   return 0;
    case GGML_BK_Q6_K:   ggml_bk_dequant_row_q6_K(x, y, n);   return 0;
    case GGML_BK_IQ4_NL: ggml_bk_dequant_row_iq4_nl(x, y, n); return 0;
    case GGML_BK_IQ3_S:  ggml_bk_dequant_row_iq3_s(x, y, n);  return 0;
    case GGML_BK_IQ4_XS: ggml_bk_dequant_row_iq4_xs(x, y, n); return 0;
    default:             return -1;
    }
}

/* ---- GEMV: y[m] = sum_k W[m,k] * x[k] over raw block rows ----
 *
 * W is m rows, each ggml_bk_row_bytes(type, k) bytes, contiguous (the layout
 * of a raw GGUF 2-D tensor).  f32 in, f32 out, f32 accumulate; scales hoisted
 * per block / sub-block.  OpenMP splits output rows, matching the engine's
 * other reference matvecs. */

static inline float ggml_bk_dot_row_q8_0(const uint8_t *row, const float *x, int k) {
    const ggml_bk_block_q8_0 *b = (const ggml_bk_block_q8_0 *)row;
    const int nb = k / GGML_BK_QK8_0;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(b[i].d);
        float s = 0.0f;
        for (int j = 0; j < GGML_BK_QK8_0; ++j) {
            s += (float)b[i].qs[j] * x[i * GGML_BK_QK8_0 + j];
        }
        acc += d * s;
    }
    return acc;
}

static inline float ggml_bk_dot_row_q6_K(const uint8_t *row, const float *x, int k) {
    const ggml_bk_block_q6_K *b = (const ggml_bk_block_q6_K *)row;
    const int nb = k / GGML_BK_QK_K;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(b[i].d);
        const uint8_t *ql = b[i].ql;
        const uint8_t *qh = b[i].qh;
        const int8_t  *sc = b[i].scales;
        const float *xb = x + i * GGML_BK_QK_K;
        for (int half = 0; half < 2; ++half) {
            /* 8 sub-scales per half: hoist d*sc over each 16-elem group. */
            float sub[8];
            int t;
            for (t = 0; t < 8; ++t) sub[t] = 0.0f;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sub[is + 0] += (float)q1 * xb[l +  0];
                sub[is + 2] += (float)q2 * xb[l + 32];
                sub[is + 4] += (float)q3 * xb[l + 64];
                sub[is + 6] += (float)q4 * xb[l + 96];
            }
            for (t = 0; t < 8; ++t) acc += d * (float)sc[t] * sub[t];
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

static inline float ggml_bk_dot_row_iq4_nl(const uint8_t *row, const float *x, int k) {
    const ggml_bk_block_iq4_nl *b = (const ggml_bk_block_iq4_nl *)row;
    const int nb = k / GGML_BK_QK4_NL;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(b[i].d);
        const uint8_t *qs = b[i].qs;
        const float *xb = x + i * GGML_BK_QK4_NL;
        float s = 0.0f;
        for (int j = 0; j < GGML_BK_QK4_NL / 2; ++j) {
            s += (float)ggml_bk_kvalues_iq4nl[qs[j] & 0xF] * xb[j +  0];
            s += (float)ggml_bk_kvalues_iq4nl[qs[j] >>  4] * xb[j + 16];
        }
        acc += d * s;
    }
    return acc;
}

static inline float ggml_bk_dot_row_iq3_s(const uint8_t *row, const float *x, int k) {
    const ggml_bk_block_iq3_s *b = (const ggml_bk_block_iq3_s *)row;
    const int nb = k / GGML_BK_QK_K;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(b[i].d);
        const uint8_t *qs    = b[i].qs;
        const uint8_t *qh    = b[i].qh;
        const uint8_t *signs = b[i].signs;
        const float *xb = x + i * GGML_BK_QK_K;
        for (int ib32 = 0; ib32 < GGML_BK_QK_K / 32; ++ib32) {
            const uint8_t sn = (ib32 & 1) ? (b[i].scales[ib32 / 2] >> 4)
                                          : (b[i].scales[ib32 / 2] & 0xF);
            const float db = d * (float)(1 + 2 * (int)sn);
            float s = 0.0f;
            for (int l = 0; l < 4; ++l) {
                const uint32_t g1 = ggml_bk_iq3s_grid[qs[2 * l + 0] |
                    (((uint32_t)qh[ib32] << (8 - 2 * l)) & 256)];
                const uint32_t g2 = ggml_bk_iq3s_grid[qs[2 * l + 1] |
                    (((uint32_t)qh[ib32] << (7 - 2 * l)) & 256)];
                const uint8_t sg = signs[l];
                for (int j = 0; j < 4; ++j) {
                    const float m1 = (float)((g1 >> (8 * j)) & 0xFF);
                    const float m2 = (float)((g2 >> (8 * j)) & 0xFF);
                    s += m1 * ((sg >> (j + 0)) & 1 ? -xb[8 * l + j + 0] : xb[8 * l + j + 0]);
                    s += m2 * ((sg >> (j + 4)) & 1 ? -xb[8 * l + j + 4] : xb[8 * l + j + 4]);
                }
            }
            acc += db * s;
            xb    += 32;
            qs    += 8;
            signs += 4;
        }
    }
    return acc;
}

static inline float ggml_bk_dot_row_iq4_xs(const uint8_t *row, const float *x, int k) {
    const ggml_bk_block_iq4_xs *b = (const ggml_bk_block_iq4_xs *)row;
    const int nb = k / GGML_BK_QK_K;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_bk_fp16_to_fp32(b[i].d);
        const uint8_t *qs = b[i].qs;
        const float *xb = x + i * GGML_BK_QK_K;
        for (int ib = 0; ib < GGML_BK_QK_K / 32; ++ib) {
            const int ls = ((b[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) |
                           (((b[i].scales_h >> 2 * ib) & 3) << 4);
            float s = 0.0f;
            for (int j = 0; j < 16; ++j) {
                s += (float)ggml_bk_kvalues_iq4nl[qs[j] & 0xF] * xb[j +  0];
                s += (float)ggml_bk_kvalues_iq4nl[qs[j] >>  4] * xb[j + 16];
            }
            acc += d * (float)(ls - 32) * s;
            xb += 32;
            qs += 16;
        }
    }
    return acc;
}

/* Returns 0 on success, -1 on unknown type / k not a block multiple. */
static inline int ggml_bk_gemv(int type, int m, int k, const void *W,
                               const float *x, float *y) {
    const size_t row_bytes = ggml_bk_row_bytes(type, (size_t)k);
    const uint8_t *w = (const uint8_t *)W;
    int r;
    if (row_bytes == 0 || m < 0) return -1;
    switch (type) {
    case GGML_BK_Q8_0:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < m; ++r) y[r] = ggml_bk_dot_row_q8_0(w + (size_t)r * row_bytes, x, k);
        return 0;
    case GGML_BK_Q6_K:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < m; ++r) y[r] = ggml_bk_dot_row_q6_K(w + (size_t)r * row_bytes, x, k);
        return 0;
    case GGML_BK_IQ4_NL:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < m; ++r) y[r] = ggml_bk_dot_row_iq4_nl(w + (size_t)r * row_bytes, x, k);
        return 0;
    case GGML_BK_IQ3_S:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < m; ++r) y[r] = ggml_bk_dot_row_iq3_s(w + (size_t)r * row_bytes, x, k);
        return 0;
    case GGML_BK_IQ4_XS:
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (r = 0; r < m; ++r) y[r] = ggml_bk_dot_row_iq4_xs(w + (size_t)r * row_bytes, x, k);
        return 0;
    default:
        return -1;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_GGML_BLOCKS_H */
