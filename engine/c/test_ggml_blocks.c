/* test_ggml_blocks.c - standalone oracle for the GGML block-quant decoders.
 *
 * Two independent implementations face off:
 *   - the library: ggml_blocks.h (f32 row dequant + f32 GEMV), and, when
 *     built with -DGGML_BLOCKS_CUDA, the ggml_blocks_cuda.cu kernels;
 *   - the reference: per-element f64 decoders written HERE, index-first from
 *     the format spec, structurally unlike the library's row loops.
 *
 * Coverage per type (Q8_0, Q6_K, IQ4_NL, IQ3_S, IQ4_XS):
 *   1. random raw blocks (every bit pattern is decodable) -> library dequant
 *      vs f64 reference, 1e-6 relative (same math, float vs double);
 *   2. encode round-trip for the linear-ish types (Q8_0, Q6_K, IQ4_NL) with
 *      simple encoders written here -> decoded output within the format's
 *      quantization step of the input (sanity that the byte layout we WRITE
 *      is the byte layout we READ);
 *   3. CPU GEMV vs naive f64 GEMV over the f64-decoded matrix, 1e-4 relative
 *      (f32 accumulation over k=2560);
 *   4. (CUDA builds) CUDA dequant + GEMV + batched GEMV vs the CPU results,
 *      1e-4 relative.
 *
 * Cross-check mode:  test_ggml_blocks --vectors <file>
 * reads externally generated vectors (e.g. from tools/convert_qwen38.py's
 * python dequantizers) and compares the CPU dequant against them.  Binary
 * format, little-endian, no padding:
 *   magic   u32  0x47425456              ("GBTV")
 *   version u32  1
 * then records until EOF, each:
 *   type    u32  0=Q8_0 1=Q6_K 2=IQ4_NL 3=IQ3_S 4=IQ4_XS (enum ggml_bk_type)
 *   n       u32  element count (multiple of the block size)
 *   nbytes  u64  raw size, must equal ggml_bk_row_bytes(type, n)
 *   raw     nbytes bytes of blocks
 *   ref     n * f32 expected dequantized values
 * Tolerance for vectors: 1e-5 relative (both sides are f32 renditions of the
 * same exact math).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml_blocks.h"
#ifdef GGML_BLOCKS_CUDA
#include "ggml_blocks_cuda.h"
#include <cuda_runtime_api.h>  /* the C-compatible runtime header */
#endif

static int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* xorshift64: deterministic across platforms */
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd_u64(void) {
    uint64_t x = g_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return g_rng = x;
}
static uint8_t  rnd_u8(void)  { return (uint8_t)(rnd_u64() >> 40); }
static float    rnd_unit(void){ return (float)((rnd_u64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }

/* ---- fp16 helpers for the reference side (independent of the header) ---- */

static double ref_fp16(uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int exp  = (h >> 10) & 0x1f;
    const int man  = h & 0x3ff;
    double v;
    if (exp == 0x1f)      v = man ? (0.0 / 0.0) : (1.0 / 0.0);
    else if (exp == 0)    v = ldexp((double)man, -24);
    else                  v = ldexp((double)(man + 1024), exp - 25);
    return sign ? -v : v;
}

static uint16_t fp16_encode(float f) {  /* round-to-nearest-even, finite only */
    union { float f; uint32_t u; } c; c.f = f;
    const uint32_t sign = (c.u >> 16) & 0x8000u;
    int32_t e = (int32_t)((c.u >> 23) & 0xff) - 127 + 15;
    uint32_t m = c.u & 0x7fffffu;
    if (e >= 31) return (uint16_t)(sign | 0x7bffu);   /* clamp to max finite */
    if (e <= 0) {                                     /* subnormal / zero */
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        { const int shift = 14 - e;
          const uint32_t q = m >> shift;
          const uint32_t rem = m & ((1u << shift) - 1);
          const uint32_t half = 1u << (shift - 1);
          uint32_t r = q;
          if (rem > half || (rem == half && (q & 1))) ++r;
          return (uint16_t)(sign | r); }
    }
    { const uint32_t q = m >> 13, rem = m & 0x1fffu;
      uint32_t r = ((uint32_t)e << 10) | q;
      if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
      if (r >= 0x7c00u) r = 0x7bffu;
      return (uint16_t)(sign | r); }
}

/* ---- independent f64 per-element decoders (from the spec, index-first) ---- */

static const int8_t ref_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                      1, 13, 25, 38, 53, 69, 89, 113 };

static double ref_decode_elem(int type, const uint8_t *row, size_t e) {
    switch (type) {
    case GGML_BK_Q8_0: {
        const uint8_t *b = row + (e / 32) * 34;
        uint16_t d; memcpy(&d, b, 2);
        return ref_fp16(d) * (double)(int8_t)b[2 + e % 32];
    }
    case GGML_BK_Q6_K: {
        const uint8_t *b = row + (e / 256) * 210;
        const size_t i = e % 256;
        const int half = (int)(i / 128), r = (int)(i % 128);
        const int l = r % 32, quad = r / 32;
        const uint8_t *ql = b + half * 64;             /* ql[128] at off 0 */
        const uint8_t *qh = b + 128 + half * 32;       /* qh[64] */
        const int8_t  *sc = (const int8_t *)(b + 192) + half * 8; /* scales[16] */
        uint16_t d; memcpy(&d, b + 208, 2);
        int q;
        if      (quad == 0) q = (ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (quad == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (quad == 2) q = (ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4);
        else                q = (ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4);
        return ref_fp16(d) * (double)sc[quad * 2 + l / 16] * (double)(q - 32);
    }
    case GGML_BK_IQ4_NL: {
        const uint8_t *b = row + (e / 32) * 18;
        const size_t i = e % 32;
        uint16_t d; memcpy(&d, b, 2);
        const uint8_t byte = b[2 + i % 16];
        return ref_fp16(d) * (double)ref_iq4nl[i < 16 ? (byte & 0xF) : (byte >> 4)];
    }
    case GGML_BK_IQ3_S: {
        const uint8_t *b = row + (e / 256) * 110;
        const size_t i = e % 256;
        const int ib32 = (int)(i / 32), j = (int)(i % 32);
        const int l = j / 8, w8 = j % 8, pair = (w8 % 8) / 4;
        uint16_t d; memcpy(&d, b, 2);
        const uint8_t *qs     = b + 2;        /* qs[64] */
        const uint8_t *qh     = b + 66;       /* qh[8] */
        const uint8_t *signs  = b + 74;       /* signs[32] */
        const uint8_t *scales = b + 106;      /* scales[4] */
        const int sn = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xF);
        const int hi = (qh[ib32] >> (2 * l + pair)) & 1;
        const uint32_t g = ggml_bk_iq3s_grid[qs[ib32 * 8 + 2 * l + pair] | (hi << 8)];
        const double mag = (double)((g >> (8 * (w8 % 4))) & 0xFF);
        const double sgn = ((signs[ib32 * 4 + l] >> w8) & 1) ? -1.0 : 1.0;
        return ref_fp16(d) * (double)(1 + 2 * sn) * mag * sgn;
    }
    case GGML_BK_IQ4_XS: {
        const uint8_t *b = row + (e / 256) * 136;
        const size_t i = e % 256;
        const int ib = (int)(i / 32), j = (int)(i % 32);
        uint16_t d, sh; memcpy(&d, b, 2); memcpy(&sh, b + 2, 2);
        const uint8_t *sl = b + 4;    /* scales_l[4] */
        const uint8_t *qs = b + 8;    /* qs[128] */
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                       (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t byte = qs[ib * 16 + j % 16];
        const int nib = (j < 16) ? (byte & 0xF) : (byte >> 4);
        return ref_fp16(d) * (double)(ls - 32) * (double)ref_iq4nl[nib];
    }
    default:
        return 0.0 / 0.0;
    }
}

/* ---- random raw rows (any bit pattern decodes; keep d finite & modest) ---- */

static void fill_random_row(int type, uint8_t *row, size_t n) {
    const size_t bb = ggml_bk_block_bytes(type);
    const size_t nb = n / (size_t)ggml_bk_block_elems(type);
    for (size_t i = 0; i < nb * bb; ++i) row[i] = rnd_u8();
    /* overwrite each block's fp16 scale with a finite value in [2^-8, 2^0) */
    for (size_t i = 0; i < nb; ++i) {
        const uint16_t d = fp16_encode(ldexpf(0.5f + 0.5f * fabsf(rnd_unit()),
                                              -(int)(rnd_u64() % 8)));
        size_t off = i * bb;
        if (type == GGML_BK_Q6_K) off += 208;  /* d is the trailing field */
        memcpy(row + off, &d, 2);
    }
}

/* ---- simple encoders (layout-writing sanity; not quality encoders) ---- */

static void encode_q8_0(const float *x, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n / 32; ++i, out += 34, x += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
        const float d = amax / 127.0f;
        const uint16_t dh = fp16_encode(d);
        const float dd = (float)ref_fp16(dh);
        const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        memcpy(out, &dh, 2);
        for (int j = 0; j < 32; ++j) {
            int q = (int)lrintf(x[j] * id);
            if (q < -127) q = -127; if (q > 127) q = 127;
            ((int8_t *)out)[2 + j] = (int8_t)q;
        }
    }
}

static void encode_iq4_nl(const float *x, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n / 32; ++i, out += 18, x += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
        const uint16_t dh = fp16_encode(amax / 127.0f);
        const float dd = (float)ref_fp16(dh);
        memcpy(out, &dh, 2);
        for (int j = 0; j < 16; ++j) {
            int best_lo = 0, best_hi = 0;
            float err_lo = 1e30f, err_hi = 1e30f;
            for (int t = 0; t < 16; ++t) {
                const float v = dd * (float)ref_iq4nl[t];
                const float e1 = fabsf(x[j] - v), e2 = fabsf(x[j + 16] - v);
                if (e1 < err_lo) { err_lo = e1; best_lo = t; }
                if (e2 < err_hi) { err_hi = e2; best_hi = t; }
            }
            out[2 + j] = (uint8_t)(best_lo | (best_hi << 4));
        }
    }
}

static void encode_q6_K(const float *x, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n / 256; ++i, out += 210, x += 256) {
        /* per-16 sub-scales; d sized so the largest fits in int8 */
        float sub_amax[16];
        float smax = 0.0f;
        for (int s = 0; s < 16; ++s) {
            float a = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const float v = fabsf(x[s * 16 + j]);
                if (v > a) a = v;
            }
            sub_amax[s] = a / 31.0f;                 /* q-32 in [-32,31] */
            if (sub_amax[s] > smax) smax = sub_amax[s];
        }
        const uint16_t dh = fp16_encode(smax > 0 ? smax / 127.0f : 0.0f);
        const double dd = ref_fp16(dh);
        memcpy(out + 208, &dh, 2);
        int8_t *sc = (int8_t *)(out + 192);
        for (int s = 0; s < 16; ++s) {
            int v = dd > 0 ? (int)lrint(sub_amax[s] / dd) : 0;
            if (v > 127) v = 127; if (v < 1) v = 1;
            sc[s] = (int8_t)v;
        }
        memset(out, 0, 192);
        for (int e = 0; e < 256; ++e) {
            const int half = e / 128, r = e % 128, l = r % 32, quad = r / 32;
            const int sidx = half * 8 + quad * 2 + l / 16;
            const double step = dd * (double)sc[sidx];
            int q = step > 0 ? (int)lrint(x[e] / step) + 32 : 32;
            if (q < 0) q = 0; if (q > 63) q = 63;
            {   uint8_t *ql = out + half * 64;
                uint8_t *qh = out + 128 + half * 32;
                const int lo = q & 0xF, hi = q >> 4;
                if      (quad == 0) { ql[l]      |= (uint8_t)lo;        qh[l] |= (uint8_t)(hi << 0); }
                else if (quad == 1) { ql[l + 32] |= (uint8_t)lo;        qh[l] |= (uint8_t)(hi << 2); }
                else if (quad == 2) { ql[l]      |= (uint8_t)(lo << 4); qh[l] |= (uint8_t)(hi << 4); }
                else                { ql[l + 32] |= (uint8_t)(lo << 4); qh[l] |= (uint8_t)(hi << 6); }
            }
        }
    }
}

static void encode_iq4_xs(const float *x, uint8_t *out, size_t n) {
    for (size_t i = 0; i < n / 256; ++i, out += 136, x += 256) {
        /* per-32 sub-scale via ls-32 in 1..31, d sized from the largest */
        float sub_amax[8];
        float smax = 0.0f;
        uint16_t sh = 0;
        int ib;
        for (ib = 0; ib < 8; ++ib) {
            float a = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const float v = fabsf(x[ib * 32 + j]);
                if (v > a) a = v;
            }
            sub_amax[ib] = a / 127.0f;   /* LUT spans +-127 */
            if (sub_amax[ib] > smax) smax = sub_amax[ib];
        }
        const uint16_t dh = fp16_encode(smax > 0 ? smax / 31.0f : 0.0f);
        const double dd = ref_fp16(dh);
        memcpy(out, &dh, 2);
        memset(out + 4, 0, 4 + 128);
        for (ib = 0; ib < 8; ++ib) {
            int ls = dd > 0 ? (int)lrint(sub_amax[ib] / dd) + 32 : 32;
            if (ls < 33) ls = 33; if (ls > 63) ls = 63;
            out[4 + ib / 2] |= (uint8_t)((ls & 0xF) << (4 * (ib % 2)));
            sh |= (uint16_t)(((ls >> 4) & 3) << (2 * ib));
            {   const double dl = dd * (double)(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    int best_lo = 0, best_hi = 0;
                    double err_lo = 1e30, err_hi = 1e30;
                    for (int t = 0; t < 16; ++t) {
                        const double v = dl * (double)ref_iq4nl[t];
                        const double e1 = fabs((double)x[ib * 32 + j] - v);
                        const double e2 = fabs((double)x[ib * 32 + j + 16] - v);
                        if (e1 < err_lo) { err_lo = e1; best_lo = t; }
                        if (e2 < err_hi) { err_hi = e2; best_hi = t; }
                    }
                    out[8 + ib * 16 + j] = (uint8_t)(best_lo | (best_hi << 4));
                }
            }
        }
        memcpy(out + 2, &sh, 2);
    }
}

/* ---- comparisons ---- */

static int close_rel(double got, double want, double tol) {
    const double diff = fabs(got - want);
    const double mag  = fabs(want);
    return diff <= tol * (mag > 1.0 ? mag : 1.0);
}

static const char *type_name(int t) {
    switch (t) {
    case GGML_BK_Q8_0:   return "q8_0";
    case GGML_BK_Q6_K:   return "q6_K";
    case GGML_BK_IQ4_NL: return "iq4_nl";
    case GGML_BK_IQ3_S:  return "iq3_s";
    case GGML_BK_IQ4_XS: return "iq4_xs";
    default:             return "?";
    }
}

/* ---- tests ---- */

static void test_row_sizes(void) {
    CHECK(ggml_bk_row_bytes(GGML_BK_Q8_0,   32)  == 34,  "q8_0 32 -> 34");
    CHECK(ggml_bk_row_bytes(GGML_BK_Q8_0,   2560) == 2720, "q8_0 2560");
    CHECK(ggml_bk_row_bytes(GGML_BK_Q6_K,   256) == 210, "q6_K 256 -> 210");
    CHECK(ggml_bk_row_bytes(GGML_BK_IQ4_NL, 160) == 90,  "iq4_nl 160 -> 90 (PLE row)");
    CHECK(ggml_bk_row_bytes(GGML_BK_IQ3_S,  256) == 110, "iq3_s 256 -> 110");
    CHECK(ggml_bk_row_bytes(GGML_BK_IQ4_XS, 256) == 136, "iq4_xs 256 -> 136");
    CHECK(ggml_bk_row_bytes(GGML_BK_IQ4_XS, 2560) == 1360, "iq4_xs 2560");
    CHECK(ggml_bk_row_bytes(GGML_BK_IQ3_S,  100) == 0,   "non-multiple rejected");
    CHECK(ggml_bk_row_bytes(99, 256) == 0, "unknown type rejected");
}

static void test_dequant_random(int type) {
    enum { N = 2560 };
    const size_t bytes = ggml_bk_row_bytes(type, N);
    uint8_t *row = (uint8_t *)malloc(bytes);
    float   *y   = (float *)malloc(N * sizeof(float));
    int bad = 0;
    fill_random_row(type, row, N);
    CHECK(ggml_bk_dequant_row(type, row, y, N) == 0, "%s dequant rc", type_name(type));
    for (size_t e = 0; e < N; ++e) {
        const double want = ref_decode_elem(type, row, e);
        if (!close_rel((double)y[e], want, 1e-6) && bad++ < 3) {
            ++g_fail;
            fprintf(stderr, "FAIL %s dequant[%zu]: got %.9g want %.9g\n",
                    type_name(type), e, (double)y[e], want);
        }
    }
    free(row); free(y);
}

static void test_encode_roundtrip(int type) {
    enum { N = 512 };
    const size_t bytes = ggml_bk_row_bytes(type, N);
    float   *x   = (float *)malloc(N * sizeof(float));
    uint8_t *row = (uint8_t *)malloc(bytes);
    float   *y   = (float *)malloc(N * sizeof(float));
    double step = 0.0;
    for (int j = 0; j < N; ++j) x[j] = 4.0f * rnd_unit();
    switch (type) {
    case GGML_BK_Q8_0:   encode_q8_0(x, row, N);   step = 8.0 / 127.0; break;
    case GGML_BK_Q6_K:   encode_q6_K(x, row, N);   step = 8.0 / 31.0;  break;
    case GGML_BK_IQ4_NL: encode_iq4_nl(x, row, N); step = 8.0 / 8.0;   break;
    case GGML_BK_IQ4_XS: encode_iq4_xs(x, row, N); step = 8.0 / 8.0;   break;
    default: free(x); free(row); free(y); return;
    }
    CHECK(ggml_bk_dequant_row(type, row, y, N) == 0, "%s rt rc", type_name(type));
    for (int j = 0; j < N; ++j) {
        /* within one quantization step of the input (loose: layout, not RD) */
        CHECK(fabs((double)y[j] - (double)x[j]) <= step + 1e-6,
              "%s roundtrip[%d]: in %.6f out %.6f step %.4f",
              type_name(type), j, (double)x[j], (double)y[j], step);
    }
    free(x); free(row); free(y);
}

static void test_gemv(int type) {
    enum { M = 64, K = 2560 };
    const size_t row_bytes = ggml_bk_row_bytes(type, K);
    uint8_t *W = (uint8_t *)malloc((size_t)M * row_bytes);
    float   *x = (float *)malloc(K * sizeof(float));
    float   *y = (float *)malloc(M * sizeof(float));
    double  *tol = (double *)malloc(M * sizeof(double));
    for (int r = 0; r < M; ++r) fill_random_row(type, W + (size_t)r * row_bytes, K);
    for (int j = 0; j < K; ++j) x[j] = rnd_unit();
    CHECK(ggml_bk_gemv(type, M, K, W, x, y) == 0, "%s gemv rc", type_name(type));
    for (int r = 0; r < M; ++r) {
        double want = 0.0, sum_abs = 0.0;
        for (int j = 0; j < K; ++j) {
            const double t = ref_decode_elem(type, W + (size_t)r * row_bytes,
                                             (size_t)j) * (double)x[j];
            want += t;
            sum_abs += fabs(t);
        }
        /* random raw blocks cancel hard (|want| << sum|terms|), so anchor the
         * tolerance to the accumulated magnitude, not the tiny result */
        tol[r] = 1e-4 * (fabs(want) > 1e-3 * sum_abs ? fabs(want)
                                                     : 1e-3 * sum_abs);
        if (tol[r] < 1e-4) tol[r] = 1e-4;
        CHECK(fabs((double)y[r] - want) <= tol[r],
              "%s gemv[%d]: got %.9g want %.9g", type_name(type), r,
              (double)y[r], want);
    }
#ifdef GGML_BLOCKS_CUDA
    {
        enum { NMAT = 7 };
        void *dW = NULL, *dx = NULL, *dy = NULL, *dyq = NULL, *dlist = NULL;
        void *dtgrp = NULL, *dtoff = NULL, *didx = NULL;
        float *yg  = (float *)malloc((size_t)NMAT * M * sizeof(float));
        float *yref = (float *)malloc((size_t)NMAT * M * sizeof(float));
        float *xg  = (float *)malloc((size_t)NMAT * K * sizeof(float));
        float *yq  = (float *)malloc(K * sizeof(float));
        float *yqc = (float *)malloc(K * sizeof(float));
        const void *hlist[NMAT];
        const int htgrp[2] = { 0, 0 }, htoff[3] = { 0, 4, NMAT };
        const int hidx[NMAT] = { 6, 0, 4, 1, 5, 2, 3 };
        for (int i = 0; i < NMAT; ++i)
            for (int j = 0; j < K; ++j) xg[i*K+j] = x[j] + 0.01f*(float)i;
        CHECK(cudaMalloc(&dW, (size_t)M * row_bytes) == cudaSuccess, "cudaMalloc W");
        CHECK(cudaMalloc(&dx, (size_t)NMAT * K * sizeof(float)) == cudaSuccess, "cudaMalloc x");
        CHECK(cudaMalloc(&dy, (size_t)NMAT * M * sizeof(float)) == cudaSuccess, "cudaMalloc y");
        CHECK(cudaMalloc(&dyq, K * sizeof(float)) == cudaSuccess, "cudaMalloc yq");
        CHECK(cudaMalloc(&dlist, NMAT * sizeof(void *)) == cudaSuccess, "cudaMalloc list");
        CHECK(cudaMalloc(&dtgrp, sizeof(htgrp)) == cudaSuccess, "cudaMalloc tile groups");
        CHECK(cudaMalloc(&dtoff, sizeof(htoff)) == cudaSuccess, "cudaMalloc tile offsets");
        CHECK(cudaMalloc(&didx, sizeof(hidx)) == cudaSuccess, "cudaMalloc pair indices");
        cudaMemcpy(dW, W, (size_t)M * row_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, xg, (size_t)NMAT * K * sizeof(float), cudaMemcpyHostToDevice);
        /* dequant of row 0 vs CPU */
        CHECK(ggml_blocks_dequant_row_cuda(type, dW, (float *)dyq, K, NULL) == 0,
              "%s cuda dequant rc", type_name(type));
        cudaMemcpy(yq, dyq, K * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK(ggml_bk_dequant_row(type, W, yqc, K) == 0, "cpu dequant rc");
        for (int j = 0; j < K; ++j) {
            CHECK(close_rel((double)yq[j], (double)yqc[j], 1e-4),
                  "%s cuda dequant[%d]: %.9g vs %.9g", type_name(type), j,
                  (double)yq[j], (double)yqc[j]);
        }
        /* single GEMV */
        CHECK(ggml_blocks_gemv_cuda(type, M, K, dW, (const float *)dx,
                                    (float *)dy, NULL) == 0,
              "%s cuda gemv rc", type_name(type));
        cudaMemcpy(yg, dy, M * sizeof(float), cudaMemcpyDeviceToHost);
        for (int r = 0; r < M; ++r) {
            CHECK(fabs((double)yg[r] - (double)y[r]) <= tol[r],
                  "%s cuda gemv[%d]: %.9g vs cpu %.9g", type_name(type), r,
                  (double)yg[r], (double)y[r]);
        }
        /* batched GEMV: NMAT copies of the same matrix must agree with CPU */
        for (int i = 0; i < NMAT; ++i) hlist[i] = dW;
        cudaMemcpy(dlist, hlist, sizeof(hlist), cudaMemcpyHostToDevice);
        CHECK(ggml_blocks_gemv_batch(type, M, K, NMAT, (const void **)dlist,
                                     (const float *)dx, (float *)dy, NULL) == 0,
              "%s cuda batch rc", type_name(type));
        cudaMemcpy(yg, dy, (size_t)NMAT * M * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < NMAT; ++i) {
            for (int r = 0; r < M; ++r) {
                CHECK(fabs((double)yg[i * M + r] - (double)y[r]) <= tol[r],
                      "%s cuda batch[%d][%d]: %.9g vs cpu %.9g", type_name(type),
                      i, r, (double)yg[i * M + r], (double)y[r]);
            }
        }
        /* grouped GEMV: one expert weight row serves several token inputs;
         * outputs remain indexed by the original pair order. */
        CHECK(ggml_blocks_gemv_batch_sx(type, M, K, NMAT,
                                        (const void **)dlist, (const float *)dx,
                                        K, (float *)dy, NULL) == 0,
              "%s cuda per-pair rc", type_name(type));
        cudaMemcpy(yref, dy, (size_t)NMAT * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dtgrp, htgrp, sizeof(htgrp), cudaMemcpyHostToDevice);
        cudaMemcpy(dtoff, htoff, sizeof(htoff), cudaMemcpyHostToDevice);
        cudaMemcpy(didx, hidx, sizeof(hidx), cudaMemcpyHostToDevice);
        CHECK(ggml_blocks_gemv_grouped_sx(type, M, K, 2,
                                          (const void **)dlist, (const int *)dtgrp,
                                          (const int *)dtoff,
                                          (const int *)didx, (const float *)dx, K,
                                          (float *)dy, NULL) == 0,
              "%s cuda grouped rc", type_name(type));
        cudaMemcpy(yg, dy, (size_t)NMAT * M * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < NMAT; ++i)
            for (int r = 0; r < M; ++r)
                CHECK(yg[i*M+r] == yref[i*M+r],
                      "%s cuda grouped[%d][%d]: %.9g vs per-pair %.9g",
                      type_name(type), i, r, (double)yg[i*M+r],
                      (double)yref[i*M+r]);
        cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dyq); cudaFree(dlist);
        cudaFree(dtgrp); cudaFree(dtoff); cudaFree(didx);
        free(yg); free(yref); free(xg); free(yq); free(yqc);
    }
#endif
    free(W); free(x); free(y); free(tol);
}

/* ---- --vectors mode ---- */

static int run_vectors(const char *path) {
    FILE *f = fopen(path, "rb");
    uint32_t magic = 0, version = 0;
    int nrec = 0;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (fread(&magic, 4, 1, f) != 1 || fread(&version, 4, 1, f) != 1 ||
        magic != 0x47425456u || version != 1) {
        fprintf(stderr, "%s: bad magic/version\n", path);
        fclose(f);
        return 1;
    }
    for (;;) {
        uint32_t type = 0, n = 0;
        uint64_t nbytes = 0;
        uint8_t *raw; float *ref, *y;
        if (fread(&type, 4, 1, f) != 1) break;   /* EOF */
        if (fread(&n, 4, 1, f) != 1 || fread(&nbytes, 8, 1, f) != 1) {
            fprintf(stderr, "record %d: truncated header\n", nrec);
            fclose(f); return 1;
        }
        if (nbytes != (uint64_t)ggml_bk_row_bytes((int)type, n)) {
            fprintf(stderr, "record %d: nbytes %llu != row_bytes(%u, %u)\n",
                    nrec, (unsigned long long)nbytes, type, n);
            fclose(f); return 1;
        }
        raw = (uint8_t *)malloc((size_t)nbytes);
        ref = (float *)malloc((size_t)n * 4);
        y   = (float *)malloc((size_t)n * 4);
        if (fread(raw, 1, (size_t)nbytes, f) != (size_t)nbytes ||
            fread(ref, 4, n, f) != n) {
            fprintf(stderr, "record %d: truncated payload\n", nrec);
            free(raw); free(ref); free(y); fclose(f); return 1;
        }
        CHECK(ggml_bk_dequant_row((int)type, raw, y, n) == 0,
              "vectors rec %d rc", nrec);
        for (uint32_t j = 0; j < n; ++j) {
            CHECK(close_rel((double)y[j], (double)ref[j], 1e-5),
                  "vectors rec %d (%s) elem %u: got %.9g want %.9g",
                  nrec, type_name((int)type), j, (double)y[j], (double)ref[j]);
        }
        free(raw); free(ref); free(y);
        ++nrec;
    }
    fclose(f);
    printf("vectors: %d record(s) from %s, %s\n", nrec, path,
           g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    int t;
    if (argc == 3 && strcmp(argv[1], "--vectors") == 0) {
        return run_vectors(argv[2]);
    }
    test_row_sizes();
    for (t = 0; t < GGML_BK_TYPE_COUNT; ++t) {
        test_dequant_random(t);
        test_encode_roundtrip(t);   /* no-op for iq3_s */
        test_gemv(t);
    }
#ifdef GGML_BLOCKS_CUDA
    printf("ggml_blocks: CPU+CUDA %s (%d failure(s))\n",
           g_fail ? "FAILED" : "all tests passed", g_fail);
#else
    printf("ggml_blocks: CPU %s (%d failure(s))\n",
           g_fail ? "FAILED" : "all tests passed", g_fail);
#endif
    return g_fail ? 1 : 0;
}
