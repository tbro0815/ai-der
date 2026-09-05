#ifndef COLIBRI_VISION_QWEN3VL_H
#define COLIBRI_VISION_QWEN3VL_H

/*
 * Qwen3-VL "qwen3vl_merger" vision tower, loaded straight from a llama.cpp
 * mmproj GGUF (F16/F32 tensors only, which is what mmproj-F16.gguf carries).
 *
 * Shape (verified against llama.cpp tools/mtmd/models/qwen3vl.cpp and the
 * actual mmproj metadata for Qwen3.8-Flash-Next):
 *
 *   - patch embed: two Conv2d [1152, 3, 16, 16] (the temporal pair), applied
 *     to the SAME frame for a still image and summed -- folded here into one
 *     [1152, 768] matrix at load;
 *   - + patch bias, + learned position embeddings (48x48 grid, bilinear
 *     align-corners interpolated to the actual patch grid);
 *   - 27 pre-norm blocks: LayerNorm (w+b) -> fused qkv (+bias, no q/k norm)
 *     -> 2-D rotary (half the head rotates against y, half against x,
 *     theta 10000) -> full attention -> proj (+bias) -> LayerNorm ->
 *     fc_up -> GELU(tanh) -> fc_down (no gate);
 *   - post LayerNorm;
 *   - merger: concat each 2x2 block's 4 tokens (adjacent, see token order)
 *     -> mm.0 [4608->4608] -> GELU -> mm.2 [4608->2560].
 *
 * Token order is BLOCK-MAJOR over the 2x2 merge blocks (llama.cpp reorders
 * the row-major conv output in-graph; here the caller must deliver patches
 * already in that order -- tools/qwen38_image.py does). Within a patch the
 * layout is [channel][row][col], 3*16*16 = 768 floats, matching the folded
 * conv weight rows.
 *
 * The rotary tables use position ids in the ROW-MAJOR patch grid coordinates
 * of each token (llama.cpp fills "positions" in the same block-major order as
 * the tokens; y = t-section, x = h-section of GGML_ROPE_TYPE_VISION -- see
 * ggml_mrope_cache_init with indep_sects=true and sections {18,18,18,18}:
 * pairs 0..17 rotate by y, pairs 18..35 by x, inv_freq = theta^(-2j/36),
 * each pair (i, i+36) NeoX-split over the 72-wide head).
 *
 * GELU is the tanh approximation -- ggml_gelu, i.e. what llama.cpp runs and
 * what SigLIP-style towers were trained with (gelu_pytorch_tanh).
 *
 * Everything is f32 on the CPU; ~0.9 GB of F16 weights become ~1.8 GB
 * resident, loaded lazily on the first image.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vision_qwen3vl_types.h"   /* Q38VisCfg / Q38VisBlk / Q38VisTower (shared with the CUDA port) */

/* ---------------- minimal GGUF reader (F16/F32 tensors) ---------------- */

typedef struct { char name[128]; int type; int ndim; int64_t ne[4]; uint64_t off; } Q38VtInfo;

static float q38v_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {                                    /* subnormal */
            int e = -1;
            do { man <<= 1; e++; } while (!(man & 0x400));
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | (man & 0x3ff) << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0xffu << 23 | man << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | man << 13;
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static int q38v_read(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }

static int q38v_skip_kv_value(FILE *f, uint32_t type, uint64_t *align_out);

static int q38v_skip_kv_scalar(FILE *f, uint32_t type, uint64_t *val_out) {
    static const int size[13] = {1,1,2,2,4,4,4,1,-1,-1,8,8,8};
    if (type == 8) {                              /* string */
        uint64_t n; if (q38v_read(f, &n, 8)) return -1;
        return fseek(f, (long)n, SEEK_CUR);
    }
    if (type > 12 || size[type] < 0) return -1;
    uint64_t v = 0;
    if (q38v_read(f, &v, (size_t)size[type])) return -1;
    if (val_out) *val_out = v;
    return 0;
}

static int q38v_skip_kv_value(FILE *f, uint32_t type, uint64_t *val_out) {
    if (type == 9) {                              /* array */
        uint32_t et; uint64_t n;
        if (q38v_read(f, &et, 4) || q38v_read(f, &n, 8)) return -1;
        for (uint64_t i = 0; i < n; i++)
            if (q38v_skip_kv_value(f, et, NULL)) return -1;
        return 0;
    }
    return q38v_skip_kv_scalar(f, type, val_out);
}

/* Parse the header; returns tensor infos (malloc'd) and the data offset. */
static Q38VtInfo *q38v_gguf_scan(FILE *f, int64_t *n_out, uint64_t *data_off) {
    char magic[4]; uint32_t version; uint64_t n_tensors, n_kv;
    if (q38v_read(f, magic, 4) || memcmp(magic, "GGUF", 4) ||
        q38v_read(f, &version, 4) || version < 2 ||
        q38v_read(f, &n_tensors, 8) || q38v_read(f, &n_kv, 8) ||
        n_tensors > 4096 || n_kv > 65536) return NULL;
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; char key[256];
        if (q38v_read(f, &klen, 8) || klen >= sizeof key) return NULL;
        if (q38v_read(f, key, (size_t)klen)) return NULL;
        key[klen] = 0;
        uint32_t type; if (q38v_read(f, &type, 4)) return NULL;
        uint64_t val = 0;
        if (q38v_skip_kv_value(f, type, &val)) return NULL;
        if (!strcmp(key, "general.alignment") && val) alignment = val;
    }
    Q38VtInfo *ti = calloc((size_t)n_tensors, sizeof(Q38VtInfo));
    if (!ti) return NULL;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        if (q38v_read(f, &nlen, 8) || nlen >= sizeof ti[i].name) { free(ti); return NULL; }
        if (q38v_read(f, ti[i].name, (size_t)nlen)) { free(ti); return NULL; }
        ti[i].name[nlen] = 0;
        uint32_t nd;
        if (q38v_read(f, &nd, 4) || nd < 1 || nd > 4) { free(ti); return NULL; }
        ti[i].ndim = (int)nd;
        ti[i].ne[0] = ti[i].ne[1] = ti[i].ne[2] = ti[i].ne[3] = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t e; if (q38v_read(f, &e, 8)) { free(ti); return NULL; }
            ti[i].ne[d] = (int64_t)e;
        }
        uint32_t type;
        if (q38v_read(f, &type, 4) || q38v_read(f, &ti[i].off, 8)) { free(ti); return NULL; }
        ti[i].type = (int)type;
    }
    long here = ftell(f);
    if (here < 0) { free(ti); return NULL; }
    *data_off = ((uint64_t)here + alignment - 1) / alignment * alignment;
    *n_out = (int64_t)n_tensors;
    return ti;
}

/* Load tensor `name` as f32; verifies the element count. Exits on error --
 * the tower either loads whole or not at all. */
static float *q38v_tensor(FILE *f, const Q38VtInfo *ti, int64_t n_tensors,
                          uint64_t data_off, const char *name, int64_t want) {
    for (int64_t i = 0; i < n_tensors; i++) {
        if (strcmp(ti[i].name, name)) continue;
        int64_t n = ti[i].ne[0] * ti[i].ne[1] * ti[i].ne[2] * ti[i].ne[3];
        if (n != want) {
            fprintf(stderr, "[vis] %s: %lld elements, expected %lld\n",
                    name, (long long)n, (long long)want);
            exit(1);
        }
        if (ti[i].type != 0 && ti[i].type != 1) {
            fprintf(stderr, "[vis] %s: type %d, only F32/F16 supported\n", name, ti[i].type);
            exit(1);
        }
        float *out = malloc((size_t)n * sizeof(float));
        if (!out) { fprintf(stderr, "[vis] OOM on %s\n", name); exit(1); }
        if (fseek(f, (long)(data_off + ti[i].off), SEEK_SET)) { perror(name); exit(1); }
        if (ti[i].type == 0) {
            if (q38v_read(f, out, (size_t)n * 4)) { fprintf(stderr, "[vis] short read %s\n", name); exit(1); }
        } else {
            uint16_t *h = malloc((size_t)n * 2);
            if (!h || q38v_read(f, h, (size_t)n * 2)) { fprintf(stderr, "[vis] short read %s\n", name); exit(1); }
            for (int64_t j = 0; j < n; j++) out[j] = q38v_f16_to_f32(h[j]);
            free(h);
        }
        return out;
    }
    fprintf(stderr, "[vis] tensor %s missing from mmproj\n", name);
    exit(1);
}

/* Load the tower from a qwen3vl_merger mmproj GGUF. Geometry is fixed by the
 * checkpoint this engine serves and verified against the tensor shapes above.
 * Returns NULL only when the file cannot be opened/parsed. */
static Q38VisTower *q38v_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    int64_t nt = 0; uint64_t data_off = 0;
    Q38VtInfo *ti = q38v_gguf_scan(f, &nt, &data_off);
    if (!ti) { fprintf(stderr, "[vis] %s: not a GGUF file this reader understands\n", path); fclose(f); return NULL; }

    Q38VisTower *t = calloc(1, sizeof(Q38VisTower));
    if (!t) { free(ti); fclose(f); return NULL; }
    t->c = (Q38VisCfg){ .depth = 27, .hidden = 1152, .heads = 16, .head_dim = 72,
                        .inter = 4304, .patch = 16, .merge = 2, .out_dim = 2560,
                        .pos_side = 48, .proj_in = 1152 * 4,
                        .eps = 1e-6f, .theta = 10000.0f };
    const Q38VisCfg *c = &t->c;
    int64_t pw = 3 * c->patch * c->patch;

    /* the temporal Conv2d pair collapses to one matrix for a still image */
    t->patch_w = q38v_tensor(f, ti, nt, data_off, "v.patch_embd.weight", (int64_t)c->hidden * pw);
    {
        float *w1 = q38v_tensor(f, ti, nt, data_off, "v.patch_embd.weight.1", (int64_t)c->hidden * pw);
        for (int64_t j = 0; j < (int64_t)c->hidden * pw; j++) t->patch_w[j] += w1[j];
        free(w1);
    }
    t->patch_b  = q38v_tensor(f, ti, nt, data_off, "v.patch_embd.bias", c->hidden);
    t->pos_embd = q38v_tensor(f, ti, nt, data_off, "v.position_embd.weight",
                              (int64_t)c->pos_side * c->pos_side * c->hidden);
    t->post_w   = q38v_tensor(f, ti, nt, data_off, "v.post_ln.weight", c->hidden);
    t->post_b   = q38v_tensor(f, ti, nt, data_off, "v.post_ln.bias", c->hidden);
    t->mm0_w    = q38v_tensor(f, ti, nt, data_off, "mm.0.weight", (int64_t)c->proj_in * c->proj_in);
    t->mm0_b    = q38v_tensor(f, ti, nt, data_off, "mm.0.bias", c->proj_in);
    t->mm2_w    = q38v_tensor(f, ti, nt, data_off, "mm.2.weight", (int64_t)c->out_dim * c->proj_in);
    t->mm2_b    = q38v_tensor(f, ti, nt, data_off, "mm.2.bias", c->out_dim);

    t->blk = calloc((size_t)c->depth, sizeof(Q38VisBlk));
    if (!t->blk) { fprintf(stderr, "[vis] OOM on blocks\n"); exit(1); }
    for (int i = 0; i < c->depth; i++) {
        char nm[96];
        Q38VisBlk *b = &t->blk[i];
        #define VT(field, fmt, count) \
            snprintf(nm, sizeof nm, fmt, i); \
            b->field = q38v_tensor(f, ti, nt, data_off, nm, count)
        VT(ln1_w,  "v.blk.%d.ln1.weight", c->hidden);
        VT(ln1_b,  "v.blk.%d.ln1.bias", c->hidden);
        VT(ln2_w,  "v.blk.%d.ln2.weight", c->hidden);
        VT(ln2_b,  "v.blk.%d.ln2.bias", c->hidden);
        VT(qkv_w,  "v.blk.%d.attn_qkv.weight", (int64_t)3 * c->hidden * c->hidden);
        VT(qkv_b,  "v.blk.%d.attn_qkv.bias", 3 * c->hidden);
        VT(o_w,    "v.blk.%d.attn_out.weight", (int64_t)c->hidden * c->hidden);
        VT(o_b,    "v.blk.%d.attn_out.bias", c->hidden);
        VT(up_w,   "v.blk.%d.ffn_up.weight", (int64_t)c->inter * c->hidden);
        VT(up_b,   "v.blk.%d.ffn_up.bias", c->inter);
        VT(down_w, "v.blk.%d.ffn_down.weight", (int64_t)c->hidden * c->inter);
        VT(down_b, "v.blk.%d.ffn_down.bias", c->hidden);
        #undef VT
    }
    free(ti);
    fclose(f);
    return t;
}

/* ---------------- forward ---------------- */

static void q38v_layernorm(float *out, const float *in, const float *w,
                           const float *b, int n, float eps) {
    float mean = 0.f;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; i++) var += (in[i] - mean) * (in[i] - mean);
    float inv = 1.0f / sqrtf(var / n + eps);
    for (int i = 0; i < n; i++) out[i] = (in[i] - mean) * inv * w[i] + b[i];
}

/* ggml_gelu: the tanh approximation the reference runs */
static float q38v_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535588f * (x + 0.044715f * x * x * x)));
}

static void q38v_matvec(float *out, const float *w, const float *b,
                        const float *in, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *wr = w + (size_t)r * cols;
        float acc = b ? b[r] : 0.f;
        for (int j = 0; j < cols; j++) acc += wr[j] * in[j];
        out[r] = acc;
    }
}

/* Y[t*ldy + r] = W[r] . X[t] + b[r] for all T tokens.  Parallel over weight
 * ROWS with the tokens inner (4-way unrolled): each weight row is streamed
 * from DRAM exactly once per image and the activations live in L3, which is
 * what makes the tower ~10x faster than the per-token matvec (that streamed
 * the full weight matrix per TOKEN). */
static void q38v_gemm(float *Y, const float *W, const float *b, const float *X,
                      int rows, int cols, int T, int ldy) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *wr = W + (size_t)r * cols;
        const float bb = b ? b[r] : 0.f;
        int t = 0;
        for (; t + 4 <= T; t += 4) {
            const float *x0 = X + (size_t)(t+0) * cols;
            const float *x1 = X + (size_t)(t+1) * cols;
            const float *x2 = X + (size_t)(t+2) * cols;
            const float *x3 = X + (size_t)(t+3) * cols;
            float a0 = bb, a1 = bb, a2 = bb, a3 = bb;
            /* simd reduction: without it gcc keeps the strict serial sum
             * order and the loop never vectorizes (no -ffast-math in this
             * build).  The reassociation moves results by ~1e-6 relative --
             * the same liberty ggml's vectorized dot products take. */
            #pragma omp simd reduction(+:a0,a1,a2,a3)
            for (int j = 0; j < cols; j++) {
                float w = wr[j];
                a0 += w * x0[j]; a1 += w * x1[j];
                a2 += w * x2[j]; a3 += w * x3[j];
            }
            Y[(size_t)(t+0)*ldy + r] = a0;
            Y[(size_t)(t+1)*ldy + r] = a1;
            Y[(size_t)(t+2)*ldy + r] = a2;
            Y[(size_t)(t+3)*ldy + r] = a3;
        }
        for (; t < T; t++) {
            const float *x = X + (size_t)t * cols;
            float acc = bb;
            for (int j = 0; j < cols; j++) acc += wr[j] * x[j];
            Y[(size_t)t*ldy + r] = acc;
        }
    }
}

/* Position embedding (bilinear, align-corners from the pos_side grid) ADDED
 * to pos_out[tokens][hidden], and the rotary cos|sin table cs[tokens][head_dim]
 * (pairs 0..quarter-1 rotate by the row, quarter..half-1 by the column),
 * both in block-major token order.  Shared by the CPU and the GPU tower. */
static void q38v_tables(const Q38VisTower *t, int grid_h, int grid_w, float *pos_out, float *cs) {
    const Q38VisCfg *c = &t->c;
    const int tokens = grid_h * grid_w, hidden = c->hidden, hd = c->head_dim;
    const int half = hd / 2, quarter = hd / 4, m = c->merge, bw = grid_w / m;
    #pragma omp parallel for schedule(static)
    for (int tk = 0; tk < tokens; tk++) {
        float *row = pos_out + (size_t)tk * hidden;
        /* token -> row-major patch coordinates */
        int blk = tk / (m * m), inner = tk % (m * m);
        int py = (blk / bw) * m + inner / m;
        int px = (blk % bw) * m + inner % m;
        float sy = grid_h > 1 ? (float)py * (c->pos_side - 1) / (float)(grid_h - 1) : 0.f;
        float sx = grid_w > 1 ? (float)px * (c->pos_side - 1) / (float)(grid_w - 1) : 0.f;
        int y0 = (int)sy, x0 = (int)sx;
        int y1 = y0 + 1 < c->pos_side ? y0 + 1 : y0;
        int x1 = x0 + 1 < c->pos_side ? x0 + 1 : x0;
        float fy = sy - y0, fx = sx - x0;
        const float *p00 = t->pos_embd + ((size_t)y0 * c->pos_side + x0) * hidden;
        const float *p01 = t->pos_embd + ((size_t)y0 * c->pos_side + x1) * hidden;
        const float *p10 = t->pos_embd + ((size_t)y1 * c->pos_side + x0) * hidden;
        const float *p11 = t->pos_embd + ((size_t)y1 * c->pos_side + x1) * hidden;
        for (int j = 0; j < hidden; j++)
            row[j] += (1 - fy) * ((1 - fx) * p00[j] + fx * p01[j]) +
                      fy * ((1 - fx) * p10[j] + fx * p11[j]);
        float *cst = cs + (size_t)tk * half * 2;
        for (int j = 0; j < quarter; j++) {
            float inv = powf(c->theta, -2.0f * j / (float)half);
            float ay = py * inv, ax = px * inv;
            cst[j] = cosf(ay);              cst[half + j] = sinf(ay);
            cst[quarter + j] = cosf(ax);    cst[half + quarter + j] = sinf(ax);
        }
    }
}

static int q38v_output_tokens(const Q38VisTower *t, int grid_h, int grid_w) {
    int m = t->c.merge;
    if (grid_h < m || grid_w < m || grid_h % m || grid_w % m) return -1;
    return (grid_h / m) * (grid_w / m);
}

/* patches: [grid_h*grid_w][3*16*16] block-major (see header comment);
 * out: [output_tokens][2560]. Returns 0, -1 on bad grid/OOM. */
static int q38v_encode(const Q38VisTower *t, const float *patches,
                       int grid_h, int grid_w, float *out) {
    const Q38VisCfg *c = &t->c;
    const int tokens = grid_h * grid_w, hd = c->head_dim, heads = c->heads;
    const int hidden = c->hidden, half = hd / 2;
    const int n_out = q38v_output_tokens(t, grid_h, grid_w);
    if (n_out < 0) return -1;
    const int pw = 3 * c->patch * c->patch;

    float *state = malloc((size_t)tokens * hidden * sizeof(float));
    float *qkv = malloc((size_t)tokens * 3 * hidden * sizeof(float));
    float *branch = malloc((size_t)tokens * hidden * sizeof(float));
    float *normed = malloc((size_t)tokens * hidden * sizeof(float));
    float *mlp = malloc((size_t)tokens * c->inter * sizeof(float));
    float *cs = malloc((size_t)tokens * half * 2 * sizeof(float));  /* cos|sin per pair */
    if (!state || !qkv || !branch || !normed || !mlp || !cs) {
        free(cs); free(mlp); free(normed); free(branch); free(qkv); free(state); return -1;
    }

    /* patch embed + interpolated position embedding, block-major order */
    q38v_gemm(state, t->patch_w, t->patch_b, patches, hidden, pw, tokens, hidden);
    q38v_tables(t, grid_h, grid_w, state, cs);

    const float scale = 1.0f / sqrtf((float)hd);
    for (int layer = 0; layer < c->depth; layer++) {
        const Q38VisBlk *b = &t->blk[layer];
        /* ln1, fused qkv, rope */
        #pragma omp parallel for schedule(static)
        for (int tk = 0; tk < tokens; tk++)
            q38v_layernorm(normed + (size_t)tk * hidden, state + (size_t)tk * hidden,
                           b->ln1_w, b->ln1_b, hidden, c->eps);
        q38v_gemm(qkv, b->qkv_w, b->qkv_b, normed, 3 * hidden, hidden, tokens, 3 * hidden);
        #pragma omp parallel for schedule(static)
        for (int tk = 0; tk < tokens; tk++) {
            float *row = qkv + (size_t)tk * 3 * hidden;
            const float *cst = cs + (size_t)tk * half * 2;
            for (int pass = 0; pass < 2; pass++) {              /* q then k */
                float *base = row + (size_t)pass * hidden;
                for (int h = 0; h < heads; h++) {
                    float *v = base + (size_t)h * hd;
                    for (int j = 0; j < half; j++) {
                        float x0 = v[j], x1 = v[j + half];
                        v[j]        = x0 * cst[j] - x1 * cst[half + j];
                        v[j + half] = x1 * cst[j] + x0 * cst[half + j];
                    }
                }
            }
        }
        /* full attention */
        #pragma omp parallel
        {
            float *score = malloc((size_t)tokens * sizeof(float));
            #pragma omp for schedule(static) collapse(2)
            for (int h = 0; h < heads; h++)
                for (int tk = 0; tk < tokens; tk++) {
                    const float *q = qkv + (size_t)tk * 3 * hidden + (size_t)h * hd;
                    float *sc = score;
                    float mx = -INFINITY;
                    for (int s = 0; s < tokens; s++) {
                        const float *k = qkv + (size_t)s * 3 * hidden + hidden + (size_t)h * hd;
                        float d = 0.f;
                        for (int j = 0; j < hd; j++) d += q[j] * k[j];
                        d *= scale;
                        sc[s] = d;
                        if (d > mx) mx = d;
                    }
                    float tot = 0.f;
                    for (int s = 0; s < tokens; s++) { sc[s] = expf(sc[s] - mx); tot += sc[s]; }
                    float *dst = branch + (size_t)tk * hidden + (size_t)h * hd;
                    for (int j = 0; j < hd; j++) dst[j] = 0.f;
                    for (int s = 0; s < tokens; s++) {
                        const float *v = qkv + (size_t)s * 3 * hidden + 2 * hidden + (size_t)h * hd;
                        float a = sc[s] / tot;
                        for (int j = 0; j < hd; j++) dst[j] += a * v[j];
                    }
                }
            free(score);
        }
        /* o proj + residual (normed reused as the projected branch) */
        q38v_gemm(normed, b->o_w, b->o_b, branch, hidden, hidden, tokens, hidden);
        #pragma omp parallel for schedule(static)
        for (int tk = 0; tk < tokens; tk++) {
            float *row = state + (size_t)tk * hidden;
            const float *add = normed + (size_t)tk * hidden;
            for (int j = 0; j < hidden; j++) row[j] += add[j];
            q38v_layernorm(normed + (size_t)tk * hidden, row, b->ln2_w, b->ln2_b, hidden, c->eps);
        }
        q38v_gemm(mlp, b->up_w, b->up_b, normed, c->inter, hidden, tokens, c->inter);
        #pragma omp parallel for schedule(static)
        for (int tk = 0; tk < tokens; tk++) {
            float *row = mlp + (size_t)tk * c->inter;
            for (int j = 0; j < c->inter; j++) row[j] = q38v_gelu(row[j]);
        }
        q38v_gemm(normed, b->down_w, b->down_b, mlp, hidden, c->inter, tokens, hidden);
        #pragma omp parallel for schedule(static)
        for (int tk = 0; tk < tokens; tk++) {
            float *row = state + (size_t)tk * hidden;
            const float *add = normed + (size_t)tk * hidden;
            for (int j = 0; j < hidden; j++) row[j] += add[j];
        }
    }

    /* post norm + merger; the 4 tokens of a block are adjacent by token order,
     * so the post-normed state IS the [n_out][4608] concatenation. */
    #pragma omp parallel for schedule(static)
    for (int tk = 0; tk < tokens; tk++)
        q38v_layernorm(normed + (size_t)tk * hidden, state + (size_t)tk * hidden,
                       t->post_w, t->post_b, hidden, c->eps);
    {
        float *mid = malloc((size_t)n_out * c->proj_in * sizeof(float));
        if (!mid) { free(cs); free(mlp); free(normed); free(branch); free(qkv); free(state); return -1; }
        q38v_gemm(mid, t->mm0_w, t->mm0_b, normed, c->proj_in, c->proj_in, n_out, c->proj_in);
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < n_out; n++) {
            float *row = mid + (size_t)n * c->proj_in;
            for (int j = 0; j < c->proj_in; j++) row[j] = q38v_gelu(row[j]);
        }
        q38v_gemm(out, t->mm2_w, t->mm2_b, mid, c->out_dim, c->proj_in, n_out, c->out_dim);
        free(mid);
    }

    free(cs); free(mlp); free(normed); free(branch); free(qkv); free(state);
    return 0;
}

#endif /* COLIBRI_VISION_QWEN3VL_H */
