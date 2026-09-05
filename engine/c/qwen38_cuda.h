/* qwen38_cuda.h -- P4: dense GPU path for the qwen38 engine.
 *
 * Moves the per-token dense backbone (hyper-connection mixers, GDN
 * projections + conv + delta-rule recurrence, QSA projections, final head)
 * onto the CUDA device.  The engine keeps orchestration, routing, the QSA
 * attention walk / indexer / KV caches and the PLE block on the CPU; this
 * module owns the device-resident copies of the dense weights (uploaded once
 * at load, re-laid-out from raw GGML blocks into a planar int8 + f32
 * group-scale form the GEMV kernels stream at full bandwidth) and the
 * device-resident GDN recurrent/conv state.
 *
 * Numerics: per-element dequantized weight VALUES are exactly the CPU
 * decoders' values (Q8_0: fp16 scale * int8; Q6_K: fp16 d * int8 sub-scale *
 * (q-32), products in f32 in the CPU's order); only summation order differs
 * (warp-parallel vs sequential), the same accepted bit-close class as the S2
 * GPU expert tier.  RMS-norm style reductions accumulate in double like the
 * CPU code.
 *
 * Per-token transfer budget: one x upload, one mixed download per layer (for
 * the CPU router / indexer), the QSA projection download + context upload on
 * the 12 attention layers, one MoE-output upload per layer, one logits
 * download.  Everything else stays on the device across tokens.
 *
 * All entries are safe no-ops / return 0 when the module is inactive.  Any
 * CUDA error marks the module dead (q38g_active() -> 0); because the GDN
 * state then only exists on the device, the engine treats a mid-run death as
 * fatal for the current generation.
 *
 * Enabled by q38g_init() (call under COLI_CUDA=1; Q38_GPU_DENSE=0 opts out
 * in the engine).  Compiled from qwen38_cuda.cu only with CUDA=1; the inline
 * stubs below keep the CPU-only build byte-identical. */
#ifndef QWEN38_CUDA_H
#define QWEN38_CUDA_H
#include <stdint.h>
#include <stddef.h>

#ifdef COLI_CUDA
#ifdef __cplusplus
extern "C" {
#endif

/* Geometry from the parsed config; is_attn[i]!=0 marks QSA layers (their GDN
 * state slots are skipped).  Returns 1 when the device context is up. */
int q38g_init(int n_layers, const uint8_t *is_attn, int D,
              int hc_rank, int vh, int vk, int kd, int vd, int convk, int cdim,
              int H, int KV, int hd, int vocab, float eps);
int  q38g_active(void);
void q38g_disable(void);       /* free everything, go inactive (upload failed) */
void q38g_shutdown(void);
size_t q38g_vram_used(void);   /* weight + state + scratch bytes on device */

/* Upload one dense matrix.  raw = the container's raw GGML blocks (Q8_0 or
 * Q6_K -- the only dense types in this container), O rows of I.  Returns a
 * handle >= 0, or -1 (unsupported type / OOM: caller should q38g_disable()). */
int q38g_up_raw(int type, int O, int I, const void *raw);
/* Upload an f32 tensor (norm gammas, inject mats, GDN scalars/conv). */
int q38g_up_f32(const float *p, int64_t n);

/* ---- per-token ops (single decode thread, one device stream) ---- */
void q38g_tok_begin(const float *x /*[D] embedding row*/);  /* res[s]=x, all streams */
void q38g_res_get(float *res /*[4*D]*/);   /* D2H sync (CPU PLE block) */
void q38g_res_set(const float *res);       /* H2D after the CPU PLE mutation */
/* hyper-connection mix at the current res -> device mixed + inject.
 * hinj < 0 skips the inject GEMV (the output head's mix). */
void q38g_hc_mix(int hdown, int hup, int hnorm, int hinj);
void q38g_mixed_get(float *mixed /*[D]*/); /* D2H sync (router / indexer x) */
/* Full GDN block from device mixed -> device blk; layer's recurrent + conv
 * ring state lives on the device (see q38g_state_put/get). */
void q38g_gdn(int layer, int hqkv, int hz, int hba,
              int ha, int hdt, int hnorm, int hconv, int hout);
/* CUDA graph over a fixed launch span.  open(id) returns 1 when a previously
 * captured graph was replayed (skip the span); 0 means "issue the span now"
 * -- and if a capture was started, close(id) instantiates + runs it.  ids:
 * layer (GDN body), nl+layer / 2*nl+layer (QSA hc mixes); Q38_GPU_GRAPHS=0
 * disables capture (spans then always run directly). */
int  q38g_graph_open(int id);
void q38g_graph_close(int id);
/* QSA q/k/v projections from device mixed, downloaded to the host (sync);
 * the attention walk stays on the CPU. */
void q38g_qsa_proj(int hq, int hk, int hv, int nq, int nkv,
                   float *qfull, float *kk, float *vv);
/* gated context back up + o-projection -> device blk. */
void q38g_qsa_out(int ho, const float *ctx, int nctx);
/* res += blk * 2*sigmoid(inject/4) with the device blk / inject. */
void q38g_combine(void);
/* upload a CPU-computed block output (the MoE result) and combine it. */
void q38g_blk_combine(const float *blk /*[D]*/);
/* output head: hc mix (no inject) + lm_head GEMV, logits D2H (sync). */
void q38g_head(int hdown, int hup, int hnorm, int hlm, float *logits);

/* ---- batched prefill ops ----
 * Host buffers are batch-major.  Dense weights are decoded once per small
 * token tile on the device; causal GDN state still advances token by token. */
int q38g_batch_max(void);
void q38g_hc_mix_batch(int hdown, int hup, int hnorm, int hinj,
                       const float *res, int batch, float *mixed, float *inject);
void q38g_qsa_proj_batch(int hq, int hk, int hv, const float *mixed, int batch,
                         float *qfull, float *kk, float *vv);
void q38g_gdn_batch(int layer, int hqkv, int hz, int hba,
                    int ha, int hdt, int hnorm, int hconv, int hout,
                    const float *mixed, int batch, float *blk, int snapshot);
void q38g_out_batch(int h, const float *x, int batch, float *y);
int  q38g_out_argmax(int h, const float *x);   /* argmax(mat[h] @ x), first maximum; -1 on failure */
int  q38g_router(int hgate, int E, float *pr);            /* opt-in GPU router (decode) */
int  q38g_router_batch(int hgate, int E, int B, float *pr); /* opt-in GPU router (prefill chunk) */
/* device-resident chunk residual: put before layer 0, get after the loop;
 * hc_mix_batch(res=NULL) / gdn_batch, qsa_proj_batch(mixed=NULL) reuse the
 * device copies, combine_batch(res=NULL) skips the download */
void q38g_res_batch_put(const float *res, int batch);
const float *q38g_b_mixed_dev(void);  /* P10 P7a: device copy of the chunk's mixed (X), NULL when inactive */
/* P9 QSA on the device for batched prefill (q8 KV mirror; host cache is the
 * source of truth).  qsa_proj_batch(qfull=NULL) leaves the projections on
 * the device for qsa_batch; qsa_out_batch projects the device context. */
int  q38g_qsa_kv_setup(int n_layers, const uint8_t *is_attn, int kv_heads, int hd,
                       int max_t, int H, int sel_cap);
int  q38g_qsa_ready(int layer, int max_t);
/* P12: lazily grown mirrors (the tier hands VRAM back through q38t_release) */
int    q38g_qsa_alloc_t(void);
size_t q38g_free_bytes(void);
size_t q38g_mirror_bytes(int n_layers, const uint8_t *is_attn, const int *ratio, int t);
int    q38g_qsa_kv_grow(int n_layers, const uint8_t *is_attn, int new_t);
int    q38g_idx_grow(int n_layers, const uint8_t *is_attn, const int *ratio, int new_t);
void q38g_qsa_kv_invalidate(int pos);
int  q38g_qsa_batch(int layer, int pos0, int batch, int hqn, int hkn,
                    int8_t *K8, float *K8s, int8_t *V8, float *V8s, int host_max_t,
                    const int *nsel, const int *sel, int n_rot, float theta,
                    float eps, int walk_max, const int *mp3, int sec1, int sec2);
void q38g_qsa_out_batch(int ho, int batch, float *blk);
void q38g_qsa_out_dev(int ho);        /* decode: o-projection from the device context into d_blk */
/* indexer block scoring on the device (CPU's serial per-head order, so the
 * scores are identical and the host selection is unchanged) */
int  q38g_idx_setup(int n_layers, const uint8_t *is_attn, const int *ratio, int max_t,
                    int nh, int id);
const float *q38g_idx_score(int layer, const float *ibk_host, int nblk, const float *qi,
                            const int *nfb, int nfb_max, int batch);
void q38g_idx_invalidate(int nblk);
void q38g_res_batch_get(float *res, int batch);
/* shared expert over a chunk: y[B*D] = down(silu(gate x) * (up x)); host scales */
void q38g_shared_batch(int hg, int hu, int hd, const float *x, int batch, float *y);
int  q38g_shared_batch_issue(int hg, int hu, int hd, const float *x, int batch);
int  q38g_shared_batch_take(int batch, float *y);
void q38g_combine_batch(const float *blk, int batch, float *res);

/* Speculative verification snapshots the device-resident GDN state after
 * each batch boundary.  Allocate before the expert tier sizes itself. */
int  q38g_spec_init(int boundaries);
void q38g_spec_restore(int boundary);

/* GDN state transfer (CPU <-> device), one GDN layer at a time:
 * rec [vh*kd*vd], ring [cdim*(convk-1)]. */
void q38g_state_put(int layer, const float *rec, const float *ring);
void q38g_state_get(int layer, float *rec, float *ring);

void q38g_sync(void);          /* drain the stream (timer boundaries) */

#ifdef __cplusplus
}
#endif

#else /* !COLI_CUDA: inline stubs, engine stays CPU-only */

static inline int q38g_init(int a,const uint8_t*b,int c,int d,int e,int f,int g,int h,int i,int j,int k,int l,int m,int n,float o){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;return 0;}
static inline int  q38g_active(void){return 0;}
static inline void q38g_disable(void){}
static inline void q38g_shutdown(void){}
static inline size_t q38g_vram_used(void){return 0;}
static inline int q38g_up_raw(int a,int b,int c,const void*d){(void)a;(void)b;(void)c;(void)d;return -1;}
static inline int q38g_up_f32(const float*a,int64_t b){(void)a;(void)b;return -1;}
static inline void q38g_tok_begin(const float*a){(void)a;}
static inline void q38g_res_get(float*a){(void)a;}
static inline void q38g_res_set(const float*a){(void)a;}
static inline void q38g_hc_mix(int a,int b,int c,int d){(void)a;(void)b;(void)c;(void)d;}
static inline void q38g_mixed_get(float*a){(void)a;}
static inline void q38g_gdn(int a,int b,int c,int d,int e,int f,int g,int h,int i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;}
static inline int  q38g_graph_open(int a){(void)a;return 0;}
static inline void q38g_graph_close(int a){(void)a;}
static inline void q38g_qsa_proj(int a,int b,int c,int d,int e,float*f,float*g,float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline void q38g_qsa_out(int a,const float*b,int c){(void)a;(void)b;(void)c;}
static inline void q38g_combine(void){}
static inline void q38g_blk_combine(const float*a){(void)a;}
static inline void q38g_head(int a,int b,int c,int d,float*e){(void)a;(void)b;(void)c;(void)d;(void)e;}
static inline int q38g_batch_max(void){return 0;}
static inline void q38g_hc_mix_batch(int a,int b,int c,int d,const float*e,int f,float*h,float*i){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)h;(void)i;}
static inline void q38g_qsa_proj_batch(int a,int b,int c,const float*d,int e,float*f,float*g,float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline void q38g_gdn_batch(int a,int b,int c,int d,int e,int f,int g,int h,int i,const float*j,int k,float*l,int m){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;}
static inline void q38g_out_batch(int a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline int  q38g_out_argmax(int a,const float*b){(void)a;(void)b;return -1;}
static inline int  q38g_router(int a,int b,float*c){(void)a;(void)b;(void)c;return 0;}
static inline int  q38g_router_batch(int a,int b,int c,float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void q38g_res_batch_put(const float*a,int b){(void)a;(void)b;}
static inline const float *q38g_b_mixed_dev(void){return NULL;}
static inline int  q38g_qsa_kv_setup(int a,const uint8_t*b,int c,int d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int    q38g_qsa_alloc_t(void){return 0;}
static inline size_t q38g_free_bytes(void){return 0;}
static inline size_t q38g_mirror_bytes(int a,const uint8_t*b,const int*c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int    q38g_qsa_kv_grow(int a,const uint8_t*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline int    q38g_idx_grow(int a,const uint8_t*b,const int*c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int  q38g_qsa_ready(int a,int b){(void)a;(void)b;return 0;}
static inline void q38g_qsa_kv_invalidate(int a){(void)a;}
static inline int  q38g_qsa_batch(int a,int b,int c,int d,int e,int8_t*f,float*g,int8_t*h,float*i,int j,const int*k,const int*l,int m,float n,float o,int p,const int*q,int r,int s){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i;(void)j;(void)k;(void)l;(void)m;(void)n;(void)o;(void)p;(void)q;(void)r;(void)s;return 0;}
static inline void q38g_qsa_out_batch(int a,int b,float*c){(void)a;(void)b;(void)c;}
static inline void q38g_qsa_out_dev(int a){(void)a;}
static inline int  q38g_idx_setup(int a,const uint8_t*b,const int*c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline const float *q38g_idx_score(int a,const float*b,int c,const float*d,const int*e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline void q38g_idx_invalidate(int a){(void)a;}
static inline void q38g_res_batch_get(float*a,int b){(void)a;(void)b;}
static inline void q38g_shared_batch(int a,int b,int c,const float*d,int e,float*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
static inline int  q38g_shared_batch_issue(int a,int b,int c,const float*d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  q38g_shared_batch_take(int a,float*b){(void)a;(void)b;return 0;}
static inline void q38g_combine_batch(const float*a,int b,float*c){(void)a;(void)b;(void)c;}
static inline int q38g_spec_init(int a){(void)a;return 0;}
static inline void q38g_spec_restore(int a){(void)a;}
static inline void q38g_state_put(int a,const float*b,const float*c){(void)a;(void)b;(void)c;}
static inline void q38g_state_get(int a,float*b,float*c){(void)a;(void)b;(void)c;}
static inline void q38g_sync(void){}

#endif /* COLI_CUDA */
#endif /* QWEN38_CUDA_H */
