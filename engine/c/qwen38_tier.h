/* qwen38_tier.h -- optional CUDA VRAM expert tier for the qwen38 engine (S2).
 *
 * qwen36_tier's concept ("route -> place -> overlap -> learn") applied to
 * Qwen3.8-Flash-Next, with two structural differences:
 *
 *  - Experts are RAW GGML blocks of per-layer-varying types (gate/up IQ3_S or
 *    IQ4_XS, down IQ4_NL or Q8_0).  VRAM slots store the raw bytes and the
 *    GPU computes with ggml_blocks_cuda GEMV -- no int4 repack, no
 *    ColiCudaTensor.  Types and row geometry are passed per layer at init.
 *  - 24,576 experts (~58 GB) exceed VRAM: residency is PARTIAL.  The LFRU
 *    heat/hysteresis contract from tier.h decides who earns a slot; a miss
 *    falls back to the S1 CPU path and overlaps with the in-flight GPU work.
 *
 * Expert sources are pointers into the mmap'd container shards (page cache
 * backs the RAM side); a background uploader thread copies mmap -> pinned
 * staging ring -> VRAM on a dedicated stream.  Decode never blocks on
 * placement.
 *
 * Per-token flow in the MoE (single decode thread):
 *   q38t_note(l, e)                    for every routed expert (heat/upload)
 *   mask = q38t_issue(l, eids, K, x)   async: x H2D + gate|up batched GEMV
 *   ... CPU computes the misses (~mask bits) ...
 *   q38t_mid(mask)                     sync gate|up, silu*mul on CPU, async
 *                                      h H2D + down batched GEMV + y D2H
 *   ... CPU computes the shared expert ...
 *   q38t_take(mask, val, K, out)       sync y, accumulate val[k]*y_k
 *
 * Enable with COLI_CUDA=1 [CUDA_EXPERT_GB=<G>|auto (default ~9)]
 * [HEAT_FILE=<path>] [COLI_GPUS=<ordinal>] [QT_NO_WARMSTART=1].  Compiled
 * only when the build sets -DCOLI_CUDA (CUDA=1); otherwise the inline stubs
 * keep the engine CPU-only with zero overhead. */
#ifndef QWEN38_TIER_H
#define QWEN38_TIER_H
#include <stdint.h>
#include <stddef.h>

static inline size_t q38t_auto_budget(size_t free_bytes) {
    const size_t margin = (size_t)2 << 30;
    return free_bytes > margin ? free_bytes - margin : 0;
}

#ifdef COLI_CUDA

/* Init after model load.  gu_type/d_type: ggml_bk_type per layer for the
 * gate&up ([inter, hidden]) and down ([hidden, inter]) matrices.
 * Auto sizing retains a 2 GB CUDA safety margin after dense weights and decode
 * graphs have been allocated. Returns 1 when the tier is active (COLI_CUDA=1
 * env + a working device). */
int  q38t_init(int n_layers, int n_experts, int hidden, int inter, int topk,
               const int *gu_type, const int *d_type);
int  q38t_ready(void);
size_t q38t_release(size_t bytes);   /* P12: free whole slabs (coldest first), shrink the budget */
const char *q38t_device_name(void);
void q38t_shutdown(void);

/* Register the mmap'd raw-block source of one expert (called once per expert
 * at startup; pointers must stay valid for the process lifetime). */
void q38t_set_src(int layer, int eid,
                  const uint8_t *g, const uint8_t *u, const uint8_t *d);

/* Heat bump + opportunistic background upload for one routed expert. */
void q38t_note(int layer, int eid);
void q38t_note_many(int layer, const int *eids, int n);   /* ids < 0 skipped */

/* Upload one layer's shared expert (permanently resident, ~110 MB total,
 * outside the expert budget); it then rides the same async chain as the
 * routed group.  q38t_shared_on says whether the engine may skip its CPU
 * shared-expert compute for this layer. */
void q38t_set_shared(int layer, const uint8_t *g, const uint8_t *u,
                     const uint8_t *d, int gu_type, int d_type);
int  q38t_shared_on(int layer);

/* Launch the full expert chain (gate|up -> silu*mul -> down -> y D2H) for
 * the VRAM-resident subset of the K selected experts, plus the shared expert
 * when uploaded.  Returns the bitmask of k handled by the GPU; compute the
 * rest on the CPU while the chain runs. */
uint32_t q38t_issue(int layer, const int *eids, int K, const float *x);

/* No-op since the chain fused (kept for the issue/mid/take call shape). */
void q38t_mid(uint32_t mask);

/* Collect: out[hidden] += val[k]*y_k for every mask bit, plus
 * shared_s * y_shared when the shared expert was issued. */
void q38t_take(uint32_t mask, const float *val, int K, float *out,
               float shared_s);

/* ---- S3: multi-token issue for the chunked prefill ----
 * One chain serves a whole chunk: every (token, routed expert) pair that is
 * VRAM-resident becomes one batched-GEMV entry (per-pair activation), the
 * shared expert rides along as one extra pair per token when its types match
 * the layer.  masks[t] gets the per-token hit bitmask; the caller computes
 * the misses (and, when q38t_multi_shared()==0, the shared experts) on the
 * CPU while the chain runs, then collects with q38t_take_multi.
 * q38t_multi_max: max tokens per issue (0 = unavailable). */
int  q38t_multi_max(void);
int  q38t_multi_shared(int layer);
void q38t_issue_multi(int layer, const int *eids /*[B*K]*/, int K, int B,
                      const float *X /*[B*hidden]*/, uint32_t *masks /*[B]*/);
void q38t_issue_multi_finish(void);   /* P10 P5: launch the staged tiles (idempotent) */
void q38t_set_device_x(const float *dx); /* P10 P7a: device X for the next issue_multi (NULL = upload) */
int  q38t_shared_issue1(int layer, const float *x);      /* P10 D2b: one-token shared expert on sh_st */
int  q38t_shared_take1(float *out, float shared_s);     /* sync + out += s * y */
void q38t_take_multi(const float *val /*[B*K]*/, int K, int B,
                     float *OUT /*[B*hidden]*/, const float *shared_s /*[B]*/);

/* ---- P4: async expert prefetcher ----
 * Policy = the ensemble that won the offline evaluation (docs/p4-analysis.md):
 *   previous-token union   -- a routed-but-nonresident expert is queued for a
 *                             low-priority upload the moment it is routed
 *                             (q38t_note does this internally), ahead of the
 *                             16-token LFRU swap tick;
 *   hot-16 pinning         -- the artifact's per-layer 16 hottest experts are
 *                             uploaded at warmstart and never evicted;
 *   shallow bigram adjunct -- once token t+1's identity is known (decode: as
 *                             soon as it is sampled; prefill: the whole chunk),
 *                             q38t_prefetch_next(bigram_mixed) looks the hash
 *                             up in the shallow-8 table and queues the
 *                             predicted top-k experts for layers 0..7 before
 *                             layer 0 of that token computes.
 * Prefetch uploads ride a separate low-priority queue on the existing uploader
 * thread and stream; demand uploads always drain first, and pending prefetch
 * bytes are capped (Q38_PREFETCH_MB, default 64) so PCIe stays available for
 * demand misses.  Prefetch changes only residency timing, never routing.
 *
 * setup: call after q38t_init/set_src, before q38t_warmstart.  artifact_path
 * is the router_v1 file (tools/router_v1_format.md); if absent or invalid the
 * prefetcher degrades gracefully to the prev-token union alone (no bigram
 * table, no hot pinning).  Env: Q38_PREFETCH=0 disables (default on whenever
 * the tier is active), Q38_PREFETCH_MB caps pending prefetch bytes,
 * Q38_PF_SLACK is the heat slack a prefetch swap may evict against. */
void q38t_prefetch_setup(const char *artifact_path);
void q38t_prefetch_next(uint64_t bigram_mixed);

/* Phase gate: speculative uploads (prev-union + bigram) are enabled per phase
 * -- Q38_PREFETCH_PREFILL (default 1: measured +6% on 2K domain-shifted
 * prefill) and Q38_PREFETCH_DECODE (default 0: measured -2..-4.5% at warm
 * steady decode).  Q38_PREFETCH=0 remains the master kill-switch; hot-pin
 * warmstart is governed by the master switch only.  The engine calls this
 * with 1 at the top of a batched prefill chunk and 0 at the top of a decode
 * token. */
void q38t_prefetch_phase(int is_prefill);

/* Warmstart: pin + upload the hot lists (when the prefetcher is armed), then
 * pre-fill the budget in HEAT_FILE order (blocking; heat fill is a no-op
 * without a loadable heat table). */
void q38t_warmstart(void);

/* One telemetry block on stderr: residency, hit rate, uploads, phase ms. */
void q38t_stats(void);

/* ---- P7: machine-readable snapshot for the serve-mode dashboard lines ----
 * The same numbers q38t_stats() prints for a human, in a struct the engine
 * turns into the gateway's TIERS / EMAP / PFETCH lines.  Counters are
 * cumulative since process start; the engine differences them per turn where
 * the dashboard wants a rate.  Returns 0 (and leaves *out zeroed) when the
 * tier is off, so the CPU-only serve path emits an honest "everything is on
 * disk" tier bar instead of nothing. */
typedef struct {
    int      n_layers, n_experts;
    uint64_t resident, resident_bytes, budget_bytes;
    uint64_t vram_total_bytes;        /* physical device VRAM (HWINFO wants the card, not the budget) */
    uint64_t hits, miss;              /* routed (layer,expert) lookups */
    uint64_t pin_hit, pf_hit;         /* hit split: pinned / first-use-of-prefetch */
    uint64_t pf_issued, pf_done, pf_evicted, pf_bytes;
    int      pinned, prefetch_on;
} Q38TierStat;
int q38t_stat(Q38TierStat *out);

/* Residency of one expert for the EMAP tier nibble: 2 = VRAM-resident,
 * 0 = streamed from the container (page cache / NVMe).  There is no
 * intermediate "RAM slot" tier in qwen38 -- experts are mmap'd raw blocks,
 * so tier 1 never occurs and the dashboard's RAM band stays empty by design. */
int q38t_resident(int layer, int eid);

#else /* !COLI_CUDA: inline stubs, engine stays CPU-only */

#include <stddef.h>
#include <string.h>
static inline int  q38t_init(int a,int b,int c,int d,int e,const int*f,const int*g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  q38t_multi_max(void){return 0;}
static inline int  q38t_multi_shared(int a){(void)a;return 0;}
static inline void q38t_issue_multi(int a,const int*b,int c,int d,const float*e,uint32_t*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
static inline void q38t_take_multi(const float*a,int b,int c,float*d,const float*e){(void)a;(void)b;(void)c;(void)d;(void)e;}
static inline void q38t_issue_multi_finish(void){}
static inline void q38t_set_device_x(const float *a){(void)a;}
static inline int  q38t_shared_issue1(int a,const float*b){(void)a;(void)b;return 0;}
static inline int  q38t_shared_take1(float*a,float b){(void)a;(void)b;return 0;}
static inline int  q38t_ready(void){return 0;}
static inline size_t q38t_release(size_t b){(void)b;return 0;}
static inline const char *q38t_device_name(void){return "";}
static inline void q38t_shutdown(void){}
static inline void q38t_set_src(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e){(void)a;(void)b;(void)c;(void)d;(void)e;}
static inline void q38t_note(int a,int b){(void)a;(void)b;}
static inline void q38t_note_many(int a,const int*b,int c){(void)a;(void)b;(void)c;}
static inline uint32_t q38t_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void q38t_mid(uint32_t a){(void)a;}
static inline void q38t_take(uint32_t a,const float*b,int c,float*d,float e){(void)a;(void)b;(void)c;(void)d;(void)e;}
static inline void q38t_set_shared(int a,const uint8_t*b,const uint8_t*c,const uint8_t*d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}
static inline int  q38t_shared_on(int a){(void)a;return 0;}
static inline void q38t_prefetch_setup(const char *a){(void)a;}
static inline void q38t_prefetch_next(uint64_t a){(void)a;}
static inline void q38t_prefetch_phase(int a){(void)a;}
static inline void q38t_warmstart(void){}
static inline void q38t_stats(void){}
typedef struct {
    int      n_layers, n_experts;
    uint64_t resident, resident_bytes, budget_bytes;
    uint64_t vram_total_bytes;
    uint64_t hits, miss;
    uint64_t pin_hit, pf_hit;
    uint64_t pf_issued, pf_done, pf_evicted, pf_bytes;
    int      pinned, prefetch_on;
} Q38TierStat;
static inline int q38t_stat(Q38TierStat *o){ if(o) memset(o,0,sizeof *o); return 0; }
static inline int q38t_resident(int a,int b){(void)a;(void)b;return 0;}

#endif /* COLI_CUDA */
#endif /* QWEN38_TIER_H */
