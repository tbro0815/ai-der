/* qwen38_tier.c -- CUDA VRAM expert tier for the qwen38 engine.  See header.
 *
 * Plain C against the CUDA runtime API (link -lcudart) + the raw-GGML-block
 * kernels in ggml_blocks_cuda.cu.  Deliberately NOT built on backend_cuda.cu's
 * ColiCudaTensor/expert-group API: that surface decodes colibri's int4/int8
 * container formats, while qwen38 experts are raw GGML blocks (IQ3_S/IQ4_XS
 * gate&up, IQ4_NL/Q8_0 down) that ggml_blocks_cuda GEMVs byte-identically to
 * the CPU decoders.  Heat/hysteresis semantics are shared via tier.h. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include "qwen38_tier.h"
#include "ggml_blocks.h"
#include "ggml_blocks_cuda.h"
#include "tier.h"

#define QT_MAXK   32          /* top-k ceiling (model uses 10) */
#define QT_MAXB   1024        /* S3: max tokens per multi-token issue (Q38_PREFILL_B ceiling) */
#define QT_MAXE   1024        /* engine config ceiling */
#define QT_GROUP_TILE 4       /* must match ggml_blocks grouped kernel ceiling */
#define QT_QCAP   64          /* upload queue depth */
#define QT_QCAP2  256         /* P4: low-priority prefetch queue depth */
#define QT_MAGIC  0x48383351u /* "Q38H" little-endian: heat file header */
#define QT_SWAPS_PER_TICK 8
#define QT_SLAB   (256ull<<20)   /* VRAM slab size */
#define QT_CLASSES 16            /* distinct expert byte sizes (<=4 in practice) */

typedef struct {
    void *dev;                       /* g | u | d raw blocks, one allocation */
    const uint8_t *g, *u, *d;        /* mmap'd container sources */
    uint32_t heat;
    uint32_t recent;                 /* P12 H: routing count of the recent window */
    uint8_t resident, queued;
    uint8_t pinned;                  /* P4: hot-list pin, never a swap victim */
    uint8_t pf;                      /* P4: resident via prefetch, unused so far */
} QSlot;

static struct {
    int on, dead, nl, ne, D, I, topk, dev;
    int gu_type[128], d_type[128];
    size_t gu_bytes[128], d_bytes[128], exp_bytes[128], max_exp;
    int    cls[128];                 /* size class per layer */
    size_t budget, used;
    /* slab allocator: per-expert cudaMalloc rounds ~2.3 MB up to the driver's
     * 2 MiB granularity (~70% waste, found the hard way); experts are carved
     * from 256 MB slabs instead, freed slots recycle through per-size-class
     * free lists (a swap victim's slot is reusable by its own layer family). */
    size_t cls_bytes[QT_CLASSES]; int ncls;
    void **freelist[QT_CLASSES]; int nfree[QT_CLASSES], capfree[QT_CLASSES];
    uint8_t *slab_cur; size_t slab_rem, slab_total;
    struct { uint8_t *base; size_t size; } slabs[512]; int nslab;   /* P12: releasable */
    int busy;                        /* uploader outside the lock (copying) */
    /* P12 heat adaptation knobs (env, see q38t_init) */
    int swaps_per_tick, tick_tokens, heat0_shift, promote_recent, recent_decay;
    /* P12 H5: promote staged misses by a device-to-device copy out of the
     * staging window (no PCIe): the tick leaves its coldest residents as a
     * victim pool, the issue picks staged experts that beat a same-class
     * victim under the hysteresis, the NEXT issue copies them into slots
     * before it overwrites the window. */
    int promote_staged;
    long vict[256]; int nvict;
    struct { int layer, eid; size_t off; long victim; } promo[64]; int npromo;
    uint64_t promos, promo_fail;
    pthread_mutex_t alloc_mx;        /* slot_alloc/slot_free: uploader + compute thread */
    uint64_t released_bytes, released_experts;
    QSlot *slot;                     /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_cond_t cv, cv_take;
    pthread_t th; int th_stop;
    struct { int layer, eid, v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    int issue_open;
    /* uploader resources.  P12 U1: two pinned staging buffers so the
     * mmap->pinned copy of the next expert overlaps the H2D of the previous
     * one (QT_UPLOAD_PIPE=0 restores the serial copy/upload/sync loop). */
    uint8_t *stage;                  /* pinned, max_exp bytes */
    uint8_t *stage2;                 /* pinned, max_exp bytes (pipelined uploader) */
    cudaEvent_t ev_up[2];
    int up_pipe;
    double up_ms, busy_t0;           /* uploader busy wall time (dequeue -> idle), for the GB/s in the stats */
    cudaStream_t up_st, co_st, sh_st;
    cudaStream_t ms_st;              /* miss-staging H2D, overlaps resident tiles */
    cudaEvent_t ev_gu, ev_y, ev_x, ev_sh, ev_ms;
    /* QT_KPROF=1: device-side timing of the grouped chain (resident gate|up,
     * staged wait + gate|up, silu + down), accumulated ms, printed in stats */
    int kprof; cudaEvent_t kp[4]; double kp_ms[3]; int kp_pending;
    /* P10 P5: the mmap->pinned bounce of a chunk's staged misses runs on a
     * worker thread so the caller can compute the shared expert meanwhile;
     * q38t_issue_multi_finish joins it and launches the staged tiles. */
    pthread_t bth; pthread_mutex_t bmx; pthread_cond_t bcv;
    int b_state, b_stop, b_layer, b_nds, b_ok;     /* 0 idle, 1 job, 2 done */
    int b_eid[1024]; size_t b_off[1024]; size_t b_used;
    int mi_deferred, mi_nt, mi_nt_res, mi_ng, mi_ng_res, mi_ok, mi_job;
    /* P10/S2 staging rounds: the misses of one issue are staged through the
     * window in rounds; rd_ds[r]..rd_ds[r+1] are round r's ds entries,
     * rd_used[r] its bytes, ev_rd marks the end of a round's gate|up tiles */
    #define QT_MS_ROUNDS 16
    int rd_ds[QT_MS_ROUNDS+1]; size_t rd_used[QT_MS_ROUNDS]; int nround;
    int ds_eid_g[1024]; size_t ds_off_g[1024]; int nds_g;
    cudaEvent_t ev_rd;
    uint64_t ms_rounds_extra;
    /* P10 D2: the per-issue tables (expert pointers, pair->token map, pair
     * list, tile table, tile offsets) and the per-take tables (token offsets,
     * pair lists, weights) each travel as ONE packed upload instead of five
     * and three small ones -- at B=1 (decode) the launches dominated. */
    uint8_t *h_meta, *d_meta, *h_meta2, *d_meta2; size_t meta_cap, meta2_cap;
    const void **mi_dptr; int *mi_dptok, *mi_dgpair, *mi_dgtile, *mi_dtoff;
    /* per-issue state (single decode thread) */
    int is_cnt, is_cc, is_k[QT_MAXK], is_eid[QT_MAXK], is_layer, is_fail, sh_issued;
    /* per-layer shared expert, permanently VRAM-resident (outside the budget,
     * ~110 MB): computed on the same async chain as the routed group */
    struct { void *dev; const uint8_t *g,*u,*d;
             int gu_type,d_type; size_t gu_bytes,d_bytes; int on; } sh[128];
    float *d_sg, *d_su, *d_sh;       /* device scratch [inter] each */
    /* device + pinned buffers for the compute path */
    float *d_x, *d_gu, *d_h, *d_y;
    const void **d_ptr;              /* 3*QT_MAXK device pointers */
    float *h_x, *h_gu, *h_h, *h_y;   /* pinned */
    const void **h_ptr;              /* pinned */
    /* S3 multi-token issue: one chain per (layer, prefill chunk).  Every
     * resident (token, expert) pair is one batched-GEMV entry with its own
     * activation (gemv_batch_sx); the shared expert adds one pair per token. */
    int maxpairs;                    /* QT_MAXB * (topk + 1) */
    float *d_xb, *d_gub, *d_hb, *d_yb;
    float *h_xb, *h_yb;              /* pinned */
    const void **d_ptrb, **h_ptrb;   /* 3*maxpairs */
    const void **h_gptr;             /* 3*maxpairs, unique expert groups */
    int *h_goff, *h_gpair, *h_gtile, *h_toff;
    int *d_gpair, *d_gtile, *d_toff;
    cudaEvent_t ev_ym;
    int mi_cnt, mi_layer, mi_fail, mi_shared;
    /* device-scatter path (grouped issue): X uploaded once per token, pairs
     * read it through a token map, y stays on the device and take_multi runs
     * the weighted accumulation there in the host's pair order */
    int mi_scatter;
    float *d_xt, *h_xt, *d_out, *h_out;      /* [QT_MAXB*D] */
    /* P10 P7a: the dense path's device copy of X (d_b_mixed) feeds the
     * grouped kernels directly; the host copy/upload of X is skipped.  Same
     * bytes as the host X (which is that buffer's download), so exact.
     * QT_X_DEV=0 keeps the upload. */
    const float *x_dev, *mi_x, *mi_dx; int x_dev_on;
    int trace;                       /* QT_TRACE=1: stage prints of the multi path (debug) */
    int *d_ptok, *h_ptok, *d_toklist, *h_toklist;   /* [maxpairs] */
    int *d_tokoff, *h_tokoff;                /* [QT_MAXB+1] */
    float *d_pw, *h_pw;                      /* [maxpairs] */
    int *mi_tok, *mi_k, *mi_eid;     /* [maxpairs]; mi_k < 0 = shared pair */
    /* miss staging: non-resident experts of a multi issue ride the chain from
     * a scratch VRAM window (mmap -> pinned -> H2D on the compute stream)
     * instead of costing ~1.4 ms each in scalar CPU decode */
    uint8_t *h_ms, *d_ms; size_t ms_cap;
    uint64_t ms_pairs, ms_bytes;
    /* statistics */
    uint64_t hits, miss, uploads, swaps, q_full_skips, tick;
    uint64_t up_bytes, win_uploads, win_hits, win_miss;
    double t_issue, t_gu_wait, t_silu, t_down, t_y_wait;
    uint32_t *heat0;
    int timers;
    char device_name[256];
    /* ---- P4 async prefetcher (docs/p4-analysis.md ensemble) ---- */
    int pf_on;                       /* Q38_PREFETCH master switch (default on) */
    int pf_prefill, pf_decode;       /* per-phase enables (default 1 / 0) */
    int pf_phase;                    /* 1 = inside a batched prefill chunk */
    uint32_t pf_slack;               /* heat slack for a prefetch swap victim */
    size_t pf_cap, pf_pend;          /* pending prefetch byte cap / in flight */
    struct { int layer, eid, v_layer, v_eid; } q2[QT_QCAP2];
    int q2h, q2t, q2n;               /* low-priority queue (drained after G.q) */
    /* Router v2 step 3 (2026-09-05): QT_RANK_TABLE=<w> credits the table's
     * predicted experts for token t+1 with w routing counts in the recent
     * window (no upload), so the LFRU tick and the H5 promotion rank by
     * "routed recently OR predicted next" and residents the table expects
     * are not evicted.  0 (default) = off.  The artifact loads for the
     * ranking even when the prefetcher is off. */
    int rank_w;                      /* credit per predicted expert */
    int rank_topk;                   /* QT_RANK_TOPK: first k of each layer's row (default 10, the routed top-k) */
    int rank_resident;               /* QT_RANK_RESIDENT=1: credit residents only (eviction protection, no promotion) */
    uint64_t rank_bumps, rank_lookups, rank_hits;
    /* router_v1 artifact (one malloc'd blob; shallow build ~9 MB) */
    uint8_t *rt_blob;
    uint32_t rt_bits, rt_predk, rt_hotm, rt_nctx, rt_tlayers, rt_nlayers;
    const uint32_t *rt_bucket;       /* [rt_nctx], sorted ascending */
    const uint16_t *rt_table;        /* [rt_nctx][rt_tlayers][rt_predk] */
    const uint16_t *rt_hot;          /* [nl][rt_hotm] */
    int pinned_cnt;
    uint64_t pf_iss_ng, pf_iss_prev, pf_done, pf_bytes;
    uint64_t pf_hit, pin_hit, pf_evict;
    uint64_t pf_drop_cap, pf_drop_vict, pf_drop_q;
    uint64_t win_pf_hit;
} G;

static double qms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static void *bounce_worker(void *arg);   /* P10 P5, defined with the multi-issue path */
void q38t_issue_multi_finish(void);
static QSlot *qs(int l,int e){ return &G.slot[(size_t)l*G.ne+e]; }

static int qck(cudaError_t e, const char *what){
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "[qtier] CUDA error in %s: %s -> tier disabled, CPU path\n",
            what, cudaGetErrorString(e));
    G.dead = 1;
    return 0;
}

/* ---- VRAM slot allocator (uploader thread only) ---- */
static void *slot_alloc_unlocked(int layer);
static void *slot_alloc(int layer){
    pthread_mutex_lock(&G.alloc_mx);
    void *p=slot_alloc_unlocked(layer);
    pthread_mutex_unlock(&G.alloc_mx);
    return p;
}
static void *slot_alloc_unlocked(int layer){
    int c=G.cls[layer]; size_t sz=G.cls_bytes[c];
    if(G.nfree[c]) return G.freelist[c][--G.nfree[c]];
    if(G.slab_rem < sz){
        size_t want = QT_SLAB;
        pthread_mutex_lock(&G.mx);
        size_t room = G.budget > G.slab_total ? G.budget - G.slab_total : 0;
        pthread_mutex_unlock(&G.mx);
        if(want > room) want = room;
        if(want < sz) return NULL;
        void *s=NULL;
        if(cudaMalloc(&s,want)!=cudaSuccess){
            if(want<=sz || cudaMalloc(&s,sz)!=cudaSuccess) return NULL;
            want=sz;
        }
        G.slab_cur=s; G.slab_rem=want; G.slab_total+=want;
        pthread_mutex_lock(&G.mx);
        if(G.nslab<512){ G.slabs[G.nslab].base=(uint8_t*)s; G.slabs[G.nslab].size=want; G.nslab++; }
        pthread_mutex_unlock(&G.mx);
    }
    void *p=G.slab_cur; G.slab_cur+=sz; G.slab_rem-=sz;
    return p;
}
/* can slot_alloc succeed without evicting anything?  (free-list entry, room
 * in the current slab, or budget room for a new slab) */
static int slot_can_alloc(int layer){
    /* called with G.mx held: no alloc_mx here (the uploader takes alloc_mx
     * then G.mx when it carves a slab, so the reverse order would deadlock);
     * a racy answer only turns into a failed, undone promotion */
    int c=G.cls[layer]; size_t sz=G.cls_bytes[c];
    return G.nfree[c]>0 || G.slab_rem>=sz || (G.budget>G.slab_total && G.budget-G.slab_total>=sz);
}
static void slot_free(int layer,void *p){
    pthread_mutex_lock(&G.alloc_mx);
    int c=G.cls[layer];
    if(G.nfree[c]==G.capfree[c]){
        G.capfree[c]=G.capfree[c]?2*G.capfree[c]:1024;
        G.freelist[c]=realloc(G.freelist[c],G.capfree[c]*sizeof(void*));
        if(!G.freelist[c]){ G.nfree[c]=G.capfree[c]=0; pthread_mutex_unlock(&G.alloc_mx); return; }   /* leak the slot */
    }
    G.freelist[c][G.nfree[c]++]=p;
    pthread_mutex_unlock(&G.alloc_mx);
}

/* ---- background uploader: mmap -> pinned staging -> VRAM ---- */
/* One queued upload in flight: its bookkeeping is published once the H2D
 * completed (P12 U1: the next item's host copy runs meanwhile). */
typedef struct { int live, layer, from_pf, ok; QSlot *s; void *dst; size_t tot; void *vdev; size_t vbytes; int buf; } UpJob;
/* G.mx held: account one finished upload. */
static void upload_publish_locked(UpJob *j){
    if(j->ok){ j->s->dev=j->dst; j->s->resident=1; G.uploads++; G.win_uploads++; G.up_bytes+=j->tot;
               if(j->from_pf){ j->s->pf=1; G.pf_done++; G.pf_bytes+=j->tot; } }
    else  { G.used -= G.exp_bytes[j->layer];   /* release the reservation */
            if(!j->vdev) G.budget = G.used;    /* device genuinely full: stop growing */ }
    if(j->vdev) G.used -= j->vbytes;           /* victim's bytes are back */
    if(j->from_pf) G.pf_pend -= G.exp_bytes[j->layer]<G.pf_pend?G.exp_bytes[j->layer]:G.pf_pend;
    j->s->queued=0;
    j->live=0;
}
static void *uploader(void *arg){
    (void)arg;
    cudaSetDevice(G.dev);
    if(getenv("QT_TRACE")) fprintf(stderr,"[qtrace] uploader tid %ld\n",(long)syscall(SYS_gettid));
    UpJob prev={0}; int nbuf=0;
    for(;;){
        pthread_mutex_lock(&G.mx);
        if(prev.live){
            /* an upload is in flight: take the next item only when one is
             * queued; otherwise finish the flight first */
            if(G.qn==0 && G.q2n==0){
                pthread_mutex_unlock(&G.mx);
                if(prev.ok && cudaEventSynchronize(G.ev_up[prev.buf])!=cudaSuccess){ prev.ok=0; slot_free(prev.layer,prev.dst); }
                pthread_mutex_lock(&G.mx);
                upload_publish_locked(&prev);
                G.busy=0; G.up_ms += qms()-G.busy_t0;
                pthread_cond_broadcast(&G.cv_take);
                pthread_mutex_unlock(&G.mx);
                continue;
            }
        } else {
            while(G.qn==0 && G.q2n==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
            if(G.th_stop && G.qn==0 && G.q2n==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        }
        /* demand queue always drains first; prefetch is strictly lower priority */
        int from_pf = (G.qn==0);
        int layer,eid,vl,ve;
        if(!from_pf){
            layer=G.q[G.qh].layer; eid=G.q[G.qh].eid;
            vl=G.q[G.qh].v_layer;  ve=G.q[G.qh].v_eid;
            G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        } else {
            layer=G.q2[G.q2h].layer; eid=G.q2[G.q2h].eid;
            vl=G.q2[G.q2h].v_layer;  ve=G.q2[G.q2h].v_eid;
            G.q2h=(G.q2h+1)%QT_QCAP2; G.q2n--;
        }
        pthread_cond_broadcast(&G.cv_take);            /* queue space */
        if(!G.busy) G.busy_t0=qms();
        G.busy=1;                                      /* P12: q38t_release waits for this */
        void *vdev=NULL; size_t vbytes=0;
        if(ve>=0){
            /* LFRU swap: free the victim only when no group is in flight */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            QSlot *v=qs(vl,ve);
            vdev=v->dev; v->dev=NULL; vbytes=G.exp_bytes[vl];
            v->queued=0;                               /* victim reservation (P4) released */
            if(v->pf){ G.pf_evict++; v->pf=0; }        /* prefetched, never used */
        }
        pthread_mutex_unlock(&G.mx);
        if(vdev) slot_free(vl, vdev);

        size_t gu=G.gu_bytes[layer], db=G.d_bytes[layer], tot=2*gu+db;
        QSlot *s=qs(layer,eid);
        uint8_t *stage = (G.up_pipe && nbuf) ? G.stage2 : G.stage;
        memcpy(stage,        s->g, gu);
        memcpy(stage+gu,     s->u, gu);
        memcpy(stage+2*gu,   s->d, db);
        void *dst=slot_alloc(layer);
        int ok = dst != NULL;
        if(ok){
            ok = cudaMemcpyAsync(dst, stage, tot, cudaMemcpyHostToDevice, G.up_st)==cudaSuccess;
            if(ok && G.up_pipe) ok = cudaEventRecord(G.ev_up[nbuf], G.up_st)==cudaSuccess;
            if(ok && !G.up_pipe) ok = cudaStreamSynchronize(G.up_st)==cudaSuccess;
            if(!ok) slot_free(layer, dst);
        }
        UpJob cur={1,layer,from_pf,ok,s,dst,tot,vdev,vbytes,nbuf};
        if(G.up_pipe){
            /* the previous flight completes before this one (stream order);
             * publish it now, then the next iteration may reuse its buffer */
            if(prev.live){
                if(prev.ok && cudaEventSynchronize(G.ev_up[prev.buf])!=cudaSuccess){ prev.ok=0; slot_free(prev.layer,prev.dst); }
            }
            pthread_mutex_lock(&G.mx);
            if(prev.live) upload_publish_locked(&prev);
            pthread_cond_broadcast(&G.cv_take);
            pthread_mutex_unlock(&G.mx);
            prev=cur; nbuf^=1;
            continue;
        }
        pthread_mutex_lock(&G.mx);
        upload_publish_locked(&cur);
        G.busy=0; G.up_ms += qms()-G.busy_t0;
        pthread_cond_broadcast(&G.cv_take);
        pthread_mutex_unlock(&G.mx);
    }
}

/* P10 P7a: the caller names the device copy of the current chunk's X (NULL
 * = none; the tier uploads its own) before an issue_multi. */
void q38t_set_device_x(const float *dx){ G.x_dev=dx; }

/* P12: hand VRAM back (the KV mirror grows with the conversation).  Frees
 * whole slabs, coldest first (sum of resident heat), evicting every expert
 * they hold; the budget shrinks by the same bytes so the allocator does not
 * take them again.  Runs with the uploader parked and no group in flight.
 * Returns the bytes freed (0 when nothing could be freed). */
size_t q38t_release(size_t bytes){
    if(!G.on||G.dead||bytes==0) return 0;
    pthread_mutex_lock(&G.mx);
    while((G.qn||G.q2n||G.busy||G.issue_open) && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    size_t freed=0; int nfreed=0; uint64_t evicted=0;
    double t0=qms();
    while(freed<bytes && G.nslab>0){
        /* slab heat: sum over resident experts inside it */
        uint64_t *heat=calloc((size_t)G.nslab,sizeof(uint64_t));
        if(!heat) break;
        for(size_t i=0;i<(size_t)G.nl*G.ne;i++){
            QSlot *s=&G.slot[i];
            if(!s->resident||!s->dev) continue;
            for(int k=0;k<G.nslab;k++)
                if((uint8_t*)s->dev>=G.slabs[k].base && (uint8_t*)s->dev<G.slabs[k].base+G.slabs[k].size){ heat[k]+=s->heat+1; break; }
        }
        int v=0; for(int k=1;k<G.nslab;k++) if(heat[k]<heat[v]) v=k;
        free(heat);
        uint8_t *base=G.slabs[v].base; size_t size=G.slabs[v].size;
        for(size_t i=0;i<(size_t)G.nl*G.ne;i++){
            QSlot *s=&G.slot[i];
            if(!s->dev||(uint8_t*)s->dev<base||(uint8_t*)s->dev>=base+size) continue;
            int layer=(int)(i/G.ne);
            if(s->resident){ s->resident=0; G.used-=G.used>=G.exp_bytes[layer]?G.exp_bytes[layer]:G.used; evicted++; }
            if(s->pinned){ s->pinned=0; if(G.pinned_cnt>0) G.pinned_cnt--; }
            s->pf=0; s->dev=NULL;
        }
        for(int c=0;c<G.ncls;c++){                    /* prune the free lists */
            int w=0;
            for(int j=0;j<G.nfree[c];j++){
                uint8_t *p=(uint8_t*)G.freelist[c][j];
                if(p>=base && p<base+size) continue;
                G.freelist[c][w++]=G.freelist[c][j];
            }
            G.nfree[c]=w;
        }
        if(G.slab_cur>=base && G.slab_cur<=base+size){ G.slab_cur=NULL; G.slab_rem=0; }
        cudaFree(base);
        G.slab_total-=size<G.slab_total?size:G.slab_total;
        G.budget = G.budget>size ? G.budget-size : 0;
        G.slabs[v]=G.slabs[--G.nslab];
        freed+=size; nfreed++;
    }
    G.released_bytes+=freed; G.released_experts+=evicted;
    pthread_mutex_unlock(&G.mx);
    if(freed) fprintf(stderr,"[qtier] released %.0f MB (%d slab%s, %llu experts evicted) in %.0f ms | budget now %.2f GB\n",
                      freed/1048576.0,nfreed,nfreed==1?"":"s",(unsigned long long)evicted,qms()-t0,G.budget/1073741824.0);
    return freed;
}

/* G.mx held.  victim<0: plain upload (budget reserved here); victim>=0: swap
 * (victim's bytes are released by the uploader). */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued||!s->g) return 0;
    if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }
    if(v_eid<0 && G.used+G.exp_bytes[layer]>G.budget) return 0;
    G.used += G.exp_bytes[layer];
    s->queued=1;
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
    pthread_cond_signal(&G.cv);
    return 1;
}

/* P4, G.mx held: enqueue a low-priority prefetch upload for (layer, eid).
 * With budget room it is a plain reserve; with the budget full it swaps
 * against the coldest same-layer non-pinned resident (same size class, so the
 * slab slot recycles directly) -- but only when that victim's heat is below
 * the incoming expert's heat + pf_slack, so a one-shot prediction cannot
 * evict a genuinely hot resident.  Pending prefetch bytes are capped
 * (Q38_PREFETCH_MB) so the PCIe link stays available for demand misses. */
static int enqueue_pf_locked(int layer,int eid){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued||!s->g) return 0;
    if(G.q2n>=QT_QCAP2){ G.pf_drop_q++; return 0; }
    size_t eb=G.exp_bytes[layer];
    if(G.pf_pend+eb>G.pf_cap){ G.pf_drop_cap++; return 0; }
    int vl=-1, ve=-1;
    if(G.used+eb>G.budget){
        /* victim = coldest same-layer non-pinned resident; a prefetched slot
         * that was never used is speculative and scores at heat/8, so churn
         * recycles speculation before it eats demand-earned residency */
        int cold=-1; uint32_t ch=0;
        for(int e2=0;e2<G.ne;e2++){
            QSlot *v=qs(layer,e2);
            if(!v->resident || v->queued || v->pinned) continue;
            uint32_t vh = v->pf ? v->heat/8 : v->heat;
            if(cold<0||vh<ch){ cold=e2; ch=vh; }
        }
        /* admission: default is the tier.h hysteresis (same bar as the LFRU
         * tick, applied immediately) -- prefetch then IS "LFRU without the
         * 16-token wait".  Q38_PF_SLACK=<n> relaxes it to heat<incoming+n,
         * which measures as high-churn/low-precision at tight budgets. */
        int okv = cold>=0 && (G.pf_slack ? (uint64_t)ch < (uint64_t)s->heat + G.pf_slack
                                         : tier_should_promote(s->heat, ch));
        if(!okv){ G.pf_drop_vict++; return 0; }
        /* the victim keeps `queued` until its slot is actually freed, so a
         * later demand upload of the same expert cannot overtake this swap in
         * the queue and have its fresh allocation freed underneath it */
        qs(layer,cold)->resident=0;              /* CPU fallback from now on */
        qs(layer,cold)->queued=1;
        vl=layer; ve=cold;
    }
    G.used += eb;                                /* swap: uploader releases the victim's bytes */
    s->queued=1;
    G.q2[G.q2t].layer=layer; G.q2[G.q2t].eid=eid;
    G.q2[G.q2t].v_layer=vl;  G.q2[G.q2t].v_eid=ve;
    G.q2t=(G.q2t+1)%QT_QCAP2; G.q2n++;
    G.pf_pend += eb;
    pthread_cond_signal(&G.cv);
    return 1;
}

int q38t_init(int nl,int ne,int D,int I,int topk,const int *gu_type,const int *d_type){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(nl>128 || topk>QT_MAXK){ fprintf(stderr,"[qtier] nl>128 or topk>%d unsupported\n",QT_MAXK); return 0; }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.I=I; G.topk=topk;
    G.timers = getenv("COLI_TIMERS") && atoi(getenv("COLI_TIMERS"));
    G.kprof = getenv("QT_KPROF") && atoi(getenv("QT_KPROF"));
    { const char *xe=getenv("QT_X_DEV"); G.x_dev_on = xe ? atoi(xe) : 1; }
    G.trace = getenv("QT_TRACE") && atoi(getenv("QT_TRACE"));
#define QT_TR(...) do{ if(G.trace){ fprintf(stderr,"[qtrace] " __VA_ARGS__); fputc('\n',stderr); } }while(0)
    { const char *e;
      /* defaults from the 2026-09-02 adaptation sweeps (docs/serve-decode-gap.md):
       * a novel conversation decoded 10.6 tok/s with 8 swaps / 16 tokens and
       * 17.7-18.4 with 64 swaps / 8 tokens ranked by the recent window plus
       * staged-miss promotion; QT_* restore any older setting */
      G.swaps_per_tick = (e=getenv("QT_SWAPS_PER_TICK")) ? atoi(e) : 64;
      G.tick_tokens    = (e=getenv("QT_TICK_TOKENS")) && atoi(e)>0 ? atoi(e) : 8;
      G.heat0_shift    = (e=getenv("QT_HEAT0_SHIFT")) ? atoi(e) : 1;
      G.promote_recent = (e=getenv("QT_PROMOTE_RECENT")) ? atoi(e) : 1;
      G.recent_decay   = (e=getenv("QT_RECENT_DECAY")) ? atoi(e) : 256;
      G.promote_staged = (e=getenv("QT_PROMOTE_STAGED")) ? atoi(e) : 1;
      pthread_mutex_init(&G.alloc_mx,NULL);
      if(G.heat0_shift<0) G.heat0_shift=0; if(G.heat0_shift>31) G.heat0_shift=31; }

    const char *gl=getenv("COLI_GPUS");
    G.dev = (gl && *gl) ? atoi(gl) : 0;
    int ndev=0;
    if(cudaGetDeviceCount(&ndev)!=cudaSuccess || ndev<=G.dev){
        fprintf(stderr,"[qtier] no CUDA device %d -> CPU path\n",G.dev); return 0;
    }
    if(cudaSetDevice(G.dev)!=cudaSuccess){ fprintf(stderr,"[qtier] cudaSetDevice failed -> CPU path\n"); return 0; }
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,G.dev)==cudaSuccess)
        snprintf(G.device_name,sizeof G.device_name,"%s",prop.name);

    G.max_exp=0;
    for(int l=0;l<nl;l++){
        size_t gur=ggml_bk_row_bytes(gu_type[l],(size_t)D);
        size_t dr =ggml_bk_row_bytes(d_type[l], (size_t)I);
        if(!gur||!dr){ fprintf(stderr,"[qtier] layer %d: bad expert types (%d,%d)\n",l,gu_type[l],d_type[l]); return 0; }
        G.gu_type[l]=gu_type[l]; G.d_type[l]=d_type[l];
        G.gu_bytes[l]=gur*(size_t)I; G.d_bytes[l]=dr*(size_t)D;
        G.exp_bytes[l]=(2*G.gu_bytes[l]+G.d_bytes[l]+255)&~(size_t)255;   /* slab carve size */
        if(G.exp_bytes[l]>G.max_exp) G.max_exp=G.exp_bytes[l];
        int ci;
        for(ci=0;ci<G.ncls;ci++) if(G.cls_bytes[ci]==G.exp_bytes[l]) break;
        if(ci==G.ncls){
            if(G.ncls>=QT_CLASSES){ fprintf(stderr,"[qtier] too many expert size classes\n"); return 0; }
            G.cls_bytes[G.ncls++]=G.exp_bytes[l];
        }
        G.cls[l]=ci;
    }

    size_t freeb=0,totb=0; cudaMemGetInfo(&freeb,&totb);
    const char *bg=getenv("CUDA_EXPERT_GB");
    double gb = (bg && strcmp(bg,"auto") && atof(bg)>0) ? atof(bg) : 0.0;
    size_t b = gb>0 ? (size_t)(gb*1073741824.0)
             : (bg && !strcmp(bg,"auto")) ? q38t_auto_budget(freeb)
             : (size_t)(9.0*1073741824.0);                 /* default ~9 GB */
    if(b > freeb-(1ull<<30)) b = freeb>(1ull<<30)?freeb-(1ull<<30):0;
    /* the multi-token scratch and the miss-staging window live outside the
     * expert budget; take them off an auto budget so the 2 GiB margin stays a
     * margin.  QT_MS_MB sets the staging window; the default holds half a
     * layer's expert set (ne/2 * largest expert, at least 512 MB): a chunk's
     * misses fit until the hit rate drops below ~50 %, and an overflowed miss
     * is computed on the CPU (the `overflow -> CPU` rule in q38t_issue_multi),
     * which cost 15 ms/token on a 13.6K-token novel prompt at a 56 % hit rate
     * with the old 512 MB default (2026-09-02: 95 % of the misses went to the
     * CPU).  A full-layer window (2.55 GB) starved the resident tier at 128K
     * context (decode 10 tok/s); window-sized staging rounds (TODO P10/S2)
     * remove the tradeoff. */
    { const char *ms=getenv("QT_MS_MB"); double msmb=(ms&&atof(ms)>0)?atof(ms):0.0;
      /* S2: with staging rounds the window only sets the round size; 512 MB
       * (~220 experts) keeps the rounds few and hands the rest to the tier */
      G.ms_cap = msmb>0 ? (size_t)(msmb*1048576.0) : (512u<<20);
      fprintf(stderr,"[qtier] miss-staging window %.0f MB (%s, staged in rounds)\n",G.ms_cap/1048576.0,msmb>0?"QT_MS_MB":"default");
      size_t mp=(size_t)QT_MAXB*(topk+1);
      size_t scratch=mp*((size_t)D*4*2+(size_t)I*4*3+3*sizeof(void*)+3*4+3*4)
                    +(size_t)QT_MAXB*D*4*2+G.ms_cap;
      /* QT_RESERVE_MB: device bytes another module will allocate later (the
       * P6.1 GPU vision tower, ~1.4 GB) come off an auto budget too */
      { const char *rv=getenv("QT_RESERVE_MB"); double rmb=(rv&&atof(rv)>0)?atof(rv):0.0;
        scratch += (size_t)(rmb*1048576.0); }
      if(bg && !strcmp(bg,"auto")) b = b>scratch ? b-scratch : 0; }
    G.budget=b;
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    if(!G.slot) return 0;

    /* pinned + device compute buffers */
    { const char *pe=getenv("QT_UPLOAD_PIPE"); G.up_pipe = pe ? atoi(pe) : 1; }
    if(G.up_pipe && (!qck(cudaHostAlloc((void**)&G.stage2,G.max_exp,cudaHostAllocDefault),"stage2") ||
                     !qck(cudaEventCreateWithFlags(&G.ev_up[0],cudaEventDisableTiming),"ev_up0") ||
                     !qck(cudaEventCreateWithFlags(&G.ev_up[1],cudaEventDisableTiming),"ev_up1"))) return 0;
    if(!qck(cudaHostAlloc((void**)&G.stage,G.max_exp,cudaHostAllocDefault),"stage") ||
       !qck(cudaHostAlloc((void**)&G.h_x,(size_t)D*sizeof(float),0),"h_x") ||
       !qck(cudaHostAlloc((void**)&G.h_gu,(size_t)2*QT_MAXK*I*sizeof(float),0),"h_gu") ||
       !qck(cudaHostAlloc((void**)&G.h_h,(size_t)QT_MAXK*I*sizeof(float),0),"h_h") ||
       !qck(cudaHostAlloc((void**)&G.h_y,(size_t)QT_MAXK*D*sizeof(float),0),"h_y") ||
       !qck(cudaHostAlloc((void**)&G.h_ptr,3*QT_MAXK*sizeof(void*),0),"h_ptr") ||
       !qck(cudaMalloc((void**)&G.d_x,(size_t)D*sizeof(float)),"d_x") ||
       !qck(cudaMalloc((void**)&G.d_gu,(size_t)2*QT_MAXK*I*sizeof(float)),"d_gu") ||
       !qck(cudaMalloc((void**)&G.d_h,(size_t)QT_MAXK*I*sizeof(float)),"d_h") ||
       !qck(cudaMalloc((void**)&G.d_y,(size_t)QT_MAXK*D*sizeof(float)),"d_y") ||
       !qck(cudaMalloc((void**)&G.d_ptr,3*QT_MAXK*sizeof(void*)),"d_ptr") ||
       !qck(cudaMalloc((void**)&G.d_sg,(size_t)I*sizeof(float)),"d_sg") ||
       !qck(cudaMalloc((void**)&G.d_su,(size_t)I*sizeof(float)),"d_su") ||
       !qck(cudaMalloc((void**)&G.d_sh,(size_t)I*sizeof(float)),"d_sh") ||
       !qck(cudaStreamCreateWithFlags(&G.up_st,cudaStreamNonBlocking),"up_st") ||
       !qck(cudaStreamCreateWithFlags(&G.co_st,cudaStreamNonBlocking),"co_st") ||
       !qck(cudaStreamCreateWithFlags(&G.sh_st,cudaStreamNonBlocking),"sh_st") ||
       !qck(cudaStreamCreateWithFlags(&G.ms_st,cudaStreamNonBlocking),"ms_st") ||
       !qck(cudaEventCreateWithFlags(&G.ev_ms,cudaEventDisableTiming),"ev_ms") ||
       !qck(cudaEventCreateWithFlags(&G.ev_rd,cudaEventDisableTiming),"ev_rd") ||
       !qck(cudaEventCreate(&G.kp[0]),"kp0") || !qck(cudaEventCreate(&G.kp[1]),"kp1") ||
       !qck(cudaEventCreate(&G.kp[2]),"kp2") || !qck(cudaEventCreate(&G.kp[3]),"kp3") ||
       !qck(cudaEventCreateWithFlags(&G.ev_gu,cudaEventDisableTiming),"ev_gu") ||
       !qck(cudaEventCreateWithFlags(&G.ev_y,cudaEventDisableTiming),"ev_y") ||
       !qck(cudaEventCreateWithFlags(&G.ev_x,cudaEventDisableTiming),"ev_x") ||
       !qck(cudaEventCreateWithFlags(&G.ev_sh,cudaEventDisableTiming),"ev_sh"))
        return 0;

    /* S3 multi-token issue buffers (~24 MB dev + ~14 MB pinned at topk=10) */
    G.maxpairs = QT_MAXB * (topk + 1);
    G.mi_tok = malloc(G.maxpairs*sizeof(int));
    G.mi_k   = malloc(G.maxpairs*sizeof(int));
    G.mi_eid = malloc(G.maxpairs*sizeof(int));
    G.h_gptr = malloc(3*(size_t)G.maxpairs*sizeof(void*));
    G.h_goff = malloc((size_t)(G.maxpairs+1)*sizeof(int));
    G.h_gpair= malloc((size_t)G.maxpairs*sizeof(int));
    G.h_gtile= malloc((size_t)G.maxpairs*sizeof(int));
    G.h_toff = malloc((size_t)(G.maxpairs+1)*sizeof(int));
    if(!G.mi_tok||!G.mi_k||!G.mi_eid||!G.h_gptr||!G.h_goff||!G.h_gpair||
       !G.h_gtile||!G.h_toff) return 0;
    if(!qck(cudaHostAlloc((void**)&G.h_xb,(size_t)G.maxpairs*D*sizeof(float),0),"h_xb") ||
       !qck(cudaHostAlloc((void**)&G.h_yb,(size_t)G.maxpairs*D*sizeof(float),0),"h_yb") ||
       !qck(cudaHostAlloc((void**)&G.h_ptrb,3*(size_t)G.maxpairs*sizeof(void*),0),"h_ptrb") ||
       !qck(cudaMalloc((void**)&G.d_xb,(size_t)G.maxpairs*D*sizeof(float)),"d_xb") ||
       !qck(cudaMalloc((void**)&G.d_gub,2*(size_t)G.maxpairs*I*sizeof(float)),"d_gub") ||
       !qck(cudaMalloc((void**)&G.d_hb,(size_t)G.maxpairs*I*sizeof(float)),"d_hb") ||
       !qck(cudaMalloc((void**)&G.d_yb,(size_t)G.maxpairs*D*sizeof(float)),"d_yb") ||
       !qck(cudaMalloc((void**)&G.d_ptrb,3*(size_t)G.maxpairs*sizeof(void*)),"d_ptrb") ||
       !qck(cudaMalloc((void**)&G.d_gpair,(size_t)G.maxpairs*sizeof(int)),"d_gpair") ||
       !qck(cudaMalloc((void**)&G.d_gtile,(size_t)G.maxpairs*sizeof(int)),"d_gtile") ||
       !qck(cudaMalloc((void**)&G.d_toff,(size_t)(G.maxpairs+1)*sizeof(int)),"d_toff") ||
       !qck(cudaEventCreateWithFlags(&G.ev_ym,cudaEventDisableTiming),"ev_ym"))
        return 0;
    G.meta_cap  = 3*(size_t)G.maxpairs*sizeof(void*) + 3*(size_t)G.maxpairs*sizeof(int) + ((size_t)G.maxpairs+1)*sizeof(int) + 256;
    G.meta2_cap = ((size_t)QT_MAXB+1)*sizeof(int) + (size_t)G.maxpairs*(sizeof(int)+sizeof(float)) + 256;
    if(!qck(cudaHostAlloc((void**)&G.h_meta,G.meta_cap,0),"h_meta") ||
       !qck(cudaMalloc((void**)&G.d_meta,G.meta_cap),"d_meta") ||
       !qck(cudaHostAlloc((void**)&G.h_meta2,G.meta2_cap,0),"h_meta2") ||
       !qck(cudaMalloc((void**)&G.d_meta2,G.meta2_cap),"d_meta2"))
        return 0;
    if(!qck(cudaMalloc((void**)&G.d_xt,(size_t)QT_MAXB*D*sizeof(float)),"d_xt") ||
       !qck(cudaMalloc((void**)&G.d_out,(size_t)QT_MAXB*D*sizeof(float)),"d_out") ||
       !qck(cudaMalloc((void**)&G.d_ptok,(size_t)G.maxpairs*sizeof(int)),"d_ptok") ||
       !qck(cudaMalloc((void**)&G.d_toklist,(size_t)G.maxpairs*sizeof(int)),"d_toklist") ||
       !qck(cudaMalloc((void**)&G.d_tokoff,(size_t)(QT_MAXB+1)*sizeof(int)),"d_tokoff") ||
       !qck(cudaMalloc((void**)&G.d_pw,(size_t)G.maxpairs*sizeof(float)),"d_pw") ||
       !qck(cudaHostAlloc((void**)&G.h_xt,(size_t)QT_MAXB*D*sizeof(float),0),"h_xt") ||
       !qck(cudaHostAlloc((void**)&G.h_out,(size_t)QT_MAXB*D*sizeof(float),0),"h_out") ||
       !qck(cudaHostAlloc((void**)&G.h_ptok,(size_t)G.maxpairs*sizeof(int),0),"h_ptok") ||
       !qck(cudaHostAlloc((void**)&G.h_toklist,(size_t)G.maxpairs*sizeof(int),0),"h_toklist") ||
       !qck(cudaHostAlloc((void**)&G.h_tokoff,(size_t)(QT_MAXB+1)*sizeof(int),0),"h_tokoff") ||
       !qck(cudaHostAlloc((void**)&G.h_pw,(size_t)G.maxpairs*sizeof(float),0),"h_pw"))
        return 0;
    if(!qck(cudaHostAlloc((void**)&G.h_ms,G.ms_cap,cudaHostAllocDefault),"h_ms") ||
       !qck(cudaMalloc((void**)&G.d_ms,G.ms_cap),"d_ms"))
        return 0;

    /* learned heat (HEAT_FILE): warmstart order + initial values */
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            /* P5: a file with one layer fewer (a 48-layer table read into the
             * 49-layer MTP-enabled tier) loads as a prefix -- the MTP layer
             * starts cold instead of throwing away the learned backbone heat. */
            if(fread(hdr,4,3,f)==3 && hdr[0]==QT_MAGIC &&
               (hdr[1]==(uint32_t)nl || hdr[1]+1==(uint32_t)nl) && hdr[2]==(uint32_t)ne){
                size_t fnl=hdr[1];
                G.heat0=calloc((size_t)nl*ne,4);
                if(G.heat0 && fread(G.heat0,4,fnl*ne,f)==fnl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>G.heat0_shift;   /* aged at load */
                    fprintf(stderr,"[qtier] HEAT_FILE loaded: %s (%zu/%d layers)\n",hf,fnl,nl);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }

    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL); pthread_cond_init(&G.cv_take,NULL);
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0) return 0;
    pthread_mutex_init(&G.bmx,NULL); pthread_cond_init(&G.bcv,NULL);
    if(pthread_create(&G.bth,NULL,bounce_worker,NULL)!=0) return 0;
    G.on=1;
    /* avg over layers for the banner */
    size_t avg=0; for(int l=0;l<nl;l++) avg+=G.exp_bytes[l]; avg/= (size_t)nl;
    fprintf(stderr,"[qtier] CUDA VRAM expert tier active: dev %d, %.1f GB free, "
            "budget %.2f GB (~%zu of %d experts, %.2f MB/expert avg)\n",
            G.dev, freeb/1073741824.0, G.budget/1073741824.0,
            G.budget/(avg?avg:1), nl*ne, avg/1048576.0);
    return 1;
}

int q38t_ready(void){ return G.on && !G.dead; }
const char *q38t_device_name(void){ return G.on ? G.device_name : ""; }

void q38t_set_src(int layer,int eid,const uint8_t *g,const uint8_t *u,const uint8_t *d){
    if(!G.on) return;
    QSlot *s=qs(layer,eid);
    s->g=g; s->u=u; s->d=d;
}

static void note_locked(int layer,int eid){
    QSlot *s=qs(layer,eid);
    if(s->heat<0xFFFFFFFFu) s->heat++;
    if(s->recent<0xFFFFFFFFu) s->recent++;
    /* P4 previous-token union: a routed expert is likely routed again at t+1.
     * The demand enqueue below fails when the budget is full; the prefetch
     * path then queues a low-priority swap immediately instead of waiting for
     * the 16-token LFRU tick + hysteresis. */
    if(!enqueue_locked(layer,eid,-1,-1) && G.pf_on && !s->resident && !s->queued
       && (G.pf_phase ? G.pf_prefill : G.pf_decode))
        if(enqueue_pf_locked(layer,eid)) G.pf_iss_prev++;
}
void q38t_note(int layer,int eid){
    if(!G.on||G.dead) return;
    pthread_mutex_lock(&G.mx);
    note_locked(layer,eid);
    pthread_mutex_unlock(&G.mx);
}
/* P10 P6: a whole chunk's routed ids under one lock (was one lock per pair) */
void q38t_note_many(int layer,const int *eids,int n){
    if(!G.on||G.dead) return;
    pthread_mutex_lock(&G.mx);
    for(int j=0;j<n;j++) if(eids[j]>=0) note_locked(layer,eids[j]);
    pthread_mutex_unlock(&G.mx);
}

/* P12 H: the LFRU tick.  Every QT_TICK_TOKENS tokens (16) up to
 * QT_SWAPS_PER_TICK (8) swaps of the hottest non-resident against the
 * coldest resident experts, under the tier.h hysteresis.  One pass over the
 * slots collects both candidate sets (partial selection), so a larger swap
 * budget costs no extra scans.  QT_PROMOTE_RECENT=1 ranks by the recent
 * window count (decayed every QT_RECENT_DECAY tokens, default 256) with the
 * long-term heat as tie-break, so a new conversation's working set is not
 * held back by the heat file's history; long-term heat decays every 1024. */
#define QT_SEL_MAX 256
static uint64_t slot_key(const QSlot *s){
    return G.promote_recent ? (((uint64_t)s->recent)<<32 | s->heat) : (((uint64_t)s->heat)<<32 | s->recent);
}
static void lfru_tick_locked(void){
    size_t n=(size_t)G.nl*G.ne;
    G.tick++;
    if(!(G.tick%1024))
        for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
    if(G.recent_decay>0 && !(G.tick%(unsigned)G.recent_decay))
        for(size_t i=0;i<n;i++) G.slot[i].recent>>=1;
    if(G.tick%(unsigned)G.tick_tokens) return;
    const int S = G.swaps_per_tick > QT_SEL_MAX ? QT_SEL_MAX : G.swaps_per_tick;
    if(S<=0) return;
    /* one pass: S hottest non-resident (descending) and SC coldest resident
     * (ascending); SC exceeds S so the H5 victim pool keeps candidates after
     * the swaps took theirs */
    const int SC = G.promote_staged ? QT_SEL_MAX : S;
    long hot[QT_SEL_MAX], cold[QT_SEL_MAX]; uint64_t hk[QT_SEL_MAX], ck[QT_SEL_MAX]; int nh=0, nc=0;
    for(size_t i=0;i<n;i++){
        QSlot *s=&G.slot[i];
        if(s->queued) continue;
        const uint64_t k=slot_key(s);
        if(s->resident){
            if(s->pinned) continue;
            if(nc<SC){ int j=nc++; while(j>0 && ck[j-1]>k){ ck[j]=ck[j-1]; cold[j]=cold[j-1]; j--; } ck[j]=k; cold[j]=(long)i; }
            else if(k<ck[nc-1]){ int j=nc-1; while(j>0 && ck[j-1]>k){ ck[j]=ck[j-1]; cold[j]=cold[j-1]; j--; } ck[j]=k; cold[j]=(long)i; }
        } else if(s->g){
            if(nh<S){ int j=nh++; while(j>0 && hk[j-1]<k){ hk[j]=hk[j-1]; hot[j]=hot[j-1]; j--; } hk[j]=k; hot[j]=(long)i; }
            else if(k>hk[nh-1]){ int j=nh-1; while(j>0 && hk[j-1]<k){ hk[j]=hk[j-1]; hot[j]=hot[j-1]; j--; } hk[j]=k; hot[j]=(long)i; }
        }
    }
    for(int r=0;r<S && r<nh && r<nc;r++){
        QSlot *h=&G.slot[hot[r]], *v=&G.slot[cold[r]];
        const uint32_t hh = G.promote_recent ? h->recent : h->heat;
        const uint32_t ch = G.promote_recent ? v->recent : v->heat;
        if(!tier_should_promote(hh,ch)) break;
        v->resident=0;                                  /* CPU fallback from now on */
        if(enqueue_locked((int)(hot[r]/G.ne),(int)(hot[r]%G.ne),(int)(cold[r]/G.ne),(int)(cold[r]%G.ne))) G.swaps++;
        else { v->resident=1; break; }                  /* queue full: revert */
    }
    /* H5: the unused cold candidates become the victim pool until the next tick */
    G.nvict=0;
    if(G.promote_staged)
        for(int r=0;r<nc && G.nvict<QT_SEL_MAX;r++){ QSlot *v=&G.slot[cold[r]]; if(v->resident && !v->queued && !v->pinned) G.vict[G.nvict++]=cold[r]; }
    if(G.timers && !(G.tick%64)){
        size_t res=0; for(size_t i=0;i<n;i++) res+=G.slot[i].resident;
        double tot=(double)(G.win_hits+G.win_miss);
        fprintf(stderr,"[qtier] t=%llu resident=%zu hit=%.1f%% (pf %llu) uploads/64tok=%llu\n",
                (unsigned long long)G.tick,res, tot>0?100.0*G.win_hits/tot:0.0,
                (unsigned long long)G.win_pf_hit,
                (unsigned long long)G.win_uploads);
        G.win_uploads=0; G.win_hits=0; G.win_miss=0; G.win_pf_hit=0;
    }
}

uint32_t q38t_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||G.dead||K>QT_MAXK) return 0;
    double t0=G.timers?qms():0;
    uint32_t mask=0;
    G.is_cnt=0; G.is_layer=layer; G.is_fail=0;
    pthread_mutex_lock(&G.mx);
    if(layer==0) lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int c=G.is_cnt;
            G.h_ptr[c]          = s->dev;                                   /* gate */
            G.h_ptr[QT_MAXK+c]  = (const uint8_t*)s->dev + G.gu_bytes[layer];    /* up */
            G.h_ptr[2*QT_MAXK+c]= (const uint8_t*)s->dev + 2*G.gu_bytes[layer];  /* down */
            G.is_k[c]=k; G.is_eid[c]=eids[k]; G.is_cnt=c+1;
            mask|=1u<<k; G.hits++; G.win_hits++;
            if(s->pf){ G.pf_hit++; G.win_pf_hit++; s->pf=0; }
            else if(s->pinned) G.pin_hit++;
        } else { G.miss++; G.win_miss++; }
    }
    pthread_mutex_unlock(&G.mx);
    int c=G.is_cnt;
    /* sh_issued: 0 = engine computes the shared expert, 1 = own-stream chain
     * (types differ from the routed group), 2 = batched as one extra expert
     * in the routed group's GEMVs (the common case: same types, zero extra
     * launches). */
    G.sh_issued = 0;
    if(G.sh[layer].on)
        G.sh_issued = (G.sh[layer].gu_type==G.gu_type[layer] &&
                       G.sh[layer].d_type ==G.d_type[layer]) ? 2 : 1;
    if(!c && !G.sh_issued) return 0;
    int cc=c;
    if(G.sh_issued==2){
        const uint8_t *sd=(const uint8_t*)G.sh[layer].dev;
        G.h_ptr[cc]          = sd;
        G.h_ptr[QT_MAXK+cc]  = sd + G.sh[layer].gu_bytes;
        G.h_ptr[2*QT_MAXK+cc]= sd + 2*G.sh[layer].gu_bytes;
        cc++;
    }
    /* compact the pointer lists: [g0..g{cc-1} | u.. | d..] */
    memmove((void*)(G.h_ptr+cc),   (void*)(G.h_ptr+QT_MAXK),   cc*sizeof(void*));
    memmove((void*)(G.h_ptr+2*cc), (void*)(G.h_ptr+2*QT_MAXK), cc*sizeof(void*));
    memcpy(G.h_x,x,(size_t)G.D*sizeof(float));
    /* one async chain: x H2D, gate|up batch, silu*mul, down batch, y D2H */
    int ok =
        cudaMemcpyAsync(G.d_x,G.h_x,(size_t)G.D*sizeof(float),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess &&
        cudaEventRecord(G.ev_x,G.co_st)==cudaSuccess;
    if(ok && G.sh_issued==1){
        /* shared expert on its own stream, overlapping the routed chain */
        const uint8_t *sd=(const uint8_t*)G.sh[layer].dev;
        ok = cudaStreamWaitEvent(G.sh_st,G.ev_x,0)==cudaSuccess &&
             ggml_blocks_gemv_cuda(G.sh[layer].gu_type,G.I,G.D,sd,                       G.d_x,G.d_sg,G.sh_st)==0 &&
             ggml_blocks_gemv_cuda(G.sh[layer].gu_type,G.I,G.D,sd+G.sh[layer].gu_bytes,  G.d_x,G.d_su,G.sh_st)==0 &&
             ggml_blocks_silu_mul_cuda(G.d_sg,G.d_su,G.d_sh,(size_t)G.I,G.sh_st)==0 &&
             ggml_blocks_gemv_cuda(G.sh[layer].d_type,G.D,G.I,sd+2*G.sh[layer].gu_bytes, G.d_sh,G.d_y+(size_t)(QT_MAXK-1)*G.D,G.sh_st)==0 &&
             cudaMemcpyAsync(G.h_y+(size_t)(QT_MAXK-1)*G.D,G.d_y+(size_t)(QT_MAXK-1)*G.D,(size_t)G.D*sizeof(float),cudaMemcpyDeviceToHost,G.sh_st)==cudaSuccess &&
             cudaEventRecord(G.ev_sh,G.sh_st)==cudaSuccess;
    }
    if(ok && cc){
        ok = cudaMemcpyAsync((void*)G.d_ptr,(const void*)G.h_ptr,3*cc*sizeof(void*),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess &&
             ggml_blocks_gemv_batch(G.gu_type[layer],G.I,G.D,2*cc,G.d_ptr,G.d_x,G.d_gu,G.co_st)==0 &&
             ggml_blocks_silu_mul_cuda(G.d_gu,G.d_gu+(size_t)cc*G.I,G.d_h,(size_t)cc*G.I,G.co_st)==0 &&
             ggml_blocks_gemv_batch_sx(G.d_type[layer],G.D,G.I,cc,G.d_ptr+2*cc,G.d_h,(size_t)G.I,G.d_y,G.co_st)==0 &&
             cudaMemcpyAsync(G.h_y,G.d_y,(size_t)cc*G.D*sizeof(float),cudaMemcpyDeviceToHost,G.co_st)==cudaSuccess;
    }
    ok = ok && cudaEventRecord(G.ev_y,G.co_st)==cudaSuccess;
    G.is_cc=cc;
    if(!ok){
        qck(cudaGetLastError(),"issue");
        G.is_fail=1;   /* take() recomputes this group + shared on the CPU */
    }
    if(G.timers) G.t_issue+=qms()-t0;
    return mask;
}

/* the S2 chain no longer needs a mid hop; kept as a no-op for the call shape */
void q38t_mid(uint32_t mask){ (void)mask; }

int q38t_shared_on(int layer){ return G.on && !G.dead && G.sh[layer].on; }

void q38t_set_shared(int layer,const uint8_t *g,const uint8_t *u,const uint8_t *d,
                     int gu_type,int d_type){
    if(!G.on||G.dead) return;
    size_t gur=ggml_bk_row_bytes(gu_type,(size_t)G.D);
    size_t dr =ggml_bk_row_bytes(d_type, (size_t)G.I);
    if(!gur||!dr) return;
    size_t gub=gur*(size_t)G.I, db=dr*(size_t)G.D;
    void *dev=NULL;
    if(cudaMalloc(&dev,2*gub+db)!=cudaSuccess) return;
    if(cudaMemcpy(dev,g,gub,cudaMemcpyHostToDevice)!=cudaSuccess ||
       cudaMemcpy((uint8_t*)dev+gub,u,gub,cudaMemcpyHostToDevice)!=cudaSuccess ||
       cudaMemcpy((uint8_t*)dev+2*gub,d,db,cudaMemcpyHostToDevice)!=cudaSuccess){
        cudaFree(dev); return;
    }
    G.sh[layer].dev=dev; G.sh[layer].g=g; G.sh[layer].u=u; G.sh[layer].d=d;
    G.sh[layer].gu_type=gu_type; G.sh[layer].d_type=d_type;
    G.sh[layer].gu_bytes=gub; G.sh[layer].d_bytes=db; G.sh[layer].on=1;
}

void q38t_take(uint32_t mask,const float *val,int K,float *out,float shared_s){
    (void)K;
    if(!G.on) return;
    int c=G.is_cnt;
    if(c || G.sh_issued){
        double t0=G.timers?qms():0;
        if(!G.is_fail && qck(cudaEventSynchronize(G.ev_y),"ev_y") &&
           (G.sh_issued!=1 || qck(cudaEventSynchronize(G.ev_sh),"ev_sh"))){
            for(int j=0;j<c;j++){
                float w=val[G.is_k[j]];
                const float *row=G.h_y+(size_t)j*G.D;
                for(int d=0;d<G.D;d++) out[d]+=w*row[d];
            }
            if(G.sh_issued){
                const float *row = G.sh_issued==2 ? G.h_y+(size_t)(G.is_cc-1)*G.D
                                                  : G.h_y+(size_t)(QT_MAXK-1)*G.D;
                for(int d=0;d<G.D;d++) out[d]+=shared_s*row[d];
            }
        } else {
            /* CUDA error after the CPU already skipped these k: recompute the
             * whole group on the CPU from the mmap'd raw blocks (correctness
             * over speed; the tier is dead after the first such error). */
            int l=G.is_layer;
            float *g=malloc((size_t)G.I*sizeof(float)), *u=malloc((size_t)G.I*sizeof(float));
            float *y=malloc((size_t)G.D*sizeof(float));
            if(g&&u&&y){
                for(int j=0;j<c;j++){
                    QSlot *s=qs(l,G.is_eid[j]);
                    ggml_bk_gemv(G.gu_type[l],G.I,G.D,s->g,G.h_x,g);
                    ggml_bk_gemv(G.gu_type[l],G.I,G.D,s->u,G.h_x,u);
                    for(int i=0;i<G.I;i++){
                        float gv=g[i];
                        float sg = gv>=0.f ? 1.f/(1.f+expf(-gv)) : expf(gv)/(1.f+expf(gv));
                        g[i]=gv*sg*u[i];
                    }
                    ggml_bk_gemv(G.d_type[l],G.D,G.I,s->d,g,y);
                    float w=val[G.is_k[j]];
                    for(int d=0;d<G.D;d++) out[d]+=w*y[d];
                }
                if(G.sh_issued && G.sh[l].on){
                    ggml_bk_gemv(G.sh[l].gu_type,G.I,G.D,G.sh[l].g,G.h_x,g);
                    ggml_bk_gemv(G.sh[l].gu_type,G.I,G.D,G.sh[l].u,G.h_x,u);
                    for(int i=0;i<G.I;i++){
                        float gv=g[i];
                        float sg = gv>=0.f ? 1.f/(1.f+expf(-gv)) : expf(gv)/(1.f+expf(gv));
                        g[i]=gv*sg*u[i];
                    }
                    ggml_bk_gemv(G.sh[l].d_type,G.D,G.I,G.sh[l].d,g,y);
                    for(int d=0;d<G.D;d++) out[d]+=shared_s*y[d];
                }
            }
            free(g); free(u); free(y);
        }
        if(G.timers) G.t_y_wait+=qms()-t0;
    }
    G.is_cnt=0; G.is_fail=0; G.sh_issued=0;
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

/* ---- S3 multi-token issue (chunked prefill) ---- */
int q38t_multi_max(void){ return (G.on && !G.dead) ? QT_MAXB : 0; }

int q38t_multi_shared(int layer){
    return G.on && !G.dead && G.sh[layer].on &&
           G.sh[layer].gu_type==G.gu_type[layer] && G.sh[layer].d_type==G.d_type[layer];
}

/* mmap -> pinned copy of the staged experts of one issue (OMP over the
 * cores: single-threaded it collapsed prefill once the heat-trained region
 * ended).  Used inline on the ungrouped path and by the bounce worker. */
static long g_pgmask=-1;
static void willneed(const void *p,size_t n){
    if(g_pgmask<0) g_pgmask=sysconf(_SC_PAGESIZE)-1;
    uintptr_t a=(uintptr_t)p, s=a&~(uintptr_t)g_pgmask;
    madvise((void*)s,(size_t)(a+n-s),MADV_WILLNEED);
}
static void bounce_copy(int layer,int nds,const int *eid,const size_t *off){
    /* the container is mmap'd MADV_RANDOM: on a cold page cache (after a
     * backend switch loaded other weights) every staged expert would fault
     * 4 KB at a time inside the memcpy below (seen as a minutes-long stall
     * per layer, 2026-09-03).  Announce the whole batch first so the NVMe
     * serves it at queue depth; QT_WILLNEED=0 disables.  Exact. */
    static int wn=-1;
    if(wn<0){ const char *e=getenv("QT_WILLNEED"); wn = e ? atoi(e) : 1; }
    if(wn) for(int j2=0;j2<nds;j2++){
        QSlot *s=qs(layer,eid[j2]);
        willneed(s->g,G.gu_bytes[layer]); willneed(s->u,G.gu_bytes[layer]); willneed(s->d,G.d_bytes[layer]);
    }
    static int tids_done=0;
    if(G.trace && !tids_done){ tids_done=1;
        #pragma omp parallel
        { 
            #pragma omp critical
            fprintf(stderr,"[qtrace] omp worker tid %ld (main tid %ld)\n",(long)syscall(SYS_gettid),(long)getpid());
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(int j2=0;j2<nds;j2++){
        QSlot *s=qs(layer,eid[j2]);
        memcpy(G.h_ms+off[j2],                     s->g, G.gu_bytes[layer]);
        memcpy(G.h_ms+off[j2]+G.gu_bytes[layer],   s->u, G.gu_bytes[layer]);
        memcpy(G.h_ms+off[j2]+2*G.gu_bytes[layer], s->d, G.d_bytes[layer]);
    }
}
static void *bounce_worker(void *arg){
    (void)arg;
    cudaSetDevice(G.dev);
    if(getenv("QT_TRACE")) fprintf(stderr,"[qtrace] bounce worker tid %ld\n",(long)syscall(SYS_gettid));
    for(;;){
        pthread_mutex_lock(&G.bmx);
        while(G.b_state!=1 && !G.b_stop) pthread_cond_wait(&G.bcv,&G.bmx);
        if(G.b_stop){ pthread_mutex_unlock(&G.bmx); return NULL; }
        pthread_mutex_unlock(&G.bmx);
        bounce_copy(G.b_layer,G.b_nds,G.b_eid,G.b_off);
        /* the window is free once the previous chain released it (ev_ym) */
        int ok = cudaStreamWaitEvent(G.ms_st,G.ev_ym,0)==cudaSuccess &&
                 cudaMemcpyAsync(G.d_ms,G.h_ms,G.b_used,cudaMemcpyHostToDevice,G.ms_st)==cudaSuccess &&
                 cudaEventRecord(G.ev_ms,G.ms_st)==cudaSuccess;
        pthread_mutex_lock(&G.bmx);
        G.b_ok=ok; G.b_state=2;
        pthread_cond_broadcast(&G.bcv);
        pthread_mutex_unlock(&G.bmx);
    }
}

/* Stable pair grouping. Computation is reordered by expert, but each result
 * keeps its original pair index so take_multi retains token/k accumulation
 * order. Returns the group count and reports the largest group. */
static int group_multi_pairs(int n,int mp,int *n_tile,int n_res,int *ng_res){
    int lookup[QT_MAXE+1], count[QT_MAXE+1], cursor[QT_MAXE+1];
    memset(lookup,0xff,sizeof(lookup));
    int ng=0; *ng_res=0;
    for(int j=0;j<n;j++){
        int key=G.mi_eid[j]+1, g=lookup[key];
        if(g<0){
            g=lookup[key]=ng++;
            if(j<n_res) *ng_res=ng;
            G.h_gptr[g]      =G.h_ptrb[j];
            G.h_gptr[mp+g]   =G.h_ptrb[mp+j];
            G.h_gptr[2*mp+g] =G.h_ptrb[2*mp+j];
            count[g]=0;
        }
        count[g]++;
    }
    G.h_goff[0]=0;
    for(int g=0;g<ng;g++){
        G.h_goff[g+1]=G.h_goff[g]+count[g];
        cursor[g]=G.h_goff[g];
    }
    for(int j=0;j<n;j++)
        G.h_gpair[cursor[lookup[G.mi_eid[j]+1]]++]=j;
    memmove((void*)(G.h_gptr+ng),   (void*)(G.h_gptr+mp),   (size_t)ng*sizeof(void*));
    memmove((void*)(G.h_gptr+2*ng), (void*)(G.h_gptr+2*mp), (size_t)ng*sizeof(void*));
    int nt=0;
    for(int g=0;g<ng;g++)
        for(int p=G.h_goff[g];p<G.h_goff[g+1];p+=QT_GROUP_TILE){
            G.h_gtile[nt]=g; G.h_toff[nt++]=p;
        }
    G.h_toff[nt]=n; *n_tile=nt;
    return ng;
}

/* H5: copy the previous issue's promoted experts out of the staging window
 * into slots (device-to-device on the compute stream, ordered before the
 * next H2D into the window).  The previous chain has completed (take). */
static void promote_pending(void){
    if(!G.npromo) return;
    for(int i=0;i<G.npromo;i++){
        int l=G.promo[i].layer, e=G.promo[i].eid; long vi=G.promo[i].victim;
        QSlot *s=qs(l,e);
        void *vdev=NULL;
        if(vi>=0){ QSlot *v=&G.slot[vi]; vdev=v->dev; v->dev=NULL; }
        if(vdev) slot_free((int)(vi/G.ne), vdev);
        void *dst=slot_alloc(l);
        size_t eb=2*G.gu_bytes[l]+G.d_bytes[l];
        int ok = dst && cudaMemcpyAsync(dst, G.d_ms+G.promo[i].off, eb, cudaMemcpyDeviceToDevice, G.co_st)==cudaSuccess;
        pthread_mutex_lock(&G.mx);
        if(ok){ s->dev=dst; s->resident=1; G.promos++; }
        else { if(dst) slot_free(l,dst); G.used -= G.used>=G.exp_bytes[l]?G.exp_bytes[l]:G.used; G.promo_fail++; }
        s->queued=0;
        if(vi>=0){ G.slot[vi].queued=0; G.used -= G.used>=G.exp_bytes[(int)(vi/G.ne)]?G.exp_bytes[(int)(vi/G.ne)]:G.used; }
        pthread_mutex_unlock(&G.mx);
    }
    G.npromo=0;
}

void q38t_issue_multi(int layer,const int *eids,int K,int B,const float *X,uint32_t *masks){
    G.mi_cnt=0; G.mi_fail=0; G.mi_shared=0; G.mi_layer=layer;
    for(int t=0;t<B;t++) masks[t]=0;
    if(!G.on||G.dead||B>QT_MAXB||K>QT_MAXK) return;
    double t0=G.timers?qms():0;
    int mp=G.maxpairs, n=0;
    if(G.promote_staged) promote_pending();
    pthread_mutex_lock(&G.mx);
    if(layer==0) for(int t=0;t<B;t++) lfru_tick_locked();
    G.issue_open=1;
    for(int t=0;t<B;t++){
        for(int k=0;k<K;k++){
            int e=eids[t*K+k];
            if(e<0) continue;
            QSlot *s=qs(layer,e);
            if(s->resident){
                G.h_ptrb[n]      = s->dev;
                G.h_ptrb[mp+n]   = (const uint8_t*)s->dev + G.gu_bytes[layer];
                G.h_ptrb[2*mp+n] = (const uint8_t*)s->dev + 2*G.gu_bytes[layer];
                G.mi_tok[n]=t; G.mi_k[n]=k; G.mi_eid[n]=e; n++;
                masks[t]|=1u<<k; G.hits++; G.win_hits++;
                if(s->pf){ G.pf_hit++; G.win_pf_hit++; s->pf=0; }
                else if(s->pinned) G.pin_hit++;
            } else { G.miss++; G.win_miss++; }
        }
    }
    pthread_mutex_unlock(&G.mx);
    int n_res=n;                     /* pairs [0,n_res) are VRAM-resident */
    /* stage non-resident experts into the scratch VRAM window (deduped);
     * they join the same chain, so a prefill miss costs PCIe bytes instead
     * of scalar CPU decode.  QT_NO_MS=1 restores the CPU miss path. */
    static int no_ms = -1;
    if(no_ms<0){ const char *e=getenv("QT_NO_MS"); no_ms = e && atoi(e); }
    size_t ms_used=0;
    int *ds_eid=G.ds_eid_g; size_t *ds_off=G.ds_off_g; int nds=0;
    G.nround=1; G.rd_ds[0]=0;
    /* P12 hybrid miss serving: on a decode token (B==1) only the first
     * QT_MS_DECODE_CAP misses are staged over PCIe; the rest fall to the
     * caller's parallel CPU pair loop, which runs in overlap with the chain.
     * 0 (default) stages every miss. */
    static int ms_dec_cap=-1;
    if(ms_dec_cap<0){ const char *e=getenv("QT_MS_DECODE_CAP"); ms_dec_cap = e ? atoi(e) : 0; if(ms_dec_cap<0) ms_dec_cap=0; }
    const int dec_cap = (B==1 && ms_dec_cap>0) ? ms_dec_cap : 0;
    if(!no_ms){
        size_t eb=2*G.gu_bytes[layer]+G.d_bytes[layer];
        for(int t=0;t<B;t++){
            for(int k=0;k<K;k++){
                if((masks[t]>>k)&1u) continue;
                int e=eids[t*K+k];
                if(e<0) continue;
                QSlot *s=qs(layer,e);
                if(!s->g) continue;
                size_t off; int f=-1;
                for(int j2=0;j2<nds;j2++) if(ds_eid[j2]==e){ f=j2; break; }
                if(f>=0) off=ds_off[f];
                else {
                    if(nds>=1024) continue;                           /* overflow -> CPU */
                    if(dec_cap && nds>=dec_cap) continue;             /* hybrid: the rest on the CPU */
                    if(ms_used+eb>G.ms_cap){                          /* S2: next round through the window */
                        if(G.nround>=QT_MS_ROUNDS || eb>G.ms_cap) continue;   /* -> CPU */
                        G.rd_used[G.nround-1]=ms_used; G.rd_ds[G.nround]=nds; G.nround++;
                        ms_used=0; G.ms_rounds_extra++;
                    }
                    off=ms_used;
                    ds_eid[nds]=e; ds_off[nds]=off; nds++; ms_used+=eb;
                    G.ms_pairs++; G.ms_bytes+=eb;
                }
                G.h_ptrb[n]      = G.d_ms + off;
                G.h_ptrb[mp+n]   = G.d_ms + off + G.gu_bytes[layer];
                G.h_ptrb[2*mp+n] = G.d_ms + off + 2*G.gu_bytes[layer];
                G.mi_tok[n]=t; G.mi_k[n]=k; G.mi_eid[n]=e; n++;
                masks[t]|=1u<<k;
            }
        }
    }
    /* H5: pick staged experts to promote after this chain: each must beat a
     * same-class victim from the tick's pool under the hysteresis (recent
     * window counts when QT_PROMOTE_RECENT=1).  Both sides are reserved now
     * (queued), the copy happens at the next issue. */
    if(G.promote_staged && nds){
        pthread_mutex_lock(&G.mx);
        /* only the LAST round's entries are still in the window at the next issue */
        for(int j=G.rd_ds[G.nround-1];j<nds && G.npromo<64;j++){
            QSlot *s=qs(layer,ds_eid[j]);
            if(s->resident||s->queued) continue;
            const uint32_t hh = G.promote_recent ? s->recent : s->heat;
            long vi=-1;
            size_t eb=G.exp_bytes[layer];
            if(G.used+eb>G.budget || !slot_can_alloc(layer)){
                int vk=-1;
                for(int q=0;q<G.nvict;q++){
                    long c=G.vict[q]; QSlot *v=&G.slot[c]; int vl=(int)(c/G.ne);
                    if(!v->resident||v->queued||v->pinned||G.cls[vl]!=G.cls[layer]) continue;
                    const uint32_t ch = G.promote_recent ? v->recent : v->heat;
                    if(!tier_should_promote(hh,ch)) continue;
                    vi=c; vk=q; break;
                }
                if(vi<0) continue;
                G.slot[vi].resident=0; G.slot[vi].queued=1;       /* CPU fallback from now on */
                G.vict[vk]=G.vict[--G.nvict];
            }
            s->queued=1; G.used+=eb;
            G.promo[G.npromo].layer=layer; G.promo[G.npromo].eid=ds_eid[j];
            G.promo[G.npromo].off=ds_off[j]; G.promo[G.npromo].victim=vi; G.npromo++;
        }
        pthread_mutex_unlock(&G.mx);
    }
    /* the mmap->pinned copy is the staging wall on a miss-heavy chunk
     * (hundreds of MB).  On the grouped path it runs on the bounce worker
     * while the caller computes the shared expert; q38t_issue_multi_finish
     * joins it and launches the staged tiles. */
    if(G.sh[layer].on && G.sh[layer].gu_type==G.gu_type[layer] &&
       G.sh[layer].d_type==G.d_type[layer]){
        G.mi_shared=1;
        const uint8_t *sd=(const uint8_t*)G.sh[layer].dev;
        for(int t=0;t<B;t++){
            G.h_ptrb[n]      = sd;
            G.h_ptrb[mp+n]   = sd + G.sh[layer].gu_bytes;
            G.h_ptrb[2*mp+n] = sd + 2*G.sh[layer].gu_bytes;
            G.mi_tok[n]=t; G.mi_k[n]=-1; G.mi_eid[n]=-1; n++;
        }
    }
    G.mi_cnt=n;
    if(!n){ if(G.timers) G.t_issue+=qms()-t0; return; }
    static int grouped=-1;
    if(grouped<0){ const char *e=getenv("QT_GROUPED_MOE"); grouped=!e||atoi(e)!=0; }
    G.mi_scatter=grouped;
    int nt=0, ng=0, ng_res=0, nt_res=0, ok=1;
    if(grouped){
        /* one activation per token; pairs map to it through mi_tok */
        G.mi_x=X;
        G.mi_dx = (G.x_dev_on && G.x_dev) ? G.x_dev : G.d_xt;
        QT_TR("issue L%d B=%d n=%d dx=%s",layer,B,n,G.mi_dx==G.d_xt?"upload":"device");
        ng=group_multi_pairs(n,mp,&nt,n_res,&ng_res);
        QT_TR("issue L%d grouped ng=%d nt=%d",layer,ng,nt);
        /* tiles come out in group order and resident groups were created
         * first, so the resident tiles are a prefix */
        while(nt_res<nt && G.h_gtile[nt_res]<ng_res) nt_res++;
        /* phase A: one packed table upload + resident tiles' gate|up */
        if(G.kprof) cudaEventRecord(G.kp[0],G.co_st);
        {
            size_t o=0;
            #define QT_PACK(dst,src,bytes) do{ memcpy(G.h_meta+o,(src),(bytes)); dst=(void*)(G.d_meta+o); o=(o+(bytes)+15)&~(size_t)15; }while(0)
            void *p; 
            QT_PACK(p,G.h_gptr,3*(size_t)ng*sizeof(void*)); G.mi_dptr=(const void**)p;
            QT_PACK(p,G.mi_tok,(size_t)n*sizeof(int));     G.mi_dptok=(int*)p;
            QT_PACK(p,G.h_gpair,(size_t)n*sizeof(int));    G.mi_dgpair=(int*)p;
            QT_PACK(p,G.h_gtile,(size_t)nt*sizeof(int));   G.mi_dgtile=(int*)p;
            QT_PACK(p,G.h_toff,((size_t)nt+1)*sizeof(int)); G.mi_dtoff=(int*)p;
            #undef QT_PACK
            ok = 1;
            if(G.mi_dx==G.d_xt){
                memcpy(G.h_xt, X, (size_t)B*G.D*sizeof(float));
                ok = cudaMemcpyAsync(G.d_xt,G.h_xt,(size_t)B*G.D*sizeof(float),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess;
            }
            ok = ok && cudaMemcpyAsync(G.d_meta,G.h_meta,o,cudaMemcpyHostToDevice,G.co_st)==cudaSuccess;
        }
        QT_TR("issue L%d meta+x issued ok=%d",layer,ok);
        if(ok && nt_res>0)
            ok = ggml_blocks_gemv_grouped_x(G.gu_type[layer],G.I,G.D,nt_res,G.mi_dptr,    G.mi_dgtile,G.mi_dtoff,G.mi_dgpair,G.mi_dptok,G.mi_dx,(size_t)G.D,G.d_gub,G.co_st)==0 &&
                 ggml_blocks_gemv_grouped_x(G.gu_type[layer],G.I,G.D,nt_res,G.mi_dptr+ng, G.mi_dgtile,G.mi_dtoff,G.mi_dgpair,G.mi_dptok,G.mi_dx,(size_t)G.D,G.d_gub+(size_t)n*G.I,G.co_st)==0;
        /* the bounce copy overlaps those kernels AND the caller's shared
         * expert: post it to the worker; finish() joins and launches the
         * staged tiles, silu and down */
        if(G.kprof) cudaEventRecord(G.kp[1],G.co_st);
        G.mi_job=0;
        /* QT_BOUNCE_ASYNC=1: bounce on the worker while the caller computes
         * the shared expert.  Default off: measured neutral-to-negative at
         * prefill-2K (88.9 vs 90.5 tok/s) -- the copy and the int8 shared
         * expert are both DRAM-bound and the staged tiles start later. */
        static int async_b=-1;
        if(async_b<0){ const char *e=getenv("QT_BOUNCE_ASYNC"); async_b = e && atoi(e); }
        G.rd_used[G.nround-1]=ms_used; G.rd_ds[G.nround]=nds; G.nds_g=nds;
        if(ok && nds){
            /* round 0 only; finish() runs the later rounds after its tiles */
            const int n0=G.rd_ds[1]; const size_t u0=G.rd_used[0];
            if(async_b){
                pthread_mutex_lock(&G.bmx);
                G.b_layer=layer; G.b_nds=n0; G.b_used=u0;
                memcpy(G.b_eid,ds_eid,(size_t)n0*sizeof(int));
                memcpy(G.b_off,ds_off,(size_t)n0*sizeof(size_t));
                G.b_state=1; G.mi_job=1;
                pthread_cond_broadcast(&G.bcv);
                pthread_mutex_unlock(&G.bmx);
            } else {
                QT_TR("issue L%d bounce start n0=%d u0=%zu",layer,n0,(size_t)u0);
                bounce_copy(layer,n0,ds_eid,ds_off);
                QT_TR("issue L%d bounce done",layer);
                ok = cudaStreamWaitEvent(G.ms_st,G.ev_ym,0)==cudaSuccess &&
                     cudaMemcpyAsync(G.d_ms,G.h_ms,u0,cudaMemcpyHostToDevice,G.ms_st)==cudaSuccess &&
                     cudaEventRecord(G.ev_ms,G.ms_st)==cudaSuccess &&
                     cudaStreamWaitEvent(G.co_st,G.ev_ms,0)==cudaSuccess;
            }
        }
        G.mi_deferred=1; G.mi_nt=nt; G.mi_nt_res=nt_res; G.mi_ng=ng; G.mi_ng_res=ng_res; G.mi_ok=ok;
        if(G.timers) G.t_issue+=qms()-t0;
        QT_TR("issue L%d phase A done ok=%d nt=%d nt_res=%d ng=%d ng_res=%d",layer,ok,nt,nt_res,ng,ng_res);
        if(!async_b) q38t_issue_multi_finish();     /* inline: launch everything now */
        QT_TR("issue L%d finish done",layer);
        return;
    } else {
        /* per-pair activation replicated into the pinned staging block */
        if(ms_used) bounce_copy(layer,nds,ds_eid,ds_off);
        for(int j=0;j<n;j++)
            memcpy(G.h_xb+(size_t)j*G.D, X+(size_t)G.mi_tok[j]*G.D, (size_t)G.D*sizeof(float));
        memmove((void*)(G.h_ptrb+n),   (void*)(G.h_ptrb+mp),   (size_t)n*sizeof(void*));
        memmove((void*)(G.h_ptrb+2*n), (void*)(G.h_ptrb+2*mp), (size_t)n*sizeof(void*));
        ok = (ms_used==0 ||
              cudaMemcpyAsync(G.d_ms,G.h_ms,ms_used,cudaMemcpyHostToDevice,G.co_st)==cudaSuccess) &&
             cudaMemcpyAsync(G.d_xb,G.h_xb,(size_t)n*G.D*sizeof(float),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess;
    }
    if(ok && !grouped)
        ok = cudaMemcpyAsync((void*)G.d_ptrb,(const void*)G.h_ptrb,3*(size_t)n*sizeof(void*),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess &&
             ggml_blocks_gemv_batch_sx(G.gu_type[layer],G.I,G.D,n,G.d_ptrb,   G.d_xb,(size_t)G.D,G.d_gub,G.co_st)==0 &&
             ggml_blocks_gemv_batch_sx(G.gu_type[layer],G.I,G.D,n,G.d_ptrb+n, G.d_xb,(size_t)G.D,G.d_gub+(size_t)n*G.I,G.co_st)==0;
    if(ok) ok=ggml_blocks_silu_mul_cuda(G.d_gub,G.d_gub+(size_t)n*G.I,G.d_hb,(size_t)n*G.I,G.co_st)==0;
    if(ok && grouped)
        ok=ggml_blocks_gemv_grouped_sx(G.d_type[layer],G.D,G.I,nt,G.d_ptrb+2*ng,G.d_gtile,G.d_toff,G.d_gpair,G.d_hb,(size_t)G.I,G.d_yb,G.co_st)==0;
    else if(ok)
        ok=ggml_blocks_gemv_batch_sx(G.d_type[layer],G.D,G.I,n,G.d_ptrb+2*n,G.d_hb,(size_t)G.I,G.d_yb,G.co_st)==0;
    if(ok && !grouped) ok=cudaMemcpyAsync(G.h_yb,G.d_yb,(size_t)n*G.D*sizeof(float),cudaMemcpyDeviceToHost,G.co_st)==cudaSuccess;
    if(G.kprof && grouped){ cudaEventRecord(G.kp[3],G.co_st); G.kp_pending=1; }
    if(ok) ok=cudaEventRecord(G.ev_ym,G.co_st)==cudaSuccess;
    if(!ok){ qck(cudaGetLastError(),"issue_multi"); G.mi_fail=1; }
    if(G.timers) G.t_issue+=qms()-t0;
}

/* P10 D2b: the shared expert of a ONE-token multi issue (decode with
 * Q38_MS_DECODE=1) on the tier's resident raw-block copy, on sh_st so it
 * overlaps the routed chain; the multi path otherwise leaves it to the host
 * when its quant types differ from the routed experts (235 MB of int8 per
 * token from DRAM).  Same chain as q38t_issue's sh_issued==1 branch. */
int q38t_shared_issue1(int layer,const float *x){
    if(!G.on||G.dead||!G.sh[layer].on) return 0;
    memcpy(G.h_x,x,(size_t)G.D*sizeof(float));
    const uint8_t *sd=(const uint8_t*)G.sh[layer].dev;
    int ok =
        cudaMemcpyAsync(G.d_x,G.h_x,(size_t)G.D*sizeof(float),cudaMemcpyHostToDevice,G.sh_st)==cudaSuccess &&
        ggml_blocks_gemv_cuda(G.sh[layer].gu_type,G.I,G.D,sd,                       G.d_x,G.d_sg,G.sh_st)==0 &&
        ggml_blocks_gemv_cuda(G.sh[layer].gu_type,G.I,G.D,sd+G.sh[layer].gu_bytes,  G.d_x,G.d_su,G.sh_st)==0 &&
        ggml_blocks_silu_mul_cuda(G.d_sg,G.d_su,G.d_sh,(size_t)G.I,G.sh_st)==0 &&
        ggml_blocks_gemv_cuda(G.sh[layer].d_type,G.D,G.I,sd+2*G.sh[layer].gu_bytes, G.d_sh,G.d_y+(size_t)(QT_MAXK-1)*G.D,G.sh_st)==0 &&
        cudaMemcpyAsync(G.h_y+(size_t)(QT_MAXK-1)*G.D,G.d_y+(size_t)(QT_MAXK-1)*G.D,(size_t)G.D*sizeof(float),cudaMemcpyDeviceToHost,G.sh_st)==cudaSuccess &&
        cudaEventRecord(G.ev_sh,G.sh_st)==cudaSuccess;
    if(!ok) qck(cudaGetLastError(),"shared_issue1");
    return ok;
}
int q38t_shared_take1(float *out,float shared_s){
    if(!G.on) return 0;
    if(!qck(cudaEventSynchronize(G.ev_sh),"ev_sh1")) return 0;
    const float *row=G.h_y+(size_t)(QT_MAXK-1)*G.D;
    for(int d=0;d<G.D;d++) out[d]+=shared_s*row[d];
    return 1;
}

/* P10 P5: second half of a grouped issue -- join the bounce worker, then the
 * staged tiles' gate|up, silu and down for every pair.  Idempotent; called
 * by the engine after the CPU shared expert and again by take_multi. */
void q38t_issue_multi_finish(void){
    if(!G.on||!G.mi_deferred) return;
    G.mi_deferred=0;
    double t0=G.timers?qms():0;
    int layer=G.mi_layer, n=G.mi_cnt, nt=G.mi_nt, nt_res=G.mi_nt_res, ng=G.mi_ng, ok=G.mi_ok;
    if(G.mi_job){
        pthread_mutex_lock(&G.bmx);
        while(G.b_state!=2) pthread_cond_wait(&G.bcv,&G.bmx);
        ok = ok && G.b_ok; G.b_state=0;
        pthread_mutex_unlock(&G.bmx);
        if(ok) ok = cudaStreamWaitEvent(G.co_st,G.ev_ms,0)==cudaSuccess;
    }
    /* S2: the whole chain per round.  The staged groups sit in ds order
     * right after the resident groups, so round r owns the tiles whose group
     * index lies in [ng_res+rd_ds[r], ng_res+rd_ds[r+1]); its gate|up, the
     * silu (all pairs, elementwise, cheap) and its DOWN tiles run before the
     * window is refilled -- the down weights live in the window too.  Round
     * 0's down launch also covers the resident tiles [0, nt_res). */
    {
        const int ng_res = G.mi_ng_res;
        const int nr = nt>nt_res ? G.nround : 1;
        int t0r=nt_res;
        for(int r=0;r<nr && ok;r++){
            const int g_end = ng_res + G.rd_ds[r+1];
            int t1r=t0r; while(t1r<nt && G.h_gtile[t1r]<g_end) t1r++;
            if(r>0){
                /* the window is read by round r-1's tiles: wait for them,
                 * refill it (h_ms is free once the previous H2D landed) */
                const int j0=G.rd_ds[r], j1=G.rd_ds[r+1];
                ok = cudaEventRecord(G.ev_rd,G.co_st)==cudaSuccess &&
                     cudaStreamSynchronize(G.ms_st)==cudaSuccess;
                if(ok){ bounce_copy(layer,j1-j0,G.ds_eid_g+j0,G.ds_off_g+j0); }
                ok = ok &&
                     cudaStreamWaitEvent(G.ms_st,G.ev_rd,0)==cudaSuccess &&
                     cudaMemcpyAsync(G.d_ms,G.h_ms,G.rd_used[r],cudaMemcpyHostToDevice,G.ms_st)==cudaSuccess &&
                     cudaEventRecord(G.ev_ms,G.ms_st)==cudaSuccess &&
                     cudaStreamWaitEvent(G.co_st,G.ev_ms,0)==cudaSuccess;
            }
            if(ok && t1r>t0r)
                ok = ggml_blocks_gemv_grouped_x(G.gu_type[layer],G.I,G.D,t1r-t0r,G.mi_dptr,    G.mi_dgtile+t0r,G.mi_dtoff+t0r,G.mi_dgpair,G.mi_dptok,G.mi_dx,(size_t)G.D,G.d_gub,G.co_st)==0 &&
                     ggml_blocks_gemv_grouped_x(G.gu_type[layer],G.I,G.D,t1r-t0r,G.mi_dptr+ng, G.mi_dgtile+t0r,G.mi_dtoff+t0r,G.mi_dgpair,G.mi_dptok,G.mi_dx,(size_t)G.D,G.d_gub+(size_t)n*G.I,G.co_st)==0;
            if(r==0 && G.kprof) cudaEventRecord(G.kp[2],G.co_st);
            if(ok) ok=ggml_blocks_silu_mul_cuda(G.d_gub,G.d_gub+(size_t)n*G.I,G.d_hb,(size_t)n*G.I,G.co_st)==0;
            { const int d0 = r==0 ? 0 : t0r;
              if(ok && t1r>d0)
                  ok=ggml_blocks_gemv_grouped_sx(G.d_type[layer],G.D,G.I,t1r-d0,G.mi_dptr+2*ng,G.mi_dgtile+d0,G.mi_dtoff+d0,G.mi_dgpair,G.d_hb,(size_t)G.I,G.d_yb,G.co_st)==0; }
            t0r=t1r;
        }
    }
    if(G.kprof){ cudaEventRecord(G.kp[3],G.co_st); G.kp_pending=1; }
    if(ok) ok=cudaEventRecord(G.ev_ym,G.co_st)==cudaSuccess;
    if(!ok){ qck(cudaGetLastError(),"issue_multi_finish"); G.mi_fail=1; }
    if(G.timers) G.t_issue+=qms()-t0;
}

void q38t_take_multi(const float *val,int K,int B,float *OUT,const float *shared_s){
    QT_TR("take B=%d enter",B);
    if(!G.on) return;
    q38t_issue_multi_finish();
    int n=G.mi_cnt, l=G.mi_layer;
    if(n){
        double t0=G.timers?qms():0;
        int done=0;
        if(!G.mi_fail && G.mi_scatter){
            /* per-token pair lists in issue order (== the host loop's visit
             * order for that token) + weights; OUT rides along so the device
             * continues the host's accumulation exactly */
            memset(G.h_tokoff,0,(size_t)(B+1)*sizeof(int));
            for(int j=0;j<n;j++) G.h_tokoff[G.mi_tok[j]+1]++;
            for(int t=0;t<B;t++) G.h_tokoff[t+1]+=G.h_tokoff[t];
            { int cur[QT_MAXB]; memcpy(cur,G.h_tokoff,(size_t)B*sizeof(int));
              for(int j=0;j<n;j++) G.h_toklist[cur[G.mi_tok[j]]++]=j; }
            for(int j=0;j<n;j++){
                int t=G.mi_tok[j];
                G.h_pw[j] = G.mi_k[j]<0 ? shared_s[t] : val[t*K+G.mi_k[j]];
            }
            memcpy(G.h_out,OUT,(size_t)B*G.D*sizeof(float));
            size_t o=0; int *dtokoff,*dtoklist; float *dpw;
            #define QT_PACK2(dst,type,src,bytes) do{ memcpy(G.h_meta2+o,(src),(bytes)); dst=(type)(G.d_meta2+o); o=(o+(bytes)+15)&~(size_t)15; }while(0)
            QT_PACK2(dtokoff,int*,G.h_tokoff,((size_t)B+1)*sizeof(int));
            QT_PACK2(dtoklist,int*,G.h_toklist,(size_t)n*sizeof(int));
            QT_PACK2(dpw,float*,G.h_pw,(size_t)n*sizeof(float));
            #undef QT_PACK2
            int ok =
                cudaMemcpyAsync(G.d_out,G.h_out,(size_t)B*G.D*sizeof(float),cudaMemcpyHostToDevice,G.co_st)==cudaSuccess &&
                cudaMemcpyAsync(G.d_meta2,G.h_meta2,o,cudaMemcpyHostToDevice,G.co_st)==cudaSuccess &&
                ggml_blocks_scatter_add_cuda(G.d_yb,G.D,B,dtokoff,dtoklist,dpw,G.d_out,G.co_st)==0 &&
                cudaMemcpyAsync(G.h_out,G.d_out,(size_t)B*G.D*sizeof(float),cudaMemcpyDeviceToHost,G.co_st)==cudaSuccess &&
                qck(cudaStreamSynchronize(G.co_st),"take_scatter");
            if(ok){ memcpy(OUT,G.h_out,(size_t)B*G.D*sizeof(float)); done=1; }
            else qck(cudaGetLastError(),"take_scatter");
            if(G.kprof && G.kp_pending){
                float a=0,b=0,c=0;
                if(cudaEventElapsedTime(&a,G.kp[0],G.kp[1])==cudaSuccess &&
                   cudaEventElapsedTime(&b,G.kp[1],G.kp[2])==cudaSuccess &&
                   cudaEventElapsedTime(&c,G.kp[2],G.kp[3])==cudaSuccess){
                    G.kp_ms[0]+=a; G.kp_ms[1]+=b; G.kp_ms[2]+=c;
                }
                G.kp_pending=0;
            }
        } else if(!G.mi_fail && qck(cudaEventSynchronize(G.ev_ym),"ev_ym")){
            for(int j=0;j<n;j++){
                int t=G.mi_tok[j];
                float w = G.mi_k[j]<0 ? shared_s[t] : val[t*K+G.mi_k[j]];
                const float *row=G.h_yb+(size_t)j*G.D;
                float *out=OUT+(size_t)t*G.D;
                for(int d=0;d<G.D;d++) out[d]+=w*row[d];
            }
            done=1;
        }
        if(!done){
            /* CUDA error mid-chunk: recompute every issued pair on the CPU
             * from the mmap'd raw blocks (tier is dead from here on). */
            float *g=malloc((size_t)G.I*sizeof(float)), *u=malloc((size_t)G.I*sizeof(float));
            float *y=malloc((size_t)G.D*sizeof(float));
            if(g&&u&&y) for(int j=0;j<n;j++){
                int t=G.mi_tok[j];
                const uint8_t *pg,*pu,*pd; int gut,dt;
                if(G.mi_k[j]<0){ pg=G.sh[l].g; pu=G.sh[l].u; pd=G.sh[l].d;
                                 gut=G.sh[l].gu_type; dt=G.sh[l].d_type; }
                else { QSlot *s=qs(l,G.mi_eid[j]); pg=s->g; pu=s->u; pd=s->d;
                       gut=G.gu_type[l]; dt=G.d_type[l]; }
                const float *x = G.mi_scatter ? G.mi_x+(size_t)t*G.D : G.h_xb+(size_t)j*G.D;
                ggml_bk_gemv(gut,G.I,G.D,pg,x,g);
                ggml_bk_gemv(gut,G.I,G.D,pu,x,u);
                for(int i=0;i<G.I;i++){
                    float gv=g[i];
                    float sg = gv>=0.f ? 1.f/(1.f+expf(-gv)) : expf(gv)/(1.f+expf(gv));
                    g[i]=gv*sg*u[i];
                }
                ggml_bk_gemv(dt,G.D,G.I,pd,g,y);
                float w = G.mi_k[j]<0 ? shared_s[t] : val[t*K+G.mi_k[j]];
                float *out=OUT+(size_t)t*G.D;
                for(int d=0;d<G.D;d++) out[d]+=w*y[d];
            }
            free(g); free(u); free(y);
        }
        if(G.timers) G.t_y_wait+=qms()-t0;
    }
    G.mi_cnt=0; G.mi_fail=0; G.mi_shared=0;
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

/* ---- P4 async prefetcher: router_v1 artifact + policy entry points ---- */

/* Load the router_v1 artifact (tools/router_v1_format.md) into one blob and
 * arm the prefetcher.  Absent/invalid artifact -> prev-token union only. */
void q38t_prefetch_setup(const char *path){
    if(!G.on||G.dead) return;
    const char *e=getenv("Q38_PREFETCH");
    G.pf_on = e ? atoi(e)!=0 : 1;                /* default on with the tier */
    e=getenv("QT_RANK_TABLE"); G.rank_w = e ? atoi(e) : 0; if(G.rank_w<0) G.rank_w=0;
    e=getenv("QT_RANK_TOPK"); G.rank_topk = e && atoi(e)>0 ? atoi(e) : 10;
    e=getenv("QT_RANK_RESIDENT"); G.rank_resident = e ? atoi(e)!=0 : 0;
    if(!G.pf_on){
        fprintf(stderr,"[qtier] prefetch disabled (Q38_PREFETCH=0)%s\n",
                G.rank_w ? "; the router table ranks residency (QT_RANK_TABLE)" : "");
        if(!G.rank_w) return;
    }
    /* per-phase defaults follow the gate measurements (p4-prefetch-status.md):
     * prefill lookahead pays (+6% on domain shift), warm-decode speculation
     * costs 2-4.5%.  Hot-pin warmstart follows the master switch only. */
    e=getenv("Q38_PREFETCH_PREFILL"); G.pf_prefill = e ? atoi(e)!=0 : 1;
    e=getenv("Q38_PREFETCH_DECODE");  G.pf_decode  = e ? atoi(e)!=0 : 0;
    G.pf_phase=0;
    e=getenv("Q38_PREFETCH_MB");
    double mb = (e && atof(e)>0) ? atof(e) : 64.0;
    G.pf_cap=(size_t)(mb*1048576.0);
    e=getenv("Q38_PF_SLACK");
    G.pf_slack = e ? (uint32_t)atoi(e) : 0;
    FILE *f = path ? fopen(path,"rb") : NULL;
    if(!f){
        fprintf(stderr,"[qtier] prefetch armed (prev-token union only): no router artifact at %s\n",
                path?path:"(null)");
        return;
    }
    fseek(f,0,SEEK_END); long fs=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b = fs>=64 ? malloc((size_t)fs) : NULL;
    if(!b || fread(b,1,(size_t)fs,f)!=(size_t)fs){ free(b); fclose(f);
        fprintf(stderr,"[qtier] prefetch: cannot read %s -> prev-token union only\n",path); return; }
    fclose(f);
    uint32_t v[10]; memcpy(v,b+8,40);
    uint64_t corpus; memcpy(&corpus,b+48,8);
    uint32_t tl; memcpy(&tl,b+56,4);
    /* v: version nl ne topk bits predk hotm nctx order flags */
    if(memcmp(b,"Q38RV1\0\0",8) || v[0]!=1 ||
       !((int)v[1]==G.nl || (int)v[1]+1==G.nl) || (int)v[2]!=G.ne ||
       v[4]>32 || v[5]>64 || tl>v[1]){
        fprintf(stderr,"[qtier] prefetch: %s is not a valid router_v1 for this model -> prev-token union only\n",path);
        free(b); return;
    }
    G.rt_bits=v[4]; G.rt_predk=v[5]; G.rt_hotm=v[6]; G.rt_nctx=v[7];
    G.rt_tlayers=tl; G.rt_nlayers=v[1];
    size_t A, off=64;
    #define ALG(x) (((x)+63)&~(size_t)63)
    A=ALG(off+(size_t)G.rt_nctx*4);
    G.rt_bucket=(const uint32_t*)(b+off);
    G.rt_table =(const uint16_t*)(b+A);
    off=ALG(A+(size_t)G.rt_nctx*G.rt_tlayers*G.rt_predk*2);
    G.rt_hot   =(const uint16_t*)(b+off);
    #undef ALG
    if(off+(size_t)G.rt_nlayers*G.rt_hotm*2 > (size_t)fs){
        fprintf(stderr,"[qtier] prefetch: %s truncated -> prev-token union only\n",path);
        G.rt_nctx=0; G.rt_hot=NULL; free(b); G.rt_bucket=NULL; G.rt_table=NULL; return;
    }
    G.rt_blob=b;
    fprintf(stderr,"[qtier] %s: %s (%u buckets, table layers 0-%u, k=%u, hot-%u"
            ", trained on %llu tokens) | prefill %s, decode %s | pending cap %.0f MB, heat slack %u | rank weight %d\n",
            G.pf_on ? "prefetch armed" : "router table loaded for ranking", path, G.rt_nctx, G.rt_tlayers?G.rt_tlayers-1:0, G.rt_predk,
            G.rt_hotm, (unsigned long long)corpus,
            G.pf_prefill?"on":"off", G.pf_decode?"on":"off",
            G.pf_cap/1048576.0, G.pf_slack, G.rank_w);
}

/* Bigram lookahead: called by the engine the moment token t+1's identity is
 * known (decode: right after sampling; prefill: once per chunk token, before
 * layer 0 runs).  One binary search covers all table layers; the predicted
 * top-k experts of layers [0, rt_tlayers) join the low-priority queue. */
void q38t_prefetch_phase(int is_prefill){ G.pf_phase = is_prefill; }

void q38t_prefetch_next(uint64_t bigram_mixed){
    if(!G.on||G.dead||!G.rt_nctx) return;
    const int pf = G.pf_on && (G.pf_phase ? G.pf_prefill : G.pf_decode);
    if(!pf && !G.rank_w) return;
    uint32_t bucket=(uint32_t)(bigram_mixed & ((1ull<<G.rt_bits)-1ull));
    uint32_t lo=0, hi=G.rt_nctx;
    while(lo<hi){ uint32_t mid=(lo+hi)/2;
        if(G.rt_bucket[mid]<bucket) lo=mid+1; else hi=mid; }
    if(G.rank_w) G.rank_lookups++;
    if(lo>=G.rt_nctx || G.rt_bucket[lo]!=bucket) return;   /* table miss */
    if(G.rank_w) G.rank_hits++;
    const uint16_t *row=G.rt_table+(size_t)lo*G.rt_tlayers*G.rt_predk;
    int tl=(int)G.rt_tlayers<G.nl?(int)G.rt_tlayers:G.nl;
    pthread_mutex_lock(&G.mx);
    for(int l=0;l<tl;l++)
        for(uint32_t k=0;k<G.rt_predk;k++){
            uint16_t ex=row[(size_t)l*G.rt_predk+k];
            if(ex==0xFFFFu||(int)ex>=G.ne) continue;
            if(G.rank_w && (int)k<G.rank_topk){       /* step 3: rank, do not upload */
                QSlot *s=qs(l,ex);
                if(!G.rank_resident || s->resident){
                    uint32_t nr=s->recent+(uint32_t)G.rank_w;
                    s->recent = nr<s->recent ? 0xFFFFFFFFu : nr;
                    G.rank_bumps++;
                }
            }
            if(pf && enqueue_pf_locked(l,ex)) G.pf_iss_ng++;
        }
    pthread_mutex_unlock(&G.mx);
}

/* Pin + upload the per-layer hot-16 (never evicted; docs/p4-analysis.md gives
 * them 0.163 standalone coverage and they anchor the ensemble).  Runs inside
 * warmstart, before the heat fill, and drains so pins land first. */
static void pf_pin_hot(void){
    if(!G.pf_on||!G.rt_hot) return;
    /* Default OFF: pins tax warm steady-state decode ~2.2% at auto budget
     * for a benefit that only shows on cold start / domain shift
     * (docs/p4-prefetch-status.md).  Multi-session serving is exactly that
     * shift-heavy regime, so the P7 server enables Q38_PIN_HOT explicitly. */
    const char *e=getenv("Q38_PIN_HOT");         /* pins per layer, default 0 */
    int want = e ? atoi(e) : 0;
    if(want<=0){ fprintf(stderr,"[qtier] prefetch: hot pinning off (Q38_PIN_HOT unset/0)\n"); return; }
    int m16=(int)G.rt_hotm<want?(int)G.rt_hotm:want;
    /* pinned bytes are capped at budget/8 unless Q38_PIN_HOT was set
     * explicitly: at tight budgets the full hot-16 (1.73 GB) displaces ~29%
     * of a 6 GB budget's workload-hot residents, which measures as a hit-rate
     * LOSS (docs/p4-prefetch-status.md).  Rank-major fill: hot-0 of every
     * layer first, then hot-1, ... so the cap trims the weakest pins. */
    size_t pin_cap = e ? G.budget : G.budget/8;
    size_t bytes=0; int n=0;
    int hot_layers=(int)G.rt_nlayers<G.nl?(int)G.rt_nlayers:G.nl;
    for(int j=0;j<m16;j++)
        for(int l=0;l<hot_layers;l++){
            uint16_t ex=G.rt_hot[(size_t)l*G.rt_hotm+j];
            if(ex==0xFFFFu||(int)ex>=G.ne) continue;
            if(bytes+G.exp_bytes[l]>pin_cap) goto done;
            pthread_mutex_lock(&G.mx);
            QSlot *s=qs(l,ex);
            if(s->resident){ if(!s->pinned){ s->pinned=1; n++; bytes+=G.exp_bytes[l]; } }
            else if(s->g){
                while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
                if(enqueue_locked(l,ex,-1,-1)){ s->pinned=1; n++; bytes+=G.exp_bytes[l]; }
            }
            pthread_mutex_unlock(&G.mx);
        }
done:;
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
    G.pinned_cnt=n;
    fprintf(stderr,"[qtier] prefetch: pinned %d hot experts (%.2f GB)\n",
            n, bytes/1073741824.0);
}

/* warmstart: enqueue in heat0-descending order until the budget is reserved,
 * then drain.  Without a loadable HEAT_FILE this is a no-op (demand fill). */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
void q38t_warmstart(void){
    if(!G.on||G.dead) return;
    if(getenv("QT_NO_WARMSTART")) return;
    pf_pin_hot();                                /* P4: hot-16 pins land first */
    if(!G.heat0) return;
    size_t n=(size_t)G.nl*G.ne;
    int *ord=malloc(n*sizeof(int));
    if(!ord) return;
    for(size_t i=0;i<n;i++) ord[i]=(int)i;
    g_sort_heat=G.heat0;
    qsort(ord,n,sizeof(int),cmp_heat_desc);
    double t0=qms();
    size_t planned=0;
    for(size_t i=0;i<n;i++){
        int gi=ord[i], l=gi/G.ne, e=gi%G.ne;
        if(G.heat0[gi]==0) break;                      /* never-routed tail */
        pthread_mutex_lock(&G.mx);
        if(G.used+G.exp_bytes[l]>G.budget){ pthread_mutex_unlock(&G.mx); break; }
        while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
        if(enqueue_locked(l,e,-1,-1)) planned++;
        pthread_mutex_unlock(&G.mx);
    }
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
    fprintf(stderr,"[qtier] warmstart: %zu experts uploaded in %.1fs (%.2f GB)\n",
            planned,(qms()-t0)/1e3,G.up_bytes/1073741824.0);
    free(ord);
}

void q38t_stats(void){
    if(!G.on) return;
    size_t res=0,resb=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++)
        if(G.slot[i].resident){ res++; resb+=G.exp_bytes[i/G.ne]; }
    double tot=(double)(G.hits+G.miss);
    fprintf(stderr,"[qtier] resident %zu/%d experts (%.2f GB / budget %.2f GB) | "
            "uploads %llu (%.2f GB, %.2f GB/s busy%s) | swaps %llu | promoted %llu (%llu failed) | q_skips %llu | staging rounds+%llu\n",
            res, G.nl*G.ne, resb/1073741824.0, G.budget/1073741824.0,
            (unsigned long long)G.uploads, G.up_bytes/1073741824.0,
            G.up_ms>0 ? (G.up_bytes/1073741824.0)/(G.up_ms/1e3) : 0.0, G.up_pipe?", pipelined":"",
            (unsigned long long)G.swaps, (unsigned long long)G.promos, (unsigned long long)G.promo_fail, (unsigned long long)G.q_full_skips, (unsigned long long)G.ms_rounds_extra);
    fprintf(stderr,"[qtier] VRAM hit rate %.1f%% (hits %llu, miss %llu of which %llu staged->GPU, %.2f GB staged)\n",
            tot>0?100.0*G.hits/tot:0.0,
            (unsigned long long)G.hits,(unsigned long long)G.miss,
            (unsigned long long)G.ms_pairs, G.ms_bytes/1073741824.0);
    if(G.rank_w)
        fprintf(stderr,"[qtier] rank-table (QT_RANK_TABLE=%d, topk %d, %s): %llu lookups, %llu table hits (%.1f%%), %llu expert credits\n",
                G.rank_w, G.rank_topk, G.rank_resident?"residents only":"all predicted",(unsigned long long)G.rank_lookups,(unsigned long long)G.rank_hits,
                G.rank_lookups?100.0*G.rank_hits/G.rank_lookups:0.0,(unsigned long long)G.rank_bumps);
    if(G.pf_on){
        uint64_t iss=G.pf_iss_ng+G.pf_iss_prev;
        uint64_t dem=G.hits-G.pf_hit-G.pin_hit;
        fprintf(stderr,"[qtier] prefetch: issued %llu (ngram %llu, prev-union %llu) | "
                "completed %llu (%.2f GB) | hit %llu | evicted-unused %llu | "
                "drops cap %llu victim %llu queue %llu | pinned %d\n",
                (unsigned long long)iss,(unsigned long long)G.pf_iss_ng,
                (unsigned long long)G.pf_iss_prev,
                (unsigned long long)G.pf_done, G.pf_bytes/1073741824.0,
                (unsigned long long)G.pf_hit,(unsigned long long)G.pf_evict,
                (unsigned long long)G.pf_drop_cap,(unsigned long long)G.pf_drop_vict,
                (unsigned long long)G.pf_drop_q, G.pinned_cnt);
        fprintf(stderr,"[qtier] hit split of %.1f%%: demand %.1f%% + pinned %.1f%% + prefetch %.1f%% "
                "(pf-hit = first use of a prefetched upload)\n",
                tot>0?100.0*G.hits/tot:0.0, tot>0?100.0*dem/tot:0.0,
                tot>0?100.0*G.pin_hit/tot:0.0, tot>0?100.0*G.pf_hit/tot:0.0);
    }
    if(G.timers)
        fprintf(stderr,"[qtier] phase ms total: issue %.0f | gu_wait %.0f | "
                "silu %.0f | down %.0f | y_wait %.0f\n",
                G.t_issue,G.t_gu_wait,G.t_silu,G.t_down,G.t_y_wait);
    if(G.kprof)
        fprintf(stderr,"[qtier] kprof ms total: resident gate|up %.0f | staged wait+gate|up %.0f | silu+down %.0f\n",
                G.kp_ms[0],G.kp_ms[1],G.kp_ms[2]);
}

/* P7: the same counters q38t_stats() prints, for the serve-mode TIERS/EMAP/
 * PFETCH lines.  Read without the mutex: every field is a plain counter the
 * uploader only increments, and a torn read costs the dashboard one stale
 * pixel -- taking G.mx here would put a telemetry printf on the decode
 * thread's critical path once per turn. */
int q38t_stat(Q38TierStat *out){
    if(!out) return 0;
    memset(out,0,sizeof *out);
    if(!G.on) return 0;
    size_t res=0,resb=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++)
        if(G.slot[i].resident){ res++; resb+=G.exp_bytes[i/G.ne]; }
    out->n_layers=G.nl; out->n_experts=G.ne;
    out->resident=res; out->resident_bytes=resb; out->budget_bytes=G.budget;
    { size_t fb=0, tb=0; if (cudaMemGetInfo(&fb,&tb)==cudaSuccess) out->vram_total_bytes=tb; }
    out->hits=G.hits; out->miss=G.miss;
    out->pin_hit=G.pin_hit; out->pf_hit=G.pf_hit;
    out->pf_issued=G.pf_iss_ng+G.pf_iss_prev; out->pf_done=G.pf_done;
    out->pf_evicted=G.pf_evict; out->pf_bytes=G.pf_bytes;
    out->pinned=G.pinned_cnt; out->prefetch_on=G.pf_on;
    return 1;
}

int q38t_resident(int layer,int eid){
    if(!G.on || layer<0 || layer>=G.nl || eid<0 || eid>=G.ne) return 0;
    return qs(layer,eid)->resident ? 2 : 0;
}

void q38t_shutdown(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.bmx); G.b_stop=1; pthread_cond_broadcast(&G.bcv); pthread_mutex_unlock(&G.bmx);
    pthread_join(G.bth,NULL);
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={QT_MAGIC,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier] HEAT_FILE saved: %s\n",hf);
        }
    }
    pthread_mutex_lock(&G.mx); G.th_stop=1;
    pthread_cond_signal(&G.cv); pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    free(G.rt_blob); G.rt_blob=NULL;
    G.rt_bucket=NULL; G.rt_table=NULL; G.rt_hot=NULL; G.rt_nctx=0;
    G.on=0;
}

#endif /* COLI_CUDA */
