/* qwen38_pool.h — P3: prompt/KV cache pool with warm cross-turn reuse.
 *
 * P7 deliberately shipped serve mode with NO cross-turn KV reuse: qwen38
 * carries GDN recurrent state, a PLE conv history and the QSA indexer caches
 * alongside the KV, so a naive truncate-and-extend that restored only the KV
 * would answer from a silently wrong recurrent state (docs/p7-status.md §2).
 * P5 then built exactly the missing machinery — byte-copy snapshots of the
 * GDN rec/ring + PLE history, and positional rollback of KV/indexer caches
 * (docs/p5-status.md "Rollback").  This header scales that idea from
 * intra-batch boundaries to whole-turn checkpoints:
 *
 *   CHECKPOINT = the COMPLETE sequence state after feeding tokens 0..n-1:
 *     - QSA KV rows 0..n-1 (f32 or q8+scales), every attn layer (+MTP slot);
 *     - indexer raw keys IK[0..n-1] + pooled block keys IBK[0..nblk-1] + nblk;
 *     - GDN recurrent state + conv rings, every GDN layer;
 *     - PLE conv history (9 x 10240) + its fill count;
 *     - the fed token ids themselves (kv_prefix.h's INVARIANT: the record of
 *       what was fed is the only description of the state anyone consults);
 *     - MTP: the pend residual + pos (the MTP KV rides the ordinary attn-slot
 *       loop, layer 48).
 *
 *   On a new request the pool returns the LONGEST checkpoint whose token ids
 *   are a strict prefix of the new prompt (exact-prefix-only, v1: multi-turn
 *   chat naturally extends the prefix); the caller restores it and prefills
 *   only the tail.  Restoring replaces the single KV slot, so kv_slots stays
 *   honestly 1.
 *
 * Two snapshot points per serve turn:
 *   - post-prefill (n = prompt length): the one a multi-turn client actually
 *     hits — its next prompt replays this turn's prompt verbatim, then
 *     diverges inside the assistant reply (thinking blocks are stripped from
 *     history by the chat template, so end-of-turn states rarely match);
 *   - end-of-turn (n = prompt + generated): matches when the client resends
 *     the transcript verbatim and re-tokenization is identity (common for
 *     plain text replies).  Under Q38_GPU_DENSE the decode advanced the GDN
 *     state on the DEVICE, so this snapshot pulls it back via q38g_state_get
 *     (the host copy is stale after a GPU decode).
 *
 * Correctness argument: everything restored is a byte copy of what a full
 * prefill of the same token ids would have produced (the whole state is a
 * deterministic function of the fed ids on the CPU path — same reason P5's
 * spec_restore is bit-identical), and chunked prefill is bit-identical to
 * token-at-a-time regardless of chunk boundaries (docs/s3-status.md), so
 * prefilling the tail from position n is bit-identical to prefilling from 0.
 * Image turns are never stored or restored: patch payloads are not described
 * by token ids (kv_prefix.h's TAINT rule).
 *
 * RAM pool, LRU, budget Q38_KV_POOL_GB (default 8, 0 disables).  Optional
 * SSD spill: every stored checkpoint is also written (tmp+rename, crash-safe
 * like kv_persist.h) to Q38_KV_POOL_DIR — defaulting to a "qwen38_kv_pool"
 * dir next to HEAT_FILE, so the serving unit gets restart persistence with
 * no config change; Q38_KV_POOL_DIR=off disables spill.  A RAM-evicted entry
 * keeps its file and reloads lazily at NVMe speed (~0.1 s for a 6K state).
 *
 * ---- spill file format (q38kv_<hash>.bin, little-endian) ----
 *   char   magic[8]  "Q38POOL1"
 *   u32    version   1
 *   u32    sig       config signature (layer/head/dim/q8/mtp geometry hash)
 *   u32    n         tokens covered
 *   u32    ple_hist_n
 *   i32    mtp_pend_pos      (-1 when MTP off)
 *   u32    reserved
 *   u64    blob_bytes
 *   i32    toks[n]
 *   u8     blob[blob_bytes]  the packed state (layout fixed by the geometry:
 *          per layer 0..NL-1: attn -> K,V per kv head (n rows each; q8 mode
 *          adds the per-32-group scales), indexer -> i32 nblk + IK + IBK,
 *          GDN -> rec + ring; then PLE hist; then MTP pend when present)
 *
 * Include from qwen38.c after Model/Cfg/reset_state/q38g_* are visible. */
#ifndef QWEN38_POOL_H
#define QWEN38_POOL_H

#include <stdint.h>
#include <sys/statvfs.h>
#include <stddef.h>

#define Q38POOL_MAGIC "Q38POOL1"
/* 256 (was 64, 2026-09-04): a long agent session stores a checkpoint per turn
 * and the small index dropped the harness's stable system/tool prefix. */
#define Q38POOL_MAX_ENTRIES 256

typedef struct {
    uint64_t hash;          /* FNV-1a over n + token ids, XOR the image key (also the file name) */
    uint64_t vkey;          /* P6.3: key of the images whose cells the checkpoint covers (0: none) */
    int      n;             /* tokens covered */
    int     *toks;          /* [n] fed token ids */
    uint8_t *blob;          /* packed state; NULL = evicted to disk */
    size_t   bytes;         /* blob size */
    int      ple_hist_n;
    int      mtp_pend_pos;
    int      on_disk;       /* spill file exists */
    uint64_t last_use;      /* LRU clock */
    /* last-position logits (P12 pool fix, 2026-09-03): stored with the
     * prompt-end checkpoint so an IDENTICAL prompt restores as well -- the
     * first token is sampled from these instead of a re-prefill.  NULL when
     * the entry has none (prefix or end-of-turn checkpoints, v1/v2 files). */
    float   *logits;
    int      lg_n;          /* vocab entries stored (0 = none); on disk when blob is */
    /* pinned (2026-09-04): a stable system/tool prefix published from the
     * gateway's prefix hint.  Never dropped from the index for room and
     * evicted from disk only when no unpinned checkpoint is left, so a
     * new session or a post-compaction history keeps restoring it.  Spill
     * format v4 carries the flag. */
    int      pinned;
} Q38PoolEnt;

static Q38PoolEnt g_pool_ent[Q38POOL_MAX_ENTRIES];
static int        g_pool_n;
static size_t     g_pool_ram;             /* sum of resident blob bytes */
static size_t     g_pool_budget;          /* Q38_KV_POOL_GB, 0 = pool off */
static char       g_pool_dir[2048];       /* "" = no SSD spill */
static uint64_t   g_pool_clock;
static double     g_pool_min_free_pct = 10.0;   /* Q38_KV_POOL_MIN_FREE_PCT: keep this share of the NVMe free */
static size_t     g_pool_disk_cap;              /* Q38_KV_POOL_DISK_GB: optional cap on spilled bytes (0 = none) */
static size_t     g_pool_disk;                  /* spilled bytes on disk (blob sizes) */
static long       g_pool_hits, g_pool_miss, g_pool_stores;
static uint32_t   g_pool_sig;          /* cached config signature */

static uint64_t q38pool_hash(const int *ids, int n) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)n; h *= 1099511628211ull;
    for (int i = 0; i < n; i++) {
        h ^= (uint32_t)ids[i]; h *= 1099511628211ull;
    }
    return h;
}

/* P6.3: the images of the current prompt whose placeholder cells all lie
 * within the first n tokens, folded into one key.  Token ids alone cannot
 * tell two images apart (their cells are the same <|image_pad|> ids), so a
 * checkpoint carries this key and only restores for the same images. */
static uint64_t q38pool_vis_key(const Model *m, int n) {
    uint64_t h = 0;
    for (int k = 0; k < m->vis_img_n; k++) {
        if (m->vis_img_end[k] > n) continue;
        h ^= m->vis_img_key[k] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    return h;
}

/* geometry signature: a checkpoint from a different config must never load */
static uint32_t q38pool_sig(const Model *m) {
    const Cfg *c = &m->c;
    int v[14] = { c->n_layers, c->kv_heads, c->head_dim, c->idx_dim,
                  c->dn_vheads, c->dn_kdim, c->dn_vdim, c->dn_convk,
                  c->dn_conv_dim, c->ple_convk, c->ple_ngram, c->hidden,
                  m->kv_q8, m->has_mtp };
    uint32_t h = 2166136261u;
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++) {
        h ^= (uint32_t)v[i]; h *= 16777619u;
    }
    return h;
}

/* Does layer i carry indexer caches in the blob?  Must depend only on the
 * loaded weights + config -- NOT on m->IK (allocated lazily by ensure_kv):
 * the startup spill-file scan runs before any KV allocation, and a
 * state-dependent layout would make it mis-size (and delete) valid files. */
static int q38pool_has_idx(const Model *m, int i) {
    return i < m->c.n_layers && m->c.is_attn[i] && m->L[i].ixk && m->c.ratio[i] > 0;
}

/* packed blob size for a checkpoint covering n tokens (layout contract) */
static size_t q38pool_blob_bytes(const Model *m, int n) {
    const Cfg *c = &m->c;
    int NL = c->n_layers + m->has_mtp, ns = c->head_dim / 32;
    size_t b = 0;
    for (int i = 0; i < NL; i++) {
        if (c->is_attn[i]) {
            size_t rows = (size_t)c->kv_heads * n;
            if (m->kv_q8) b += rows * c->head_dim * 2 + rows * ns * sizeof(float) * 2;
            else          b += rows * c->head_dim * sizeof(float) * 2;
            if (q38pool_has_idx(m, i)) {
                int nb = c->ratio[i] > 0 ? n / c->ratio[i] : 0;
                b += sizeof(int32_t);
                b += (size_t)n  * c->idx_dim * sizeof(float);
                b += (size_t)nb * c->idx_dim * sizeof(float);
            }
        } else {
            b += ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim +
                  (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * sizeof(float);
        }
    }
    b += (size_t)(c->ple_convk - 1) * c->ple_ngram * HC * c->hidden * sizeof(float);
    if (m->has_mtp) b += (size_t)HC * c->hidden * sizeof(float);
    return b;
}

/* pack the live state (n = m->n_toks) into blob.  pull_gpu: the GDN state
 * advanced on the device (GPU decode ran since the last gpu_state_push), so
 * read it from there — the host copy is stale. */
static void q38pool_pack(Model *m, uint8_t *blob, int n, int pull_gpu) {
    Cfg *c = &m->c;
    int NL = c->n_layers + m->has_mtp, hd = c->head_dim, ns = hd / 32;
    size_t recsz  = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    size_t ringsz = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    uint8_t *b = blob;
    for (int i = 0; i < NL; i++) {
        if (c->is_attn[i]) {
            for (int h = 0; h < c->kv_heads; h++) {
                int64_t off = (int64_t)h * m->max_t;
                if (m->kv_q8) {
                    memcpy(b, m->K8[i]  + off*hd, (size_t)n*hd);              b += (size_t)n*hd;
                    memcpy(b, m->K8s[i] + off*ns, (size_t)n*ns*sizeof(float)); b += (size_t)n*ns*sizeof(float);
                    memcpy(b, m->V8[i]  + off*hd, (size_t)n*hd);              b += (size_t)n*hd;
                    memcpy(b, m->V8s[i] + off*ns, (size_t)n*ns*sizeof(float)); b += (size_t)n*ns*sizeof(float);
                } else {
                    memcpy(b, m->K[i] + off*hd, (size_t)n*hd*sizeof(float));  b += (size_t)n*hd*sizeof(float);
                    memcpy(b, m->V[i] + off*hd, (size_t)n*hd*sizeof(float));  b += (size_t)n*hd*sizeof(float);
                }
            }
            if (q38pool_has_idx(m, i)) {
                int32_t nb = m->nblk[i];
                memcpy(b, &nb, 4); b += 4;
                memcpy(b, m->IK[i],  (size_t)n  * c->idx_dim * sizeof(float)); b += (size_t)n  * c->idx_dim * sizeof(float);
                memcpy(b, m->IBK[i], (size_t)nb * c->idx_dim * sizeof(float)); b += (size_t)nb * c->idx_dim * sizeof(float);
            }
        } else {
            if (pull_gpu) q38g_state_get(i, (float*)b, (float*)(b + recsz*sizeof(float)));
            else {
                memcpy(b,                      m->DN_rec[i],  recsz  * sizeof(float));
                memcpy(b + recsz*sizeof(float), m->DN_ring[i], ringsz * sizeof(float));
            }
            b += (recsz + ringsz) * sizeof(float);
        }
    }
    size_t plesz = (size_t)(c->ple_convk - 1) * c->ple_ngram * HC * c->hidden;
    memcpy(b, m->ple_hist, plesz * sizeof(float)); b += plesz * sizeof(float);
    if (m->has_mtp) { memcpy(b, m->mtp_pend, (size_t)HC*c->hidden*sizeof(float)); b += (size_t)HC*c->hidden*sizeof(float); }
    (void)b;
}

/* unpack blob into the live state; caller has run reset_state + ensure_kv */
static void q38pool_unpack(Model *m, const Q38PoolEnt *e) {
    Cfg *c = &m->c;
    int n = e->n, NL = c->n_layers + m->has_mtp, hd = c->head_dim, ns = hd / 32;
    size_t recsz  = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    size_t ringsz = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    const uint8_t *b = e->blob;
    for (int i = 0; i < NL; i++) {
        if (c->is_attn[i]) {
            for (int h = 0; h < c->kv_heads; h++) {
                int64_t off = (int64_t)h * m->max_t;
                if (m->kv_q8) {
                    memcpy(m->K8[i]  + off*hd, b, (size_t)n*hd);              b += (size_t)n*hd;
                    memcpy(m->K8s[i] + off*ns, b, (size_t)n*ns*sizeof(float)); b += (size_t)n*ns*sizeof(float);
                    memcpy(m->V8[i]  + off*hd, b, (size_t)n*hd);              b += (size_t)n*hd;
                    memcpy(m->V8s[i] + off*ns, b, (size_t)n*ns*sizeof(float)); b += (size_t)n*ns*sizeof(float);
                } else {
                    memcpy(m->K[i] + off*hd, b, (size_t)n*hd*sizeof(float));  b += (size_t)n*hd*sizeof(float);
                    memcpy(m->V[i] + off*hd, b, (size_t)n*hd*sizeof(float));  b += (size_t)n*hd*sizeof(float);
                }
            }
            if (q38pool_has_idx(m, i)) {
                int32_t nb; memcpy(&nb, b, 4); b += 4;
                m->nblk[i] = nb;
                memcpy(m->IK[i],  b, (size_t)n  * c->idx_dim * sizeof(float)); b += (size_t)n  * c->idx_dim * sizeof(float);
                memcpy(m->IBK[i], b, (size_t)nb * c->idx_dim * sizeof(float)); b += (size_t)nb * c->idx_dim * sizeof(float);
            }
        } else {
            memcpy(m->DN_rec[i],  b, recsz  * sizeof(float));
            memcpy(m->DN_ring[i], b + recsz*sizeof(float), ringsz * sizeof(float));
            b += (recsz + ringsz) * sizeof(float);
        }
    }
    size_t plesz = (size_t)(c->ple_convk - 1) * c->ple_ngram * HC * c->hidden;
    memcpy(m->ple_hist, b, plesz * sizeof(float)); b += plesz * sizeof(float);
    m->ple_hist_n = e->ple_hist_n;
    if (m->has_mtp) { memcpy(m->mtp_pend, b, (size_t)HC*c->hidden*sizeof(float)); b += (size_t)HC*c->hidden*sizeof(float); }
    m->mtp_pend_pos = e->mtp_pend_pos;
    memcpy(m->toks, e->toks, (size_t)n * sizeof(int));
    m->n_toks = n;
}

/* ---- SSD spill ---- */
static void q38pool_drop(int idx, int delete_file);
static void q38pool_path(char *out, size_t cap, uint64_t hash) {
    snprintf(out, cap, "%s/q38kv_%016llx.bin", g_pool_dir, (unsigned long long)hash);
}

/* Disk eviction (2026-09-03): the spill directory had grown to 103 GB of
 * checkpoints with 14 % of the NVMe left.  Before a spill (and once at
 * startup) the oldest on-disk checkpoints are removed until the filesystem
 * keeps Q38_KV_POOL_MIN_FREE_PCT (10 %) free beyond the bytes about to be
 * written, and the spilled total stays under Q38_KV_POOL_DISK_GB when set.
 * A resident blob survives its file's removal (it is simply unspilled again);
 * a disk-only entry is dropped.  `keep` is never evicted. */
static void q38pool_disk_room(size_t need, const Q38PoolEnt *keep) {
    if (!g_pool_dir[0]) return;
    for (;;) {
        struct statvfs st;
        int over_cap = g_pool_disk_cap && g_pool_disk + need > g_pool_disk_cap;
        int low_free = 0;
        if (statvfs(g_pool_dir, &st) == 0) {
            double total = (double)st.f_blocks * st.f_frsize, avail = (double)st.f_bavail * st.f_frsize;
            low_free = avail - (double)need < total * g_pool_min_free_pct / 100.0;
        }
        if (!over_cap && !low_free) return;
        int victim = -1; uint64_t oldest = ~0ull;
        for (int pass = 0; pass < 2 && victim < 0; pass++)   /* unpinned first, pinned last */
            for (int i = 0; i < g_pool_n; i++) {
                Q38PoolEnt *e = &g_pool_ent[i];
                if (!e->on_disk || e == keep || (pass == 0 && e->pinned)) continue;
                if (e->last_use < oldest) { oldest = e->last_use; victim = i; }
            }
        if (victim < 0) return;                     /* nothing left to evict */
        Q38PoolEnt *e = &g_pool_ent[victim];
        char path[2304]; q38pool_path(path, sizeof path, e->hash);
        remove(path);
        g_pool_disk -= e->bytes < g_pool_disk ? e->bytes : g_pool_disk;
        fprintf(stderr, "[pool] disk evicted %d-token %scheckpoint (%.2f GB) for %s\n",
                e->n, e->pinned ? "PINNED " : "", e->bytes / 1073741824.0,
                over_cap ? "the disk cap" : "the free-space floor");
        if (e->blob) e->on_disk = 0;                /* still usable from RAM */
        else q38pool_drop(victim, 0);
    }
}

static void q38pool_spill(Q38PoolEnt *e) {
    if (!g_pool_dir[0] || e->on_disk || !e->blob) return;
    q38pool_disk_room(e->bytes, e);
    char path[2304], tmp[2320];
    q38pool_path(path, sizeof path, e->hash);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    /* version 2 adds the 8-byte image key after the blob size; version 3
     * adds lg_n last-position logits (f32) after the blob (h32[5] = lg_n);
     * version 4 adds a u32 flags word after the image key (bit 0: pinned) */
    uint32_t h32[6] = { 4, g_pool_sig, (uint32_t)e->n,
                        (uint32_t)e->ple_hist_n, (uint32_t)e->mtp_pend_pos,
                        (uint32_t)(e->logits ? e->lg_n : 0) };
    uint64_t bb = e->bytes, vk = e->vkey; uint32_t flags = e->pinned ? 1u : 0u;
    int ok = fwrite(Q38POOL_MAGIC, 1, 8, f) == 8 &&
             fwrite(h32, 4, 6, f) == 6 &&
             fwrite(&bb, 8, 1, f) == 1 &&
             fwrite(&vk, 8, 1, f) == 1 &&
             fwrite(&flags, 4, 1, f) == 1 &&
             fwrite(e->toks, 4, (size_t)e->n, f) == (size_t)e->n &&
             fwrite(e->blob, 1, e->bytes, f) == e->bytes &&
             (!e->logits || fwrite(e->logits, sizeof(float), (size_t)e->lg_n, f) == (size_t)e->lg_n);
    ok = ok && fflush(f) == 0 && fsync(fileno(f)) == 0;   /* data before name */
    fclose(f);
    if (!ok || rename(tmp, path) != 0) { remove(tmp); return; }
    e->on_disk = 1;
    g_pool_disk += e->bytes;
}

/* reload a RAM-evicted entry's blob from its spill file; 0 = failed */
static int q38pool_reload(Model *m, Q38PoolEnt *e) {
    if (!e->on_disk || !g_pool_dir[0]) return 0;
    char path[2304]; q38pool_path(path, sizeof path, e->hash);
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char mg[8]; uint32_t h32[6], flags = 0; uint64_t bb = 0, vk = 0;
    int ok = fread(mg, 1, 8, f) == 8 && !memcmp(mg, Q38POOL_MAGIC, 8) &&
             fread(h32, 4, 6, f) == 6 && (h32[0] >= 1 && h32[0] <= 4) &&
             h32[1] == g_pool_sig && (int)h32[2] == e->n &&
             fread(&bb, 8, 1, f) == 1 && (h32[0] == 1 || fread(&vk, 8, 1, f) == 1) &&
             (h32[0] < 4 || fread(&flags, 4, 1, f) == 1) &&
             bb == (uint64_t)q38pool_blob_bytes(m, e->n);
    if (ok) {
        e->blob = malloc((size_t)bb);
        ok = e->blob &&
             fseek(f, (long)((size_t)e->n * 4), SEEK_CUR) == 0 &&
             fread(e->blob, 1, (size_t)bb, f) == (size_t)bb;
        if (!ok) { free(e->blob); e->blob = NULL; }
        else { e->bytes = (size_t)bb; g_pool_ram += e->bytes; }
        /* v3: the last-position logits follow the blob */
        int lg = h32[0] >= 3 ? (int)h32[5] : 0;
        if (ok && lg > 0 && lg == m->c.vocab) {
            e->logits = malloc((size_t)lg * sizeof(float));
            if (e->logits && fread(e->logits, sizeof(float), (size_t)lg, f) == (size_t)lg) {
                e->lg_n = lg; g_pool_ram += (size_t)lg * sizeof(float);
            } else { free(e->logits); e->logits = NULL; e->lg_n = 0; }
        }
    }
    fclose(f);
    return e->blob != NULL;
}

static void q38pool_drop(int idx, int delete_file) {
    Q38PoolEnt *e = &g_pool_ent[idx];
    if (e->blob) { g_pool_ram -= e->bytes; free(e->blob); }
    if (e->logits) { g_pool_ram -= (size_t)e->lg_n * sizeof(float); free(e->logits); e->logits = NULL; }
    if (delete_file && e->on_disk && g_pool_dir[0]) {
        char path[2304]; q38pool_path(path, sizeof path, e->hash);
        remove(path);
        g_pool_disk -= e->bytes < g_pool_disk ? e->bytes : g_pool_disk;
    }
    free(e->toks);
    g_pool_ent[idx] = g_pool_ent[--g_pool_n];
}

static void q38pool_clear(void) {
    int cleared = g_pool_n;
    while (g_pool_n) q38pool_drop(g_pool_n - 1, 1);
    if (g_pool_dir[0]) {
        DIR *dp = opendir(g_pool_dir);
        struct dirent *de;
        while (dp && (de = readdir(dp))) {
            unsigned long long hash; int end = 0;
            if (sscanf(de->d_name, "q38kv_%16llx.bin%n", &hash, &end) == 1 &&
                end == 26 && !de->d_name[26]) {
                char path[2304];
                snprintf(path, sizeof path, "%s/%s", g_pool_dir, de->d_name);
                remove(path);
            }
        }
        if (dp) closedir(dp);
    }
    fprintf(stderr, "[pool] reset: cleared %d checkpoints\n", cleared);
}

/* evict resident blobs (oldest first) until `need` more bytes fit the budget.
 * A spilled entry keeps its file and stays in the index; an unspilled one is
 * gone for good. */
static void q38pool_make_room(size_t need) {
    while (g_pool_ram + need > g_pool_budget && g_pool_ram > 0) {
        int victim = -1; uint64_t oldest = ~0ull;
        for (int i = 0; i < g_pool_n; i++)
            if (g_pool_ent[i].blob && g_pool_ent[i].last_use < oldest) {
                oldest = g_pool_ent[i].last_use; victim = i;
            }
        if (victim < 0) break;
        Q38PoolEnt *e = &g_pool_ent[victim];
        if (!e->on_disk) q38pool_spill(e);         /* keep it reachable on SSD */
        if (e->on_disk) {
            g_pool_ram -= e->bytes; free(e->blob); e->blob = NULL;
            if (e->logits) { g_pool_ram -= (size_t)e->lg_n * sizeof(float); free(e->logits); e->logits = NULL; }
        }
        else q38pool_drop(victim, 0);
    }
}

/* spill every resident-but-unspilled entry; called after the turn's DONE so
 * the fsync'd file writes never sit between prefill and the first token */
static void q38pool_flush(void) {
    for (int i = 0; i < g_pool_n; i++)
        if (g_pool_ent[i].blob && !g_pool_ent[i].on_disk)
            q38pool_spill(&g_pool_ent[i]);
}

/* snapshot the live state (n = m->n_toks).  pull_gpu: see q38pool_pack.
 * The blob is packed NOW (the state changes as decode continues) but spilled
 * later by q38pool_flush. */
static int q38pool_can_store(Model *m, int n) {
    return g_pool_budget && n >= 16 &&
           q38pool_blob_bytes(m, n) <= g_pool_budget;
}

/* logits: the last position's logits when this is a prompt-end checkpoint
 * (an identical prompt then restores fully), NULL otherwise. */
/* pinned: 1 for the stable system/tool prefix published from the prefix
 * hint (see Q38PoolEnt.pinned); 0 for per-turn checkpoints. */
static void q38pool_store(Model *m, int pull_gpu, const float *logits, int pinned) {
    if (!g_pool_budget) return;
    int n = m->n_toks;
    if (n < 16) return;                             /* not worth the copies */
    uint64_t vkey = q38pool_vis_key(m, n);          /* P6.3: image turns are keyed, not refused */
    uint64_t hash = q38pool_hash(m->toks, n) ^ vkey;
    for (int i = 0; i < g_pool_n; i++)
        if (g_pool_ent[i].hash == hash && g_pool_ent[i].n == n && g_pool_ent[i].vkey == vkey &&
            !memcmp(g_pool_ent[i].toks, m->toks, (size_t)n * sizeof(int))) {
            Q38PoolEnt *e = &g_pool_ent[i];
            e->last_use = ++g_pool_clock;               /* already have it */
            if (pinned && !e->pinned) {                 /* promote: the file carries the flag */
                e->pinned = 1;
                if (e->on_disk && e->blob) {
                    char path[2304]; q38pool_path(path, sizeof path, e->hash);
                    remove(path); g_pool_disk -= e->bytes < g_pool_disk ? e->bytes : g_pool_disk;
                    e->on_disk = 0; q38pool_spill(e);
                }
            }
            if (logits && !e->logits) {                 /* upgrade a logits-less entry */
                e->logits = malloc((size_t)m->c.vocab * sizeof(float));
                if (e->logits) {
                    memcpy(e->logits, logits, (size_t)m->c.vocab * sizeof(float));
                    e->lg_n = m->c.vocab; g_pool_ram += (size_t)e->lg_n * sizeof(float);
                    if (e->on_disk) {                   /* rewrite the file with the logits */
                        char path[2304]; q38pool_path(path, sizeof path, e->hash);
                        remove(path); g_pool_disk -= e->bytes < g_pool_disk ? e->bytes : g_pool_disk;
                        e->on_disk = 0;
                        if (e->blob) q38pool_spill(e);
                    }
                }
            }
            return;
        }
    size_t bytes = q38pool_blob_bytes(m, n);
    if (bytes > g_pool_budget) return;              /* larger than the pool */
    if (g_pool_n == Q38POOL_MAX_ENTRIES) {          /* index full: drop the LRU unpinned entry */
        int victim = -1;
        for (int pass = 0; pass < 2 && victim < 0; pass++)
            for (int i = 0; i < g_pool_n; i++)
                if ((pass || !g_pool_ent[i].pinned) &&
                    (victim < 0 || g_pool_ent[i].last_use < g_pool_ent[victim].last_use)) victim = i;
        q38pool_drop(victim, 1);
    }
    q38pool_make_room(bytes);
    double t0 = now_s();
    Q38PoolEnt *e = &g_pool_ent[g_pool_n];
    memset(e, 0, sizeof *e);
    e->blob = malloc(bytes);
    e->toks = malloc((size_t)n * sizeof(int));
    if (!e->blob || !e->toks) { free(e->blob); free(e->toks); return; }
    memcpy(e->toks, m->toks, (size_t)n * sizeof(int));
    e->hash = hash; e->vkey = vkey; e->n = n; e->bytes = bytes;
    e->ple_hist_n = m->ple_hist_n;
    e->mtp_pend_pos = m->has_mtp ? m->mtp_pend_pos : -1;
    e->last_use = ++g_pool_clock;
    e->pinned = pinned;
    q38pool_pack(m, e->blob, n, pull_gpu);
    g_pool_ram += bytes;
    if (logits) {
        e->logits = malloc((size_t)m->c.vocab * sizeof(float));
        if (e->logits) { memcpy(e->logits, logits, (size_t)m->c.vocab * sizeof(float));
                         e->lg_n = m->c.vocab; g_pool_ram += (size_t)e->lg_n * sizeof(float); }
    }
    g_pool_n++;
    g_pool_stores++;
    fprintf(stderr, "[pool] stored %d tokens (%.1f MB) in %.2fs | ram %.2f/%.2f GB, %d entries%s\n",
            n, bytes / 1048576.0, now_s() - t0,
            g_pool_ram / 1073741824.0, g_pool_budget / 1073741824.0, g_pool_n,
            pinned ? " (pinned stable prefix)" : "");
}

/* longest stored checkpoint that is a STRICT prefix of ids[0..np-1]; restores
 * it (reset_state + unpack) and returns its length — the caller prefills only
 * ids[n..np-1].  0 = no usable checkpoint (caller does the ordinary reset).
 * n == np is rejected: the state would need zero tail tokens, but the last
 * prompt position's logits are not stored (kv_prefix.h has the same rule). */
/* logits_out (vocab floats, may be NULL): receives the stored last-position
 * logits on an exact-length hit (n == np), which is then a complete restore
 * -- the caller prefills nothing and samples the first token from them. */
static int q38pool_restore(Model *m, const int *ids, int np, float *logits_out) {
    if (!g_pool_budget) return 0;
    int best = -1;
    for (int i = 0; i < g_pool_n; i++) {
        Q38PoolEnt *e = &g_pool_ent[i];
        if (e->n > np || (best >= 0 && e->n <= g_pool_ent[best].n)) continue;
        if (e->n == np && !(logits_out && (e->logits || (e->on_disk && e->lg_n > 0)))) continue;
        if (!memcmp(e->toks, ids, (size_t)e->n * sizeof(int)) &&
            e->vkey == q38pool_vis_key(m, e->n)) best = i;   /* same tokens AND same images */
    }
    if (best < 0) { g_pool_miss++; return 0; }
    Q38PoolEnt *e = &g_pool_ent[best];
    double t0 = now_s();
    int from_disk = !e->blob;
    if (!e->blob) {
        q38pool_make_room(e->bytes ? e->bytes : q38pool_blob_bytes(m, e->n));
        if (!q38pool_reload(m, e)) {                /* stale/corrupt file */
            q38pool_drop(best, 1);
            g_pool_miss++;
            return 0;
        }
    }
    if (e->n == np && (!e->logits || e->lg_n != m->c.vocab)) {
        g_pool_miss++;                              /* file lost its logits: not a full restore */
        return 0;
    }
    e->last_use = ++g_pool_clock;
    reset_state(m);
    q38pool_unpack(m, e);
    if (e->n == np) memcpy(logits_out, e->logits, (size_t)e->lg_n * sizeof(float));
    g_pool_hits++;
    fprintf(stderr, "[pool] hit: %d/%d prompt tokens restored (%s, %.2fs) -- prefilling %d%s\n",
            e->n, np, from_disk ? "ssd" : "ram", now_s() - t0, np - e->n,
            e->n == np ? " (identical prompt, logits restored)" : "");
    return e->n;
}

/* startup: parse env, and index any spill files a previous process left (the
 * blob stays on disk until a hit — only header + token ids are read here). */
static void q38pool_init(Model *m) {
    g_pool_sig = q38pool_sig(m);
    const char *gb = getenv("Q38_KV_POOL_GB");
    double v = gb && *gb ? atof(gb) : 8.0;
    g_pool_budget = v > 0 ? (size_t)(v * 1073741824.0) : 0;
    if (!g_pool_budget) { fprintf(stderr, "[pool] disabled (Q38_KV_POOL_GB=0)\n"); return; }
    g_pool_dir[0] = 0;
    const char *d = getenv("Q38_KV_POOL_DIR");
    if (d && (!strcmp(d, "off") || !strcmp(d, "none") || !strcmp(d, "0"))) d = NULL;
    else if (!d || !*d) {
        static char def[2048];
        const char *hf = getenv("HEAT_FILE");
        if (hf && *hf) {                            /* sibling of the heat file */
            snprintf(def, sizeof def, "%s", hf);
            char *slash = strrchr(def, '/');
            if (slash) *slash = 0; else snprintf(def, sizeof def, ".");
            size_t l = strlen(def);
            snprintf(def + l, sizeof def - l, "/qwen38_kv_pool");
            d = def;
        } else d = NULL;
    }
    if (d) {
        if (mkdir(d, 0755) != 0 && errno != EEXIST)
            fprintf(stderr, "[pool] cannot create %s -- SSD spill off\n", d);
        else snprintf(g_pool_dir, sizeof g_pool_dir, "%s", d);
    }
    int loaded = 0;
    if (g_pool_dir[0]) {                            /* index leftover spills */
        DIR *dp = opendir(g_pool_dir);
        struct dirent *de;
        while (dp && (de = readdir(dp)) && g_pool_n < Q38POOL_MAX_ENTRIES) {
            unsigned long long fh; int end = 0;
            if (sscanf(de->d_name, "q38kv_%16llx.bin%n", &fh, &end) != 1 ||
                end != 26 || de->d_name[26]) continue;    /* exact name only (skips .tmp) */
            char path[2304]; snprintf(path, sizeof path, "%s/%s", g_pool_dir, de->d_name);
            FILE *f = fopen(path, "rb"); if (!f) continue;
            char mg[8]; uint32_t h32[6], flags = 0; uint64_t bb, vk = 0;
            int ok = fread(mg, 1, 8, f) == 8 && !memcmp(mg, Q38POOL_MAGIC, 8) &&
                     fread(h32, 4, 6, f) == 6 && (h32[0] >= 1 && h32[0] <= 4) &&
                     h32[1] == g_pool_sig && fread(&bb, 8, 1, f) == 1 &&
                     (h32[0] == 1 || fread(&vk, 8, 1, f) == 1) &&
                     (h32[0] < 4 || fread(&flags, 4, 1, f) == 1) &&
                     h32[2] >= 1 && h32[2] <= (uint32_t)Q38_MAX_CTX &&
                     bb == (uint64_t)q38pool_blob_bytes(m, (int)h32[2]);
            if (ok) {
                Q38PoolEnt *e = &g_pool_ent[g_pool_n];
                memset(e, 0, sizeof *e);
                e->n = (int)h32[2];
                e->vkey = vk;
                e->toks = malloc((size_t)e->n * sizeof(int));
                if (e->toks && fread(e->toks, 4, (size_t)e->n, f) == (size_t)e->n &&
                    (q38pool_hash(e->toks, e->n) ^ vk) == (uint64_t)fh) {
                    e->hash = (uint64_t)fh;
                    e->bytes = (size_t)bb;
                    e->ple_hist_n = (int)h32[3];
                    e->mtp_pend_pos = (int32_t)h32[4];
                    e->on_disk = 1;
                    e->lg_n = h32[0] >= 3 ? (int)h32[5] : 0;   /* logits stay on disk until a hit */
                    e->pinned = (flags & 1u) != 0;
                    e->last_use = ++g_pool_clock;
                    g_pool_disk += e->bytes;
                    g_pool_n++; loaded++;
                } else free(e->toks);
            } else remove(path);                    /* different model/config */
            fclose(f);
        }
        if (dp) closedir(dp);
    }
    { const char *pf = getenv("Q38_KV_POOL_MIN_FREE_PCT"); if (pf && *pf) g_pool_min_free_pct = atof(pf);
      const char *dc = getenv("Q38_KV_POOL_DISK_GB");
      g_pool_disk_cap = dc && *dc ? (size_t)(atof(dc) * 1073741824.0) : 0; }
    if (g_pool_dir[0]) q38pool_disk_room(0, NULL);  /* trim a stale pool at start */
    fprintf(stderr, "[pool] kv pool on: budget %.1f GB, spill %s%s",
            g_pool_budget / 1073741824.0,
            g_pool_dir[0] ? g_pool_dir : "off", loaded ? "" : "\n");
    if (loaded) fprintf(stderr, " (%d checkpoints indexed from disk)\n", loaded);
}

#endif /* QWEN38_POOL_H */
