/* Qwen3.8-Flash-Next (arch "qwen4exp") inference engine in pure C.
 * Stage S1: CORRECT CPU-only forward pass, validated against llama.cpp.
 *
 * Architecture (docs/qwen4exp-reference.md):
 *   48 layers, hidden 2560, vocab 248320.
 *   36 GDN (Gated DeltaNet) layers + 12 QSA (sparse attention) layers at
 *   (i+1)%4==0.  Hyper-connections: 4 residual streams (res_hc [4][2560])
 *   with per-site mix (norm -> down/silu -> up/sigmoid gate -> mean) and
 *   combine (res += out * 2*sigmoid(inject/4)).  No per-layer layernorm, no
 *   final output norm -- the HC mixers replace both.
 *   MoE every layer: softmax over 512 experts -> top-10 -> renorm, SwiGLU
 *   inter 640, plus shared expert with scalar sigmoid gate.
 *   PLE n-gram block at ONE layer (gguf_kv ple.layers, 0-based layer 1):
 *   16 hashed rows x 160 dims gathered from the mmap'd 320M-row IQ4_NL
 *   table, key/query gate + depthwise conv (kernel 4, dilation 3), additive.
 *
 * S3 additions (docs/s3-status.md):
 *   - QSA indexer + top-k block selection (docs/qwen4exp-reference.md §3.2):
 *     raw indexer keys cached per QSA layer, mean-pooled per block of
 *     r = compress_ratios[il], norm+rope AFTER pooling at pos b*r, 4 RoPE'd
 *     query heads, ReLU-per-head-then-sum scores, +1e9 tail bias, budget
 *     min(n_kv, indexer_top_k + r - 1) cells.  While n_kv <= budget the
 *     selection covers every cell, so the engine short-circuits to the S1
 *     full-attention path (bit-identical regression guard).
 *   - Optional q8 KV cache (Q38_KV_Q8=1): K/V rows stored as int8 with one
 *     f32 scale per 32-dim group (q8_0-style); f32 stays the parity default.
 *   - Chunked layer-major prefill (Q38_PREFILL_B, default 1024, max 1024): dense mats are
 *     GEMM'd over the chunk, GDN/PLE recurrences stay sequential inside it,
 *     and MoE uses one multi-token tier issue.  CPU batching preserves the
 *     scalar summation order; CUDA dense batching is the same accepted
 *     bit-close class as the decode GPU path.
 *   - Load-time host-memory report; CUDA_EXPERT_GB=auto sizes the expert tier
 *     from CUDA free memory after dense weights and graphs are allocated.
 *
 * S1 simplifications still present:
 *   - Prefill logits only for the last prompt token.
 *   - Experts are decoded ON DEMAND per token from the container via
 *     ggml_blocks GEMV directly on the raw GGML blocks (no cache; the box's
 *     page cache holds the container).
 *
 * Weights: every quantized tensor in the container is a byte-identical raw
 * copy of its GGML blocks (safetensors U8).  Dense raw tensors are held in
 * RAM as raw blocks and multiplied with ggml_bk_gemv (same dequant math as
 * llama.cpp, f32 activations).  F32/BF16 tensors load via st_read_f32.
 *
 * Env: SNAP=<container dir> (required), TOK=<tokenizer.json> (default
 * SNAP/tokenizer.json), N_NEW=<n> (default 32), Q38_MAXT=<ctx> (default
 * 8192), QWEN38_DEBUG=1 (per-layer residual RMS for the first token),
 * DUMP=<path> (last-token logits raw f32), PROMPT=<text> (inline prompt,
 * overrides the argv file).  Argv: qwen38 <prompt.txt>
 *
 * P7 additions (docs/p7-status.md):
 *   - SERVE=1 puts the engine on the colibri gateway wire protocol
 *     (docs/serve_protocol.md) instead of the argv run, so `coli chat`,
 *     `coli serve` and `coli web` can drive it; argv[1] is then the launcher's
 *     cache-slot count, not a prompt file.  Sampling: temperature and top_p
 *     come from the SUBMIT frame, Q38_TOP_K / Q38_MIN_P from the environment,
 *     SEED seeds the RNG.  Greedy (temperature 0) stays the default for the
 *     reference gates.
 *   - HWINFO / TIERS / EMAP / HITS dashboard lines, plus a PFETCH line
 *     carrying the P4 prefetcher's hit split.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <errno.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <dirent.h>
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "st.h"
#include "json.h"
#include "ggml_blocks.h"
#include "qwen38_tier.h"
#include "qwen38_cuda.h"
#include "compat.h"
#include "serve_codec.h"
#include "qwen38_control.h"
#include "qwen38_reasoning.h"
#include "qwen38_sampler.h"
#include "vision_qwen3vl.h"
#include "vision_qwen3vl_cuda.h"   /* P6.1: the tower on the device */

#define Q38_MAX_CTX 131072      /* S3 validation ceiling (model native 262144) */
#define Q38_DEFAULT_MAX_CTX 8192

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r);
#if defined(__APPLE__)
    return r.ru_maxrss / (1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss / (1024.0*1024.0);
#endif
}
static float *falloc(int64_t n) { float *p = malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static float *fcalloc(int64_t n) { float *p = calloc((size_t)n,sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static float sigmoidf_(float v){ return v>=0.f ? 1.f/(1.f+expf(-v)) : expf(v)/(1.f+expf(v)); }
static float siluf_(float v){ return v * sigmoidf_(v); }
static float softplus_f(float z) { return z > 20.f ? z : log1pf(expf(z)); }

/* ==================== tokenizer (HF tokenizer.json byte-level BPE) =====
 * Taken from qwen36.c (same Qwen GPT-2-style BPE, same pretokenizer regex,
 * vocab 248,320).  encode_text / decode_id_to_bytes are the public surface. */
static char **g_tok = NULL;
static int    g_tok_n = 0;
static int hexnib(char c){
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return 0;
}
typedef struct { char **keys; int *vals; int *used; int cap; } SMap;
static unsigned shash(const char *s){ unsigned h=2166136261u; while(*s){ h^=(unsigned char)*s++; h*=16777619u; } return h; }
static void smap_init(SMap *m,int cap){ m->cap=cap; m->keys=calloc(cap,sizeof(char*)); m->vals=malloc(cap*sizeof(int)); m->used=calloc(cap,sizeof(int)); }
static void smap_put(SMap *m,const char *k,int v){ if(!k)return; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0){m->vals[h]=v;return;} h=(h+1)&(m->cap-1);} m->used[h]=1; m->keys[h]=(char*)k; m->vals[h]=v; }
static int smap_get(SMap *m,const char *k){ if(!m||!m->cap||!k)return -1; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0)return m->vals[h]; h=(h+1)&(m->cap-1);} return -1; }

static SMap  g_rev;
static SMap  g_merge;
static char  byte_sym_utf8[256][8];
static short g_unmap[512];
static int   g_nspecial = 0;
static char **g_sp_str = NULL; static int *g_sp_id = NULL; static int *g_sp_len = NULL;

static const char *jstr(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_STR)?v->str:NULL; }
static double jnum(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_NUM)?v->num:0; }

enum { U_W=0, U_L=1, U_M=2, U_N=3, U_P=4, U_O=5 };
static int uclass(unsigned cp){
    if (cp==0x20||cp==0x09||cp==0x0A||cp==0x0D||cp==0x0B||cp==0x0C) return U_W;
    if (cp==0x00A0||cp==0x2000||cp==0x2001||cp==0x2002||cp==0x2003||cp==0x2004||cp==0x2005||cp==0x2006||cp==0x2007||cp==0x2008||cp==0x2009||cp==0x200A||cp==0x2028||cp==0x2029||cp==0x202F||cp==0x205F||cp==0x3000||cp==0xFEFF) return U_W;
    if (cp>=0x30&&cp<=0x39) return U_N;
    if (cp>=0xFF10&&cp<=0xFF19) return U_N;
    if (cp>=0x0660&&cp<=0x0669) return U_N;
    if ((cp>=0x41&&cp<=0x5A)||(cp>=0x61&&cp<=0x7A)) return U_L;
    if (cp>=0x00C0&&cp<=0x024F) return U_L;
    if (cp>=0x0400&&cp<=0x04FF) return U_L;
    if (cp>=0x0600&&cp<=0x06FF) return U_L;
    if (cp>=0x1F00&&cp<=0x1FFF) return U_L;
    if (cp>=0x3040&&cp<=0x30FF) return U_L;
    if (cp>=0x3400&&cp<=0x4DBF) return U_L;
    if (cp>=0x4E00&&cp<=0x9FFF) return U_L;
    if (cp>=0xAC00&&cp<=0xD7A3) return U_L;
    if (cp>=0x300&&cp<=0x36F) return U_M;
    if (cp>=0x1AB0&&cp<=0x1AFF) return U_M;
    if (cp>=0x1DC0&&cp<=0x1DFF) return U_M;
    if (cp>=0x20D0&&cp<=0x20FF) return U_M;
    if (cp>=0xFE20&&cp<=0xFE2F) return U_M;
    if (cp>=0x21&&cp<=0x2F) return U_P;
    if (cp>=0x3A&&cp<=0x40) return U_P;
    if (cp>=0x5B&&cp<=0x60) return U_P;
    if (cp>=0x7B&&cp<=0x7E) return U_P;
    if (cp>=0x3000&&cp<=0x303F) return U_P;
    if (cp>=0xFF01&&cp<=0xFF0F) return U_P;
    if (cp>=0xFF1A&&cp<=0xFF20) return U_P;
    if (cp>=0xFF3B&&cp<=0xFF40) return U_P;
    if (cp>=0xFF5B&&cp<=0xFF65) return U_P;
    if (cp>=0x2010&&cp<=0x2027) return U_P;
    if (cp>=0x2030&&cp<=0x205E) return U_P;
    return U_O;
}
static int utf8_decode(const char *s,int i,int n,int *adv){
    unsigned char c=(unsigned char)s[i]; int cp,a;
    if(c<0x80){cp=c;a=1;}
    else if((c>>5)==6){cp=c&0x1F;a=2;}
    else if((c>>4)==14){cp=c&0x0F;a=3;}
    else if((c>>3)==30){cp=c&0x07;a=4;}
    else {cp=c;a=1;}
    for(int k=1;k<a;k++){ if(i+k<n && ((unsigned char)s[i+k]&0xC0)==0x80) cp=(cp<<6)|((unsigned char)s[i+k]&0x3F); }
    if(adv)*adv=a; return cp;
}
static int utf8_adv(const char *s,int i){ int a; utf8_decode(s,i,0x7fffffff,&a); return a; }

static void build_byte_sym(void){
    for(int i=0;i<512;i++) g_unmap[i]=-1;
    int bs[256]; for(int b=0;b<256;b++) bs[b]=0;
    for(int b=33;b<=126;b++) bs[b]=1;
    for(int b=161;b<=172;b++) bs[b]=1;
    for(int b=174;b<=255;b++) bs[b]=1;
    int cn=0;
    for(int b=0;b<256;b++){
        int cp = bs[b]?b:(256+cn); if(!bs[b]) cn++;
        int k=0; unsigned c=(unsigned)cp;
        if(c<0x80) byte_sym_utf8[b][k++]=(char)c;
        else if(c<0x800){ byte_sym_utf8[b][k++]=0xC0|(c>>6); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        else { byte_sym_utf8[b][k++]=0xE0|(c>>12); byte_sym_utf8[b][k++]=0x80|((c>>6)&0x3F); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        byte_sym_utf8[b][k]=0;
        g_unmap[cp]=(short)b;
    }
}
static void push_id(int **ids,int *n,int *cap,int v){ if(*n==*cap){*cap*=2; *ids=realloc(*ids,*cap*sizeof(int));} (*ids)[(*n)++]=v; }

static int try_special(const char *s,int i,int n,int *id_out){
    int best_len=0,best_id=-1;
    for(int k=0;k<g_nspecial;k++){
        int L=g_sp_len[k]; if(L<=0||i+L>n) continue;
        if(memcmp(s+i,g_sp_str[k],L)==0){ if(L>best_len){best_len=L;best_id=g_sp_id[k];} }
    }
    *id_out=best_id; return best_len;
}
static int pretok_end(const char *s,int i,int n){
    if (s[i]=='\''){
        const char *cands[]={"ll","ve","re","s","t","m","d"}; int clen[]={2,2,2,1,1,1,1};
        int best=0;
        for(int c=0;c<7;c++){ int L=clen[c]; if(i+1+L>n) continue; int ok=1; for(int k=0;k<L;k++){ char a=(char)tolower((unsigned char)s[i+1+k]); if(a!=cands[c][k]){ok=0;break;} } if(ok&&L>best)best=L; }
        if(best>0) return i+1+best;
    }
    int adv; unsigned c0=utf8_decode(s,i,n,&adv);
    { int k=i; unsigned c=c0; int prefix=0;
        if(k<n && c!='\r'&&c!='\n'&&uclass(c)!=U_L&&uclass(c)!=U_N){
            int a2; unsigned c1=utf8_decode(s,k+adv,n,&a2);
            if(uclass(c1)==U_L||uclass(c1)==U_M){ prefix=1; k+=adv; }
        }
        if(prefix || uclass(c)==U_L || uclass(c)==U_M){
            while(k<n){ int a; unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)==U_L||uclass(cc)==U_M) k+=a; else break; }
            return k;
        }
    }
    if(uclass(c0)==U_N) return i+adv;
    { int k=i;
        if(s[i]==' '&&i+1<n){ int a1; unsigned c1=utf8_decode(s,i+1,n,&a1); if(uclass(c1)!=U_W&&uclass(c1)!=U_L&&uclass(c1)!=U_N&&c1!='\r'&&c1!='\n'){ k=i+1; while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k+=a; else break;} while(k<n&&(s[k]=='\r'||s[k]=='\n'))k++; return k; } }
        if(uclass(c0)!=U_W&&uclass(c0)!=U_L&&uclass(c0)!=U_N&&c0!='\r'&&c0!='\n'){ int k2=i; while(k2<n){int a;unsigned cc=utf8_decode(s,k2,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k2+=a; else break;} while(k2<n&&(s[k2]=='\r'||s[k2]=='\n'))k2++; return k2; }
    }
    if(uclass(c0)==U_W){
        /* whitespace rules, in regex order:  \s*[\r\n]+  |  \s+(?!\S)  |  \s+
         * (qwen36's version swallowed the whole run, which mis-splits
         * "\n    return" -- fixed here, verified against llama-tokenize) */
        int k=i, last_nl_end=-1, prev_start=i, cnt=0;
        while(k<n){ int a; unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)!=U_W) break;
                    if(cc=='\r'||cc=='\n') last_nl_end=k+a; prev_start=k; k+=a; cnt++; }
        if(last_nl_end>i) return last_nl_end;   /* \s*[\r\n]+ */
        if(k<n && cnt>=2) return prev_start;    /* \s+(?!\S): leave last ws char to the next piece */
        return k;                               /* \s+ */
    }
    return i+adv;
}
static void bpe_piece(const char *piece,int len,int **ids,int *n,int *cap){
    if(len<=0) return;
    int sc=0,scap=16; char **syms=malloc(scap*sizeof(char*));
    for(int b=0;b<len;b++){
        const char *sym=byte_sym_utf8[(unsigned char)piece[b]];
        int sl=(int)strlen(sym); char *d=malloc(sl+1); memcpy(d,sym,sl); d[sl]=0;
        if(sc==scap){scap*=2; syms=realloc(syms,scap*sizeof(char*));} syms[sc++]=d;
    }
    while(sc>1){
        int best=-1,besti=-1;
        for(int k=0;k<sc-1;k++){
            const char *a=syms[k],*b=syms[k+1];
            size_t kl=(size_t)strlen(a)+1+(size_t)strlen(b)+1;
            char *key=malloc(kl); snprintf(key,kl,"%s\x1F%s",a,b);
            int r=smap_get(&g_merge,key); free(key);
            if(r>=0 && (best<0||r<best)){best=r;besti=k;}
        }
        if(besti<0) break;
        char *m=malloc(strlen(syms[besti])+strlen(syms[besti+1])+1);
        strcpy(m,syms[besti]); strcat(m,syms[besti+1]);
        free(syms[besti]); free(syms[besti+1]); syms[besti]=m;
        for(int k=besti+1;k<sc-1;k++) syms[k]=syms[k+1]; sc--;
    }
    for(int k=0;k<sc;k++){ int id=smap_get(&g_rev,syms[k]); if(id<0) id=0; push_id(ids,n,cap,id); free(syms[k]); }
    free(syms);
}
static void encode_text(const char *text,int **out_ids,int *out_n){
    int cap=1024,n=0; int *ids=malloc(cap*sizeof(int));
    int tlen=(int)strlen(text); int i=0;
    while(i<tlen){
        int sid; int L=try_special(text,i,tlen,&sid);
        if(L>0){ push_id(&ids,&n,&cap,sid); i+=L; continue; }
        int j=pretok_end(text,i,tlen); if(j<=i) j=i+utf8_adv(text,i);
        /* HF splits the text on special tokens BEFORE pretokenization; a
         * special starting inside this piece therefore truncates it
         * (qwen36's walker missed e.g. "?<|im_end|>" -> "?<|" + "im_end..."). */
        for (int p=i+1; p<j; p++){ int sid2; if (try_special(text,p,tlen,&sid2) > 0){ j=p; break; } }
        bpe_piece(text+i,j-i,&ids,&n,&cap);
        i=j;
    }
    *out_ids=ids; *out_n=n;
}
static void load_tokenizer(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[tok] cannot open %s\n", path); return; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1);
    if (fread(buf,1,(size_t)n,f) != (size_t)n) { }
    buf[n] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    jval *model = json_get(root, "model"); if (!model) model = root;
    jval *vocab = json_get(model, "vocab");
    if (!vocab) vocab = json_get(model, "tokens");
    if (!vocab) { fprintf(stderr, "[tok] no model.vocab/tokens in %s\n", path); free(buf); return; }
    int mx = 0;
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>mx)mx=id; }
    } else mx = vocab->len - 1;
    g_tok = calloc((size_t)mx+1, sizeof(char*));
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>=0 && id<=mx) g_tok[id]=strdup(vocab->keys[i]); }
    } else {
        for (int i=0;i<vocab->len;i++){ if(vocab->kids[i] && vocab->kids[i]->t==J_STR) g_tok[i]=strdup(vocab->kids[i]->str); }
    }
    g_tok_n = mx+1;
    smap_init(&g_rev, 1<<19);
    for (int i=0;i<g_tok_n;i++) if (g_tok[i]) smap_put(&g_rev, g_tok[i], i);
    smap_init(&g_merge, 1<<19);
    jval *merges = json_get(model, "merges");
    if (merges && merges->t==J_ARR){
        for (int r=0;r<merges->len;r++){
            const char *e = merges->kids[r]->str; if(!e) continue;
            const char *sp = strchr(e, ' '); if(!sp) continue;
            int la=(int)(sp-e), lb=(int)strlen(sp+1);
            char *key=malloc(la+1+lb+1);
            memcpy(key,e,la); key[la]=0x1F; memcpy(key+la+1,sp+1,lb); key[la+1+lb]=0;
            smap_put(&g_merge, key, r);
        }
    }
    jval *adds = json_get(root, "added_tokens");
    if (adds && adds->t==J_ARR && g_nspecial==0){
        g_nspecial = adds->len;
        g_sp_str = malloc(g_nspecial*sizeof(char*));
        g_sp_id   = malloc(g_nspecial*sizeof(int));
        g_sp_len  = malloc(g_nspecial*sizeof(int));
        for (int k=0;k<adds->len;k++){
            jval *t = adds->kids[k];
            const char *c = jstr(t,"content");
            g_sp_str[k] = c?strdup(c):strdup("");
            g_sp_id[k]  = (int)jnum(t,"id");
            g_sp_len[k] = (int)strlen(g_sp_str[k]);
        }
    }
    build_byte_sym();
    fprintf(stderr, "[tok] loaded %d pieces (max id %d) from %s\n", vocab->len, mx, path);
    free(buf);
}
static void decode_id_to_bytes(int id, unsigned char *out, int *outn){
    *outn = 0;
    for (int k = 0; k < g_nspecial; k++) if (g_sp_id[k] == id) {
        int n = g_sp_len[k] < 255 ? g_sp_len[k] : 255;
        memcpy(out, g_sp_str[k], (size_t)n);
        *outn = n;
        return;
    }
    if (!g_tok || id<0 || id>=g_tok_n || !g_tok[id]) return;
    const unsigned char *pc = (const unsigned char*)g_tok[id];
    if (pc[0]=='<' && pc[1]=='0' && pc[2]=='x' && pc[5]=='>'){
        out[(*outn)++] = (unsigned char)(hexnib((char)pc[3])*16 + hexnib((char)pc[4]));
        return;
    }
    int i = 0;
    while (pc[i]){
        int cp, extra;
        if (pc[i] < 0x80){ cp = pc[i]; extra = 0; }
        else if ((pc[i] & 0xE0) == 0xC0){ cp = pc[i] & 0x1F; extra = 1; }
        else if ((pc[i] & 0xF0) == 0xE0){ cp = pc[i] & 0x0F; extra = 2; }
        else if ((pc[i] & 0xF8) == 0xF0){ cp = pc[i] & 0x07; extra = 3; }
        else { i++; continue; }
        int ok = 1;
        for (int e=0; e<extra; e++){ if (!pc[i+1+e]){ ok=0; break; } cp = (cp<<6) | (pc[i+1+e] & 0x3F); }
        i += 1 + extra;
        if (!ok) continue;
        if (cp == 0x2581) out[(*outn)++] = ' ';
        else if (cp < 512 && g_unmap[cp] >= 0) out[(*outn)++] = (unsigned char)g_unmap[cp];
        else out[(*outn)++] = (unsigned char)cp;
        if (*outn >= 255) break;
    }
}
static void out_bytes(unsigned char *buf, int *bn, const unsigned char *b, int n){
    for (int k=0; k<n; k++){
        if (*bn < 16) buf[(*bn)++] = b[k];
        int j = 0;
        while (j < *bn){
            unsigned char lead = buf[j]; int need;
            if (lead < 0x80) need = 1;
            else if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else { putchar(buf[j]); memmove(buf+j, buf+j+1, *bn-j-1); (*bn)--; continue; }
            if (j+need > *bn) break;
            fwrite(buf+j, 1, (size_t)need, stdout);
            memmove(buf+j, buf+j+need, *bn-j-need);
            *bn -= need;
        }
    }
}
static void print_decoded(const int *arr, int from, int to){
    unsigned char buf[16]; int bn = 0;
    for (int i=from;i<to;i++){
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(arr[i], tmp, &tn);
        out_bytes(buf, &bn, tmp, tn);
    }
    if (bn) fwrite(buf, 1, (size_t)bn, stdout);
}

/* ==================== raw GGML matrices ==================== */
typedef struct {
    uint8_t *raw;      /* raw GGML blocks, O rows of row_bytes each */
    int type;          /* ggml_bk_type */
    int O, I;
    size_t row_bytes;
    int8_t *q8;        /* S2 dense-i8 (Q38_DENSE_I8=1): per-row int8 requant */
    float  *s8;        /* [O] row scales */
} RawMat;

/* S2 dense-i8 (qwen36's "quantize-once" approach): dequantize each raw-block
 * row once at load and re-quantize to symmetric per-row int8.  The GEMV then
 * runs an AVX2 int8->f32 FMA kernel instead of the scalar per-element block
 * decoders -- ~2-4x faster on the big dense mats.  NOT bit-identical to the
 * S1 block path (a second ~0.4%-per-weight quantization); opt-in via
 * Q38_DENSE_I8=1 and validated against the greedy-parity gates. */
static int g_dense_i8 = 0;

/* int8 x int8 GEMV: the activation is quantized ONCE per call to symmetric
 * int8 with a scale per 32-element block, then each row runs the AVX2
 * sign/maddubs/madd dot (llama.cpp's q8 pattern: |x| stays <=127 so the u8*s8
 * i16 accumulation cannot saturate).  DRAM traffic is the 1 B/elem weight
 * rows; the compute per 32 weights drops from 12 f32 converts+FMAs to one
 * maddubs+madd+FMA, which is what lets 24 Zen2 threads reach the bandwidth
 * ceiling instead of ~2/3 of it. */
/* one row's int8 dot (shared by mati8 and its S3 batched twin so the two
 * paths stay bit-identical) */
static inline float mati8_row_dot(const int8_t *w, const int8_t *xq, const float *xs, int nb) {
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int b = 0; b + 2 <= nb; b += 2) {
        __m256i xv0 = _mm256_loadu_si256((const __m256i*)(xq + b*32));
        __m256i wv0 = _mm256_loadu_si256((const __m256i*)(w + b*32));
        __m256i xv1 = _mm256_loadu_si256((const __m256i*)(xq + b*32 + 32));
        __m256i wv1 = _mm256_loadu_si256((const __m256i*)(w + b*32 + 32));
        __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(xv0, xv0), _mm256_sign_epi8(wv0, xv0)), ones);
        __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(xv1, xv1), _mm256_sign_epi8(wv1, xv1)), ones);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(xs[b]),   _mm256_cvtepi32_ps(p0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(xs[b+1]), _mm256_cvtepi32_ps(p1), a1);
    }
    if (nb & 1) {
        int b = nb - 1;
        __m256i xv = _mm256_loadu_si256((const __m256i*)(xq + b*32));
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + b*32));
        __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(xv, xv), _mm256_sign_epi8(wv, xv)), ones);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(xs[b]), _mm256_cvtepi32_ps(p), a0);
    }
    __m256 s8v = _mm256_add_ps(a0, a1);
    __m128 lo = _mm256_castps256_ps128(s8v), hi = _mm256_extractf128_ps(s8v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    return _mm_cvtss_f32(s4);
#else
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        int32_t p = 0;
        for (int i = 0; i < 32; i++) p += (int32_t)xq[b*32+i] * (int32_t)w[b*32+i];
        acc += xs[b] * (float)p;
    }
    return acc;
#endif
}

static void mati8(float *y, const float *x, const int8_t *q, const float *sc, int I, int O) {
    if (I % 32) {   /* no such matrix in this model; keep a correct fallback */
        #pragma omp parallel for schedule(static) if(O >= 32)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
            y[o] = acc * sc[o];
        }
        return;
    }
    const int nb = I / 32;
    /* shared with the OpenMP workers below; rm_gemv only runs on the single
     * decode thread, so one buffer suffices (NOT __thread: workers must see
     * the block written by the caller thread). */
    static int8_t *xq = NULL; static float *xs = NULL;
    if (!xq) { xq = malloc(16384); xs = malloc(512 * sizeof(float)); if(!xq||!xs){fprintf(stderr,"OOM xq\n");exit(1);} }
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b*32;
        float mx = 0.f;
        for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > mx) mx = a; }
        float s = mx > 0.f ? mx / 127.f : 0.f, inv = mx > 0.f ? 127.f / mx : 0.f;
        xs[b] = s;
        for (int i = 0; i < 32; i++) xq[b*32+i] = (int8_t)lrintf(xb[i] * inv);
    }
    #pragma omp parallel for schedule(static) if(O >= 32)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        y[o] = mati8_row_dot(w, xq, xs, nb) * sc[o];
    }
}

/* S3: batched twin of mati8.  B activations (X row-major, stride I), each
 * quantized EXACTLY as mati8 quantizes its single x, each row dot running the
 * same kernel in the same order -> outputs are bit-identical to B separate
 * mati8 calls.  The win: a weight row is read from DRAM once per chunk. */
static void mati8_b(float *Y, const float *X, const int8_t *q, const float *sc,
                    int I, int O, int B, int ldy) {
    if (I % 32) {
        for (int b = 0; b < B; b++) mati8(Y + (int64_t)b*ldy, X + (int64_t)b*I, q, sc, I, O);
        return;
    }
    const int nb = I / 32;
    int8_t *xqb = malloc((size_t)B * I);
    float  *xsb = malloc((size_t)B * nb * sizeof(float));
    if (!xqb || !xsb) { fprintf(stderr, "OOM mati8_b\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < B; t++) {
        const float *x = X + (int64_t)t * I;
        int8_t *xq = xqb + (int64_t)t * I; float *xs = xsb + (int64_t)t * nb;
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float mx = 0.f;
            for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > mx) mx = a; }
            float s = mx > 0.f ? mx / 127.f : 0.f, inv = mx > 0.f ? 127.f / mx : 0.f;
            xs[b] = s;
            for (int i = 0; i < 32; i++) xq[b*32+i] = (int8_t)lrintf(xb[i] * inv);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        for (int t = 0; t < B; t++)
            Y[(int64_t)t*ldy + o] = mati8_row_dot(w, xqb + (int64_t)t*I, xsb + (int64_t)t*nb, nb) * sc[o];
    }
    free(xqb); free(xsb);
}

/* build the int8 twin of a loaded raw matrix (skipped for the embedding,
 * which is only row-gathered, never GEMV'd). */
static void rm_build_i8(RawMat *rm) {
    rm->q8 = malloc((size_t)rm->O * rm->I);
    rm->s8 = malloc((size_t)rm->O * sizeof(float));
    if (!rm->q8 || !rm->s8) { fprintf(stderr, "OOM dense-i8\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < rm->O; o++) {
        float row[16384];
        ggml_bk_dequant_row(rm->type, rm->raw + (size_t)o * rm->row_bytes, row, (size_t)rm->I);
        float mx = 0.f;
        for (int i = 0; i < rm->I; i++) { float a = fabsf(row[i]); if (a > mx) mx = a; }
        float s = mx > 0.f ? mx / 127.f : 1.f, inv = mx > 0.f ? 127.f / mx : 0.f;
        rm->s8[o] = s;
        int8_t *q = rm->q8 + (size_t)o * rm->I;
        for (int i = 0; i < rm->I; i++) q[i] = (int8_t)lrintf(row[i] * inv);
    }
}

/* Infer the GGML type from the on-disk byte count.  The five block formats
 * have pairwise distinct bytes-per-element ratios, so (O, I, nbytes) pins the
 * type without parsing the (converter-authoritative) meta map. */
static int rm_infer_type(int64_t nbytes, int O, int I, const char *name) {
    static const int cands[] = { GGML_BK_Q8_0, GGML_BK_Q6_K, GGML_BK_IQ4_NL, GGML_BK_IQ3_S, GGML_BK_IQ4_XS };
    for (unsigned c = 0; c < sizeof(cands)/sizeof(cands[0]); c++) {
        size_t rb = ggml_bk_row_bytes(cands[c], (size_t)I);
        if (rb && (int64_t)rb * O == nbytes) return cands[c];
    }
    fprintf(stderr, "%s: cannot infer GGML type from %lld bytes for [%d,%d] -- refusing\n",
            name, (long long)nbytes, O, I);
    exit(1);
}

static void rm_load(shards *S, RawMat *rm, const char *name, int O, int I) {
    st_tensor *t = st_find(S, name);
    if (!t) st_die_missing(S, name);
    rm->type = rm_infer_type(t->nbytes, O, I, name);
    rm->O = O; rm->I = I;
    rm->row_bytes = ggml_bk_row_bytes(rm->type, (size_t)I);
    rm->raw = malloc((size_t)t->nbytes);
    if (!rm->raw) { fprintf(stderr, "OOM %lld raw bytes for %s\n", (long long)t->nbytes, name); exit(1); }
    st_read_raw(S, name, rm->raw, 0);
    rm->q8 = NULL; rm->s8 = NULL;
    if (g_dense_i8 && !strstr(name, "embed_tokens")) rm_build_i8(rm);
}

static void rm_gemv(float *y, const RawMat *rm, const float *x) {
    if (rm->q8) { mati8(y, x, rm->q8, rm->s8, rm->I, rm->O); return; }
    if (ggml_bk_gemv(rm->type, rm->O, rm->I, rm->raw, x, y) != 0) {
        fprintf(stderr, "rm_gemv: bad type/shape (%d, %d x %d)\n", rm->type, rm->O, rm->I); exit(1);
    }
}

/* dequant one row of a raw matrix to f32 (embedding gather) */
static void rm_row_f32(const RawMat *rm, int row, float *out) {
    ggml_bk_dequant_row(rm->type, rm->raw + (size_t)row * rm->row_bytes, out, (size_t)rm->I);
}

/* f32 GEMV: y[O] = W[O,I] @ x, W row-major f32 */
static void matf(float *y, const float *W, const float *x, int I, int O) {
    #pragma omp parallel for schedule(static) if(O >= 32)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += w[i] * x[i];
        y[o] = acc;
    }
}

/* S3 batched twins: per-token math order identical to the single-token
 * kernels (bit-identical outputs), weight rows re-used across the chunk. */
/* Token-blocked f32 GEMM.  Eight tokens share one pass over a weight row,
 * each with its own accumulator that still sums i = 0..I-1 in order, so the
 * result is bit-identical to the one-token loop while the eight independent
 * FMA chains hide the add latency that bounded the serial version (the MoE
 * router at 512 x 2560 per token was 4.0 ms/token of prefill). */
static void matf_b(float *Y, const float *W, const float *X, int I, int O, int B, int ldy) {
    #pragma omp parallel for schedule(static) if(O >= 32)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        int t = 0;
        for (; t + 8 <= B; t += 8) {
            const float *x0 = X + (int64_t)t * I;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
            for (int i = 0; i < I; i++) {
                float wi = w[i];
                a0 += wi * x0[i];                a1 += wi * x0[(int64_t)1*I + i];
                a2 += wi * x0[(int64_t)2*I + i]; a3 += wi * x0[(int64_t)3*I + i];
                a4 += wi * x0[(int64_t)4*I + i]; a5 += wi * x0[(int64_t)5*I + i];
                a6 += wi * x0[(int64_t)6*I + i]; a7 += wi * x0[(int64_t)7*I + i];
            }
            float *y = Y + (int64_t)t*ldy + o;
            y[0] = a0; y[(int64_t)1*ldy] = a1; y[(int64_t)2*ldy] = a2; y[(int64_t)3*ldy] = a3;
            y[(int64_t)4*ldy] = a4; y[(int64_t)5*ldy] = a5; y[(int64_t)6*ldy] = a6; y[(int64_t)7*ldy] = a7;
        }
        for (; t < B; t++) {
            const float *x = X + (int64_t)t * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += w[i] * x[i];
            Y[(int64_t)t*ldy + o] = acc;
        }
    }
}
/* raw-block matrices have no reuse-friendly batched decoder; the parity (f32)
 * path just loops the S1 GEMV per token (bit-identical by construction). */
static void rm_gemv_b(float *Y, const RawMat *rm, const float *X, int B, int ldy) {
    if (rm->q8) { mati8_b(Y, X, rm->q8, rm->s8, rm->I, rm->O, B, ldy); return; }
    for (int t = 0; t < B; t++) rm_gemv(Y + (int64_t)t*ldy, rm, X + (int64_t)t*rm->I);
}

/* llama.cpp-style RMS norm: y = x * rsqrt(mean(x^2)+eps) * w  (PLAIN weight,
 * unlike qwen36's (1+w) variant). in-place capable. w may be NULL (no gamma). */
static void rmsnorm_plain(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    if (w) for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
    else   for (int i = 0; i < D; i++) out[i] = x[i] * r;
}
static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static int g_debug = 0;
static int g_dbg_now = -1;   /* layer index currently under value-debug, -1 = off */
static double vec_rms(const float *x, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i]*x[i];
    return sqrt(s / n);
}
/* first-3 + last-3 of a row, matching llama-eval-callback's print format */
static void dbg_row(const char *tag, const float *x, int n) {
    fprintf(stderr, "[dbg] %-14s %9.4f %9.4f %9.4f ... %9.4f %9.4f %9.4f\n",
            tag, x[0], x[1], x[2], x[n-3], x[n-2], x[n-1]);
}

/* ==================== config / model ==================== */
#define HC 4                 /* hyper-connection streams */
typedef struct {
    int hidden, n_layers, vocab;
    int n_experts, topk, inter, shared_inter;
    int q_heads, kv_heads, head_dim;      /* QSA: 24 / 2 / 256 */
    int idx_heads, idx_dim, idx_topk;     /* S3 indexer: 4 / 128 / 2048 */
    int *ratio;                           /* [n_layers] compress ratio (0 = dense) */
    int n_rot;                            /* rope dims (64) */
    int rope_sec[4];                      /* IMRoPE sections (pairs; [11,11,10,0]) */
    int ple_img;                          /* ple.image_token_id (= <|image_pad|>) */
    float theta, eps;
    int hc_rank;                          /* 320 */
    /* GDN dims */
    int dn_vheads, dn_kheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    uint8_t *is_attn;                     /* [n_layers] */
    /* PLE */
    int ple_layer;                        /* 0-based inject layer */
    int ple_ngram, ple_hpn, ple_dim;      /* 3, 8, 160 */
    int ple_convk;                        /* 4 */
    int ple_eos;
    int64_t ple_rows;
    uint64_t ple_mult[8];
    uint64_t ple_vocab[64], ple_off[64];
    int ple_n_heads;                      /* 16 */
} Cfg;

typedef struct {
    /* hyper-connections (both sites) */
    RawMat hca_down, hca_up, hcf_down, hcf_up;
    float *hca_norm, *hca_inj, *hcf_norm, *hcf_inj;   /* [10240], [4*10240] */
    /* QSA */
    RawMat q, k, v, o;
    float *qn, *kn;                       /* [256] */
    /* S3 indexer (QSA layers with compress_ratio > 0; BF16 -> f32) */
    float *ixq, *ixk;                     /* [4*128,2560], [128,2560] */
    float *ixqn, *ixkn;                   /* [128] */
    /* GDN */
    RawMat dn_qkv, dn_z, dn_out;
    float *dn_b, *dn_alpha;               /* [48,2560] f32 */
    float *dn_a, *dn_dtbias, *dn_norm;    /* [48],[48],[128] */
    float *dn_conv;                       /* [10240,4] */
    /* MoE */
    float *gate;                          /* [512,2560] f32 */
    float *sh_gate;                       /* [2560] */
    RawMat sh_g, sh_u, sh_d;
    /* P4: qwen38_cuda handles for the device copies (-1 = not uploaded) */
    int g_hca_down, g_hca_up, g_hca_norm, g_hca_inj;
    int g_hcf_down, g_hcf_up, g_hcf_norm, g_hcf_inj;
    int g_q, g_k, g_v, g_o;
    int g_qkv, g_z, g_out, g_ba, g_a, g_dt, g_norm, g_conv;
    int g_shg, g_shu, g_shd;   /* shared expert on the dense GEMM path (-1 = CPU) */
    int g_qn, g_kn;            /* QSA q/k norm gammas on the device (P9 GPU QSA) */
    int g_gate;                /* P10 D4: f32 router weights on the device (-1 = host) */
} Layer;

/* images per request; the gateway mirrors it (QWEN38_MAX_IMAGES). 32 since
 * 2026-09-04: at 8 an agent session slid past it one screenshot at a time. */
#define Q38_VIS_MAX_IMG 32

typedef struct {
    Cfg c;
    shards S;
    Layer *L;
    RawMat embed, lm_head, ohc_down, ohc_up;
    float *ohc_norm;
    /* PLE weights */
    RawMat ple_key, ple_value;
    float *ple_conv, *ple_nk, *ple_nq, *ple_nc;   /* [10240*4],[10240]x3 */
    /* PLE table mmap */
    const uint8_t *ple_table;   /* rows at 64 + r*90 */
    size_t ple_map_len;
    /* state */
    float **DN_rec;             /* [layer] -> [48*128*128] */
    float **DN_ring;            /* [layer] -> [10240*(convk-1)] */
    float *ple_hist;            /* [9][10240] normed history, hist[j]=t-1-j */
    int ple_hist_n;
    float **K, **V;             /* per QSA layer: [kvh][max_t][256] (f32 mode) */
    /* S3: q8 KV mode (Q38_KV_Q8=1): int8 rows + one f32 scale per 32 dims */
    int kv_q8;
    int8_t **K8, **V8;          /* [kvh][max_t][256] int8 */
    float  **K8s, **V8s;        /* [kvh][max_t][256/32] scales */
    /* S3 indexer caches (per QSA layer with ratio > 0) */
    float **IK;                 /* raw (pre-norm, pre-rope) keys [max_t][128] */
    float **IBK;                /* pooled+normed+roped block keys [max_t/r][128] */
    int   *nblk;                /* completed (pooled) blocks so far */
    int max_t;
    int *toks;                  /* fed token history (for the PLE hash) */
    int n_toks;
    /* P6 vision: tower lazy-loaded from <snap>/mmproj-F16.gguf (Q38_MMPROJ
     * overrides).  During an image prefill vis_map[pos] >= 0 names the
     * projected-embedding row that replaces the token embedding at that cell,
     * and mp[3*pos..] carries the cell's IMRoPE (t,y,x) positions -- NULL for
     * text-only turns, where the sequential position doubles as all three. */
    const char *snap;
    Q38VisTower *vis;
    int vis_gpu;                    /* P6.1: tower weights resident on the device */
    float *vis_embd;            /* [vis_n][2560] projected image embeddings */
    int vis_n;
    /* P6.3: per image of the current prompt, its content key (patch hash)
     * and the cell just past its placeholders: the prompt pool keys a
     * checkpoint by the images its covered cells contain */
    uint64_t vis_img_key[Q38_VIS_MAX_IMG];
    int vis_img_end[Q38_VIS_MAX_IMG];
    int vis_img_n;
    int *vis_map;               /* [max_t] row index into vis_embd, -1 = text */
    int *mp;                    /* [3*max_t] IMRoPE t,y,x per cell, or NULL */
    int mp_next;                /* rope position of the next appended cell */
    /* P5: MTP draft module (Q38_MTP=1; docs/p5-status.md).  The module is ONE
     * extra hyper-connected QSA+MoE decoder layer stored at L[n_layers] (slot
     * 48), fed by a fold of the backbone's final wide residual and the next
     * token's embedding, projecting through the shared lm_head.  Its experts
     * join the VRAM tier as layer 48.  has_mtp==0 leaves every path below
     * byte-identical to the pre-P5 engine. */
    int has_mtp;
    shards mtpS;                    /* <snap>/mtp/{mtp-dense,mtp-experts}.safetensors */
    RawMat mtp_fce, mtp_fch;        /* fc_embedding / fc_hidden [2560,2560] */
    float *mtp_nfe;                 /* pre_fc_norm_embedding [2560] */
    float *mtp_nfh;                 /* pre_fc_norm_hidden [10240] (stream space) */
    RawMat mtp_mix_down, mtp_mix_up;/* hyper_connection_mixer (output head mix) */
    float *mtp_mix_norm;            /* [10240] */
    void *mtp_map[8]; size_t mtp_map_len[8];
    float *mtp_pend;                /* [4*2560] backbone res_hc of position mtp_pend_pos */
    int mtp_pend_pos;               /* -1 = invalid */
    double dense_load_s;
    /* S2: experts via whole-shard mmap (page cache backs the RAM side).
     * e_g/e_u/e_d[layer*n_experts+eid] point at the raw GGML blocks; types
     * are per layer (gate/up share one, down has its own).  have_mmap==0
     * falls back to the S1 per-call st_read_raw path. */
    void   *shard_map[512];
    size_t  shard_len[512];
    const uint8_t **e_g, **e_u, **e_d;
    int    *e_gu_type, *e_d_type;   /* [n_layers] */
    int     have_mmap;
    /* P4: device handles for the head-side dense mats */
    int g_ohc_down, g_ohc_up, g_ohc_norm, g_lm_head;
    int g_mtp_fce, g_mtp_fch, g_mtp_mix_down, g_mtp_mix_up, g_mtp_mix_norm;
} Model;

/* P4: 1 once the dense backbone runs on the GPU (decode path). */
static int g_gpu_dense = 0;
/* P5.1: MTP draft weights and target-state rollback are independently usable. */
#define MTP_MAX_N 7
static int g_mtp_n = 3;
static int g_gpu_mtp = 0, g_gpu_spec = 0;

/* ---- S2 phase timers (COLI_TIMERS=1): accumulated seconds ---- */
static int g_timers = 0;
static struct { double dense_hc, gdn, qsa, ple, moe_route, moe_cpu, moe_gpu,
                moe_shared, head; long tokens; } g_tm;

static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config/meta: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static jval *load_json(const char *snap, const char *fn){
    char path[2048]; snprintf(path, sizeof(path), "%s/%s", snap, fn);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc((size_t)n+1); if(!buf){fprintf(stderr,"OOM %s\n",path);exit(1);}
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); }
    buf[n]=0; fclose(f);
    char *arena=NULL;
    jval *r = json_parse(buf, &arena);   /* buf/arena intentionally leak (startup only) */
    if(!r){ fprintf(stderr,"%s: JSON parse failed\n",path); exit(1); }
    return r;
}

static void load_cfg(Cfg *c, const char *snap) {
    jval *r = load_json(snap, "config.json");
    c->hidden   = (int)req_num(r,"hidden_size");
    c->n_layers = (int)req_num(r,"num_hidden_layers");
    c->vocab    = (int)req_num(r,"vocab_size");
    c->eps      = (float)req_num(r,"rms_norm_eps");
    c->theta    = (float)req_num(r,"rope_theta");

    jval *m = load_json(snap, "qwen38_meta.json");
    c->n_experts    = (int)req_num(m,"num_experts");
    c->topk         = (int)req_num(m,"topk");
    c->inter        = (int)req_num(m,"moe_inter");
    c->shared_inter = (int)req_num(m,"shared_inter");
    c->q_heads      = (int)req_num(m,"q_heads");
    c->kv_heads     = (int)req_num(m,"kv_heads");
    c->head_dim     = (int)req_num(m,"head_dim");
    c->dn_vheads    = (int)req_num(m,"dn_vheads");
    c->dn_kheads    = (int)req_num(m,"dn_kheads");
    c->dn_kdim      = (int)req_num(m,"dn_kdim");
    c->dn_vdim      = (int)req_num(m,"dn_vdim");
    c->dn_convk     = (int)req_num(m,"dn_convk");
    c->dn_conv_dim  = (int)req_num(m,"dn_conv_dim");
    jval *hc = json_get(m,"hc");
    c->hc_rank = 320;
    if (hc) { jval *v=json_get(hc,"rank"); if(v&&v->t==J_NUM) c->hc_rank=(int)v->num; }
    { jval *v = hc?json_get(hc,"streams"):NULL;
      if (v && v->t==J_NUM && (int)v->num != HC) { fprintf(stderr,"hc streams %d != %d\n",(int)v->num,HC); exit(1); } }

    /* +1 slot everywhere a per-layer array is indexed by layer id: the P5 MTP
     * module lives at layer index n_layers (48) when loaded.  One spare entry
     * costs nothing when MTP is off. */
    c->is_attn = calloc(c->n_layers + 1, sizeof(uint8_t));
    jval *lt = json_get(m,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i=0;i<lt->len && i<c->n_layers;i++){
            const char *s = (lt->kids[i]->t==J_STR)? lt->kids[i]->str : "";
            if (s && (strcmp(s,"sparse_attention")==0 || strcmp(s,"full_attention")==0)) c->is_attn[i]=1;
        }
    } else {
        for (int i=0;i<c->n_layers;i++) c->is_attn[i] = ((i+1)%4==0);
    }

    /* PLE dims from meta.ple */
    jval *p = json_get(m,"ple");
    if (!p) { fprintf(stderr,"qwen38_meta.json: missing ple block\n"); exit(1); }
    c->ple_ngram = (int)req_num(p,"ngram_size");
    c->ple_hpn   = (int)req_num(p,"heads_per_ngram");
    c->ple_dim   = (int)req_num(p,"dim");
    c->ple_rows  = (int64_t)req_num(p,"rows");
    c->ple_n_heads = (c->ple_ngram-1)*c->ple_hpn;
    if (c->ple_n_heads > 64 || c->ple_ngram > 8) { fprintf(stderr,"ple dims out of range\n"); exit(1); }

    /* Verbatim GGUF KVs: hash constants, rope dims, PLE layer index.
     * NOTE: meta's ple.inject_layer says 2, but the GGUF KV ple.layers=[1]
     * (0-based) is authoritative per llama.cpp -- documented deviation. */
    jval *kv = json_get(m,"gguf_kv");
    if (!kv) { fprintf(stderr,"qwen38_meta.json: missing gguf_kv block\n"); exit(1); }
    c->n_rot   = (int)req_num(kv,"qwen4exp.rope.dimension_count");
    /* S3: indexer geometry + per-layer compress ratios (verbatim GGUF KVs) */
    c->idx_heads = (int)req_num(kv,"qwen4exp.attention.indexer.head_count");
    c->idx_dim   = (int)req_num(kv,"qwen4exp.attention.indexer.key_length");
    c->idx_topk  = (int)req_num(kv,"qwen4exp.attention.indexer.top_k");
    c->ratio = calloc(c->n_layers + 1, sizeof(int));
    { jval *cr = json_get(kv,"qwen4exp.attention.compress_ratios");
      if (!cr || cr->t!=J_ARR || cr->len!=c->n_layers) {
          fprintf(stderr,"gguf_kv: bad qwen4exp.attention.compress_ratios\n"); exit(1);
      }
      for (int i=0;i<cr->len;i++) c->ratio[i] = (int)cr->kids[i]->num; }
    if (c->idx_heads < 1 || c->idx_heads > 8 || c->idx_dim < 1 || c->idx_dim > 512 ||
        c->idx_topk < 1) { fprintf(stderr,"[cfg] indexer dims out of range\n"); exit(1); }
    c->ple_eos = (int)req_num(kv,"qwen4exp.ple.eos_token_id");
    c->ple_convk = (int)req_num(kv,"qwen4exp.ple.conv_kernel");
    { jval *v = json_get(kv,"qwen4exp.ple.image_token_id");     /* optional, falls back to EOS */
      c->ple_img = (v && v->t==J_NUM) ? (int)v->num : c->ple_eos; }
    /* IMRoPE sections (pair units); required for image positions, and with
     * equal t/y/x (text) the interleaving cancels out entirely. */
    { jval *sec = json_get(kv,"qwen4exp.rope.dimension_sections");
      if (!sec || sec->t!=J_ARR || sec->len!=4) { fprintf(stderr,"gguf_kv: bad rope.dimension_sections\n"); exit(1); }
      int sum = 0;
      for (int i=0;i<4;i++) { c->rope_sec[i] = (int)sec->kids[i]->num; sum += c->rope_sec[i]; }
      if (sum != c->n_rot/2) { fprintf(stderr,"rope sections sum %d != n_rot/2\n", sum); exit(1); } }
    jval *pl = json_get(kv,"qwen4exp.ple.layers");
    if (!pl || pl->t!=J_ARR || pl->len!=1) { fprintf(stderr,"gguf_kv ple.layers must be a 1-elem array\n"); exit(1); }
    c->ple_layer = (int)pl->kids[0]->num;
    jval *mult = json_get(kv,"qwen4exp.ple.layer_multipliers");
    jval *vsz  = json_get(kv,"qwen4exp.ple.head_vocab_sizes");
    jval *off  = json_get(kv,"qwen4exp.ple.head_offsets");
    if (!mult||mult->t!=J_ARR||mult->len!=c->ple_ngram ||
        !vsz||vsz->t!=J_ARR||vsz->len!=c->ple_n_heads ||
        !off||off->t!=J_ARR||off->len!=c->ple_n_heads) {
        fprintf(stderr,"gguf_kv: bad ple hash constant arrays\n"); exit(1);
    }
    for (int i=0;i<mult->len;i++) c->ple_mult[i] = (uint64_t)mult->kids[i]->num;
    for (int i=0;i<vsz->len;i++){
        c->ple_vocab[i] = (uint64_t)vsz->kids[i]->num;
        c->ple_off[i]   = (uint64_t)off->kids[i]->num;
        if (c->ple_vocab[i]==0 || c->ple_off[i]+c->ple_vocab[i] > (uint64_t)c->ple_rows) {
            fprintf(stderr,"ple head %d: offset+vocab exceeds table rows\n", i); exit(1);
        }
    }

    /* sanity */
    if (c->hidden != 2560 || c->dn_conv_dim != 2*c->dn_kheads*c->dn_kdim + c->dn_vheads*c->dn_vdim ||
        c->q_heads % c->kv_heads != 0 || c->topk > 256 || c->n_experts > 1024 ||
        c->ple_layer < 0 || c->ple_layer >= c->n_layers ||
        c->ple_dim * c->ple_n_heads != c->hidden) {
        fprintf(stderr,"[cfg] dimension sanity check failed -- refusing\n"); exit(1);
    }
}

static float *load_f32_ns(shards *S, const char *name, int64_t want) {
    int64_t n = st_numel(S, name);
    if (n < 0) st_die_missing(S, name);
    if (n != want) {
        fprintf(stderr, "%s: %lld elements, config implies %lld -- refusing\n",
                name, (long long)n, (long long)want); exit(1);
    }
    float *p = falloc(n);
    st_read_f32(S, name, p, 0);
    return p;
}
static float *load_f32_n(Model *m, const char *name, int64_t want) {
    return load_f32_ns(&m->S, name, want);
}

/* ==================== P5: MTP module loader ====================
 * Reads the mtp/ container (docs/mtp-container.md) into the extra layer slot
 * L[n_layers].  Tensor names are the source HF names verbatim.  Enabled by
 * Q38_MTP=1; any missing piece disables cleanly with a notice (has_mtp=0
 * leaves the engine byte-identical to Q38_MTP unset). */
static void mtp_load(Model *m, const char *snap) {
    Cfg *c = &m->c;
    const char *en = getenv("Q38_MTP");
    if (!en || atoi(en) != 1) return;
    char dir[2048], pb[2304];
    const char *d = getenv("Q38_MTP_DIR");
    if (d && *d) snprintf(dir, sizeof dir, "%s", d);
    else snprintf(dir, sizeof dir, "%s/mtp", snap);
    snprintf(pb, sizeof pb, "%s/mtp_meta.json", dir);
    struct stat sb;
    if (stat(pb, &sb) != 0) {
        fprintf(stderr, "[mtp] Q38_MTP=1 but %s missing -> MTP disabled\n", pb);
        return;
    }
    if (!m->have_mmap) {
        fprintf(stderr, "[mtp] expert mmap unavailable -> MTP disabled\n");
        return;
    }
    double t0 = now_s();
    /* The MTP container keeps HF weights VERBATIM (docs/mtp-container.md).
     * Qwen3-Next RMSNorm gammas are zero-centered (effective gamma = 1 + w);
     * the backbone came from a GGUF where the conversion had already folded
     * the +1, so the engine's rmsnorm_plain expects folded weights.  Fold
     * every MTP norm gamma here (norms only -- inject/gate/router are used
     * as stored). */
    #define MTP_FOLD1(p, n) do { for (int64_t _i = 0; _i < (int64_t)(n); _i++) (p)[_i] += 1.f; } while (0)
    st_init(&m->mtpS, dir);
    shards *S = &m->mtpS;
    int D = c->hidden, W = HC * D, ML = c->n_layers;   /* MTP layer slot */
    Layer *l = &m->L[ML];

    /* fold + output mixer */
    rm_load(S, &m->mtp_fce, "mtp.fc_embedding.weight", D, D);
    rm_load(S, &m->mtp_fch, "mtp.fc_hidden.weight",    D, D);
    m->mtp_nfe = load_f32_ns(S, "mtp.pre_fc_norm_embedding.weight", D);
    m->mtp_nfh = load_f32_ns(S, "mtp.pre_fc_norm_hidden.weight", W);
    MTP_FOLD1(m->mtp_nfe, D); MTP_FOLD1(m->mtp_nfh, W);
    rm_load(S, &m->mtp_mix_down, "mtp.hyper_connection_mixer.input_mix_weight_down.weight", c->hc_rank, W);
    rm_load(S, &m->mtp_mix_up,   "mtp.hyper_connection_mixer.input_mix_weight_up.weight",   W, c->hc_rank);
    m->mtp_mix_norm = load_f32_ns(S, "mtp.hyper_connection_mixer.hc_norm.weight", W);
    MTP_FOLD1(m->mtp_mix_norm, W);

    /* the decoder layer, mapped onto the backbone Layer struct */
    rm_load(S, &l->hca_down, "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", c->hc_rank, W);
    rm_load(S, &l->hca_up,   "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",   W, c->hc_rank);
    l->hca_norm = load_f32_ns(S, "mtp.layers.0.attn_hyper_connection.hc_norm.weight", W);
    l->hca_inj  = load_f32_ns(S, "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", (int64_t)HC*W);
    rm_load(S, &l->hcf_down, "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", c->hc_rank, W);
    rm_load(S, &l->hcf_up,   "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",   W, c->hc_rank);
    l->hcf_norm = load_f32_ns(S, "mtp.layers.0.mlp_hyper_connection.hc_norm.weight", W);
    l->hcf_inj  = load_f32_ns(S, "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", (int64_t)HC*W);
    rm_load(S, &l->q, "mtp.layers.0.self_attn.q_proj.weight", c->q_heads * 2 * c->head_dim, D);
    rm_load(S, &l->k, "mtp.layers.0.self_attn.k_proj.weight", c->kv_heads * c->head_dim, D);
    rm_load(S, &l->v, "mtp.layers.0.self_attn.v_proj.weight", c->kv_heads * c->head_dim, D);
    rm_load(S, &l->o, "mtp.layers.0.self_attn.o_proj.weight", D, c->q_heads * c->head_dim);
    l->qn = load_f32_ns(S, "mtp.layers.0.self_attn.q_norm.weight", c->head_dim);
    l->kn = load_f32_ns(S, "mtp.layers.0.self_attn.k_norm.weight", c->head_dim);
    MTP_FOLD1(l->hca_norm, W); MTP_FOLD1(l->hcf_norm, W);
    MTP_FOLD1(l->qn, c->head_dim); MTP_FOLD1(l->kn, c->head_dim);
    /* indexer tensors (index_qk_proj [640,2560] fused q4x128|k128) exist in
     * the container but are NOT loaded: draft attention runs dense (v1; the
     * top-k budget covers every validated context anyway, and drafts carry no
     * correctness weight -- verification does). */
    l->ixq = l->ixk = l->ixqn = l->ixkn = NULL;
    l->gate    = load_f32_ns(S, "mtp.layers.0.mlp.gate.weight", (int64_t)c->n_experts * D);
    l->sh_gate = load_f32_ns(S, "mtp.layers.0.mlp.shared_expert_gate.weight", D);
    rm_load(S, &l->sh_g, "mtp.layers.0.mlp.shared_expert.gate_proj.weight", c->shared_inter, D);
    rm_load(S, &l->sh_u, "mtp.layers.0.mlp.shared_expert.up_proj.weight",   c->shared_inter, D);
    rm_load(S, &l->sh_d, "mtp.layers.0.mlp.shared_expert.down_proj.weight", D, c->shared_inter);

    /* experts: mmap the mtp shards, register raw-block pointers at slot ML */
    for (int f = 0; f < S->nfd && f < 8; f++) {
        void *mp = mmap(NULL, (size_t)S->sizes[f], PROT_READ, MAP_SHARED, S->fds[f], 0);
        if (mp == MAP_FAILED) {
            fprintf(stderr, "[mtp] shard mmap failed -> MTP disabled\n");
            return;
        }
        m->mtp_map[f] = mp; m->mtp_map_len[f] = (size_t)S->sizes[f];
    }
    char nm[128];
    int64_t gu_nb = -1, d_nb = -1;
    for (int e = 0; e < c->n_experts; e++) {
        static const char *sfx[3] = { "gate_raw", "up_raw", "down_raw" };
        const uint8_t **dstv[3];
        dstv[0] = &m->e_g[(int64_t)ML*c->n_experts+e];
        dstv[1] = &m->e_u[(int64_t)ML*c->n_experts+e];
        dstv[2] = &m->e_d[(int64_t)ML*c->n_experts+e];
        for (int wch = 0; wch < 3; wch++) {
            snprintf(nm, sizeof(nm), "mtp.experts.%d.%s", e, sfx[wch]);
            st_tensor *t = st_find(S, nm);
            int fidx = t ? st_fidx(S, t->fd) : -1;
            if (fidx < 0 || fidx >= 8 || !m->mtp_map[fidx]) {
                fprintf(stderr, "[mtp] %s missing -> MTP disabled\n", nm);
                return;
            }
            *dstv[wch] = (const uint8_t*)m->mtp_map[fidx] + t->off;
            if (wch < 2) { if (gu_nb < 0) gu_nb = t->nbytes; else if (t->nbytes != gu_nb) { fprintf(stderr,"[mtp] expert size mismatch\n"); return; } }
            else         { if (d_nb  < 0) d_nb  = t->nbytes; else if (t->nbytes != d_nb ) { fprintf(stderr,"[mtp] expert size mismatch\n"); return; } }
        }
    }
    m->e_gu_type[ML] = rm_infer_type(gu_nb, c->inter, D, "mtp expert gate/up");
    m->e_d_type[ML]  = rm_infer_type(d_nb, D, c->inter, "mtp expert down");

    c->is_attn[ML] = 1;          /* QSA-shaped layer (own KV, no GDN state) */
    c->ratio[ML] = 0;            /* dense attention, no indexer selection */
    m->mtp_pend = falloc(W);
    m->mtp_pend_pos = -1;
    m->has_mtp = 1;
    fprintf(stderr, "[mtp] module loaded from %s in %.1fs (experts %s/%s, layer slot %d)\n",
            dir, now_s()-t0,
            gu_nb == 1740800 ? "Q8_0" : "?", d_nb == 1740800 ? "Q8_0" : "?", ML);
}

static void model_init(Model *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    m->snap = snap;                 /* for the lazy mmproj load (P6) */
    load_cfg(&m->c, snap);
    Cfg *c = &m->c;
    st_init(&m->S, snap);
    double t0 = now_s();
    int D = c->hidden, W = HC*D;      /* 2560, 10240 */

    rm_load(&m->S, &m->embed,   "model.embed_tokens.weight", c->vocab, D);
    rm_load(&m->S, &m->lm_head, "lm_head.weight",            c->vocab, D);
    rm_load(&m->S, &m->ohc_down,"model.output_hc.down.weight", c->hc_rank, W);
    rm_load(&m->S, &m->ohc_up,  "model.output_hc.up.weight",   W, c->hc_rank);
    m->ohc_norm = load_f32_n(m, "model.output_hc.norm.weight", W);

    rm_load(&m->S, &m->ple_key,   "model.ple.key.weight",   W, D);
    rm_load(&m->S, &m->ple_value, "model.ple.value.weight", D, D);
    m->ple_conv = load_f32_n(m, "model.ple.conv1d.weight", (int64_t)W * c->ple_convk);
    m->ple_nk = load_f32_n(m, "model.ple.norm_key.weight", W);
    m->ple_nq = load_f32_n(m, "model.ple.norm_query.weight", W);
    m->ple_nc = load_f32_n(m, "model.ple.norm_conv.weight", W);

    m->L = calloc(c->n_layers + 1, sizeof(Layer));   /* +1: P5 MTP layer slot */
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define NM(suffix) (snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i), nm)
        rm_load(&m->S, &l->hca_down, NM("hc_attn.down.weight"), c->hc_rank, W);
        rm_load(&m->S, &l->hca_up,   NM("hc_attn.up.weight"),   W, c->hc_rank);
        l->hca_norm = load_f32_n(m, NM("hc_attn.norm.weight"), W);
        l->hca_inj  = load_f32_n(m, NM("hc_attn.inject.weight"), (int64_t)HC*W);
        rm_load(&m->S, &l->hcf_down, NM("hc_ffn.down.weight"), c->hc_rank, W);
        rm_load(&m->S, &l->hcf_up,   NM("hc_ffn.up.weight"),   W, c->hc_rank);
        l->hcf_norm = load_f32_n(m, NM("hc_ffn.norm.weight"), W);
        l->hcf_inj  = load_f32_n(m, NM("hc_ffn.inject.weight"), (int64_t)HC*W);
        l->gate    = load_f32_n(m, NM("mlp.gate.weight"), (int64_t)c->n_experts * D);
        l->sh_gate = load_f32_n(m, NM("mlp.shared_expert_gate.weight"), D);
        rm_load(&m->S, &l->sh_g, NM("mlp.shared_expert.gate_proj.weight"), c->shared_inter, D);
        rm_load(&m->S, &l->sh_u, NM("mlp.shared_expert.up_proj.weight"),   c->shared_inter, D);
        rm_load(&m->S, &l->sh_d, NM("mlp.shared_expert.down_proj.weight"), D, c->shared_inter);
        if (c->is_attn[i]) {
            rm_load(&m->S, &l->q, NM("self_attn.q_proj.weight"), c->q_heads * 2 * c->head_dim, D);
            rm_load(&m->S, &l->k, NM("self_attn.k_proj.weight"), c->kv_heads * c->head_dim, D);
            rm_load(&m->S, &l->v, NM("self_attn.v_proj.weight"), c->kv_heads * c->head_dim, D);
            rm_load(&m->S, &l->o, NM("self_attn.o_proj.weight"), D, c->q_heads * c->head_dim);
            l->qn = load_f32_n(m, NM("self_attn.q_norm.weight"), c->head_dim);
            l->kn = load_f32_n(m, NM("self_attn.k_norm.weight"), c->head_dim);
            /* S3: indexer tensors (BF16/F32 -> f32).  Absent tensors or
             * ratio == 0 leave the layer dense, as in llama.cpp. */
            if (c->ratio[i] > 0 &&
                st_numel(&m->S, NM("self_attn.indexer.q_proj.weight")) > 0) {
                l->ixq  = load_f32_n(m, NM("self_attn.indexer.q_proj.weight"),
                                     (int64_t)c->idx_heads * c->idx_dim * D);
                l->ixk  = load_f32_n(m, NM("self_attn.indexer.k_proj.weight"),
                                     (int64_t)c->idx_dim * D);
                l->ixqn = load_f32_n(m, NM("self_attn.indexer.q_norm.weight"), c->idx_dim);
                l->ixkn = load_f32_n(m, NM("self_attn.indexer.k_norm.weight"), c->idx_dim);
            }
        } else {
            int vtot = c->dn_vheads * c->dn_vdim;
            rm_load(&m->S, &l->dn_qkv, NM("linear_attn.in_proj_qkv.weight"), c->dn_conv_dim, D);
            rm_load(&m->S, &l->dn_z,   NM("linear_attn.in_proj_z.weight"),   vtot, D);
            rm_load(&m->S, &l->dn_out, NM("linear_attn.out_proj.weight"),    D, vtot);
            l->dn_b     = load_f32_n(m, NM("linear_attn.in_proj_b.weight"),     (int64_t)c->dn_vheads * D);
            l->dn_alpha = load_f32_n(m, NM("linear_attn.in_proj_alpha.weight"), (int64_t)c->dn_vheads * D);
            l->dn_a      = load_f32_n(m, NM("linear_attn.in_proj_a.weight"), c->dn_vheads);
            l->dn_dtbias = load_f32_n(m, NM("linear_attn.dt_bias"), c->dn_vheads);
            l->dn_norm   = load_f32_n(m, NM("linear_attn.norm.weight"), c->dn_vdim);
            l->dn_conv   = load_f32_n(m, NM("linear_attn.conv1d.weight"), (int64_t)c->dn_conv_dim * c->dn_convk);
        }
        #undef NM
    }

    /* PLE table mmap (the one sanctioned mmap exception) */
    {
        char path[2048]; snprintf(path, sizeof(path), "%s/ple_table.bin", snap);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); exit(1); }
        struct stat sb; fstat(fd, &sb);
        int64_t want = 64 + (int64_t)c->ple_rows * 90;
        if (sb.st_size < want) { fprintf(stderr, "%s: %lld bytes, need %lld\n", path, (long long)sb.st_size, (long long)want); exit(1); }
        void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { perror("mmap ple_table"); exit(1); }
        madvise(map, (size_t)sb.st_size, MADV_RANDOM);
        const uint8_t *h = (const uint8_t*)map;
        uint32_t qtype; uint64_t nrows; uint32_t rdim, rbytes;
        memcpy(&qtype, h+12, 4); memcpy(&nrows, h+16, 8); memcpy(&rdim, h+24, 4); memcpy(&rbytes, h+28, 4);
        if (memcmp(h, "COLIPLE1", 8) != 0 || qtype != 20 || nrows != (uint64_t)c->ple_rows ||
            rdim != (uint32_t)c->ple_dim || rbytes != 90) {
            fprintf(stderr, "%s: bad COLIPLE1 header\n", path); exit(1);
        }
        m->ple_table = h;
        m->ple_map_len = (size_t)sb.st_size;
        close(fd);
    }

    /* S2: mmap every shard once and resolve expert raw-block pointers +
     * per-layer GGML types.  Byte-identical to the S1 st_read_raw path (same
     * bytes, same decoders) but without the per-token pread/copy, and it
     * gives the CUDA tier stable RAM sources.  Any failure -> S1 path. */
    m->have_mmap = 1;
    for (int f = 0; f < m->S.nfd && f < 512; f++) {
        void *mp = mmap(NULL, (size_t)m->S.sizes[f], PROT_READ, MAP_SHARED, m->S.fds[f], 0);
        if (mp == MAP_FAILED) { m->have_mmap = 0; break; }
        m->shard_map[f] = mp; m->shard_len[f] = (size_t)m->S.sizes[f];
    }
    if (m->have_mmap) {
        int64_t nslot = (int64_t)(c->n_layers + 1) * c->n_experts;  /* +1: MTP */
        m->e_g = calloc(nslot, sizeof(uint8_t*));
        m->e_u = calloc(nslot, sizeof(uint8_t*));
        m->e_d = calloc(nslot, sizeof(uint8_t*));
        m->e_gu_type = calloc(c->n_layers + 1, sizeof(int));
        m->e_d_type  = calloc(c->n_layers + 1, sizeof(int));
        if (!m->e_g || !m->e_u || !m->e_d || !m->e_gu_type || !m->e_d_type) { fprintf(stderr,"OOM expert map\n"); exit(1); }
        for (int i = 0; i < c->n_layers && m->have_mmap; i++) {
            int64_t gu_nb = -1, d_nb = -1;
            for (int e = 0; e < c->n_experts; e++) {
                static const char *sfx[3] = { "gate_raw", "up_raw", "down_raw" };
                const uint8_t **dstv[3];
                dstv[0] = &m->e_g[(int64_t)i*c->n_experts+e];
                dstv[1] = &m->e_u[(int64_t)i*c->n_experts+e];
                dstv[2] = &m->e_d[(int64_t)i*c->n_experts+e];
                for (int wch = 0; wch < 3; wch++) {
                    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.%s", i, e, sfx[wch]);
                    st_tensor *t = st_find(&m->S, nm);
                    int fidx = t ? st_fidx(&m->S, t->fd) : -1;
                    if (fidx < 0 || !m->shard_map[fidx]) { m->have_mmap = 0; break; }
                    *dstv[wch] = (const uint8_t*)m->shard_map[fidx] + t->off;
                    if (wch < 2) { if (gu_nb < 0) gu_nb = t->nbytes; else if (t->nbytes != gu_nb) m->have_mmap = 0; }
                    else         { if (d_nb  < 0) d_nb  = t->nbytes; else if (t->nbytes != d_nb ) m->have_mmap = 0; }
                }
                if (!m->have_mmap) break;
            }
            if (m->have_mmap) {
                m->e_gu_type[i] = rm_infer_type(gu_nb, c->inter, D, "expert gate/up");
                m->e_d_type[i]  = rm_infer_type(d_nb, D, c->inter, "expert down");
            }
        }
    }
    if (!m->have_mmap) fprintf(stderr, "[qwen38] expert mmap unavailable -> S1 per-call read path\n");

    /* recurrent state */
    m->DN_rec  = calloc(c->n_layers, sizeof(float*));
    m->DN_ring = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        m->DN_rec[i]  = fcalloc((int64_t)c->dn_vheads * c->dn_kdim * c->dn_vdim);
        m->DN_ring[i] = fcalloc((int64_t)c->dn_conv_dim * (c->dn_convk - 1));
    }
    m->ple_hist = fcalloc((int64_t)(c->ple_convk-1) * c->ple_ngram * W);   /* 9 x 10240 */
    mtp_load(m, snap);                                /* P5: no-op unless Q38_MTP=1 */
    m->dense_load_s = now_s() - t0;
}

static void reset_state(Model *m) {
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_ring[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk-1) * sizeof(float));
    }
    memset(m->ple_hist, 0, (size_t)(c->ple_convk-1) * c->ple_ngram * HC * c->hidden * sizeof(float));
    m->ple_hist_n = 0;
    m->n_toks = 0;
    if (m->nblk) memset(m->nblk, 0, (size_t)(c->n_layers + 1) * sizeof(int));
    m->mtp_pend_pos = -1;                     /* P5: no folded hidden pending */
    q38g_qsa_kv_invalidate(0);                /* P9: device KV mirror is stale */
    q38g_idx_invalidate(0);
}

static void ensure_kv(Model *m, int max_t) {
    Cfg *c = &m->c;
    int NL = c->n_layers + 1;                 /* +1: P5 MTP layer slot */
    if (m->max_t >= max_t && (m->K || m->K8)) return;
    if (m->K)  { for (int i = 0; i < NL; i++) { free(m->K[i]);  free(m->V[i]);  } free(m->K);  free(m->V);  m->K = m->V = NULL; }
    if (m->K8) { for (int i = 0; i < NL; i++) { free(m->K8[i]); free(m->V8[i]);
                                                free(m->K8s[i]); free(m->V8s[i]); }
                 free(m->K8); free(m->V8); free(m->K8s); free(m->V8s); m->K8 = m->V8 = NULL; m->K8s = m->V8s = NULL; }
    if (m->IK) { for (int i = 0; i < NL; i++) { free(m->IK[i]); free(m->IBK[i]); }
                 free(m->IK); free(m->IBK); free(m->nblk); m->IK = m->IBK = NULL; m->nblk = NULL; }
    m->max_t = max_t;
    if (m->kv_q8) {
        m->K8  = calloc(NL, sizeof(int8_t*));
        m->V8  = calloc(NL, sizeof(int8_t*));
        m->K8s = calloc(NL, sizeof(float*));
        m->V8s = calloc(NL, sizeof(float*));
    } else {
        m->K = calloc(NL, sizeof(float*));
        m->V = calloc(NL, sizeof(float*));
    }
    m->IK   = calloc(NL, sizeof(float*));
    m->IBK  = calloc(NL, sizeof(float*));
    m->nblk = calloc(NL, sizeof(int));
    for (int i = 0; i < c->n_layers + m->has_mtp; i++) {
        if (!c->is_attn[i]) continue;
        int64_t rows = (int64_t)c->kv_heads * max_t;
        if (m->kv_q8) {
            int ns = c->head_dim / 32;
            m->K8[i]  = malloc((size_t)rows * c->head_dim);
            m->V8[i]  = malloc((size_t)rows * c->head_dim);
            m->K8s[i] = falloc(rows * ns);
            m->V8s[i] = falloc(rows * ns);
            if (!m->K8[i] || !m->V8[i]) { fprintf(stderr,"OOM q8 KV\n"); exit(1); }
        } else {
            m->K[i] = falloc(rows * c->head_dim);
            m->V[i] = falloc(rows * c->head_dim);
        }
        if (m->L[i].ixk && c->ratio[i] > 0) {
            m->IK[i]  = falloc((int64_t)max_t * c->idx_dim);
            m->IBK[i] = falloc((int64_t)(max_t / c->ratio[i] + 1) * c->idx_dim);
        }
    }
    free(m->toks);
    m->toks = malloc((size_t)max_t * sizeof(int));
}

/* P3: prompt/KV cache pool -- full-state checkpoints keyed by the fed token
 * ids, restored when a new prompt extends a stored prefix (serve mode). */
#include "qwen38_pool.h"

/* ==================== hyper-connections ==================== */
/* mix: res_hc [4][2560] -> mixed [2560] + inject [4] for the combine.
 * xn = per-stream rmsnorm(res, eps) * w_norm[10240] (plain gamma, folded).
 * lo = silu((down @ xn)/4); gate = sigmoid(up @ lo);
 * mixed = mean_s(xn*gate); inject = w_inject[4,10240] @ xn. */
static void hc_mix(const Cfg *c, const float *res, const RawMat *down, const RawMat *up,
                   const float *w_norm, const float *w_inj, float *mixed, float *inject,
                   float *xn /* scratch [W] */, float *lo /* [rank] */, float *gate /* [W] */) {
    int D = c->hidden, W = HC*D;
    for (int s = 0; s < HC; s++)
        rmsnorm_plain(xn + s*D, res + s*D, NULL, D, c->eps);
    for (int j = 0; j < W; j++) xn[j] *= w_norm[j];
    rm_gemv(lo, down, xn);
    for (int r = 0; r < c->hc_rank; r++) lo[r] = siluf_(lo[r] / (float)HC);
    rm_gemv(gate, up, lo);
    for (int j = 0; j < W; j++) gate[j] = sigmoidf_(gate[j]);
    for (int j = 0; j < D; j++) {
        float acc = 0.f;
        for (int s = 0; s < HC; s++) acc += xn[s*D+j] * gate[s*D+j];
        mixed[j] = acc / (float)HC;
    }
    if (inject && w_inj) matf(inject, w_inj, xn, W, HC);
}
/* combine: res_hc[s] += out * 2*sigmoid(inject[s]/4) */
static void hc_combine(const Cfg *c, float *res, const float *out, const float *inject) {
    int D = c->hidden;
    for (int s = 0; s < HC; s++) {
        float w = 2.f * sigmoidf_(inject[s] / (float)HC);
        float *r = res + s*D;
        for (int j = 0; j < D; j++) r[j] += out[j] * w;
    }
}

/* S3 batched mix over a prefill chunk: the two skinny GEMVs run as chunk
 * GEMMs, everything else loops tokens with the exact hc_mix math. */
static void hc_mix_b(const Cfg *c, const float *RES, const RawMat *down, const RawMat *up,
                     const float *w_norm, const float *w_inj, float *MIXED, float *INJECT,
                     int B) {
    int D = c->hidden, W = HC*D, R = c->hc_rank;
    float *XN = falloc((int64_t)B*W), *LO = falloc((int64_t)B*R), *GT = falloc((int64_t)B*W);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < B; t++) {
        float *xn = XN + (int64_t)t*W;
        const float *res = RES + (int64_t)t*W;
        for (int s = 0; s < HC; s++)
            rmsnorm_plain(xn + s*D, res + s*D, NULL, D, c->eps);
        for (int j = 0; j < W; j++) xn[j] *= w_norm[j];
    }
    rm_gemv_b(LO, down, XN, B, R);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < B; t++) {
        float *lo = LO + (int64_t)t*R;
        for (int r = 0; r < R; r++) lo[r] = siluf_(lo[r] / (float)HC);
    }
    rm_gemv_b(GT, up, LO, B, W);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < B; t++) {
        float *gate = GT + (int64_t)t*W, *xn = XN + (int64_t)t*W;
        float *mixed = MIXED + (int64_t)t*D;
        for (int j = 0; j < W; j++) gate[j] = sigmoidf_(gate[j]);
        for (int j = 0; j < D; j++) {
            float acc = 0.f;
            for (int s = 0; s < HC; s++) acc += xn[s*D+j] * gate[s*D+j];
            mixed[j] = acc / (float)HC;
        }
        if (INJECT && w_inj) {
            float *inj = INJECT + (int64_t)t*HC;
            for (int o = 0; o < HC; o++) {              /* matf, inlined per token */
                const float *w = w_inj + (int64_t)o * W;
                float acc = 0.f;
                for (int i = 0; i < W; i++) acc += w[i] * xn[i];
                inj[o] = acc;
            }
        }
    }
    free(XN); free(LO); free(GT);
}

/* ==================== GDN layer ==================== */
/* core for one token, the four input projections precomputed (shared by the
 * decode path and the batched prefill; per-token math identical). */
static void gdn_core(Model *m, Layer *l, int layer, const float *qkv, const float *z,
                     const float *b, const float *alpha, float *out) {
    Cfg *c = &m->c;
    int vh = c->dn_vheads, vk = c->dn_kheads, kd = c->dn_kdim, vd = c->dn_vdim;
    int convk = c->dn_convk, cdim = c->dn_conv_dim;
    int rep = vh / vk;
    int keytot = vk * kd;              /* 2048 */
    int vtot = vh * vd;                /* 6144 */
    float scale = 1.f / sqrtf((float)kd);

    float *beta = falloc(vh), *gg = falloc(vh);
    float *conv = falloc(cdim);
    float *q = falloc((int64_t)vh*kd), *k = falloc((int64_t)vh*kd);
    float *outv = falloc(vtot), *outr = falloc(vtot);

    for (int h = 0; h < vh; h++) {
        beta[h] = sigmoidf_(b[h]);
        gg[h] = l->dn_a[h] * softplus_f(alpha[h] + l->dn_dtbias[h]);   /* a = -exp(A_log) */
    }
    if (g_dbg_now == layer) {
        dbg_row("qkv_mixed", qkv, cdim); dbg_row("z", z, vtot);
        fprintf(stderr, "[dbg] beta_sig[0]=%.4f alpha=[%.4f %.4f %.4f %.4f %.4f %.4f]\n",
                beta[0], alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5]);
        fprintf(stderr, "[dbg] gate(g)=[%.4f %.4f %.4f %.4f %.4f %.4f]\n", gg[0],gg[1],gg[2],gg[3],gg[4],gg[5]);
    }
    /* causal depthwise conv over 10240 channels, kernel 4, ring state */
    float *ring = m->DN_ring[layer];
    for (int cc = 0; cc < cdim; cc++) {
        const float *w = l->dn_conv + (int64_t)cc * convk;
        float *rg = ring + (int64_t)cc * (convk - 1);
        float acc = 0.f;
        for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
        acc += w[convk - 1] * qkv[cc];
        conv[cc] = siluf_(acc);
        for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
        rg[convk - 2] = qkv[cc];
    }
    if (g_dbg_now == layer) dbg_row("conv_silu", conv, cdim);
    /* split + L2 norm per k/q head, broadcast 16 -> 48.  NOTE: llama.cpp uses
     * ggml_repeat_4d, which TILES the head axis: v-head h reads k-head h % 16
     * (not the HF repeat_interleave h/3 that qwen36's DeltaNet uses). */
    const float *q_in = conv, *k_in = conv + keytot, *v_in = conv + 2*keytot;
    (void)rep;
    for (int h = 0; h < vh; h++) {
        int kh = h % vk;
        memcpy(q + (int64_t)h*kd, q_in + (int64_t)kh*kd, kd*sizeof(float));
        memcpy(k + (int64_t)h*kd, k_in + (int64_t)kh*kd, kd*sizeof(float));
    }
    for (int h = 0; h < vh; h++) {
        float *qh = q + (int64_t)h*kd, *kh = k + (int64_t)h*kd;
        double sq = 0, sk = 0;
        for (int d = 0; d < kd; d++) { sq += (double)qh[d]*qh[d]; sk += (double)kh[d]*kh[d]; }
        float nq = fmaxf(sqrtf((float)sq), c->eps), nk = fmaxf(sqrtf((float)sk), c->eps);
        float sq_inv = scale / nq, sk_inv = 1.f / nk;
        for (int d = 0; d < kd; d++) { qh[d] *= sq_inv; kh[d] *= sk_inv; }
    }
    if (g_dbg_now == layer) { dbg_row("k_predelta", k, vh*kd); dbg_row("v_predelta", v_in, vtot); }
    /* gated delta rule per v-head */
    float *rec = m->DN_rec[layer];
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < vh; h++) {
        float kvl[512], dl[512];
        float *Sh = rec + (int64_t)h * kd * vd;
        float egh = expf(gg[h]);
        const float *kh = k + (int64_t)h*kd;
        const float *vd_ = v_in + (int64_t)h*vd;
        /* two fused state sweeps (element-wise identical to the S1 4-pass
         * version, ~2/3 of the DRAM traffic over the 128x128 states):
         * sweep 1: S *= exp(g) fused with kvl += k*S
         * sweep 2: S += k*dl fused with o += q*S */
        for (int v2 = 0; v2 < vd; v2++) kvl[v2] = 0.f;
        for (int kk = 0; kk < kd; kk++) {
            float kv_ = kh[kk]; float *Sr = Sh + (int64_t)kk*vd;
            for (int v2 = 0; v2 < vd; v2++) { Sr[v2] *= egh; kvl[v2] += kv_ * Sr[v2]; }
        }
        for (int v2 = 0; v2 < vd; v2++) dl[v2] = (vd_[v2] - kvl[v2]) * beta[h];
        const float *qh = q + (int64_t)h*kd;
        float *ov = outv + (int64_t)h*vd;
        for (int v2 = 0; v2 < vd; v2++) ov[v2] = 0.f;
        for (int kk = 0; kk < kd; kk++) {
            float kv_ = kh[kk], qv = qh[kk]; float *Sr = Sh + (int64_t)kk*vd;
            for (int v2 = 0; v2 < vd; v2++) { Sr[v2] += kv_ * dl[v2]; ov[v2] += qv * Sr[v2]; }
        }
    }
    /* per-head RMS norm (plain weight) * SIGMOID(z)  (the qwen4exp diff) */
    for (int h = 0; h < vh; h++) {
        const float *o = outv + (int64_t)h*vd;
        const float *zr = z + (int64_t)h*vd;
        float *orr = outr + (int64_t)h*vd;
        double ms = 0; for (int d = 0; d < vd; d++) ms += (double)o[d]*o[d];
        float r = 1.f / sqrtf((float)(ms/vd) + c->eps);
        for (int d = 0; d < vd; d++)
            orr[d] = o[d] * r * l->dn_norm[d] * sigmoidf_(zr[d]);
    }
    if (g_dbg_now == layer) dbg_row("final_output", outr, vtot);
    rm_gemv(out, &l->dn_out, outr);
    if (g_dbg_now == layer) dbg_row("linear_attn_out", out, c->hidden);
    free(beta); free(gg);
    free(conv); free(q); free(k); free(outv); free(outr);
}

static void gdn_forward(Model *m, Layer *l, int layer, const float *x, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, vh = c->dn_vheads;
    int cdim = c->dn_conv_dim, vtot = vh * c->dn_vdim;
    float *qkv = falloc(cdim), *z = falloc(vtot);
    float *b = falloc(vh), *alpha = falloc(vh);
    rm_gemv(qkv, &l->dn_qkv, x);
    rm_gemv(z,   &l->dn_z,   x);
    matf(b,     l->dn_b,     x, D, vh);
    matf(alpha, l->dn_alpha, x, D, vh);
    gdn_core(m, l, layer, qkv, z, b, alpha, out);
    free(qkv); free(z); free(b); free(alpha);
}

/* ==================== QSA layer (S1: full causal GQA) ==================== */
/* NEOX-style rope on the first n_rot dims of a head: pair (j, j+n_rot/2),
 * angle = pos * theta^(-2j/n_rot).  IMRoPE with equal t/y/x positions (text)
 * reduces to exactly this regardless of the section interleaving. */
static void rope_head(float *x, int pos, int n_rot, float theta) {
    int h = n_rot / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / n_rot);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b2 = x[j+h];
        x[j]   = a*cs - b2*sn;
        x[j+h] = b2*cs + a*sn;
    }
}

/* P6: rope for the QSA query/key (and indexer query) of the cell at `cell`.
 * Text-only turns carry no m->mp and take rope_head unchanged (bit-identical
 * path).  With an image in the prompt, m->mp holds each cell's (t,y,x) and
 * the interleaved IMRoPE section rule picks which position rotates each pair
 * (ggml_mrope_cache_init, is_imrope: pair p uses y if p%3==1 && p<3*sec[1],
 * x if p%3==2 && p<3*sec[2], else t; sections [11,11,10,0] leave the fourth
 * section empty).  Equal t==y==x reduces exactly to rope_head, which is why
 * decode cells after an image can keep the scalar path. */
static void rope_head_m(const Model *m, float *x, int cell) {
    const Cfg *c = &m->c;
    if (!m->mp) { rope_head(x, cell, c->n_rot, c->theta); return; }
    int pt = m->mp[3*cell], py = m->mp[3*cell+1], px = m->mp[3*cell+2];
    if (pt == py && py == px) { rope_head(x, pt, c->n_rot, c->theta); return; }
    int h = c->n_rot / 2;
    for (int j = 0; j < h; j++) {
        int pos = pt;
        if      (j % 3 == 1 && j < 3*c->rope_sec[1]) pos = py;
        else if (j % 3 == 2 && j < 3*c->rope_sec[2]) pos = px;
        float inv = powf(c->theta, -2.0f * j / c->n_rot);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b2 = x[j+h];
        x[j]   = a*cs - b2*sn;
        x[j+h] = b2*cs + a*sn;
    }
}

static int g_full_attn = 0;      /* Q38_FULL_ATTN=1: force dense QSA at any depth */
static int g_gpu_qsa;            /* P9: QSA walk/scoring on the device (defined below) */
static int g_gpu_router;         /* P10 D4: Q38_GPU_ROUTER=1 -> gate weights on the device */
static int g_req_router;         /* per request (serve control bit 30) or the env default (CLI) */
static const float *g_pr_pre;    /* router logits precomputed on the device for the next
                                  * moe_forward/moe_forward_b call (NULL = host matf) */
/* Hybrid exactness: the device GEMV's summation order perturbs the logits by
 * ~1e-6; if the top-K cutoff margin (K-th selected logit minus the best
 * unselected one) is below Q38_ROUTER_GAP (default 1e-3), that token's
 * router is recomputed on the host and the selection redone, so the routed
 * set equals the host path's whenever the device error is below the gap. */
static float g_router_gap = -1.f;
static long g_router_tok = 0, g_router_fallback = 0;
/* Q38_ROUTER_CHECK=1: after the (hybrid) device selection, recompute the host
 * router from the same input and count selection mismatches and the max
 * device-vs-host logit error -- the per-decision exactness measurement. */
static void moe_topk_renorm(const float *pr, int E, int K, int *idx, float *val);
static int g_router_check = -1;
static long g_router_checked = 0, g_router_mismatch = 0;
static double g_router_maxerr = 0.0;
static void router_check(const float *lg_dev, const float *w, const float *x, int D, int E, int K, const int *idx) {
    float *h = falloc(E);
    for (int o = 0; o < E; o++) { const float *ww = w + (int64_t)o * D; float acc = 0.f; for (int i = 0; i < D; i++) acc += ww[i] * x[i]; h[o] = acc; }
    double me = 0.0; for (int e = 0; e < E; e++) { double d = fabs((double)h[e] - lg_dev[e]); if (d > me) me = d; }
    softmax_row(h, E);
    int hidx[256]; float hval[256]; moe_topk_renorm(h, E, K, hidx, hval);
    int mism = 0;
    for (int kk = 0; kk < K; kk++) { int f = 0; for (int j = 0; j < K; j++) if (hidx[j] == idx[kk]) { f = 1; break; } if (!f) mism++; }
    #pragma omp critical
    { g_router_checked++; if (mism) g_router_mismatch++; if (me > g_router_maxerr) g_router_maxerr = me; }
    free(h);
}
static float router_gap_env(void) {
    if (g_router_gap < 0.f) { const char *e = getenv("Q38_ROUTER_GAP"); g_router_gap = e ? (float)atof(e) : 1e-3f; }
    return g_router_gap;
}
/* margin between the K-th selected logit and the best unselected logit */
static float router_margin(const float *logit, int E, const int *idx, int K) {
    float lk = 3.4e38f;
    for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0 && logit[idx[kk]] < lk) lk = logit[idx[kk]];
    float best = -3.4e38f;
    for (int e = 0; e < E; e++) {
        int sel = 0; for (int kk = 0; kk < K; kk++) if (idx[kk] == e) { sel = 1; break; }
        if (!sel && logit[e] > best) best = logit[e];
    }
    return lk - best;
}
static int g_ms_decode = 0;      /* Q38_MS_DECODE=1: decode MoE through the multi
                                  * path (GPU miss staging; opt-in, not S2-exact) */
static int g_idx_check_pos = -1; /* Q38_IDX_CHECK=1: bruteforce selection recheck at this pos */

/* q8_0-style row quantization: one f32 scale per 32-dim group */
static void kv_q8_row(const float *x, int n, int8_t *q, float *s) {
    for (int g = 0; g < n/32; g++) {
        const float *xb = x + g*32;
        float mx = 0.f;
        for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > mx) mx = a; }
        float sc = mx > 0.f ? mx / 127.f : 0.f, inv = mx > 0.f ? 127.f / mx : 0.f;
        s[g] = sc;
        for (int i = 0; i < 32; i++) q[g*32+i] = (int8_t)lrintf(xb[i] * inv);
    }
}

/* mean-pool block b's raw indexer keys, then rms_norm + rope at pos b*r
 * (RoPE strictly AFTER pooling, reference §3.2). */
static void idx_pool_block(Model *m, Layer *l, int layer, int b, float *out) {
    Cfg *c = &m->c;
    int r = c->ratio[layer], id = c->idx_dim;
    const float *ik = m->IK[layer];
    for (int d = 0; d < id; d++) {
        float acc = ik[(int64_t)(b*r)*id + d];
        for (int j = 1; j < r; j++) acc += ik[(int64_t)(b*r+j)*id + d];
        out[d] = acc / (float)r;
    }
    rmsnorm_plain(out, out, l->ixkn, id, c->eps);
    rope_head(out, b*r, c->n_rot, c->theta);
}

/* sort helper: block order by (score desc, index asc) -- deterministic where
 * ggml's std::partial_sort leaves equal scores unspecified */
static __thread const float *g_blk_score;   /* per thread: the batched selection runs tokens in parallel */
static int cmp_blk_desc(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    float sa = g_blk_score[ia], sb = g_blk_score[ib];
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return ia - ib;
}
static int cmp_int_asc(const void *a, const void *b) { return *(const int*)a - *(const int*)b; }

/* partial selection under cmp_blk_desc's strict total order (score desc,
 * index asc): rearrange ord so the best k live in ord[0..k-1] (unordered).
 * Same set as a full sort -- the order is total -- but O(n) instead of
 * O(n log n) over n_blocks, which dominates qsa time at 128K. */
static void blk_qselect(int *ord, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = ord[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (cmp_blk_desc(&ord[i], &p) < 0) i++;
            while (cmp_blk_desc(&p, &ord[j]) < 0) j--;
            if (i <= j) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
}

/* selection from block scores: the winning whole blocks (score desc, index
 * asc), the boundary block's rem lowest positions, then the tail cells.
 * `ord` is caller scratch of nfb ints.  Shared by the per-token path and the
 * batched prefill path, so both select identically from identical scores. */
static int qsa_select_emit(const float *score, int nfb, int *ord, int r, int pos,
                           int tail_start, int nfull, int rem, int *sel) {
    for (int b = 0; b < nfb; b++) ord[b] = b;
    g_blk_score = score;
    int ksel = nfull + (rem > 0 ? 1 : 0);
    if (ksel > 0 && ksel < nfb) blk_qselect(ord, nfb, ksel);
    /* the boundary block is the worst of the winning set: move it to
     * ord[nfull] where the emit code below expects it */
    if (rem > 0 && nfull < nfb) {
        int bi = 0;
        for (int i = 1; i < ksel; i++) if (cmp_blk_desc(&ord[i], &ord[bi]) > 0) bi = i;
        int t = ord[bi]; ord[bi] = ord[ksel-1]; ord[ksel-1] = t;
    }
    /* winning whole blocks, ascending for a cache-friendly attention walk */
    qsort(ord, (size_t)nfull, sizeof(int), cmp_int_asc);
    int n = 0;
    for (int b = 0; b < nfull; b++)
        for (int j = 0; j < r; j++) sel[n++] = ord[b]*r + j;
    /* the boundary block contributes its rem LOWEST positions (llama.cpp's
     * per-cell top-k leaves the equal-score choice unspecified; documented) */
    if (rem > 0 && nfull < nfb)
        for (int j = 0; j < rem; j++) sel[n++] = ord[nfull]*r + j;
    if (rem > 0 && nfull < nfb) qsort(sel, (size_t)n, sizeof(int), cmp_int_asc);
    for (int t = tail_start; t <= pos; t++) sel[n++] = t;
    return n;
}

/* Score blocks with the 4-head MQA indexer and select the attention-visible
 * cells for the token at `pos` (input x = the layer's mixed vector).
 * Appends this token's raw key to the cache and pools any block it completes.
 * Returns the number of selected cells written to sel[] (ascending), or -1
 * when the layer must run full attention (below budget / dense / forced). */
static int qsa_select(Model *m, Layer *l, int layer, int pos, const float *x, int *sel) {
    Cfg *c = &m->c;
    int r = c->ratio[layer];
    if (!l->ixk || r <= 0 || !m->IK[layer]) return -1;
    int id = c->idx_dim, nh = c->idx_heads;

    /* append raw key (cached pre-norm, pre-rope); pool completed blocks */
    float *ikrow = m->IK[layer] + (int64_t)pos * id;
    matf(ikrow, l->ixk, x, c->hidden, id);
    while ((m->nblk[layer] + 1) * r <= pos + 1) {
        idx_pool_block(m, l, layer, m->nblk[layer],
                       m->IBK[layer] + (int64_t)m->nblk[layer] * id);
        m->nblk[layer]++;
    }

    int n_kv = pos + 1;
    if (g_full_attn) return -1;
    if (n_kv <= c->idx_topk + r - 1) return -1;   /* budget covers every cell */
    int width = c->idx_topk + r - 1;

    /* 4 query heads: rms_norm then rope at the token position */
    float qi[8*512];
    matf(qi, l->ixq, x, c->hidden, nh*id);
    for (int h = 0; h < nh; h++) {
        float *qh = qi + (int64_t)h*id;
        rmsnorm_plain(qh, qh, l->ixqn, id, c->eps);
        rope_head_m(m, qh, pos);
    }

    /* tail cells [tail_start, pos] are always visible (+1e9 in the ref);
     * every block fully below tail_start is complete in this contiguous
     * single-sequence cache, so the -inf incomplete-block bias never fires
     * outside the tail (the tail cells it would protect are forced in). */
    int tail_start = ((pos + 1) / r) * r;
    int nfb = tail_start / r;                       /* scoreable blocks */
    int cnt_tail = n_kv - tail_start;               /* 0..r-1 */
    int budget = width - cnt_tail;
    int nfull = budget / r, rem = budget % r;
    if (nfull > nfb) { nfull = nfb; rem = 0; }

    float *score = falloc(nfb);
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nfb; b++) {
        const float *kb = m->IBK[layer] + (int64_t)b*id;
        float sacc = 0.f;
        for (int h = 0; h < nh; h++) {
            const float *qh = qi + (int64_t)h*id;
            float d = 0.f;
            for (int j = 0; j < id; j++) d += qh[j]*kb[j];
            if (d > 0.f) sacc += d;                 /* ReLU per head, then sum */
        }
        score[b] = sacc;
    }

    int *ord = malloc((size_t)nfb * sizeof(int));
    int n = qsa_select_emit(score, nfb, ord, r, pos, tail_start, nfull, rem, sel);

    /* Q38_IDX_CHECK: recompute every pooled key + the whole selection from
     * the raw key cache and compare (self-consistency of the incremental
     * bookkeeping vs a bruteforce evaluation of the same formula). */
    if (pos == g_idx_check_pos) {
        float *pool2 = falloc(id);
        int bad_pool = 0;
        for (int b = 0; b < m->nblk[layer]; b++) {
            idx_pool_block(m, l, layer, b, pool2);
            if (memcmp(pool2, m->IBK[layer] + (int64_t)b*id, (size_t)id*sizeof(float))) bad_pool++;
        }
        float *score2 = falloc(nfb);
        for (int b = 0; b < nfb; b++) {
            idx_pool_block(m, l, layer, b, pool2);
            float sacc = 0.f;
            for (int h = 0; h < nh; h++) {
                const float *qh = qi + (int64_t)h*id;
                float d = 0.f;
                for (int j = 0; j < id; j++) d += qh[j]*pool2[j];
                if (d > 0.f) sacc += d;
            }
            score2[b] = sacc;
        }
        int *ord2 = malloc((size_t)nfb * sizeof(int));
        for (int b = 0; b < nfb; b++) ord2[b] = b;
        g_blk_score = score2;
        qsort(ord2, (size_t)nfb, sizeof(int), cmp_blk_desc);
        int *sel2 = malloc((size_t)(n+1) * sizeof(int));
        qsort(ord2, (size_t)nfull, sizeof(int), cmp_int_asc);
        int n2 = 0;
        for (int b = 0; b < nfull; b++)
            for (int j = 0; j < r; j++) sel2[n2++] = ord2[b]*r + j;
        if (rem > 0 && nfull < nfb)
            for (int j = 0; j < rem; j++) sel2[n2++] = ord2[nfull]*r + j;
        if (rem > 0 && nfull < nfb) qsort(sel2, (size_t)n2, sizeof(int), cmp_int_asc);
        for (int t = tail_start; t <= pos; t++) sel2[n2++] = t;
        int bad_sel = (n2 != n);
        if (!bad_sel) for (int j = 0; j < n; j++) if (sel[j] != sel2[j]) { bad_sel = 1; break; }
        fprintf(stderr, "[idx-check] L%02d pos=%d blocks=%d sel=%d tail=%d: pooled %s (%d bad), "
                        "selection %s\n", layer, pos, m->nblk[layer], n, cnt_tail,
                bad_pool ? "MISMATCH" : "ok", bad_pool, bad_sel ? "MISMATCH" : "ok");
        free(pool2); free(score2); free(ord2); free(sel2);
    }

    free(score); free(ord);
    return n;
}

/* P9: selection for a whole prefill chunk.  Raw keys and block pooling
 * advance in position order exactly as the per-token path; every token
 * then scores only the blocks fully below its own tail (the set the
 * sequential path would have had), on the device when the mirror is up
 * (bit-identical scores: same serial per-head dot, same head order) and
 * otherwise on the host; the selection itself is qsa_select_emit either
 * way.  NSEL[t] < 0 means the full walk (below budget / Q38_FULL_ATTN).
 * The Q38_IDX_CHECK bruteforce stays on the per-token path. */
static void qsa_select_batch(Model *m, Layer *l, int layer, int pos0, int B,
                             const float *X, int *NSEL, int *SEL, int sel_cap) {
    Cfg *c = &m->c;
    int r = c->ratio[layer];
    if (!l->ixk || r <= 0 || !m->IK[layer]) { for (int t = 0; t < B; t++) NSEL[t] = -1; return; }
    int id = c->idx_dim, nh = c->idx_heads;
    matf_b(m->IK[layer] + (int64_t)pos0 * id, l->ixk, X, c->hidden, id, B, id);
    int last = pos0 + B - 1;
    while ((m->nblk[layer] + 1) * r <= last + 1) {
        idx_pool_block(m, l, layer, m->nblk[layer],
                       m->IBK[layer] + (int64_t)m->nblk[layer] * id);
        m->nblk[layer]++;
    }
    int *nfb = malloc((size_t)B * sizeof(int)), nfb_max = 0, need = 0;
    for (int t = 0; t < B; t++) {
        int pos = pos0 + t, n_kv = pos + 1;
        nfb[t] = 0; NSEL[t] = -1;
        if (g_full_attn || n_kv <= c->idx_topk + r - 1) continue;
        nfb[t] = ((pos + 1) / r) * r / r;
        if (nfb[t] > nfb_max) nfb_max = nfb[t];
        need = 1;
    }
    if (!need) { free(nfb); return; }
    float *QI = falloc((int64_t)B * nh * id);
    matf_b(QI, l->ixq, X, c->hidden, nh*id, B, nh*id);
    for (int t = 0; t < B; t++) {
        if (!nfb[t]) continue;
        for (int h = 0; h < nh; h++) {
            float *qh = QI + ((int64_t)t*nh + h)*id;
            rmsnorm_plain(qh, qh, l->ixqn, id, c->eps);
            rope_head_m(m, qh, pos0 + t);
        }
    }
    /* P6.3: the queries were roped on the host (rope_head_m), so image
     * prompts score on the device like text */
    const float *score = g_gpu_qsa
        ? q38g_idx_score(layer, m->IBK[layer], m->nblk[layer], QI, nfb, nfb_max, B) : NULL;
    float *hscore = NULL;
    if (!score) {
        hscore = falloc((int64_t)B * nfb_max);
        #pragma omp parallel for schedule(dynamic)
        for (int q = 0; q < B * nfb_max; q++) {
            int t = q / nfb_max, b = q % nfb_max;
            if (b >= nfb[t]) continue;
            const float *kb = m->IBK[layer] + (int64_t)b*id;
            float sacc = 0.f;
            for (int h = 0; h < nh; h++) {
                const float *qh = QI + ((int64_t)t*nh + h)*id;
                float d = 0.f;
                for (int j = 0; j < id; j++) d += qh[j]*kb[j];
                if (d > 0.f) sacc += d;
            }
            hscore[(int64_t)t*nfb_max + b] = sacc;
        }
        score = hscore;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < B; t++) {
        if (!nfb[t]) continue;
        int pos = pos0 + t, n_kv = pos + 1;
        int width = c->idx_topk + r - 1;
        int tail_start = ((pos + 1) / r) * r;
        int cnt_tail = n_kv - tail_start;
        int budget = width - cnt_tail;
        int nfull = budget / r, rem = budget % r;
        if (nfull > nfb[t]) { nfull = nfb[t]; rem = 0; }
        int *ord = malloc((size_t)nfb[t] * sizeof(int));
        NSEL[t] = qsa_select_emit(score + (int64_t)t*nfb_max, nfb[t], ord, r, pos,
                                  tail_start, nfull, rem, SEL + (int64_t)t*sel_cap);
        free(ord);
    }
    free(hscore); free(QI); free(nfb);
}

/* core attention for one token, projections precomputed (shared by the
 * decode path and the batched prefill; per-token math identical). */
/* ctx_out != NULL (P4 GPU path): write the gated context there and leave the
 * o-projection to the caller (it runs on the device); `out` is then unused. */
static void qsa_core_x(Model *m, Layer *l, int layer, const float *x, int pos,
                       float *qfull, float *kv_k, float *kv_v, float *out,
                       float *ctx_out) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int q_per_kv = H / KV, ns = hd / 32;
    float scale = 1.f / sqrtf((float)hd);

    /* per-head [q(256) | gate(256)] interleaved */
    float *query = falloc((int64_t)H * hd);
    float *gate  = falloc((int64_t)H * hd);
    for (int h = 0; h < H; h++) {
        float *qh = query + (int64_t)h*hd;
        rmsnorm_plain(qh, qfull + (int64_t)h*2*hd, l->qn, hd, c->eps);
        rope_head_m(m, qh, pos);
        memcpy(gate + (int64_t)h*hd, qfull + (int64_t)h*2*hd + hd, hd*sizeof(float));
    }
    for (int kvh = 0; kvh < KV; kvh++) {
        float *kh = kv_k + (int64_t)kvh*hd;
        rmsnorm_plain(kh, kh, l->kn, hd, c->eps);
        rope_head_m(m, kh, pos);
        if (m->kv_q8) {
            int64_t row = (int64_t)kvh*m->max_t + pos;
            kv_q8_row(kh, hd, m->K8[layer] + row*hd, m->K8s[layer] + row*ns);
            kv_q8_row(kv_v + (int64_t)kvh*hd, hd, m->V8[layer] + row*hd, m->V8s[layer] + row*ns);
        } else {
            memcpy(m->K[layer] + ((int64_t)kvh*m->max_t + pos)*hd, kh, hd*sizeof(float));
            memcpy(m->V[layer] + ((int64_t)kvh*m->max_t + pos)*hd, kv_v + (int64_t)kvh*hd, hd*sizeof(float));
        }
    }

    /* S3: sparse cell selection (NULL walk = all cells = S1 full attention) */
    int *sel = malloc((size_t)(c->idx_topk + (c->ratio[layer] > 0 ? c->ratio[layer] : 1) + 8) * sizeof(int));
    int nsel = qsa_select(m, l, layer, pos, x, sel);
    int nwalk = nsel < 0 ? pos + 1 : nsel;

    float *ctx = falloc((int64_t)H * hd);
    #pragma omp parallel
    {
        float *sc = malloc((size_t)nwalk * sizeof(float));
        #pragma omp for schedule(static)
        for (int h = 0; h < H; h++) {
            int kvh = h / q_per_kv;
            const float *qv = query + (int64_t)h*hd;
            for (int j = 0; j < nwalk; j++) {
                int t = nsel < 0 ? j : sel[j];
                float acc = 0.f;
                if (m->kv_q8) {
                    int64_t row = (int64_t)kvh*m->max_t + t;
                    const int8_t *kr = m->K8[layer] + row*hd;
                    const float  *ks = m->K8s[layer] + row*ns;
                    for (int g = 0; g < ns; g++) {
                        float p = 0.f;
                        for (int d = 0; d < 32; d++) p += qv[g*32+d]*(float)kr[g*32+d];
                        acc += p * ks[g];
                    }
                } else {
                    const float *kr = m->K[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                    for (int d = 0; d < hd; d++) acc += qv[d]*kr[d];
                }
                sc[j] = acc * scale;
            }
            softmax_row(sc, nwalk);
            float *cx = ctx + (int64_t)h*hd;
            for (int d = 0; d < hd; d++) cx[d] = 0.f;
            for (int j = 0; j < nwalk; j++) {
                int t = nsel < 0 ? j : sel[j];
                float a = sc[j];
                if (m->kv_q8) {
                    int64_t row = (int64_t)kvh*m->max_t + t;
                    const int8_t *vr = m->V8[layer] + row*hd;
                    const float  *vs = m->V8s[layer] + row*ns;
                    for (int g = 0; g < ns; g++) {
                        float as = a * vs[g];
                        for (int d = 0; d < 32; d++) cx[g*32+d] += as * (float)vr[g*32+d];
                    }
                } else {
                    const float *vr = m->V[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                    for (int d = 0; d < hd; d++) cx[d] += a * vr[d];
                }
            }
            /* output gate: sigmoid, per head per dim */
            const float *gh = gate + (int64_t)h*hd;
            for (int d = 0; d < hd; d++) cx[d] *= sigmoidf_(gh[d]);
        }
        free(sc);
    }
    if (ctx_out) memcpy(ctx_out, ctx, (size_t)H * hd * sizeof(float));
    else         rm_gemv(out, &l->o, ctx);
    free(query); free(gate); free(ctx); free(sel);
}

static void qsa_core(Model *m, Layer *l, int layer, const float *x, int pos,
                     float *qfull, float *kv_k, float *kv_v, float *out) {
    qsa_core_x(m, l, layer, x, pos, qfull, kv_k, kv_v, out, NULL);
}

static void qsa_forward(Model *m, Layer *l, int layer, const float *x, int pos, float *out) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    float *qfull = falloc((int64_t)H * 2 * hd);
    float *kv_k = falloc((int64_t)KV * hd);
    float *kv_v = falloc((int64_t)KV * hd);
    rm_gemv(qfull, &l->q, x);
    rm_gemv(kv_k, &l->k, x);
    rm_gemv(kv_v, &l->v, x);
    qsa_core(m, l, layer, x, pos, qfull, kv_k, kv_v, out);
    free(qfull); free(kv_k); free(kv_v);
}

/* ============ routing telemetry: Q38_ROUTE_TRACE (additive, flag-gated) ============
 *
 * Purpose: build an (n-gram -> expert) statistics corpus for a speculative expert
 * prefetcher.  Off unless Q38_ROUTE_TRACE=<path> is set; every entry point below
 * returns immediately when g_rt_fp is NULL, so the flag-off path is byte-identical
 * to the untraced engine (no allocation, no branch inside any inner loop).
 *
 * Why not route_trace.h: that facility emits TEXT lines "<call> <row> <layer>
 * <id>:<gate>..." and owns a [layer][expert] counter table keyed by an engine
 * identity hash.  It carries no absolute position, no token id, and above all no
 * PLE n-gram hash -- which is the whole point here -- and its record is one LAYER,
 * not one TOKEN.  Reshaping it would change the bytes tools/route_pairs.py and the
 * .coli_pairs pipeline already consume.  So: a self-contained binary writer, and
 * route_trace.h is left untouched.
 *
 * ---- file format (also in tools/route_trace_format.md) ----
 * Little-endian, no padding beyond what is written.  32-byte header, then one
 * fixed-size record per token, in generation order (prefill tokens included).
 *
 *   header (32 B):
 *     char     magic[8]   "Q38RT1\0\0"
 *     uint32   version    1
 *     uint32   n_layers   48
 *     uint32   topk       10
 *     uint32   n_experts  512
 *     uint64   reserved   0
 *
 *   record (24 + 2*n_layers*topk bytes = 984 B at 48x10):
 *     uint32   pos              absolute position of the token in the sequence
 *     uint32   token_id
 *     uint64   ngram_mixed[0]   BIGRAM  hash, XOR-mixed, BEFORE per-head modulo
 *     uint64   ngram_mixed[1]   TRIGRAM hash, XOR-mixed, BEFORE per-head modulo
 *     uint16   expert[n_layers][topk]   routed expert ids, layer-major, gate order
 *
 * ngram_mixed[n-2] is exactly the `mixed` value ple_rows_for() computes for n-gram
 * order n before it takes `mixed % ple_vocab[h] + ple_off[h]` for each of the 16
 * heads -- i.e. the cheapest complete n-gram signal, from which every per-head row
 * id is recoverable offline given the meta constants.  Same EOS cut rule.
 *
 * An unrouted slot (router degraded to -1) is stored as 0xFFFF.
 */
#define Q38RT_MAGIC "Q38RT1\0\0"
#define Q38RT_HDR   32
#define Q38RT_BUFSZ (4u << 20)          /* ~4 MB: ~4200 records between fwrites */

static FILE     *g_rt_fp;               /* NULL = tracing off; the only gate */
static uint8_t  *g_rt_buf;              /* output staging, flushed at BUFSZ */
static size_t    g_rt_len;
static uint16_t *g_rt_e;                /* [maxb][n_layers*topk] staged ids */
static uint32_t *g_rt_pos, *g_rt_tok;   /* [maxb] */
static uint64_t *g_rt_ng;               /* [maxb][2] */
static int       g_rt_nb, g_rt_maxb, g_rt_nl, g_rt_k, g_rt_rec;

/* grow the per-batch staging arrays to hold B tokens */
static void q38rt_reserve(int B) {
    if (B <= g_rt_maxb) return;
    g_rt_maxb = B;
    g_rt_e   = realloc(g_rt_e,   (size_t)B * (size_t)g_rt_nl * g_rt_k * sizeof(uint16_t));
    g_rt_pos = realloc(g_rt_pos, (size_t)B * sizeof(uint32_t));
    g_rt_tok = realloc(g_rt_tok, (size_t)B * sizeof(uint32_t));
    g_rt_ng  = realloc(g_rt_ng,  (size_t)B * 2 * sizeof(uint64_t));
    if (!g_rt_e || !g_rt_pos || !g_rt_tok || !g_rt_ng) {
        fprintf(stderr, "[route_trace] OOM staging %d tokens -- tracing off\n", B);
        fclose(g_rt_fp); g_rt_fp = NULL;
    }
}

/* called once from main() after the config is parsed */
static void q38rt_open(const Cfg *c) {
    const char *p = getenv("Q38_ROUTE_TRACE");
    if (!p || !*p) return;
    /* P4 Router v2 corpus: the served engine appends across restarts, so
     * real traffic accumulates one trace file (the header is written once,
     * when the file is empty; records are fixed-size, so a torn tail from a
     * killed process is at most one record and the reader can drop it). */
    g_rt_fp = fopen(p, "ab");
    if (!g_rt_fp) { fprintf(stderr, "[route_trace] cannot open %s\n", p); return; }
    g_rt_nl = c->n_layers; g_rt_k = c->topk;
    g_rt_rec = 24 + 2 * g_rt_nl * g_rt_k;
    g_rt_buf = malloc(Q38RT_BUFSZ + (size_t)g_rt_rec);
    if (!g_rt_buf) { fclose(g_rt_fp); g_rt_fp = NULL; return; }
    long existing = ftell(g_rt_fp);
    if (existing > Q38RT_HDR && (existing - Q38RT_HDR) % g_rt_rec) {
        /* a killed run left a partial record: cut it so the appended records stay aligned */
        existing -= (existing - Q38RT_HDR) % g_rt_rec;
        if (ftruncate(fileno(g_rt_fp), (off_t)existing) == 0) fseek(g_rt_fp, 0, SEEK_END);
        fprintf(stderr, "[route_trace] trimmed a partial record at the tail of %s\n", p);
    }
    if (existing <= 0) {
        uint8_t h[Q38RT_HDR]; memset(h, 0, sizeof h);
        memcpy(h, Q38RT_MAGIC, 8);
        uint32_t v[4] = { 1u, (uint32_t)c->n_layers, (uint32_t)c->topk, (uint32_t)c->n_experts };
        memcpy(h + 8, v, sizeof v);
        fwrite(h, 1, sizeof h, g_rt_fp);
    }
    q38rt_reserve(64);
    fprintf(stderr, "[route_trace] logging %d layers x %d experts -> %s (%d B/token%s)\n",
            c->n_layers, c->topk, p, g_rt_rec, existing > 0 ? ", appending" : "");
}

/* start a batch of B tokens at absolute pos0 (B=1 in decode).  Their ids must
 * already be in m->toks, which both callers guarantee. */
static void q38rt_begin(const Model *m, int pos0, int B) {
    if (!g_rt_fp) return;
    q38rt_reserve(B);
    if (!g_rt_fp) return;
    const Cfg *c = &m->c;
    g_rt_nb = B;
    memset(g_rt_e, 0xFF, (size_t)B * (size_t)g_rt_nl * g_rt_k * sizeof(uint16_t));
    for (int t = 0; t < B; t++) {
        int pos = pos0 + t;
        g_rt_pos[t] = (uint32_t)pos;
        g_rt_tok[t] = (uint32_t)m->toks[pos];
        /* same context build + EOS cut as ple_rows_for(); mixed kept pre-modulo */
        uint64_t ctx[8]; ctx[0] = (uint64_t)m->toks[pos];
        int cut = 0;
        for (int s = 1; s < c->ple_ngram; s++) {
            int64_t tk = -1;
            if (!cut && pos - s >= 0) tk = m->toks[pos - s];
            if (tk < 0 || tk == c->ple_eos) cut = 1;
            ctx[s] = cut ? (uint64_t)c->ple_eos : (uint64_t)tk;
        }
        for (int n = 2; n <= 3 && n <= c->ple_ngram; n++) {
            uint64_t mixed = ctx[0] * c->ple_mult[0];
            for (int j = 1; j <= n - 1; j++) mixed ^= ctx[j] * c->ple_mult[j];
            g_rt_ng[t*2 + (n-2)] = mixed;
        }
    }
}

/* record one layer's routed ids for batch row t */
static void q38rt_note(int t, int layer, const int *idx, int K) {
    if (!g_rt_fp || t < 0 || t >= g_rt_nb || layer < 0 || layer >= g_rt_nl) return;
    uint16_t *dst = g_rt_e + ((size_t)t * g_rt_nl + layer) * g_rt_k;
    for (int kk = 0; kk < K && kk < g_rt_k; kk++)
        dst[kk] = idx[kk] >= 0 ? (uint16_t)idx[kk] : 0xFFFFu;
}

/* emit the staged batch (call once all layers have run) */
static void q38rt_flush(void) {
    if (!g_rt_fp) return;
    for (int t = 0; t < g_rt_nb; t++) {
        uint8_t *p = g_rt_buf + g_rt_len;
        memcpy(p,      &g_rt_pos[t], 4);
        memcpy(p + 4,  &g_rt_tok[t], 4);
        memcpy(p + 8,  &g_rt_ng[t*2], 16);
        memcpy(p + 24, g_rt_e + (size_t)t * g_rt_nl * g_rt_k,
               (size_t)g_rt_nl * g_rt_k * sizeof(uint16_t));
        g_rt_len += (size_t)g_rt_rec;
        if (g_rt_len >= Q38RT_BUFSZ) {
            fwrite(g_rt_buf, 1, g_rt_len, g_rt_fp);
            g_rt_len = 0;
        }
    }
    g_rt_nb = 0;
}

/* served turns: land the buffered records so a long-running service's trace
 * is on disk after every request (a killed service loses nothing) */
static void q38rt_sync(void) {
    if (!g_rt_fp) return;
    if (g_rt_len) fwrite(g_rt_buf, 1, g_rt_len, g_rt_fp);
    g_rt_len = 0;
    fflush(g_rt_fp);
}

static void q38rt_close(void) {
    if (!g_rt_fp) return;
    if (g_rt_len) fwrite(g_rt_buf, 1, g_rt_len, g_rt_fp);
    g_rt_len = 0;
    fflush(g_rt_fp);
    fclose(g_rt_fp);
    g_rt_fp = NULL;
    fprintf(stderr, "[route_trace] closed\n");
}

/* ============ P7: dashboard routing telemetry (EMAP / HITS) ============
 * The web dashboard's Brain page wants two per-expert grids: a cumulative
 * heat/tier map (EMAP) and "who was routed since the last frame" (HITS).
 * Both are counted HERE rather than inside qwen38_tier.c, because the
 * dashboard must work on a CPU-only build too -- routing happens whether or
 * not a VRAM tier exists, and only the tier NIBBLE of EMAP needs the tier.
 *
 * Cost is one increment and one byte store per routed expert per token
 * (10 per layer per token), unconditional: a flag test in the same cache line
 * costs the same as the store, and gating it would give the serve path a
 * different routing hot loop from the CLI path that the reference gates
 * measure.  g_tele_cnt is never reset; g_tele_hit is cleared by each emit. */
static uint32_t *g_tele_cnt;    /* [nl*ne] cumulative routed count -> EMAP heat */
static uint8_t  *g_tele_hit;    /* [nl*ne] 1 = routed since the last HITS frame */
static int g_tele_nl, g_tele_ne;

static void tele_init(const Cfg *c) {
    g_tele_nl = c->n_layers; g_tele_ne = c->n_experts;
    size_t n = (size_t)g_tele_nl * g_tele_ne;
    g_tele_cnt = calloc(n, sizeof(uint32_t));
    g_tele_hit = calloc(n, 1);
    if (!g_tele_cnt || !g_tele_hit) { fprintf(stderr, "OOM telemetry grid\n"); exit(1); }
}

static void tele_note(int layer, const int *idx, int K) {
    if (!g_tele_cnt || layer < 0 || layer >= g_tele_nl) return;
    size_t base = (size_t)layer * g_tele_ne;
    for (int k = 0; k < K; k++) {
        int e = idx[k];
        if (e < 0 || e >= g_tele_ne) continue;
        g_tele_cnt[base + e]++;
        g_tele_hit[base + e] = 1;
    }
}

/* ==================== MoE ==================== */
/* top-k over the softmaxed router probs + renorm (order identical everywhere) */
static void moe_topk_renorm(const float *pr, int E, int K, int *idx, float *val) {
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1.f;
        for (int e = 0; e < E; e++) {
            int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
            if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
        }
        idx[kk] = best; val[kk] = bv;
    }
    float sm = 0.f; for (int kk = 0; kk < K; kk++) sm += val[kk];
    float den = fmaxf(sm, 6.103515625e-05f);
    for (int kk = 0; kk < K; kk++) val[kk] /= den;
}

/* one routed expert on the CPU (raw blocks): y[D] = down(silu(gate x)*up x).
 * g/u scratch of size I.  Safe inside an OMP region (the inner GEMV pragmas
 * collapse to serial there). */
static void moe_expert_cpu_y(Model *m, int lidx, int e, const float *x,
                             float *y, float *g, float *u) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, I = c->inter;
    char nm[256];
    static __thread uint8_t *eraw = NULL;   /* scratch for the S1 read path */
    size_t emax = 4 * 1024 * 1024;
    RawMat rg = {0}, ru = {0}, rd = {0};    /* stack views: q8/s8 must be NULL */
    if (m->have_mmap) {
        int64_t si = (int64_t)lidx * E + e;
        rg.raw = (uint8_t*)m->e_g[si]; rg.type = m->e_gu_type[lidx]; rg.O=I; rg.I=D; rg.row_bytes=ggml_bk_row_bytes(rg.type,(size_t)D);
        ru.raw = (uint8_t*)m->e_u[si]; ru.type = rg.type;             ru.O=I; ru.I=D; ru.row_bytes=rg.row_bytes;
        rd.raw = (uint8_t*)m->e_d[si]; rd.type = m->e_d_type[lidx];  rd.O=D; rd.I=I; rd.row_bytes=ggml_bk_row_bytes(rd.type,(size_t)I);
    } else {
        st_tensor *tg, *tu, *td;
        if (!eraw) { eraw = malloc(3*emax); if(!eraw){fprintf(stderr,"OOM expert scratch\n");exit(1);} }
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.gate_raw", lidx, e);
        tg = st_find(&m->S, nm); if (!tg) st_die_missing(&m->S, nm);
        if ((size_t)tg->nbytes > emax) { fprintf(stderr,"%s too big\n",nm); exit(1); }
        st_read_raw(&m->S, nm, eraw, 0);
        rg.raw = eraw; rg.type = rm_infer_type(tg->nbytes, I, D, nm); rg.O=I; rg.I=D; rg.row_bytes=ggml_bk_row_bytes(rg.type,(size_t)D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.up_raw", lidx, e);
        tu = st_find(&m->S, nm); if (!tu) st_die_missing(&m->S, nm);
        if ((size_t)tu->nbytes > emax) { fprintf(stderr,"%s too big\n",nm); exit(1); }
        st_read_raw(&m->S, nm, eraw + emax, 0);
        ru.raw = eraw + emax; ru.type = rm_infer_type(tu->nbytes, I, D, nm); ru.O=I; ru.I=D; ru.row_bytes=ggml_bk_row_bytes(ru.type,(size_t)D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.down_raw", lidx, e);
        td = st_find(&m->S, nm); if (!td) st_die_missing(&m->S, nm);
        if ((size_t)td->nbytes > emax) { fprintf(stderr,"%s too big\n",nm); exit(1); }
        st_read_raw(&m->S, nm, eraw + 2*emax, 0);
        rd.raw = eraw + 2*emax; rd.type = rm_infer_type(td->nbytes, D, I, nm); rd.O=D; rd.I=I; rd.row_bytes=ggml_bk_row_bytes(rd.type,(size_t)I);
    }
    rm_gemv(g, &rg, x);
    rm_gemv(u, &ru, x);
    for (int i = 0; i < I; i++) g[i] = siluf_(g[i]) * u[i];
    rm_gemv(y, &rd, g);
}

static void moe_expert_cpu(Model *m, int lidx, int e, const float *x, float w,
                           float *out, float *g, float *u, float *hh) {
    int D = m->c.hidden;
    moe_expert_cpu_y(m, lidx, e, x, hh, g, u);
    for (int d = 0; d < D; d++) out[d] += w * hh[d];
}

static void moe_post_route(Model *m, Layer *l, int lidx, const float *x,
                           const int *idx, const float *val, float *out);

static void moe_forward(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk;
    double tt0 = g_timers ? now_s() : 0;
    float *pr = falloc(E);
    int from_dev = g_pr_pre != NULL;
    float *lg = from_dev ? falloc(E) : NULL;
    if (g_pr_pre) { memcpy(pr, g_pr_pre, (size_t)E * sizeof(float)); g_pr_pre = NULL; memcpy(lg, pr, (size_t)E * sizeof(float)); }
    else matf(pr, l->gate, x, D, E);
    softmax_row(pr, E);                          /* softmax over ALL experts, then top-k */
    int idx[256]; float val[256];
    moe_topk_renorm(pr, E, K, idx, val);
    if (from_dev) {
        g_router_tok++;
        if (router_margin(lg, E, idx, K) < router_gap_env()) {
            g_router_fallback++;
            matf(pr, l->gate, x, D, E);
            softmax_row(pr, E);
            moe_topk_renorm(pr, E, K, idx, val);
        }
        if (g_router_check < 0) g_router_check = getenv("Q38_ROUTER_CHECK") && atoi(getenv("Q38_ROUTER_CHECK"));
        if (g_router_check) router_check(lg, l->gate, x, D, E, K, idx);
        free(lg);
    }
    if (g_dbg_now >= 0 && &m->L[g_dbg_now] == l) {
        fprintf(stderr, "[dbg] moe_topk:");
        for (int kk = 0; kk < K; kk++) fprintf(stderr, " %d:%.4f", idx[kk], val[kk]);
        fprintf(stderr, "\n");
    }

    int lidx = (int)(l - m->L);
    q38rt_note(0, lidx, idx, K);
    tele_note(lidx, idx, K);
    double tt1 = g_timers ? now_s() : 0;
    if (g_timers) g_tm.moe_route += tt1 - tt0;
    moe_post_route(m, l, lidx, x, idx, val, out);
    free(pr);
}

/* everything after routing for ONE token (also the batched-prefill fallback
 * when the multi-token tier issue is unavailable) */
static void moe_post_route(Model *m, Layer *l, int lidx, const float *x,
                           const int *idx, const float *val, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, K = c->topk, I = c->inter;
    double tt1 = g_timers ? now_s() : 0;
    memset(out, 0, (size_t)D*sizeof(float));

    /* S2: hand the VRAM-resident subset to the GPU (async), compute the
     * misses on the CPU in overlap, collect at the end.  qmask==0 when the
     * tier is off, so the CPU loop below is exactly the S1 computation. */
    uint32_t qmask = 0;
    if (m->have_mmap && q38t_ready()) {
        for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) q38t_note(lidx, idx[kk]);
        qmask = q38t_issue(lidx, idx, K, x);
    }

    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    for (int kk = 0; kk < K; kk++) {
        int e = idx[kk]; if (e < 0 || (qmask >> kk) & 1u) continue;
        moe_expert_cpu(m, lidx, e, x, val[kk], out, g, u, hh);
    }
    double tt2 = g_timers ? now_s() : 0;
    if (g_timers) g_tm.moe_cpu += tt2 - tt1;
    if (g_dbg_now >= 0 && &m->L[g_dbg_now] == l && !qmask) dbg_row("ffn_moe_out", out, D);
    /* shared expert, scalar sigmoid gate (compute on CPU only when the tier
     * did not put it on the GPU chain; the gate scalar is always CPU) */
    float shs;
    {
        double ts0 = g_timers ? now_s() : 0;
        float dot = 0.f; for (int d = 0; d < D; d++) dot += x[d] * l->sh_gate[d];
        shs = sigmoidf_(dot);
        if (!q38t_shared_on(lidx)) {
            int Is = c->shared_inter;
            float *sg = falloc(Is), *su = falloc(Is), *sd = falloc(D);
            rm_gemv(sg, &l->sh_g, x);
            rm_gemv(su, &l->sh_u, x);
            for (int i = 0; i < Is; i++) sg[i] = siluf_(sg[i]) * su[i];
            rm_gemv(sd, &l->sh_d, sg);
            for (int d = 0; d < D; d++) out[d] += shs * sd[d];
            free(sg); free(su); free(sd);
        }
        if (g_dbg_now >= 0 && &m->L[g_dbg_now] == l) {
            fprintf(stderr, "[dbg] shexp_gate_sig=%.4f\n", shs);
            if (!qmask) dbg_row("ffn_out", out, D);
        }
        if (g_timers) g_tm.moe_shared += now_s() - ts0;
    }
    { double tg0 = g_timers ? now_s() : 0;
      q38t_take(qmask, val, K, out, shs);
      if (g_timers) g_tm.moe_gpu += now_s() - tg0; }
    free(g); free(u); free(hh);
}

/* S3 batched MoE over a prefill chunk: routing as one chunk GEMM, then either
 * a single multi-token tier issue (CUDA; amortizes the per-issue chain
 * latency over the chunk) or the per-token S2 flow.  Per-token math and
 * accumulation order are identical to moe_forward. */
static void moe_forward_b(Model *m, Layer *l, const float *X, float *OUT, int B) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    int lidx = (int)(l - m->L);
    double tt0 = g_timers ? now_s() : 0;
    float *PR = falloc((int64_t)B*E);
    int from_dev = g_pr_pre != NULL;
    if (from_dev && g_router_check < 0) g_router_check = getenv("Q38_ROUTER_CHECK") && atoi(getenv("Q38_ROUTER_CHECK"));
    float *LG = from_dev ? falloc((int64_t)B * E) : NULL;
    if (g_pr_pre) { memcpy(PR, g_pr_pre, (size_t)B * E * sizeof(float)); g_pr_pre = NULL; memcpy(LG, PR, (size_t)B * E * sizeof(float)); }
    else matf_b(PR, l->gate, X, D, E, B, E);
    int *IDX = malloc((size_t)B*K*sizeof(int));
    float *VAL = falloc((int64_t)B*K);
    long fb = 0;
    #pragma omp parallel for schedule(static) reduction(+:fb)
    for (int t = 0; t < B; t++) {
        softmax_row(PR + (int64_t)t*E, E);
        moe_topk_renorm(PR + (int64_t)t*E, E, K, IDX + (int64_t)t*K, VAL + (int64_t)t*K);
        if (from_dev && router_margin(LG + (int64_t)t*E, E, IDX + (int64_t)t*K, K) < router_gap_env()) {
            /* exact host recompute for this token (serial matf order) */
            const float *x = X + (int64_t)t*D; float *pr = PR + (int64_t)t*E;
            for (int o = 0; o < E; o++) {
                const float *w = l->gate + (int64_t)o * D;
                float acc = 0.f;
                for (int i = 0; i < D; i++) acc += w[i] * x[i];
                pr[o] = acc;
            }
            softmax_row(pr, E);
            moe_topk_renorm(pr, E, K, IDX + (int64_t)t*K, VAL + (int64_t)t*K);
            fb++;
        }
        if (from_dev && g_router_check > 0) router_check(LG + (int64_t)t*E, l->gate, X + (int64_t)t*D, D, E, K, IDX + (int64_t)t*K);
    }
    if (from_dev) { g_router_tok += B; g_router_fallback += fb; free(LG); }
    if (g_rt_fp) for (int t = 0; t < B; t++) q38rt_note(t, lidx, IDX + (int64_t)t*K, K);
    for (int t = 0; t < B; t++) tele_note(lidx, IDX + (int64_t)t*K, K);
    double tt1 = g_timers ? now_s() : 0;
    if (g_timers) g_tm.moe_route += tt1 - tt0;

    if (m->have_mmap && q38t_ready() && q38t_multi_max() >= B) {
        q38t_note_many(lidx, IDX, B*K);
        uint32_t *masks = calloc(B, sizeof(uint32_t));
        float *SHS = falloc(B);
        for (int t = 0; t < B; t++) {
            const float *x = X + (int64_t)t*D;
            float dot = 0.f; for (int d = 0; d < D; d++) dot += x[d] * l->sh_gate[d];
            SHS[t] = sigmoidf_(dot);
            memset(OUT + (int64_t)t*D, 0, (size_t)D*sizeof(float));
        }
        q38t_issue_multi(lidx, IDX, K, B, X, masks);
        /* P9: shared expert as an async dense-path GEMM behind the tier
         * chain; d_b_mixed already holds X (the chunk's mixed) */
        static int gsh_env = -1;
        if (gsh_env < 0) { const char *e = getenv("Q38_GPU_SHARED_PREFILL"); gsh_env = e && atoi(e); }
        int gsh = gsh_env && g_gpu_dense && B > 1 && l->g_shg >= 0 && q38g_batch_max() >= B &&
                  !q38t_multi_shared(lidx);
        if (gsh) gsh = q38g_shared_batch_issue(l->g_shg, l->g_shu, l->g_shd, NULL, B);
        /* P10 D2b: one-token issue (decode) -- the shared expert on the
         * tier's resident copy instead of 235 MB/token of int8 from DRAM */
        int gsh1 = 0;
        if (B == 1 && !q38t_multi_shared(lidx) && q38t_shared_on(lidx))
            gsh1 = q38t_shared_issue1(lidx, X);
        double tm0 = g_timers ? now_s() : 0;
        /* CPU misses: gather (token-major, k-ascending -- the accumulation
         * order of the sequential path), compute in parallel over pairs
         * (whole experts per thread; inner GEMV pragmas collapse to serial),
         * then accumulate serially in gather order. */
        int nmiss = 0;
        int *mt = malloc((size_t)B*K*sizeof(int)), *me = malloc((size_t)B*K*sizeof(int));
        for (int t = 0; t < B; t++)
            for (int kk = 0; kk < K; kk++) {
                int e = IDX[t*K+kk];
                if (e < 0 || (masks[t] >> kk) & 1u) continue;
                mt[nmiss] = t*K + kk; me[nmiss] = e; nmiss++;
            }
        float *YM = falloc((int64_t)(nmiss > 0 ? nmiss : 1) * D);
        if (nmiss && m->have_mmap) {
            /* row-chunked tasks over all miss pairs: every thread streams raw
             * block rows at its own pace (a whole IQ3_S expert decodes too
             * slowly for one thread, a lone 640-row GEMV spreads too thin) */
            enum { MCH = 64 };
            int gu_t = m->e_gu_type[lidx], d_t = m->e_d_type[lidx];
            size_t gu_rb = ggml_bk_row_bytes(gu_t, (size_t)D);
            size_t d_rb  = ggml_bk_row_bytes(d_t, (size_t)I);
            float *GM = falloc((int64_t)nmiss * 2 * I);
            int nchA = (I + MCH - 1) / MCH;
            #pragma omp parallel for schedule(dynamic)
            for (int q = 0; q < nmiss * 2 * nchA; q++) {
                int j = q / (2*nchA), rr = q % (2*nchA), mat = rr / nchA, ch = rr % nchA;
                int r0 = ch * MCH, rn = I - r0 < MCH ? I - r0 : MCH;
                const uint8_t *src = (mat ? m->e_u : m->e_g)[(int64_t)lidx*E + me[j]];
                ggml_bk_gemv(gu_t, rn, D, src + (size_t)r0*gu_rb,
                             X + (int64_t)(mt[j]/K)*D, GM + (int64_t)j*2*I + mat*I + r0);
            }
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < nmiss; j++) {
                float *gm = GM + (int64_t)j*2*I;
                for (int i = 0; i < I; i++) gm[i] = siluf_(gm[i]) * gm[I+i];
            }
            int nchB = (D + MCH - 1) / MCH;
            #pragma omp parallel for schedule(dynamic)
            for (int q = 0; q < nmiss * nchB; q++) {
                int j = q / nchB, ch = q % nchB;
                int r0 = ch * MCH, rn = D - r0 < MCH ? D - r0 : MCH;
                const uint8_t *src = m->e_d[(int64_t)lidx*E + me[j]];
                ggml_bk_gemv(d_t, rn, I, src + (size_t)r0*d_rb,
                             GM + (int64_t)j*2*I, YM + (int64_t)j*D + r0);
            }
            free(GM);
        } else if (nmiss) {
            float *g = falloc(I), *u = falloc(I);
            for (int j = 0; j < nmiss; j++)
                moe_expert_cpu_y(m, lidx, me[j], X + (int64_t)(mt[j]/K)*D,
                                 YM + (int64_t)j*D, g, u);
            free(g); free(u);
        }
        for (int j = 0; j < nmiss; j++) {
            float w = VAL[mt[j]];
            float *out = OUT + (int64_t)(mt[j]/K)*D;
            const float *y = YM + (int64_t)j*D;
            for (int d = 0; d < D; d++) out[d] += w * y[d];
        }
        free(mt); free(me); free(YM);
        if (!q38t_multi_shared(lidx)) {
            if (gsh1) {
                if (!q38t_shared_take1(OUT, SHS[0])) gsh1 = 0;
            }
            if (gsh1) { /* done on the tier */ }
            else if (gsh) {
                /* collect the async GEMM; the gate scale and the accumulation
                 * stay on the host in the same per-token order as the loop */
                float *YS = falloc((int64_t)B*D);
                if (!q38g_shared_batch_take(B, YS)) gsh = 0;
                else for (int t = 0; t < B; t++) {
                    float *out = OUT + (int64_t)t*D;
                    const float *sd = YS + (int64_t)t*D;
                    for (int d = 0; d < D; d++) out[d] += SHS[t] * sd[d];
                }
                free(YS);
            }
            if (!gsh && !gsh1) {
                /* P10 P1: the batched twins read the int8 shared expert once
                 * per chunk instead of once per token (rm_gemv_b is
                 * bit-identical to B single rm_gemv calls); the per-token
                 * SiLU, down projection and accumulation keep their order. */
                int Is = c->shared_inter;
                float *SG = falloc((int64_t)B*Is), *SU = falloc((int64_t)B*Is), *SD = falloc((int64_t)B*D);
                rm_gemv_b(SG, &l->sh_g, X, B, Is);
                rm_gemv_b(SU, &l->sh_u, X, B, Is);
                #pragma omp parallel for schedule(static)
                for (int t = 0; t < B; t++) {
                    float *sg = SG + (int64_t)t*Is; const float *su = SU + (int64_t)t*Is;
                    for (int i = 0; i < Is; i++) sg[i] = siluf_(sg[i]) * su[i];
                }
                rm_gemv_b(SD, &l->sh_d, SG, B, D);
                for (int t = 0; t < B; t++) {
                    float *out = OUT + (int64_t)t*D;
                    const float *sd = SD + (int64_t)t*D;
                    for (int d = 0; d < D; d++) out[d] += SHS[t] * sd[d];
                }
                free(SG); free(SU); free(SD);
            }
        }
        double tm1 = g_timers ? now_s() : 0;
        if (g_timers) g_tm.moe_cpu += tm1 - tm0;
        q38t_issue_multi_finish();              /* P10 P5: staged tiles after the CPU work */
        q38t_take_multi(VAL, K, B, OUT, SHS);
        if (g_timers) g_tm.moe_gpu += now_s() - tm1;
        free(masks); free(SHS);
    } else {
        for (int t = 0; t < B; t++)
            moe_post_route(m, l, lidx, X + (int64_t)t*D, IDX + (int64_t)t*K,
                           VAL + (int64_t)t*K, OUT + (int64_t)t*D);
    }
    free(PR); free(IDX); free(VAL);
}

/* ==================== PLE n-gram block ==================== */
/* grouped_norm: RMS over each 2560-stream, gamma over the whole 10240. */
static void ple_grouped_norm(const Cfg *c, float *out, const float *x, const float *w) {
    int D = c->hidden;
    for (int s = 0; s < HC; s++)
        rmsnorm_plain(out + s*D, x + s*D, NULL, D, c->eps);
    for (int j = 0; j < HC*D; j++) out[j] *= w[j];
}

/* Compute the 16 table rows for the token at absolute position `pos`
 * (token history in m->toks[0..pos]). */
static void ple_rows_for(Model *m, int pos, int64_t *rows) {
    Cfg *c = &m->c;
    int ngram = c->ple_ngram, hpn = c->ple_hpn;
    uint64_t ctx[8];
    ctx[0] = (uint64_t)m->toks[pos];
    int cut = 0;
    for (int s = 1; s < ngram; s++) {
        int64_t t = -1;
        if (!cut && pos - s >= 0) t = m->toks[pos - s];
        if (t < 0 || t == c->ple_eos) cut = 1;
        ctx[s] = cut ? (uint64_t)c->ple_eos : (uint64_t)t;
    }
    for (int n = 2; n <= ngram; n++) {
        uint64_t mixed = ctx[0] * c->ple_mult[0];
        for (int j = 1; j <= n-1; j++) mixed ^= ctx[j] * c->ple_mult[j];
        for (int g = 0; g < hpn; g++) {
            int h = (n-2)*hpn + g;
            rows[h] = (int64_t)(mixed % c->ple_vocab[h] + c->ple_off[h]);
        }
    }
}

/* P4 prefetch: the bigram `mixed` for the token at `pos` -- exactly the value
 * ple_rows_for() computes for n-gram order 2 before its per-head modulo (same
 * context build + EOS cut; see also q38rt_begin, which stages the same hash
 * for the route trace).  Available the moment the token id is known, i.e.
 * before layer 0 of step `pos` computes -- the prefetch trigger. */
static uint64_t q38_bigram_mixed(const Model *m, int pos) {
    const Cfg *c = &m->c;
    uint64_t prev = (uint64_t)c->ple_eos;
    if (pos - 1 >= 0) {
        int64_t tk = m->toks[pos - 1];
        if (tk >= 0 && tk != c->ple_eos) prev = (uint64_t)tk;
    }
    return (uint64_t)m->toks[pos] * c->ple_mult[0] ^ prev * c->ple_mult[1];
}

/* ==================== P3: PLE cold-read batching ====================
 * The 28 GB PLE table is mmap'd with MADV_RANDOM, so on a cold page cache
 * every row gather is a serial 4K demand fault (~1.4 ms each at QD1 on this
 * NVMe; docs/bench.md).  But the row addresses are pure functions of the
 * token ids (ple_rows_for) -- in a prefill chunk ALL of them are known before
 * layer 0 runs, and at decode the sampled token names its 16 rows before its
 * forward starts.  Warming them up front turns the serial fault chain into a
 * batched readahead the SSD serves at queue depth (213K IOPS r4K vs ~11K
 * serial).  madvise(MADV_WILLNEED) issues the async readahead; the optional
 * touch threads (Q38_PLE_TOUCH=1) additionally hard-fault the pages from an
 * OpenMP loop for kernels whose WILLNEED readahead underdelivers.
 * No computation changes -- rows are read by ple_forward exactly as before,
 * so every path stays bit-identical.  Q38_PLE_BATCH=0 disables. */
static int g_ple_batch = -1, g_ple_touch;
static long g_ple_pgmask;
static void ple_warm_init(void) {
    if (g_ple_batch >= 0) return;
    g_ple_batch = getenv("Q38_PLE_BATCH") ? atoi(getenv("Q38_PLE_BATCH")) : 1;
    g_ple_touch = getenv("Q38_PLE_TOUCH") ? atoi(getenv("Q38_PLE_TOUCH")) : 0;
    g_ple_pgmask = sysconf(_SC_PAGESIZE) - 1;
}
static void ple_warm_rows(Model *m, const int64_t *rows, int nrows) {
    for (int r = 0; r < nrows; r++) {
        uintptr_t a = (uintptr_t)(m->ple_table + 64 + (uint64_t)rows[r] * 90);
        uintptr_t s = a & ~(uintptr_t)g_ple_pgmask;
        madvise((void*)s, (size_t)(a + 90 - s), MADV_WILLNEED);
    }
    if (g_ple_touch) {
        volatile uint64_t sink = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:sink)
        for (int r = 0; r < nrows; r++) {
            const volatile uint8_t *p = m->ple_table + 64 + (uint64_t)rows[r] * 90;
            sink += p[0] + p[89];            /* rows can straddle a page */
        }
        (void)sink;
    }
}
/* warm every row a prefill chunk (pos0..pos0+B-1, ids already in m->toks)
 * will gather at the PLE layer */
static void ple_warm_chunk(Model *m, int pos0, int B) {
    ple_warm_init();
    if (!g_ple_batch || !m->ple_table) return;
    int nh = m->c.ple_n_heads;
    int64_t *rows = malloc((size_t)B * nh * sizeof(int64_t));
    if (!rows) return;
    for (int t = 0; t < B; t++) {
        int64_t rb[64];
        ple_rows_for(m, pos0 + t, rb);
        memcpy(rows + (int64_t)t * nh, rb, (size_t)nh * sizeof(int64_t));
    }
    ple_warm_rows(m, rows, B * nh);
    free(rows);
}
/* decode: the sampled token's 16 rows, immediately after sampling */
static void ple_warm_token(Model *m, int pos) {
    ple_warm_init();
    if (!g_ple_batch || !m->ple_table) return;
    int64_t rows[64];
    ple_rows_for(m, pos, rows);
    ple_warm_rows(m, rows, m->c.ple_n_heads);
}

/* PLE injection: modifies res_hc in place (applied before the layer's hc_attn
 * mix).  Keeps a 9-column history of the conv input per sequence. */
static void ple_forward(Model *m, float *res, int pos) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    int hist_len = (c->ple_convk - 1) * c->ple_ngram;   /* 9 */
    int dil = c->ple_ngram;                              /* dilation 3 */

    int64_t rows[64];
    ple_rows_for(m, pos, rows);
    float *emb = falloc(D);
    for (int h = 0; h < c->ple_n_heads; h++) {
        const uint8_t *src = m->ple_table + 64 + (uint64_t)rows[h] * 90;
        ggml_bk_dequant_row_iq4_nl(src, emb + h*c->ple_dim, (size_t)c->ple_dim);
    }

    float *key = falloc(W), *value = falloc(D);
    rm_gemv(key, &m->ple_key, emb);
    rm_gemv(value, &m->ple_value, emb);

    float *keyn = falloc(W), *qryn = falloc(W);
    ple_grouped_norm(c, keyn, key, m->ple_nk);
    ple_grouped_norm(c, qryn, res, m->ple_nq);   /* QUERY = the wide residual itself */

    float gatev[HC];
    float inv = 1.f / sqrtf((float)D);
    for (int s = 0; s < HC; s++) {
        double acc = 0;
        for (int j = 0; j < D; j++) acc += (double)keyn[s*D+j] * qryn[s*D+j];
        float sv = (float)acc * inv;
        float mag = sqrtf(fmaxf(fabsf(sv), 1e-6f));
        gatev[s] = sigmoidf_(sv >= 0.f ? mag : -mag);
    }
    float *gated = falloc(W);
    for (int s = 0; s < HC; s++)
        for (int j = 0; j < D; j++) gated[s*D+j] = value[j] * gatev[s];
    if (g_dbg_now == c->ple_layer) {
        dbg_row("ple_embd", emb, D);
        fprintf(stderr, "[dbg] ple_gate=[%.4f %.4f %.4f %.4f]\n", gatev[0],gatev[1],gatev[2],gatev[3]);
        dbg_row("ple_gated", gated, W);
    }

    float *normed = falloc(W);
    ple_grouped_norm(c, normed, gated, m->ple_nc);

    /* depthwise causal conv over TIME, kernel 4, dilation 3, then SiLU.
     * hist[j][ch] = normed at time pos-1-j (j = 0..hist_len-1). */
    float *conv = falloc(W);
    for (int ch = 0; ch < W; ch++) {
        const float *w = m->ple_conv + (int64_t)ch * c->ple_convk;
        float acc = w[c->ple_convk - 1] * normed[ch];
        for (int k = 0; k < c->ple_convk - 1; k++) {
            int back = (c->ple_convk - 1 - k) * dil - 1;    /* taps t-3,t-6,t-9 -> hist idx 2,5,8 */
            if (back < m->ple_hist_n) acc += w[k] * m->ple_hist[(int64_t)back * W + ch];
        }
        conv[ch] = siluf_(acc);
    }
    /* push current normed column into the history (front = most recent) */
    int keep = hist_len - 1 < m->ple_hist_n ? hist_len - 1 : m->ple_hist_n;
    memmove(m->ple_hist + W, m->ple_hist, (size_t)keep * W * sizeof(float));
    memcpy(m->ple_hist, normed, (size_t)W * sizeof(float));
    if (m->ple_hist_n < hist_len) m->ple_hist_n++;

    if (g_dbg_now == c->ple_layer) dbg_row("ple_conv_out", conv, W);
    for (int j = 0; j < W; j++) res[j] += gated[j] + conv[j];

    free(emb); free(key); free(value); free(keyn); free(qryn);
    free(gated); free(normed); free(conv);
}

/* ==================== P5: MTP draft layer forward ====================
 * Derived graph (docs/p5-status.md has the shape evidence):
 *
 *   en    = rmsnorm(embed(tok), pre_fc_norm_embedding)          # [2560]
 *   hn[s] = rmsnorm(res_hc[s]) * pre_fc_norm_hidden[s*2560..]   # stream space
 *   res[s]= fc_hidden @ hn[s] + fc_embedding @ en                # MTP wide residual
 *           (per-stream fold; norm gammas are zero-centered -> folded +1 at load)
 *   ... one hyper-connected QSA + MoE layer (backbone machinery, layer slot 48,
 *       own KV cache, dense attention) ...
 *   mixed = hc_mix(res, hyper_connection_mixer, no inject)       # = output_hc role
 *   draft logits = lm_head @ mixed                               # shared head
 *
 * The hidden branch's stream reduction was decided EMPIRICALLY (both readings
 * of the [10240] pre_fc_norm_hidden are shape-consistent): the per-stream fold
 *   res[s] = fc_hidden @ hn[s] + fc_embedding @ en
 * measured 81.5%/54.9% acceptance (chat/raw, greedy-64) vs 43.2%/34.4% for
 * mean-then-fc, so per-stream is the default; Q38_MTP_FOLD=mean keeps the
 * alternate for reference. */
static int g_mtp_fold_stream = 1;

static void mtp_fold(Model *m, const float *prev_res, int tok, float *res_out) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    float en[2560+64], fe[2560+64], hn_s[2560+64];
    float *hn = falloc(W);
    rm_row_f32(&m->embed, tok, en);
    rmsnorm_plain(en, en, m->mtp_nfe, D, c->eps);
    if (g_gpu_mtp) q38g_out_batch(m->g_mtp_fce, en, 1, fe);
    else           rm_gemv(fe, &m->mtp_fce, en);
    for (int s = 0; s < HC; s++)
        rmsnorm_plain(hn + s*D, prev_res + s*D, NULL, D, c->eps);
    for (int j = 0; j < W; j++) hn[j] *= m->mtp_nfh[j];
    if (g_mtp_fold_stream) {
        if (g_gpu_mtp) q38g_out_batch(m->g_mtp_fch, hn, HC, res_out);
        for (int s = 0; s < HC; s++) {
            if (!g_gpu_mtp) {
                rm_gemv(hn_s, &m->mtp_fch, hn + s*D);
                memcpy(res_out + s*D, hn_s, (size_t)D*sizeof(float));
            }
            for (int j = 0; j < D; j++) res_out[s*D+j] += fe[j];
        }
    } else {
        float *h = hn_s;
        for (int j = 0; j < D; j++) {
            float acc = 0.f;
            for (int s = 0; s < HC; s++) acc += hn[s*D+j];
            h[j] = acc / (float)HC;
        }
        float fh[2560+64];
        if (g_gpu_mtp) q38g_out_batch(m->g_mtp_fch, h, 1, fh);
        else           rm_gemv(fh, &m->mtp_fch, h);
        for (int j = 0; j < D; j++) fh[j] += fe[j];
        for (int s = 0; s < HC; s++) memcpy(res_out + s*D, fh, (size_t)D*sizeof(float));
    }
    free(hn);
}

/* Teacher-forced KV absorb for one committed position: fold the backbone's
 * final res_hc at `slot` with the NEXT token's embedding, run the attn hc mix
 * and the k/v projections, append to the MTP layer's KV cache at `slot`.
 * The q projection, attention walk, MoE and head are skipped -- committed
 * slots only ever serve as attention context for later drafts. */
static void mtp_absorb(Model *m, const float *prev_res, int slot, int next_tok) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D, ML = c->n_layers;
    int KV = c->kv_heads, hd = c->head_dim, ns = hd / 32;
    Layer *l = &m->L[ML];
    float *res = falloc(W), *xn = falloc(W), *gt = falloc(W);
    float *lo = falloc(c->hc_rank), *mixed = falloc(D);
    float kv_k[2*256+8], kv_v[2*256+8];
    mtp_fold(m, prev_res, next_tok, res);
    hc_mix(c, res, &l->hca_down, &l->hca_up, l->hca_norm, NULL, mixed, NULL, xn, lo, gt);
    rm_gemv(kv_k, &l->k, mixed);
    rm_gemv(kv_v, &l->v, mixed);
    for (int kvh = 0; kvh < KV; kvh++) {
        float *kh = kv_k + (int64_t)kvh*hd;
        rmsnorm_plain(kh, kh, l->kn, hd, c->eps);
        rope_head_m(m, kh, slot);
        if (m->kv_q8) {
            int64_t row = (int64_t)kvh*m->max_t + slot;
            kv_q8_row(kh, hd, m->K8[ML] + row*hd, m->K8s[ML] + row*ns);
            kv_q8_row(kv_v + (int64_t)kvh*hd, hd, m->V8[ML] + row*hd, m->V8s[ML] + row*ns);
        } else {
            memcpy(m->K[ML] + ((int64_t)kvh*m->max_t + slot)*hd, kh, hd*sizeof(float));
            memcpy(m->V[ML] + ((int64_t)kvh*m->max_t + slot)*hd, kv_v + (int64_t)kvh*hd, hd*sizeof(float));
        }
    }
    free(res); free(xn); free(gt); free(lo); free(mixed);
}

/* One draft step: fold(prev_res, tok) -> full MTP layer at `slot` (KV write
 * included) -> shared lm_head logits.  res_out gets the MTP layer's output
 * wide residual (the chained "previous hidden" for the next draft). */
static int spec_argmax(const float *lo, int V);
/* logits == NULL: only the greedy token is wanted (draft rows); on the GPU
 * path it is then taken on the device (P10 D5) and returned via *argmax. */
static void mtp_draft(Model *m, const float *prev_res, int tok, int slot,
                      float *res_out, float *logits, int *argmax) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D, ML = c->n_layers;
    Layer *l = &m->L[ML];
    float *res = res_out;
    float *xn = falloc(W), *gt = falloc(W), *lo = falloc(c->hc_rank);
    float *mixed = falloc(D), *blk = falloc(D);
    float inject[HC];
    mtp_fold(m, prev_res, tok, res);
    if (g_gpu_mtp) {
        int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
        float *qf = falloc((int64_t)H*2*hd), *kk = falloc((int64_t)KV*hd);
        float *vv = falloc((int64_t)KV*hd), *ctx = falloc((int64_t)H*hd);
        q38g_hc_mix_batch(l->g_hca_down, l->g_hca_up, l->g_hca_norm,
                          l->g_hca_inj, res, 1, mixed, inject);
        q38g_qsa_proj_batch(l->g_q, l->g_k, l->g_v, mixed, 1, qf, kk, vv);
        qsa_core_x(m, l, ML, mixed, slot, qf, kk, vv, NULL, ctx);
        q38g_out_batch(l->g_o, ctx, 1, blk);
        q38g_combine_batch(NULL, 1, res);
        q38g_hc_mix_batch(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm,
                          l->g_hcf_inj, res, 1, mixed, inject);
        moe_forward(m, l, mixed, blk);
        q38g_combine_batch(blk, 1, res);
        q38g_hc_mix_batch(m->g_mtp_mix_down, m->g_mtp_mix_up,
                          m->g_mtp_mix_norm, -1, res, 1, mixed, NULL);
        if (logits) q38g_out_batch(m->g_lm_head, mixed, 1, logits);
        else if (argmax) *argmax = q38g_out_argmax(m->g_lm_head, mixed);
        if (!q38g_active() || (argmax && !logits && *argmax < 0)) { fprintf(stderr, "[mtp] device died during draft -- aborting\n"); exit(1); }
        free(qf); free(kk); free(vv); free(ctx);
        free(xn); free(gt); free(lo); free(mixed); free(blk);
        return;
    }
    hc_mix(c, res, &l->hca_down, &l->hca_up, l->hca_norm, l->hca_inj, mixed, inject, xn, lo, gt);
    qsa_forward(m, l, ML, mixed, slot, blk);       /* dense walk over the MTP KV */
    hc_combine(c, res, blk, inject);
    hc_mix(c, res, &l->hcf_down, &l->hcf_up, l->hcf_norm, l->hcf_inj, mixed, inject, xn, lo, gt);
    moe_forward(m, l, mixed, blk);                 /* lidx = 48 -> tier layer 48 */
    hc_combine(c, res, blk, inject);
    hc_mix(c, res, &m->mtp_mix_down, &m->mtp_mix_up, m->mtp_mix_norm, NULL, mixed, NULL, xn, lo, gt);
    if (logits) rm_gemv(logits, &m->lm_head, mixed);
    else if (argmax) {
        float *tmp = falloc(c->vocab);
        rm_gemv(tmp, &m->lm_head, mixed);
        *argmax = spec_argmax(tmp, c->vocab);
        free(tmp);
    }
    free(xn); free(gt); free(lo); free(mixed); free(blk);
}

/* ==================== forward pass ==================== */

/* Run one token through all layers; logits (if wanted) into `logits`.
 * `pos` is the absolute position; the token must already be recorded in
 * m->toks[pos]. */
static void forward_token(Model *m, int tok, int pos, float *logits) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    if (tok < 0 || tok >= c->vocab) { fprintf(stderr, "token id %d out of range\n", tok); exit(1); }
    int dbg = g_debug && pos == 0;
    int dbg2 = g_debug >= 2 && pos == 0;

    float *x = falloc(D);
    if (m->vis_map && m->vis_map[pos] >= 0)         /* P6: image cell */
        memcpy(x, m->vis_embd + (size_t)m->vis_map[pos]*D, (size_t)D*sizeof(float));
    else
        rm_row_f32(&m->embed, tok, x);
    float *res = falloc(W);
    for (int s = 0; s < HC; s++) memcpy(res + s*D, x, (size_t)D*sizeof(float));
    if (dbg2) dbg_row("hc_init", res, D);

    float *xn = falloc(W), *lo = falloc(c->hc_rank), *gt = falloc(W);
    float *mixed = falloc(D), *blk = falloc(D);
    float inject[HC];

    int dbg_layer = -1;
    { const char *e = getenv("QWEN38_DEBUG_LAYER"); if (dbg2) dbg_layer = e ? atoi(e) : 0; }
    q38rt_begin(m, pos, 1);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        g_dbg_now = (i == dbg_layer) ? i : -1;
        double tl0 = g_timers ? now_s() : 0;
        if (i == c->ple_layer) {
            ple_forward(m, res, pos);
            if (dbg) fprintf(stderr, "[dbg] L%02d after-PLE res_rms=%.6f\n", i, vec_rms(res, W));
            if (g_timers) { double t = now_s(); g_tm.ple += t - tl0; tl0 = t; }
        }
        hc_mix(c, res, &l->hca_down, &l->hca_up, l->hca_norm, l->hca_inj, mixed, inject, xn, lo, gt);
        if (g_timers) { double t = now_s(); g_tm.dense_hc += t - tl0; tl0 = t; }
        if (g_dbg_now == i) {
            dbg_row("hc_norm(a)", xn, HC*D); dbg_row("hc_mixed(a)", mixed, D);
            fprintf(stderr, "[dbg] hc_inject(a)=[%.4f %.4f %.4f %.4f]\n", inject[0],inject[1],inject[2],inject[3]);
        }
        if (c->is_attn[i]) qsa_forward(m, l, i, mixed, pos, blk);
        else               gdn_forward(m, l, i, mixed, blk);
        hc_combine(c, res, blk, inject);
        if (g_timers) { double t = now_s(); if (c->is_attn[i]) g_tm.qsa += t - tl0; else g_tm.gdn += t - tl0; tl0 = t; }
        if (dbg) fprintf(stderr, "[dbg] L%02d %s res_rms=%.6f blk_rms=%.6f\n",
                         i, c->is_attn[i] ? "qsa" : "gdn", vec_rms(res, W), vec_rms(blk, D));
        hc_mix(c, res, &l->hcf_down, &l->hcf_up, l->hcf_norm, l->hcf_inj, mixed, inject, xn, lo, gt);
        if (g_timers) g_tm.dense_hc += now_s() - tl0;
        if (g_dbg_now == i) {
            dbg_row("hc_norm(f)", xn, HC*D);
            dbg_row("hc_gate(f)", gt, HC*D);
            dbg_row("hc_mixed(f)", mixed, D);
            fprintf(stderr, "[dbg] hc_inject(f)=[%.4f %.4f %.4f %.4f]\n", inject[0],inject[1],inject[2],inject[3]);
        }
        if (g_ms_decode) moe_forward_b(m, l, mixed, blk, 1);
        else             moe_forward(m, l, mixed, blk);
        hc_combine(c, res, blk, inject);
        if (dbg) fprintf(stderr, "[dbg] L%02d moe res_rms=%.6f blk_rms=%.6f\n",
                         i, vec_rms(res, W), vec_rms(blk, D));
        if (dbg2) { char tag[32]; snprintf(tag, sizeof tag, "l_last-%d", i); dbg_row(tag, res, D); }
    }
    g_dbg_now = -1;
    if (logits) {
        /* final head: output_hc mix (no inject); its mixed IS result_norm */
        double th0 = g_timers ? now_s() : 0;
        hc_mix(c, res, &m->ohc_down, &m->ohc_up, m->ohc_norm, NULL, mixed, NULL, xn, lo, gt);
        rm_gemv(logits, &m->lm_head, mixed);
        if (g_timers) g_tm.head += now_s() - th0;
        if (dbg) fprintf(stderr, "[dbg] head mixed_rms=%.6f\n", vec_rms(mixed, D));
        if (g_debug >= 2) { dbg_row("result_norm", mixed, D); dbg_row("result_output", logits, c->vocab); }
    }
    if (g_timers) g_tm.tokens++;
    q38rt_flush();
    free(x); free(res); free(xn); free(lo); free(gt); free(mixed); free(blk);
}

/* ==================== P4: dense backbone on the GPU ==================== */
/* Upload every dense matrix to VRAM and switch prefill + decode to
 * qwen38_cuda.  The GDN recurrent/conv state stays resident on the device;
 * the layer-major prefill path batches dense projections while its causal
 * recurrence still advances in token order.
 * The embedding table stays on the CPU (row gather only) and so do the PLE
 * block, the MoE router/top-k, the QSA indexer and the attention walk. */
static int q38_max_ctx(void);
/* g_gpu_qsa: declared above (P9: QSA walk on the device in batched prefill) */

/* P12: make the device mirrors cover `need` rows, doubling up to the
 * context; the expert tier releases whole slabs when free VRAM is short.
 * Failure keeps the old mirror (the callers then take the host walk). */
static int g_mirror_fail_t = 0;
static int q38_mirror_ensure(Model *m, int need) {
    Cfg *c = &m->c;
    if (!g_gpu_qsa) return 0;
    int have = q38g_qsa_alloc_t();
    if (need <= have) return 1;
    int ctx = q38_max_ctx();
    if (need > ctx) return 0;
    int new_t = have * 2;
    while (new_t < need) new_t *= 2;
    if (new_t > ctx) new_t = ctx;
    if (g_mirror_fail_t && new_t <= g_mirror_fail_t) return 0;
    size_t cur = q38g_mirror_bytes(c->n_layers, c->is_attn, c->ratio, have);
    size_t want = q38g_mirror_bytes(c->n_layers, c->is_attn, c->ratio, new_t);
    size_t delta = want > cur ? want - cur : 0;
    const size_t margin = (size_t)512 << 20;           /* keep the driver's headroom */
    size_t freeb = q38g_free_bytes();
    double t0 = now_s();
    if (freeb < delta + margin) q38t_release(delta + margin - freeb);
    q38g_sync();
    int ok = q38g_qsa_kv_grow(c->n_layers, c->is_attn, new_t) &&
             q38g_idx_grow(c->n_layers, c->is_attn, c->ratio, new_t);
    if (ok) {
        fprintf(stderr, "[q38g] KV mirror grown %d -> %d rows (+%.2f GB) in %.2fs\n",
                have, new_t, delta / 1073741824.0, now_s() - t0);
        g_mirror_fail_t = 0;
    } else {
        fprintf(stderr, "[q38g] KV mirror growth to %d rows failed (free %.2f GB) -> host attention walk beyond %d\n",
                new_t, q38g_free_bytes() / 1073741824.0, q38g_qsa_alloc_t());
        g_mirror_fail_t = new_t;
    }
    return ok;
}
static int g_sel_cap = 0;        /* ints per token in the selection list buffer */
static int gpu_dense_setup(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    if (!getenv("COLI_CUDA") || atoi(getenv("COLI_CUDA")) != 1) return 0;
    g_gpu_router = getenv("Q38_GPU_ROUTER") && atoi(getenv("Q38_GPU_ROUTER"));
    g_req_router = g_gpu_router;      /* CLI default; serve requests set it per turn */
    if (getenv("Q38_GPU_DENSE") && atoi(getenv("Q38_GPU_DENSE")) == 0) return 0;
    if (g_debug) { fprintf(stderr, "[q38g] QWEN38_DEBUG set -> dense stays on the CPU\n"); return 0; }
    if (!q38g_init(c->n_layers, c->is_attn, D, c->hc_rank, c->dn_vheads, c->dn_kheads,
                   c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim,
                   c->q_heads, c->kv_heads, c->head_dim, c->vocab, c->eps))
        return 0;
    double t0 = now_s();
    int bad = 0;
    #define UPR(dst, rm) do { (dst) = q38g_up_raw((rm).type, (rm).O, (rm).I, (rm).raw); \
                              if ((dst) < 0) bad = 1; } while (0)
    #define UPF(dst, p, n) do { (dst) = q38g_up_f32((p), (int64_t)(n)); \
                                if ((dst) < 0) bad = 1; } while (0)
    UPR(m->g_ohc_down, m->ohc_down);
    UPR(m->g_ohc_up,   m->ohc_up);
    UPF(m->g_ohc_norm, m->ohc_norm, W);
    UPR(m->g_lm_head,  m->lm_head);
    int dense_layers = c->n_layers + (m->has_mtp ? 1 : 0);
    for (int i = 0; i < dense_layers && !bad; i++) {
        Layer *l = &m->L[i];
        UPR(l->g_hca_down, l->hca_down); UPR(l->g_hca_up, l->hca_up);
        UPF(l->g_hca_norm, l->hca_norm, W); UPF(l->g_hca_inj, l->hca_inj, (int64_t)HC*W);
        UPR(l->g_hcf_down, l->hcf_down); UPR(l->g_hcf_up, l->hcf_up);
        UPF(l->g_hcf_norm, l->hcf_norm, W); UPF(l->g_hcf_inj, l->hcf_inj, (int64_t)HC*W);
        /* shared expert for the batched prefill (its Q8_0 differs from the
         * routed experts' types, so the tier cannot batch it as a pair);
         * an unsupported type just leaves it on the CPU */
        l->g_shg = l->g_shu = l->g_shd = -1;
        l->g_gate = -1;
        if (g_gpu_router && i < c->n_layers) UPF(l->g_gate, l->gate, (int64_t)c->n_experts * D);
        if (getenv("Q38_GPU_SHARED_PREFILL") && atoi(getenv("Q38_GPU_SHARED_PREFILL"))) {
            l->g_shg = q38g_up_raw(l->sh_g.type, l->sh_g.O, l->sh_g.I, l->sh_g.raw);
            l->g_shu = q38g_up_raw(l->sh_u.type, l->sh_u.O, l->sh_u.I, l->sh_u.raw);
            l->g_shd = q38g_up_raw(l->sh_d.type, l->sh_d.O, l->sh_d.I, l->sh_d.raw);
            if (l->g_shg < 0 || l->g_shu < 0 || l->g_shd < 0) l->g_shg = l->g_shu = l->g_shd = -1;
        }
        if (i == c->n_layers || c->is_attn[i]) { /* the extra MTP layer is QSA */
            UPR(l->g_q, l->q); UPR(l->g_k, l->k); UPR(l->g_v, l->v); UPR(l->g_o, l->o);
            UPF(l->g_qn, l->qn, c->head_dim); UPF(l->g_kn, l->kn, c->head_dim);
        } else {
            UPR(l->g_qkv, l->dn_qkv); UPR(l->g_z, l->dn_z); UPR(l->g_out, l->dn_out);
            {   /* one [2*vh, D] f32 mat: b rows then alpha rows (one GEMV) */
                int64_t n = (int64_t)c->dn_vheads * D;
                float *ba = falloc(2 * n);
                memcpy(ba, l->dn_b, (size_t)n * sizeof(float));
                memcpy(ba + n, l->dn_alpha, (size_t)n * sizeof(float));
                UPF(l->g_ba, ba, 2 * n);
                free(ba);
            }
            UPF(l->g_a,     l->dn_a,     c->dn_vheads);
            UPF(l->g_dt,    l->dn_dtbias, c->dn_vheads);
            UPF(l->g_norm,  l->dn_norm,  c->dn_vdim);
            UPF(l->g_conv,  l->dn_conv,  (int64_t)c->dn_conv_dim*c->dn_convk);
        }
    }
    if (m->has_mtp && !bad) {
        UPR(m->g_mtp_fce, m->mtp_fce); UPR(m->g_mtp_fch, m->mtp_fch);
        UPR(m->g_mtp_mix_down, m->mtp_mix_down); UPR(m->g_mtp_mix_up, m->mtp_mix_up);
        UPF(m->g_mtp_mix_norm, m->mtp_mix_norm, W);
        g_gpu_mtp = !bad;
    }
    #undef UPR
    #undef UPF
    if (bad || !q38g_active()) {
        fprintf(stderr, "[q38g] dense upload failed -> CPU dense path\n");
        q38g_disable();
        return 0;
    }
    if (m->has_mtp) {
        g_gpu_spec = q38g_spec_init(g_mtp_n);
        if (!g_gpu_spec)
            fprintf(stderr, "[mtp] GPU rollback allocation failed -> speculation disabled\n");
    }
    /* P9: q8 KV mirror for the device QSA walk (Q38_GPU_QSA=0 keeps the CPU
     * walk; q8 KV is required, f32 KV falls back automatically).  P12: sized
     * for Q38_KV_MIRROR_T rows (default 32768, capped by the context) and
     * grown on demand -- the expert tier hands the VRAM back then
     * (q38_mirror_ensure) -- so a 128K context no longer costs the tier
     * 2.5 GB before any long conversation exists. */
    if (!(getenv("Q38_GPU_QSA") && atoi(getenv("Q38_GPU_QSA")) == 0) && m->kv_q8) {
        int rmax = 1;
        for (int i = 0; i < c->n_layers; i++) if (c->ratio[i] > rmax) rmax = c->ratio[i];
        int sel_cap = c->idx_topk + rmax + 8;
        g_sel_cap = sel_cap;
        int t0 = getenv("Q38_KV_MIRROR_T") ? atoi(getenv("Q38_KV_MIRROR_T")) : 32768;
        if (t0 < 4096) t0 = 4096;
        if (t0 > q38_max_ctx()) t0 = q38_max_ctx();
        g_gpu_qsa = q38g_qsa_kv_setup(c->n_layers, c->is_attn, c->kv_heads, c->head_dim,
                                      t0, c->q_heads, sel_cap);
        if (!g_gpu_qsa) fprintf(stderr, "[q38g] QSA KV mirror allocation failed -> CPU attention walk\n");
        else if (!q38g_idx_setup(c->n_layers, c->is_attn, c->ratio, t0,
                                 c->idx_heads, c->idx_dim))
            fprintf(stderr, "[q38g] indexer mirror allocation failed -> host block scoring\n");
        else fprintf(stderr, "[q38g] KV mirror for %d rows (%.2f GB), grows on demand up to %d\n",
                     t0, q38g_mirror_bytes(c->n_layers, c->is_attn, c->ratio, t0) / 1073741824.0, q38_max_ctx());
    }
    fprintf(stderr, "[q38g] dense backbone on GPU: %.2f GB VRAM, uploaded in %.1fs%s\n",
            q38g_vram_used()/1073741824.0, now_s()-t0,
            g_gpu_qsa ? " (incl. QSA q8 KV mirror)" : "");
    return 1;
}

/* push the (host-authoritative) GDN state to the device; called after the CPU
 * prefill and after reset_state. */
static void gpu_state_push(Model *m) {
    Cfg *c = &m->c;
    if (!g_gpu_dense) return;
    for (int i = 0; i < c->n_layers; i++)
        if (!c->is_attn[i]) q38g_state_put(i, m->DN_rec[i], m->DN_ring[i]);
}

static void gpu_state_pull(Model *m) {
    Cfg *c = &m->c;
    if (!g_gpu_dense) return;
    for (int i = 0; i < c->n_layers; i++)
        if (!c->is_attn[i]) q38g_state_get(i, m->DN_rec[i], m->DN_ring[i]);
}

/* One decode token with the dense backbone on the device.  Mirrors
 * forward_token's structure exactly; the CPU keeps PLE, the QSA indexer +
 * attention walk, MoE routing and the expert tier. */
static void forward_token_gpu(Model *m, int tok, int pos, float *logits) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    if (tok < 0 || tok >= c->vocab) { fprintf(stderr, "token id %d out of range\n", tok); exit(1); }

    float *x = falloc(D);
    rm_row_f32(&m->embed, tok, x);
    q38g_tok_begin(x);

    float *res = falloc(W), *mixed = falloc(D), *blk = falloc(D);
    float *qfull = falloc((int64_t)H*2*hd), *kk = falloc((int64_t)KV*hd), *vv = falloc((int64_t)KV*hd);
    float *ctx = falloc((int64_t)H*hd);
    int *dsel = g_gpu_qsa ? malloc((size_t)(g_sel_cap > 0 ? g_sel_cap : 1) * sizeof(int)) : NULL;

    q38rt_begin(m, pos, 1);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        double tl0 = g_timers ? now_s() : 0;
        if (i == c->ple_layer) {
            q38g_res_get(res);
            ple_forward(m, res, pos);
            q38g_res_set(res);
            if (g_timers) { double t = now_s(); g_tm.ple += t - tl0; tl0 = t; }
        }
        if (c->is_attn[i]) {
            if (!q38g_graph_open(c->n_layers + i)) {
                q38g_hc_mix(l->g_hca_down, l->g_hca_up, l->g_hca_norm, l->g_hca_inj);
                q38g_graph_close(c->n_layers + i);
            }
            q38g_mixed_get(mixed);   /* the indexer needs x on the host */
            if (g_timers) { double t = now_s(); g_tm.dense_hc += t - tl0; tl0 = t; }
            /* P10 D1: the P9 device walk with B=1 -- selection stays on the
             * host, projections/context never leave the device; the same
             * fallbacks as the batched path keep the host walk */
            if (g_gpu_qsa && m->kv_q8 && l->g_qn >= 0 && !q38g_qsa_ready(i, pos + 1))
                q38_mirror_ensure(m, pos + 1);
            int gq = g_gpu_qsa && m->kv_q8 && l->g_qn >= 0 &&
                     q38g_qsa_ready(i, pos + 1) &&
                     (!g_full_attn || pos + 1 <= 4096) &&
                     (c->ratio[i] > 0 && l->ixk && m->IK[i] ? 1 : pos + 1 <= 4096);
            if (gq) {
                int nsel1 = -1;
                q38g_qsa_proj_batch(l->g_q, l->g_k, l->g_v, mixed, 1, NULL, NULL, NULL);
                qsa_select_batch(m, l, i, pos, 1, mixed, &nsel1, dsel, g_sel_cap);
                int walk_max = nsel1 < 0 ? pos + 1 : nsel1;
                gq = q38g_qsa_batch(i, pos, 1, l->g_qn, l->g_kn,
                                    m->K8[i], m->K8s[i], m->V8[i], m->V8s[i], m->max_t,
                                    &nsel1, dsel, c->n_rot, c->theta, c->eps, walk_max,
                                    m->mp ? m->mp + 3*pos : NULL, 3*c->rope_sec[1], 3*c->rope_sec[2]);
                if (gq) q38g_qsa_out_dev(l->g_o);
                else { fprintf(stderr, "[q38g] QSA device walk unavailable at decode pos %d -- aborting\n", pos); exit(1); }
            } else {
            q38g_qsa_proj(l->g_q, l->g_k, l->g_v, H*2*hd, KV*hd, qfull, kk, vv);
            qsa_core_x(m, l, i, mixed, pos, qfull, kk, vv, NULL, ctx);
            q38g_qsa_out(l->g_o, ctx, H*hd);
            }
            q38g_combine();
            if (g_timers) { q38g_sync(); double t = now_s(); g_tm.qsa += t - tl0; tl0 = t; }
            if (!q38g_graph_open(2*c->n_layers + i)) {
                q38g_hc_mix(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm, l->g_hcf_inj);
                q38g_graph_close(2*c->n_layers + i);
            }
            q38g_mixed_get(mixed);
            if (g_timers) { double t = now_s(); g_tm.dense_hc += t - tl0; tl0 = t; }
        } else {
            /* the whole dense GDN layer body is one captured span (both hc
             * mixes included); under COLI_TIMERS it lands in the gdn bucket */
            if (!q38g_graph_open(i)) {
                q38g_hc_mix(l->g_hca_down, l->g_hca_up, l->g_hca_norm, l->g_hca_inj);
                q38g_gdn(i, l->g_qkv, l->g_z, l->g_ba,
                         l->g_a, l->g_dt, l->g_norm, l->g_conv, l->g_out);
                q38g_combine();
                q38g_hc_mix(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm, l->g_hcf_inj);
                q38g_graph_close(i);
            }
            q38g_mixed_get(mixed);
            if (g_timers) { double t = now_s(); g_tm.gdn += t - tl0; tl0 = t; }
        }
        if (g_req_router && l->g_gate >= 0) {   /* P10 D4: router logits from d_mixed */
            static float *pr1 = NULL; if (!pr1) pr1 = falloc(c->n_experts);
            if (q38g_router(l->g_gate, c->n_experts, pr1)) g_pr_pre = pr1;
        }
        if (g_ms_decode) moe_forward_b(m, l, mixed, blk, 1);
        else             moe_forward(m, l, mixed, blk);
        q38g_blk_combine(blk);
        if (!q38g_active()) { fprintf(stderr, "[q38g] device died mid-token -- aborting\n"); exit(1); }
    }
    if (logits) {
        double th0 = g_timers ? now_s() : 0;
        q38g_head(m->g_ohc_down, m->g_ohc_up, m->g_ohc_norm, m->g_lm_head, logits);
        if (g_timers) g_tm.head += now_s() - th0;
    }
    if (g_timers) g_tm.tokens++;
    q38rt_flush();
    free(x); free(res); free(mixed); free(blk);
    free(qfull); free(kk); free(vv); free(ctx); free(dsel);
}

/* Capture every fixed decode graph before the expert tier measures and fills
 * the remaining VRAM.  MTP verification uses the batched path, so a greedy
 * smoke turn can otherwise leave these graphs uncaptured until a later
 * reasoning/non-greedy turn, when cudaGraphInstantiate may have no headroom. */
static int gpu_graph_prime(Model *m) {
    const char *graphs = getenv("Q38_GPU_GRAPHS");
    const char *kprof = getenv("Q38_GPU_KPROF");
    if (!g_gpu_dense || (graphs && atoi(graphs) == 0) || (kprof && atoi(kprof)))
        return 1;
    Cfg *c = &m->c;
    float *x = fcalloc(c->hidden), *logits = falloc(c->vocab);
    double t0 = now_s();
    gpu_state_push(m);                          /* q38g recurrent slabs start uninitialised */
    q38g_tok_begin(x);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (c->is_attn[i]) {
            if (!q38g_graph_open(c->n_layers + i)) {
                q38g_hc_mix(l->g_hca_down, l->g_hca_up, l->g_hca_norm, l->g_hca_inj);
                q38g_graph_close(c->n_layers + i);
            }
            if (!q38g_graph_open(2*c->n_layers + i)) {
                q38g_hc_mix(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm, l->g_hcf_inj);
                q38g_graph_close(2*c->n_layers + i);
            }
        } else if (!q38g_graph_open(i)) {
            q38g_hc_mix(l->g_hca_down, l->g_hca_up, l->g_hca_norm, l->g_hca_inj);
            q38g_gdn(i, l->g_qkv, l->g_z, l->g_ba,
                     l->g_a, l->g_dt, l->g_norm, l->g_conv, l->g_out);
            q38g_combine();
            q38g_hc_mix(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm, l->g_hcf_inj);
            q38g_graph_close(i);
        }
        if (!q38g_active()) break;
    }
    if (q38g_active())
        q38g_head(m->g_ohc_down, m->g_ohc_up, m->g_ohc_norm, m->g_lm_head, logits);
    int ok = q38g_active();
    gpu_state_push(m);                          /* discard the synthetic recurrent step */
    free(x); free(logits);
    if (ok) fprintf(stderr, "[q38g] decode graphs primed before expert warmstart in %.2fs\n",
                    now_s() - t0);
    else fprintf(stderr, "[q38g] decode graph priming failed\n");
    return ok;
}

/* ==================== P5: intra-batch state snapshots ====================
 * The MTP verify runs the backbone over N+1 tokens in one batched forward;
 * on a partial acceptance the recurrent state must return to the acceptance
 * point.  The batch is layer-major, so "state after batch token j" is
 * captured layer-by-layer: after gdn_core advances token j in layer i, that
 * layer's rec+ring is copied into boundary j's snapshot slice (same for the
 * PLE history at the PLE layer).  Restore memcpys boundary A back -- byte
 * copies of state the sequential path would have produced, so the resumed
 * decode is bit-identical to a sequential decode that stopped at A.
 * QSA KV rows, indexer raw keys and the MTP KV are rolled back positionally
 * (later writes at the same slots overwrite them before any read); the pooled
 * block count nblk is trimmed to the committed length. */
static int    g_spec_snap;              /* boundaries 0..B-2 recorded when set */
static int    g_spec_verify;            /* verify batch: skip the prefill-phase prefetch */
static float *g_snap_rec[MTP_MAX_N], *g_snap_ring[MTP_MAX_N], *g_snap_ple[MTP_MAX_N];
static int    g_snap_plen[MTP_MAX_N];
static int    g_gdn_idx[64];            /* layer -> compact GDN slot, -1 = QSA */
static int    g_n_gdn;
static int64_t g_recsz, g_ringsz, g_plesz;

static void fcopy_par(float *dst, const float *src, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 65536) {
        int64_t cnt = n - i < 65536 ? n - i : 65536;
        memcpy(dst + i, src + i, (size_t)cnt * sizeof(float));
    }
}

static void spec_snap_init(Model *m, int nmax) {
    Cfg *c = &m->c;
    if (g_snap_ple[0]) return;
    g_n_gdn = 0;
    for (int i = 0; i < c->n_layers && i < 64; i++)
        g_gdn_idx[i] = c->is_attn[i] ? -1 : g_n_gdn++;
    g_recsz  = (int64_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    g_ringsz = (int64_t)c->dn_conv_dim * (c->dn_convk - 1);
    g_plesz  = (int64_t)(c->ple_convk-1) * c->ple_ngram * HC * c->hidden;
    for (int j = 0; j < nmax && j < MTP_MAX_N; j++) {
        if (!g_gpu_dense) {
            g_snap_rec[j]  = falloc(g_recsz * g_n_gdn);
            g_snap_ring[j] = falloc(g_ringsz * g_n_gdn);
        }
        g_snap_ple[j]  = falloc(g_plesz);
    }
}

/* restore boundary j (state after batch token j) into the live state */
static void spec_restore(Model *m, int j) {
    Cfg *c = &m->c;
    if (g_gpu_dense) q38g_spec_restore(j);
    else for (int i = 0; i < c->n_layers; i++) {
            int gi = g_gdn_idx[i];
            if (gi < 0) continue;
            fcopy_par(m->DN_rec[i],  g_snap_rec[j]  + gi*g_recsz,  g_recsz);
            fcopy_par(m->DN_ring[i], g_snap_ring[j] + gi*g_ringsz, g_ringsz);
    }
    memcpy(m->ple_hist, g_snap_ple[j], (size_t)g_plesz * sizeof(float));
    m->ple_hist_n = g_snap_plen[j];
}

/* S3: layer-major chunked prefill.  B tokens walk the layers together so the
 * dense matrices stream from DRAM once per chunk (and the MoE can batch its
 * tier issue); every stateful piece (GDN conv/rec, PLE history, KV writes,
 * indexer pooling) still advances token by token inside the chunk, in the
 * exact forward_token order, so the results are bit-identical to a
 * token-at-a-time prefill.  Tokens toks[0..B-1] sit at pos0..pos0+B-1 and
 * must already be recorded in m->toks. */
/* P5 extensions (all NULL/0 on the prefill path, which stays bit-identical):
 *   ALL     [B*vocab]  head logits for EVERY batch token (verify wants each
 *                      position's argmax; the last row equals `logits`)
 *   RES_OUT [B*4*2560] the final wide residual per token (MTP fold input)
 *   g_spec_snap        record rec/ring/PLE snapshots at boundaries 0..B-2 */
static void forward_tokens_batch_ex(Model *m, const int *toks, int pos0, int B,
                                    float *logits, float *ALL, float *RES_OUT) {
    Cfg *c = &m->c;
    int D = c->hidden, W = HC*D;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int vh = c->dn_vheads, cdim = c->dn_conv_dim, vtot = vh * c->dn_vdim;
    int gpu = g_gpu_dense && (!g_spec_snap || g_gpu_spec) && q38g_batch_max() >= B;

    float *RES = falloc((int64_t)B*W);
    float *MIXED = falloc((int64_t)B*D), *BLK = falloc((int64_t)B*D);
    float *INJ = falloc((int64_t)B*HC);
    for (int t = 0; t < B; t++) {
        int tok = toks[t];
        if (tok < 0 || tok >= c->vocab) { fprintf(stderr, "token id %d out of range\n", tok); exit(1); }
        float *res = RES + (int64_t)t*W;
        if (m->vis_map && m->vis_map[pos0 + t] >= 0)    /* P6: image cell */
            memcpy(res, m->vis_embd + (size_t)m->vis_map[pos0 + t]*D, (size_t)D*sizeof(float));
        else
            rm_row_f32(&m->embed, tok, res);
        for (int s = 1; s < HC; s++) memcpy(res + s*D, res, (size_t)D*sizeof(float));
    }

    /* attention/GDN projection buffers, chunk-wide */
    float *QF = falloc((int64_t)B*H*2*hd);
    float *KK = falloc((int64_t)B*KV*hd), *VV = falloc((int64_t)B*KV*hd);
    float *CTX = gpu ? falloc((int64_t)B*H*hd) : NULL;
    int *NSEL = g_gpu_qsa ? malloc((size_t)B * sizeof(int)) : NULL;
    int *SEL  = g_gpu_qsa ? malloc((size_t)B * g_sel_cap * sizeof(int)) : NULL;
    float *QKV = gpu ? NULL : falloc((int64_t)B*cdim);
    float *Z = gpu ? NULL : falloc((int64_t)B*vtot);
    float *BB = gpu ? NULL : falloc((int64_t)B*vh);
    float *AL = gpu ? NULL : falloc((int64_t)B*vh);

    /* P4: the whole chunk's token ids are known up front -- prefetch the
     * shallow-table predictions for every chunk token before layer 0 runs.
     * The phase flag routes this chunk's speculation through the
     * Q38_PREFETCH_PREFILL enable (default on: measured +6%). */
    if (q38t_ready() && !g_spec_verify) {
        q38t_prefetch_phase(1);
        for (int t = 0; t < B; t++) q38t_prefetch_next(q38_bigram_mixed(m, pos0 + t));
    }
    /* P3: batch-fault the chunk's PLE rows before layer 0 (no math changes) */
    ple_warm_chunk(m, pos0, B);

    q38rt_begin(m, pos0, B);
    /* P9: the chunk residual lives on the device for the whole layer loop;
     * it comes back to the host only around the PLE layer and at the end.
     * Same kernels and data as the per-helper transfers, so bit-identical. */
    if (gpu) q38g_res_batch_put(RES, B);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        double tl0 = g_timers ? now_s() : 0;
        if (i == c->ple_layer) {
            if (gpu) q38g_res_batch_get(RES, B);
            for (int t = 0; t < B; t++) {
                ple_forward(m, RES + (int64_t)t*W, pos0 + t);
                if (g_spec_snap && t < B-1) {
                    memcpy(g_snap_ple[t], m->ple_hist, (size_t)g_plesz * sizeof(float));
                    g_snap_plen[t] = m->ple_hist_n;
                }
            }
            if (gpu) q38g_res_batch_put(RES, B);
            if (g_timers) { double tt = now_s(); g_tm.ple += tt - tl0; tl0 = tt; }
        }
        if (gpu) q38g_hc_mix_batch(l->g_hca_down, l->g_hca_up, l->g_hca_norm,
                                   l->g_hca_inj, NULL, B, MIXED, INJ);
        else     hc_mix_b(c, RES, &l->hca_down, &l->hca_up, l->hca_norm,
                          l->hca_inj, MIXED, INJ, B);
        if (g_timers) { double tt = now_s(); g_tm.dense_hc += tt - tl0; tl0 = tt; }
        if (c->is_attn[i]) {
            /* P9: device walk when the chunk's cell lists fit the kernel's
             * shared-memory bound (selection stays on the host, so the exact
             * block/tie semantics are untouched); f32 KV keeps the CPU walk;
             * image prompts pass their IMRoPE positions (P6.3) */
            if (gpu && g_gpu_qsa && m->kv_q8 && l->g_qn >= 0 && !q38g_qsa_ready(i, pos0 + B))
                q38_mirror_ensure(m, pos0 + B);
            int gq = gpu && g_gpu_qsa && m->kv_q8 && l->g_qn >= 0 &&
                     q38g_qsa_ready(i, pos0 + B) &&
                     (!g_full_attn || pos0 + B <= 4096) &&
                     (c->ratio[i] > 0 && l->ixk && m->IK[i] ? 1 : pos0 + B <= 4096);
            if (gpu) q38g_qsa_proj_batch(l->g_q, l->g_k, l->g_v, NULL, B,
                                         gq ? NULL : QF, KK, VV);
            else {
                rm_gemv_b(QF, &l->q, MIXED, B, H*2*hd);
                rm_gemv_b(KK, &l->k, MIXED, B, KV*hd);
                rm_gemv_b(VV, &l->v, MIXED, B, KV*hd);
            }
            if (gq) {
                int walk_max = 0;
                qsa_select_batch(m, l, i, pos0, B, MIXED, NSEL, SEL, g_sel_cap);
                for (int t = 0; t < B; t++) {
                    int nw = NSEL[t] < 0 ? pos0 + t + 1 : NSEL[t];
                    if (nw > walk_max) walk_max = nw;
                }
                gq = q38g_qsa_batch(i, pos0, B, l->g_qn, l->g_kn,
                                    m->K8[i], m->K8s[i], m->V8[i], m->V8s[i], m->max_t,
                                    NSEL, SEL, c->n_rot, c->theta, c->eps, walk_max,
                                    m->mp ? m->mp + 3*pos0 : NULL, 3*c->rope_sec[1], 3*c->rope_sec[2]);
                if (gq) q38g_qsa_out_batch(l->g_o, B, BLK);
                else {
                    /* device refused (walk too long / CUDA error): finish this
                     * layer on the host from the projections; the selection
                     * already advanced the indexer, so qsa_core_x is not
                     * re-entered -- fetch projections and use the cell lists */
                    fprintf(stderr, "[q38g] QSA device walk unavailable at layer %d pos %d -- aborting\n", i, pos0);
                    exit(1);
                }
            } else {
            for (int t = 0; t < B; t++) {
                float *ctx = gpu ? CTX + (int64_t)t*H*hd : NULL;
                qsa_core_x(m, l, i, MIXED + (int64_t)t*D, pos0 + t,
                           QF + (int64_t)t*H*2*hd, KK + (int64_t)t*KV*hd,
                           VV + (int64_t)t*KV*hd,
                           gpu ? NULL : BLK + (int64_t)t*D, ctx);
            }
            if (gpu) q38g_out_batch(l->g_o, CTX, B, BLK);
            }
        } else {
            if (gpu) {
                q38g_gdn_batch(i, l->g_qkv, l->g_z, l->g_ba,
                               l->g_a, l->g_dt, l->g_norm, l->g_conv, l->g_out,
                               NULL, B, BLK, g_spec_snap);
            } else {
                rm_gemv_b(QKV, &l->dn_qkv, MIXED, B, cdim);
                rm_gemv_b(Z,   &l->dn_z,   MIXED, B, vtot);
                matf_b(BB, l->dn_b,     MIXED, D, vh, B, vh);
                matf_b(AL, l->dn_alpha, MIXED, D, vh, B, vh);
                for (int t = 0; t < B; t++) {
                    gdn_core(m, l, i, QKV + (int64_t)t*cdim, Z + (int64_t)t*vtot,
                             BB + (int64_t)t*vh, AL + (int64_t)t*vh, BLK + (int64_t)t*D);
                    if (g_spec_snap && t < B-1 && g_gdn_idx[i] >= 0) {
                        fcopy_par(g_snap_rec[t]  + g_gdn_idx[i]*g_recsz,  m->DN_rec[i],  g_recsz);
                        fcopy_par(g_snap_ring[t] + g_gdn_idx[i]*g_ringsz, m->DN_ring[i], g_ringsz);
                    }
                }
            }
        }
        if (gpu) q38g_combine_batch(NULL, B, NULL);
        else for (int t = 0; t < B; t++)
                 hc_combine(c, RES + (int64_t)t*W, BLK + (int64_t)t*D,
                            INJ + (int64_t)t*HC);
        if (g_timers) { double tt = now_s(); if (c->is_attn[i]) g_tm.qsa += tt - tl0; else g_tm.gdn += tt - tl0; tl0 = tt; }
        if (gpu) q38g_hc_mix_batch(l->g_hcf_down, l->g_hcf_up, l->g_hcf_norm,
                                   l->g_hcf_inj, NULL, B, MIXED, INJ);
        else     hc_mix_b(c, RES, &l->hcf_down, &l->hcf_up, l->hcf_norm,
                          l->hcf_inj, MIXED, INJ, B);
        if (g_timers) g_tm.dense_hc += now_s() - tl0;
        /* P10 P7a: d_b_mixed holds MIXED after the mix (synced): the tier's
         * grouped kernels read it there instead of re-uploading MIXED */
        q38t_set_device_x(gpu ? q38g_b_mixed_dev() : NULL);
        if (gpu && g_req_router && l->g_gate >= 0) {   /* P10 D4: router logits from d_b_mixed */
            float *prb = falloc((int64_t)B * c->n_experts);
            if (q38g_router_batch(l->g_gate, c->n_experts, B, prb)) g_pr_pre = prb;
            moe_forward_b(m, l, MIXED, BLK, B);
            free(prb);
        } else
        moe_forward_b(m, l, MIXED, BLK, B);
        q38t_set_device_x(NULL);
        if (gpu) q38g_combine_batch(BLK, B, NULL);
        else for (int t = 0; t < B; t++)
                 hc_combine(c, RES + (int64_t)t*W, BLK + (int64_t)t*D,
                            INJ + (int64_t)t*HC);
        /* Q38_RES_DUMP=<path> + Q38_RES_DUMP_POS=<pos>: the residual after
         * every layer at one prompt position (48 x W f32), for the
         * hidden-state error measurement around a routing divergence */
        {
            static const char *rd = NULL; static int rdpos = -2;
            if (rdpos == -2) { rd = getenv("Q38_RES_DUMP"); rdpos = getenv("Q38_RES_DUMP_POS") ? atoi(getenv("Q38_RES_DUMP_POS")) : -1; }
            if (rd && rdpos >= pos0 && rdpos < pos0 + B) {
                if (gpu) q38g_res_batch_get(RES, B);
                FILE *f = fopen(rd, i == 0 ? "wb" : "ab");
                if (f) { fwrite(RES + (int64_t)(rdpos - pos0) * W, sizeof(float), (size_t)W, f); fclose(f); }
            }
        }
        if (gpu && !q38g_active()) {
            fprintf(stderr, "[q38g] device died during batched prefill -- aborting\n");
            exit(1);
        }
    }
    if (gpu) q38g_res_batch_get(RES, B);
    if (ALL) {
        /* P5 verify: head logits for every batch token.  Per-token hc_mix in
         * the exact single-token order, then one batched lm_head GEMV
         * (mati8_b / looped rm_gemv are bit-identical to B single calls). */
        double th0 = g_timers ? now_s() : 0;
        float *xn = NULL, *lo = NULL, *gt = NULL;
        if (gpu) {
            q38g_hc_mix_batch(m->g_ohc_down, m->g_ohc_up, m->g_ohc_norm, -1,
                              RES, B, MIXED, NULL);
            q38g_out_batch(m->g_lm_head, MIXED, B, ALL);
        } else {
            xn = falloc(W); lo = falloc(c->hc_rank); gt = falloc(W);
            for (int t = 0; t < B; t++)
                hc_mix(c, RES + (int64_t)t*W, &m->ohc_down, &m->ohc_up, m->ohc_norm,
                       NULL, MIXED + (int64_t)t*D, NULL, xn, lo, gt);
            rm_gemv_b(ALL, &m->lm_head, MIXED, B, c->vocab);
        }
        if (logits) memcpy(logits, ALL + (int64_t)(B-1)*c->vocab, (size_t)c->vocab*sizeof(float));
        if (g_timers) g_tm.head += now_s() - th0;
        free(xn); free(lo); free(gt);
    } else if (logits) {
        double th0 = g_timers ? now_s() : 0;
        float *mixed = MIXED;   /* reuse token 0's slot for the head */
        if (gpu) {
            q38g_hc_mix_batch(m->g_ohc_down, m->g_ohc_up, m->g_ohc_norm, -1,
                              RES + (int64_t)(B-1)*W, 1, mixed, NULL);
            q38g_out_batch(m->g_lm_head, mixed, 1, logits);
        } else {
            float *xn = falloc(W), *lo = falloc(c->hc_rank), *gt = falloc(W);
            hc_mix(c, RES + (int64_t)(B-1)*W, &m->ohc_down, &m->ohc_up, m->ohc_norm,
                   NULL, mixed, NULL, xn, lo, gt);
            rm_gemv(logits, &m->lm_head, mixed);
            free(xn); free(lo); free(gt);
        }
        if (g_timers) g_tm.head += now_s() - th0;
    }
    if (RES_OUT) memcpy(RES_OUT, RES, (size_t)B * W * sizeof(float));
    if (g_timers) g_tm.tokens += B;
    q38rt_flush();
    free(RES); free(MIXED); free(BLK); free(INJ);
    free(QF); free(KK); free(VV); free(CTX); free(QKV); free(Z); free(BB); free(AL);
    free(NSEL); free(SEL);
}

static void forward_tokens_batch(Model *m, const int *toks, int pos0, int B, float *logits) {
    forward_tokens_batch_ex(m, toks, pos0, B, logits, NULL, NULL);
}

/* ==================== main ==================== */
static int q38_max_ctx(void) {
    const char *e = getenv("Q38_CTX");           /* S3 name; Q38_MAXT kept */
    if (!e || !*e) e = getenv("Q38_MAXT");
    int v = (e && *e) ? atoi(e) : Q38_DEFAULT_MAX_CTX;
    if (v < 1) v = Q38_DEFAULT_MAX_CTX;
    return v > Q38_MAX_CTX ? Q38_MAX_CTX : v;
}

/* S3: load-time host-memory report for the configured context. */
static void memory_report(Model *m, int max_ctx) {
    Cfg *c = &m->c;
    int n_qsa = 0, n_idx = 0, n_gdn = 0;
    size_t idx_bytes = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->is_attn[i]) { n_gdn++; continue; }
        n_qsa++;
        if (m->L[i].ixk && c->ratio[i] > 0) {
            n_idx++;
            idx_bytes += ((size_t)max_ctx + (size_t)max_ctx / c->ratio[i] + 1)
                         * c->idx_dim * sizeof(float);
        }
    }
    if (m->has_mtp) n_qsa++;       /* dense-attention MTP layer has its own KV */
    size_t rowb = m->kv_q8 ? (size_t)c->head_dim + (c->head_dim/32)*sizeof(float)
                           : (size_t)c->head_dim * sizeof(float);
    size_t kv_bytes  = (size_t)n_qsa * c->kv_heads * max_ctx * rowb * 2;
    size_t gdn_bytes = (size_t)n_gdn *
        ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim +
         (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * sizeof(float);
    size_t ple_bytes = (size_t)(c->ple_convk-1) * c->ple_ngram * HC * c->hidden * sizeof(float);
    fprintf(stderr, "[mem] ctx=%d | attn KV %.2f GB (%s, %d layers x %d heads x %d dim x2) | "
            "indexer cache %.2f GB (%d layers) | GDN state %.3f GB (fixed) | PLE hist %.1f MB\n",
            max_ctx, kv_bytes/1073741824.0, m->kv_q8 ? "q8" : "f32",
            n_qsa, c->kv_heads, c->head_dim,
            idx_bytes/1073741824.0, n_idx, gdn_bytes/1073741824.0, ple_bytes/1048576.0);
}

/* ---- prefill / decode, shared by the CLI and serve entry points ----
 * These are the argv path's own loops lifted verbatim into functions so the
 * two entry points cannot drift: the greedy-32 reference gates measure the
 * CLI, and a serve mode with its own copy of the chunking would eventually
 * be a second, unmeasured forward path.  Chunk size, the B==1 special case,
 * and the "logits only for the last prompt token" rule remain shared. */
static int q38_prefill_chunk(void) {
    int pfb = getenv("Q38_PREFILL_B") ? atoi(getenv("Q38_PREFILL_B")) : 1024;
    if (pfb < 1) pfb = 1;
    if (pfb > 1024) pfb = 1024;
    if (g_debug >= 2) pfb = 1;        /* the debug taps are per-token */
    return pfb;
}

/* P5: after a chunk's backbone layers finish, absorb its tokens into the MTP
 * layer's KV cache.  Slot t folds the backbone res_hc AFTER token t with the
 * embedding of token t+1, so the chunk's last position waits (in mtp_pend)
 * until the next chunk -- or the first sampled token -- names its successor. */
static void mtp_absorb_chunk(Model *m, const float *RES, int pos0, int B) {
    int W = HC * m->c.hidden;
    if (pos0 > 0 && m->mtp_pend_pos == pos0 - 1)
        mtp_absorb(m, m->mtp_pend, pos0 - 1, m->toks[pos0]);
    for (int t = 0; t < B - 1; t++)
        mtp_absorb(m, RES + (int64_t)t*W, pos0 + t, m->toks[pos0 + t + 1]);
    memcpy(m->mtp_pend, RES + (int64_t)(B-1)*W, (size_t)W * sizeof(float));
    m->mtp_pend_pos = pos0 + B - 1;
}

/* P3: `start` > 0 resumes a pool-restored state -- m->toks[0..start-1] and
 * every cache already cover those positions, so only the tail is prefilled.
 * Chunk boundaries shift with `start`; every causal state still advances in
 * position order, so the tail resumes the restored prefix exactly. */
/* P9 preparation UX: while a serve request prefills, emit
 *   PROGRESS <id> <np> <restored> <done> <prefix_end> <elapsed>
 * after a chunk at most once per second (and at the end of each pass), so the
 * gateway can show an ETA and whether an unseen stable prefix is being
 * computed (done <= prefix_end) or the private prompt tail. */
static struct { const char *id; int np, restored, prefix_end; double t0, last; } g_sprog;
static void q38_serve_progress(int done, int force) {
    if (!g_sprog.id) return;
    double now = now_s();
    if (!force && now - g_sprog.last < 1.0) return;
    g_sprog.last = now;
    printf("PROGRESS %s %d %d %d %d %.3f\n", g_sprog.id, g_sprog.np, g_sprog.restored,
           done, g_sprog.prefix_end, now - g_sprog.t0);
    fflush(stdout);
}

static int q38_prefill_from(Model *m, const int *prompt, int np, int start,
                            float *logits, int progress,
                            int (*poll_control)(void *), void *poll_context) {
    int pfb = q38_prefill_chunk();
    int gpu = g_gpu_dense && pfb > 1 && q38g_batch_max() >= pfb;
    if (gpu) gpu_state_push(m);       /* reset/restored host state -> device */
    double tp0 = now_s();
    float *RES = m->has_mtp ? falloc((int64_t)pfb * HC * m->c.hidden) : NULL;
    for (int i = start; i < np; ) {
        int B = np - i < pfb ? np - i : pfb;
        for (int j = 0; j < B; j++) m->toks[i+j] = prompt[i+j];
        m->n_toks = i + B;
        if (!gpu && B == 1 && !m->has_mtp)
            forward_token(m, prompt[i], i, i+1 == np ? logits : NULL);
        else
            forward_tokens_batch_ex(m, prompt + i, i, B, i+B == np ? logits : NULL,
                                    NULL, RES);
        if (RES) mtp_absorb_chunk(m, RES, i, B);
        i += B;
        if (poll_control) q38_serve_progress(i, i == np);
        if (progress && (g_debug || (i % 2048 < B && i >= 2048)))
            fprintf(stderr, "[dbg] prefill %d/%d done (%.1fs, %.1f tok/s)\n",
                    i, np, now_s()-tp0, i/(now_s()-tp0));
        if (poll_control && poll_control(poll_context)) break;
    }
    free(RES);
    return gpu;
}
static int q38_prefill(Model *m, const int *prompt, int np, float *logits, int progress) {
    return q38_prefill_from(m, prompt, np, 0, logits, progress, NULL, NULL);
}

/* One decode step for token `tok` at absolute position `pos`, including the
 * P4 bigram prefetch hook that fires as soon as the next token's identity is
 * known.  Mirrors the argv decode loop's body exactly. */
static void q38_decode_step(Model *m, int tok, int pos, float *logits) {
    m->toks[pos] = tok; m->n_toks = pos + 1;
    /* P6: after an image the rope position runs behind the cell index */
    if (m->mp) m->mp[3*pos] = m->mp[3*pos+1] = m->mp[3*pos+2] = m->mp_next++;
    ple_warm_token(m, pos);      /* P3: fault this token's 16 PLE rows early */
    if (q38t_ready()) {
        q38t_prefetch_phase(0);
        q38t_prefetch_next(q38_bigram_mixed(m, pos));
    }
    if (g_gpu_dense) forward_token_gpu(m, tok, pos, logits);
    else             forward_token(m, tok, pos, logits);
}

/* ==================== P5: draft-verify decode step ====================
 * Standard speculative-decoding semantics under greedy target sampling:
 * given the freshly sampled token `c` at position p (not yet forwarded),
 *   1. chain N MTP drafts d1..dN from (pend res_hc @ p-1, c);
 *   2. verify: ONE batched backbone forward over [c, d1..dN] at p..p+N,
 *      head logits for every position;
 *   3. accept the longest prefix with argmax(logits[p+j]) == d_{j+1};
 *   4. roll recurrent state back to the acceptance point (snapshots),
 *      teacher-force the committed slots into the MTP KV;
 *   5. return the A accepted drafts in extra[]; `logits` holds position
 *      p+A's logits, whose argmax IS the corrected token -- the caller's own
 *      greedy sampler recovers it, so emitted output is token-identical to
 *      the non-speculative greedy decode.
 * N=0 degrades to a plain (batched, bit-identical) decode step that keeps
 * the pend chain alive. */
static int  g_mtp_debug;                 /* Q38_MTP_DEBUG=1: per-step draft-vs-target dump */
static long g_mtp_steps, g_mtp_drafted, g_mtp_accepted;
static long g_mtp_pos_acc[MTP_MAX_N], g_mtp_pos_tot[MTP_MAX_N];

static int spec_argmax(const float *lo, int V) {
    int b = 0;
    for (int i = 1; i < V; i++) if (lo[i] > lo[b]) b = i;
    return b;
}

static int q38_spec_step(Model *m, int c, int p, int max_extra,
                         float *logits, int *extra) {
    Cfg *cc = &m->c;
    int D = cc->hidden, W = HC*D, V = cc->vocab;
    int N = g_mtp_n;
    if (N > max_extra) N = max_extra;
    if (p + 1 + N > m->max_t) N = m->max_t - 1 - p;
    if (N < 0) N = 0;
    if (m->mtp_pend_pos != p - 1) N = 0;     /* pend chain broken: plain step */
    int B = N + 1;
    int toks[MTP_MAX_N + 2]; toks[0] = c;
    /* P6.4: after an image the rope position is not the cell index; the
     * draft and verify cells p..p+B-1 get their (t,y,x) = sequential
     * positions now (the MTP draft ropes its KV rows through rope_head_m and
     * the batched verify passes them to the device), and mp_next resumes at
     * the accepted length below. */
    const int mp0 = m->mp_next;
    if (m->mp) for (int j = 0; j < B; j++) m->mp[3*(p+j)] = m->mp[3*(p+j)+1] = m->mp[3*(p+j)+2] = mp0 + j;

    if (N > 0) {
        spec_snap_init(m, g_mtp_n);
        float *r0 = falloc(W), *r1 = falloc(W), *dl = falloc(V);
        const float *prev = m->mtp_pend;
        for (int j = 0; j < N; j++) {
            /* KV slot = position of the HIDDEN (p-1+j), matching the absorb
             * convention -- draft 0's write at slot p-1 is byte-for-byte the
             * teacher-forced content (backbone res @ p-1 + committed token c),
             * so it permanently completes the pending slot. */
            mtp_draft(m, prev, toks[j], p - 1 + j, r1, NULL, &toks[j+1]);
            float *t2 = r0; r0 = r1; r1 = t2; prev = r0;
        }
        free(r0); free(r1); free(dl);
    }

    for (int j = 0; j < B; j++) m->toks[p + j] = toks[j];
    m->n_toks = p + B;
    if (q38t_ready()) {                       /* draft tokens extend the P4
                                               * bigram lookahead (decode phase:
                                               * Q38_PREFETCH_DECODE gates it,
                                               * default off) */
        q38t_prefetch_phase(0);
        for (int j = 0; j < B; j++) q38t_prefetch_next(q38_bigram_mixed(m, p + j));
    }

    static float *ALL = NULL, *RESV = NULL;
    if (!ALL) { ALL = falloc((int64_t)(MTP_MAX_N+1) * V); RESV = falloc((int64_t)(MTP_MAX_N+1) * W); }
    g_spec_snap = (B > 1); g_spec_verify = 1;
    forward_tokens_batch_ex(m, toks, p, B, NULL, ALL, RESV);
    g_spec_snap = 0; g_spec_verify = 0;

    int A = 0;
    while (A < N && spec_argmax(ALL + (int64_t)A*V, V) == toks[A+1]) A++;
    if (m->mp) m->mp_next = mp0 + A + 1;      /* cells p..p+A stay */
    if (g_mtp_debug) {
        fprintf(stderr, "[mtp-dbg] p=%d c=%d draft:", p, c);
        for (int j = 1; j <= N; j++) fprintf(stderr, " %d", toks[j]);
        fprintf(stderr, " | target:");
        for (int j = 0; j < N; j++) {
            int tg = spec_argmax(ALL + (int64_t)j*V, V);
            fprintf(stderr, " %d%s", tg, j == 0 ? "" : "?");   /* rows >0 only valid on-path */
        }
        fprintf(stderr, " | A=%d\n", A);
    }

    if (A < N) {                              /* partial acceptance: roll back */
        spec_restore(m, A);
        m->n_toks = p + A + 1;
        int nbmin = -1;
        for (int i = 0; i < cc->n_layers; i++)
            if (cc->is_attn[i] && m->IK[i] && cc->ratio[i] > 0) {
                int nb = (p + A + 1) / cc->ratio[i];
                if (m->nblk[i] > nb) m->nblk[i] = nb;
                if (nbmin < 0 || nb < nbmin) nbmin = nb;
            }
        /* P10 D1: the device KV/indexer mirrors are ahead of the rollback */
        q38g_qsa_kv_invalidate(p + A + 1);
        if (nbmin >= 0) q38g_idx_invalidate(nbmin);
    }

    /* teacher-forced MTP KV for the committed slots; pend = res @ p+A */
    for (int j = 0; j < A; j++) mtp_absorb(m, RESV + (int64_t)j*W, p + j, toks[j+1]);
    memcpy(m->mtp_pend, RESV + (int64_t)A*W, (size_t)W * sizeof(float));
    m->mtp_pend_pos = p + A;

    g_mtp_steps++; g_mtp_drafted += N; g_mtp_accepted += A;
    for (int j = 0; j < N && j < MTP_MAX_N; j++) {
        g_mtp_pos_tot[j]++;
        if (j < A) g_mtp_pos_acc[j]++;
    }

    memcpy(logits, ALL + (int64_t)A*V, (size_t)V * sizeof(float));
    for (int j = 0; j < A; j++) extra[j] = toks[j+1];
    return A;
}

static void mtp_print_stats(void) {
    if (!g_mtp_steps) return;
    fprintf(stderr, "[mtp] steps=%ld drafted=%ld accepted=%ld (%.1f%%) tok/step=%.2f | acc/pos:",
            g_mtp_steps, g_mtp_drafted, g_mtp_accepted,
            g_mtp_drafted ? 100.0*g_mtp_accepted/g_mtp_drafted : 0.0,
            (double)(g_mtp_steps + g_mtp_accepted) / g_mtp_steps);
    for (int j = 0; j < MTP_MAX_N && g_mtp_pos_tot[j]; j++)
        fprintf(stderr, " %ld/%ld", g_mtp_pos_acc[j], g_mtp_pos_tot[j]);
    fprintf(stderr, "\n");
}

/* ==================== P6: vision ====================
 * The tower (vision_qwen3vl.h) is loaded lazily from <snap>/mmproj-F16.gguf
 * (Q38_MMPROJ overrides) on the first image, so text-only startup cost is
 * unchanged.  The gateway preprocesses (tools/qwen38_image.py: llama.cpp
 * smart-resize, mean/std 0.5, block-major 2x2 patch order) and ships f32
 * patches of 3*16*16 floats; the engine encodes them and substitutes the
 * projected embeddings at the <|image_pad|> cells of the token stream.
 * Positions follow llama.cpp mtmd: every image cell keeps t = the position
 * where the image starts, y/x walk the merged grid, and the sequential
 * position resumes at t + max(grid) -- which is why m->mp exists at all. */

#define Q38_VIS_PATCH_FLOATS (3*16*16)

static Q38VisTower *q38_vis_ensure(Model *m) {
    if (m->vis) return m->vis;
    char pb[2048];
    const char *path = getenv("Q38_MMPROJ");
    if (!path || !*path) { snprintf(pb, sizeof pb, "%s/mmproj-F16.gguf", m->snap); path = pb; }
    double t0 = now_s();
    m->vis = q38v_load(path);
    if (m->vis)
        fprintf(stderr, "[vis] tower loaded from %s in %.1fs (RSS %.2f GB)\n",
                path, now_s()-t0, rss_gb());
    /* P6.1: F16 weights on the device unless Q38_GPU_VIT=0 (or no device) */
    if (m->vis && g_gpu_dense && !(getenv("Q38_GPU_VIT") && !atoi(getenv("Q38_GPU_VIT")))) {
        double t1 = now_s();
        m->vis_gpu = q38vg_load(m->vis) == 0;
        fprintf(stderr, m->vis_gpu ? "[vis] tower on the GPU: %.2f GB uploaded in %.1fs\n"
                                   : "[vis] GPU tower unavailable (%.0f), CPU path\n",
                m->vis_gpu ? q38vg_vram_bytes() / 1073741824.0 : 0.0, now_s()-t1);
    }
    return m->vis;
}

static void q38_vision_clear(Model *m) {
    free(m->vis_embd); m->vis_embd = NULL; m->vis_n = 0;
    m->vis_img_n = 0;
    free(m->vis_map);  m->vis_map = NULL;
    free(m->mp);       m->mp = NULL;
    m->mp_next = 0;
}

/* P6.2: projected-embedding cache.  The tower output is a pure function of
 * the patches, and a chat resends every image of its history each turn, so
 * the rows are kept by patch hash (FNV-1a over the payload plus the grid);
 * Q38_VIS_CACHE entries (default 16, 0 disables), oldest evicted. */
typedef struct { uint64_t key; int n_out; float *rows; uint64_t stamp; } Q38VisCacheEnt;
#define Q38_VIS_CACHE_MAX 64
static Q38VisCacheEnt g_vis_cache[Q38_VIS_CACHE_MAX];
static int g_vis_cache_cap = -1;
static uint64_t g_vis_cache_stamp = 0;
static long g_vis_cache_hits = 0, g_vis_cache_misses = 0;

static uint64_t q38_vis_hash(const float *patches, size_t n, int gh, int gw) {
    const unsigned char *p = (const unsigned char *)patches;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n * sizeof(float); i++) { h ^= p[i]; h *= 1099511628211ull; }
    h ^= (uint64_t)gh * 0x9E3779B97F4A7C15ull; h *= 1099511628211ull;
    h ^= (uint64_t)gw * 0xC2B2AE3D27D4EB4Full; h *= 1099511628211ull;
    return h;
}

static int q38_vis_cache_cap_get(void) {
    if (g_vis_cache_cap < 0) {
        const char *e = getenv("Q38_VIS_CACHE");
        g_vis_cache_cap = e ? atoi(e) : 16;
        if (g_vis_cache_cap > Q38_VIS_CACHE_MAX) g_vis_cache_cap = Q38_VIS_CACHE_MAX;
        if (g_vis_cache_cap < 0) g_vis_cache_cap = 0;
    }
    return g_vis_cache_cap;
}

/* Encode one image into dst[n_out][hidden], through the cache. 0 ok, -1 fail.
 * key_out receives the image's content key (also the pool's image key). */
static int q38_vision_encode_one(Model *m, const float *patches, int gh, int gw, int n_out, float *dst,
                                 uint64_t *key_out) {
    Cfg *c = &m->c;
    const size_t npatch = (size_t)gh * gw * Q38_VIS_PATCH_FLOATS;
    const int cap = q38_vis_cache_cap_get();
    uint64_t key = q38_vis_hash(patches, npatch, gh, gw);
    if (key_out) *key_out = key;
    if (cap) {
        for (int i = 0; i < cap; i++)
            if (g_vis_cache[i].rows && g_vis_cache[i].key == key && g_vis_cache[i].n_out == n_out) {
                memcpy(dst, g_vis_cache[i].rows, (size_t)n_out * c->hidden * sizeof(float));
                g_vis_cache[i].stamp = ++g_vis_cache_stamp;
                g_vis_cache_hits++;
                fprintf(stderr, "[vis] cache hit %dx%d -> %d tokens (%ld hits, %ld misses)\n",
                        gh, gw, n_out, g_vis_cache_hits, g_vis_cache_misses);
                return 0;
            }
        g_vis_cache_misses++;
    }
    double t0 = now_s();
    int rc = -1, on_gpu = 0;
    if (m->vis_gpu) {
        /* the position/rotary tables are the CPU tower's own code (q38v_tables) */
        const Q38VisCfg *vc = &m->vis->c;
        const int T = gh * gw;
        float *pos = calloc((size_t)T * vc->hidden, sizeof(float));
        float *cs = malloc((size_t)T * vc->head_dim * sizeof(float));
        if (pos && cs) {
            q38v_tables(m->vis, gh, gw, pos, cs);
            /* P6.6: with the tier and the grown KV mirrors VRAM can be full
             * (23.8 of 24.5 GB at 93K context, 2026-09-03); the encode then
             * failed and the CPU tower took 40 s per screenshot.  Make room
             * the way the KV growth does: release tier slabs for the scratch. */
            {
                const size_t need = q38vg_encode_bytes(m->vis, gh, gw) + ((size_t)256 << 20);
                size_t freeb = q38g_free_bytes();
                if (freeb < need) {
                    size_t got = q38t_release(need - freeb);
                    if (got) fprintf(stderr, "[vis] released %.0f MB of expert tier for the encode scratch\n",
                                     got / 1048576.0);
                }
            }
            rc = q38vg_encode(m->vis, patches, gh, gw, pos, cs, dst);
            if (rc) {                                   /* one retry after a forced release */
                q38t_release(q38vg_encode_bytes(m->vis, gh, gw) + ((size_t)512 << 20));
                rc = q38vg_encode(m->vis, patches, gh, gw, pos, cs, dst);
            }
            on_gpu = rc == 0;
            if (rc) fprintf(stderr, "[vis] GPU encode failed, CPU fallback\n");
        }
        free(pos); free(cs);
    }
    if (rc != 0) rc = q38v_encode(m->vis, patches, gh, gw, dst);
    if (rc != 0) return -1;
    fprintf(stderr, "[vis] encode %dx%d patches -> %d tokens in %.2fs (%s)\n",
            gh, gw, n_out, now_s()-t0, on_gpu ? "gpu" : "cpu");
    if (cap) {
        int slot = -1;
        for (int i = 0; i < cap; i++) if (!g_vis_cache[i].rows) { slot = i; break; }
        if (slot < 0) {
            slot = 0;
            for (int i = 1; i < cap; i++) if (g_vis_cache[i].stamp < g_vis_cache[slot].stamp) slot = i;
            free(g_vis_cache[slot].rows); g_vis_cache[slot].rows = NULL;
        }
        float *copy = malloc((size_t)n_out * c->hidden * sizeof(float));
        if (copy) {
            memcpy(copy, dst, (size_t)n_out * c->hidden * sizeof(float));
            g_vis_cache[slot].key = key; g_vis_cache[slot].n_out = n_out;
            g_vis_cache[slot].rows = copy; g_vis_cache[slot].stamp = ++g_vis_cache_stamp;
        }
    }
    return 0;
}

/* Encode the request's images and wire them into the prompt: the k-th run of
 * <|image_pad|> cells takes the k-th image; vis_map names the embedding row
 * per cell, mp carries the IMRoPE positions for prompt AND decode cells (room
 * = np + max_tokens): every cell of an image keeps t = the position where the
 * image starts, y/x walk its merged grid, text resumes at t + max(nx, ny).
 * Returns 0, or -1 with a message in err. */
static int q38_vision_prepare(Model *m, const int *ids, int np, int room,
                              float *const *patches, const int *ghs, const int *gws, int n_img,
                              char *err, size_t errsz) {
    Cfg *c = &m->c;
    int gstart[Q38_VIS_MAX_IMG + 1], glen[Q38_VIS_MAX_IMG + 1], ng = 0;
    for (int i = 0; i < np; i++) {
        if (ids[i] != c->ple_img) continue;
        if (ng && gstart[ng-1] + glen[ng-1] == i) { glen[ng-1]++; continue; }
        if (ng >= Q38_VIS_MAX_IMG) { snprintf(err, errsz, "too many image groups in the prompt"); return -1; }
        gstart[ng] = i; glen[ng] = 1; ng++;
    }
    if (ng != n_img) {
        snprintf(err, errsz, "prompt has %d image group(s), request carries %d image(s)", ng, n_img);
        return -1;
    }
    int n_out[Q38_VIS_MAX_IMG], off[Q38_VIS_MAX_IMG + 1];
    off[0] = 0;
    for (int k = 0; k < n_img; k++) {
        int gh = ghs[k], gw = gws[k];
        if (gh < 2 || gw < 2 || gh % 2 || gw % 2 || (int64_t)gh*gw > (int64_t)1<<20) {
            snprintf(err, errsz, "bad image grid %dx%d", gh, gw); return -1;
        }
        n_out[k] = (gw/2) * (gh/2);
        if (glen[k] != n_out[k]) {
            snprintf(err, errsz, "image %d: grid %dx%d needs %d image tokens, prompt group has %d",
                     k + 1, gh, gw, n_out[k], glen[k]);
            return -1;
        }
        off[k+1] = off[k] + n_out[k];
    }
    if (!q38_vis_ensure(m)) { snprintf(err, errsz, "vision tower unavailable (mmproj-F16.gguf)"); return -1; }

    q38_vision_clear(m);
    m->vis_embd = falloc((int64_t)off[n_img] * c->hidden);
    m->vis_n = off[n_img];
    for (int k = 0; k < n_img; k++) {
        uint64_t key = 0;
        if (q38_vision_encode_one(m, patches[k], ghs[k], gws[k], n_out[k],
                                  m->vis_embd + (size_t)off[k] * c->hidden, &key) != 0) {
            q38_vision_clear(m);
            snprintf(err, errsz, "vision encode failed for grid %dx%d", ghs[k], gws[k]);
            return -1;
        }
        m->vis_img_key[k] = key;
        m->vis_img_end[k] = gstart[k] + glen[k];
    }
    m->vis_img_n = n_img;
    if (getenv("Q38_VIS_DUMP")) {
        FILE *df = fopen(getenv("Q38_VIS_DUMP"), "wb");
        if (df) { fwrite(m->vis_embd, sizeof(float), (size_t)m->vis_n*c->hidden, df); fclose(df);
                  fprintf(stderr, "[vis] dumped %dx%d embeddings -> %s\n", m->vis_n, c->hidden, getenv("Q38_VIS_DUMP")); }
    }

    m->vis_map = malloc((size_t)room * sizeof(int));
    m->mp = malloc((size_t)room * 3 * sizeof(int));
    if (!m->vis_map || !m->mp) { fprintf(stderr, "OOM on vision maps\n"); exit(1); }
    for (int i = 0; i < room; i++) m->vis_map[i] = -1;
    int cur = 0, k = 0;
    for (int i = 0; i < np; i++) {
        if (k < n_img && i >= gstart[k] && i < gstart[k] + glen[k]) {
            int j = i - gstart[k], nx = gws[k]/2, ny = ghs[k]/2;   /* row-major over the merged grid */
            m->vis_map[i] = off[k] + j;
            m->mp[3*i]   = cur;
            m->mp[3*i+1] = cur + j / nx;
            m->mp[3*i+2] = cur + j % nx;
            if (j == n_out[k] - 1) { cur += ny > nx ? ny : nx; k++; }
        } else {
            m->mp[3*i] = m->mp[3*i+1] = m->mp[3*i+2] = cur;
            cur++;
        }
    }
    m->mp_next = cur;                            /* decode cells continue here */
    return 0;
}

/* ===================== P7: coli serve mode (SERVE=1) =====================
 * The colibri gateway wire protocol, so `coli chat` / `coli serve` /
 * `coli web` can drive this engine.  Structure mirrors qwen36.c's serve mode
 * (the closest sibling): one KV slot, one turn at a time, state reset per
 * request, length-prefixed SUBMIT in / DATA + DONE out.  Frame parsing and
 * emission go through serve_codec.h, which every serve-capable engine shares.
 *
 *   engine:  \x01\x01READY\x01\x01 / STAT / HWINFO / TIERS / EMAP
 *   gateway: SUBMIT <id> <slot> <bytes> <max_tok> <temp> <top_p>\n<payload>\n
 *   engine:  ACCEPT <id> <np>
 *            PROGRESS <id> <np> <restored> <done> <prefix_end> <seconds>  (<= 1/s)
 *            PREFILL <id> <np> <cached> <seconds>
 *            DATA <id> <n>\n<bytes>\n            (repeated, one per token)
 *            HWINFO / TIERS / EMAP / HITS / PFETCH
 *            DONE <id> STAT <gen> <tok_s> <hit%> <rss> <np> <limited>
 *
 * Mid-turn CANCEL/STOP is checked on a coarse 10-second watchdog.  Prefill
 * checks at batch boundaries; decode checks the clock every 16 emitted tokens,
 * while the nonblocking stdin poll itself runs at most once per watchdog
 * interval.  CANCEL discards the unpublished state and emits ERROR CANCELLED;
 * STOP publishes the generated prefix and emits DONE.
 *
 * Prefix reuse restores a complete qwen38_pool checkpoint (KV, GDN, PLE,
 * indexer and token history), never KV alone. The gateway can also identify
 * the stable system/tool boundary so a first request publishes that state for
 * later new chats before continuing with its private user tail. */

/* Wire limits.  max_tokens is bounded by the context ceiling rather than a
 * constant: serve_one re-checks np + max_tok against it anyway, and a
 * profile-level cap here would reject before ACCEPT with a BAD_REQUEST the
 * gateway renders as a generic 400 instead of the CONTEXT_EXCEEDED message. */
static const ColiServeWireProfile Q38_WIRE = {
    .max_header_bytes = 4096,
    .max_payload_bytes = 1u << 24,
    .max_extension_bytes = 8,
    .max_tokens = 0,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
    .allow_extension_bytes = 1,
    .allow_prefix_hint = 1,
};

/* The gateway derives this byte boundary from the structured chat request:
 * everything before the first user turn is stable system/tool input. Retokenize
 * and verify it here before using it; a stale or malformed hint then degrades to
 * ordinary prefill instead of publishing a checkpoint at the wrong position. */
static int q38_prefix_hint_tokens(Model *m, const ColiServeCommand *cmd,
                                  const int *prompt_ids, int prompt_count) {
    int prefix_bytes = cmd->prefix_bytes;
    if (prefix_bytes <= 0 || (uint64_t)prefix_bytes >= cmd->payload_bytes)
        return 0;
    char *prefix = malloc((size_t)prefix_bytes + 1);
    if (!prefix) return 0;
    memcpy(prefix, cmd->payload, (size_t)prefix_bytes);
    prefix[prefix_bytes] = 0;
    int *prefix_ids = NULL, prefix_count = 0;
    encode_text(prefix, &prefix_ids, &prefix_count);
    free(prefix);
    int valid = prefix_count >= 16 && prefix_count < prompt_count &&
                q38pool_can_store(m, prefix_count) &&
                !memcmp(prefix_ids, prompt_ids,
                        (size_t)prefix_count * sizeof(*prompt_ids));
    free(prefix_ids);
    return valid ? prefix_count : 0;
}

/* --- P6: vision over the wire -------------------------------------------
 * The gateway announces an image with an IMAGE frame immediately before the
 * SUBMIT it belongs to (docs/serve_protocol.md).  serve_codec.h does not know
 * that frame -- glm53 parses it itself -- so this wrapper consumes it and
 * hands the codec the next line.  The payload is f32 patches from
 * tools/qwen38_image.py ([grid_h*grid_w][3*16*16], block-major 2x2 order);
 * it is held until the SUBMIT with the same id arrives, and a second image
 * replaces the first (answering about the previous photo silently would be
 * worse than dropping it). */
static struct {
    char id[COLI_SERVE_ID_CAP];
    int n, bad;                                  /* bad: a frame was refused */
    float *patches[Q38_VIS_MAX_IMG];
    int gh[Q38_VIS_MAX_IMG], gw[Q38_VIS_MAX_IMG];
} g_pending_img;

static void q38_pending_clear(void) {
    for (int k = 0; k < g_pending_img.n; k++) free(g_pending_img.patches[k]);
    g_pending_img.n = 0;
    g_pending_img.bad = 0;
    g_pending_img.id[0] = 0;
}

static ColiServeReadResult q38_serve_read(FILE *in, ColiServeCommand *cmd) {
    int ch = fgetc(in);
    if (ch == EOF) return COLI_SERVE_READ_EOF;
    if (ch != 'I') { ungetc(ch, in); return coli_serve_read_command(in, &Q38_WIRE, cmd); }
    /* "IMAGE <id> <bytes> <grid_h> <grid_w>\n<payload>\n" -- the 'I' is eaten. */
    char line[512]; size_t n = 0;
    for (;;) {
        ch = fgetc(in);
        if (ch == EOF) return COLI_SERVE_READ_EOF;
        if (ch == '\n') break;
        if (n + 1 < sizeof line) line[n++] = (char)ch;
    }
    line[n] = 0;
    char id[COLI_SERVE_ID_CAP]; unsigned long long bytes = 0; int gh = 0, gw = 0;
    if (sscanf(line, "MAGE %63s %llu %d %d", id, &bytes, &gh, &gw) != 4 ||
        bytes > (1ull << 28))
        return COLI_SERVE_READ_BAD_FRAME;
    /* frames of one request accumulate in order; a different id starts over */
    if (g_pending_img.id[0] && strcmp(g_pending_img.id, id)) q38_pending_clear();
    uint64_t want = (uint64_t)gh * (uint64_t)gw * Q38_VIS_PATCH_FLOATS * sizeof(float);
    int usable = gh > 0 && gw > 0 && bytes == want && g_pending_img.n < Q38_VIS_MAX_IMG;
    float *patches = usable ? malloc((size_t)bytes) : NULL;
    if (patches) {
        if (fread(patches, 1, (size_t)bytes, in) != (size_t)bytes) { free(patches); return COLI_SERVE_READ_EOF; }
    } else {
        /* wrong size or OOM: drain to keep the pipe in sync, refuse at SUBMIT */
        for (unsigned long long i = 0; i < bytes; i++) if (fgetc(in) == EOF) return COLI_SERVE_READ_EOF;
    }
    if (fgetc(in) != '\n') { free(patches); return COLI_SERVE_READ_BAD_FRAME; }
    if (!patches)
        fprintf(stderr, "[serve] IMAGE %s: %llu bytes for grid %dx%d rejected (want %llu, %d pending)\n",
                id, bytes, gh, gw, (unsigned long long)want, g_pending_img.n);
    snprintf(g_pending_img.id, sizeof g_pending_img.id, "%s", id);
    if (patches) {
        g_pending_img.patches[g_pending_img.n] = patches;
        g_pending_img.gh[g_pending_img.n] = gh;
        g_pending_img.gw[g_pending_img.n] = gw;
        g_pending_img.n++;
    } else {
        g_pending_img.bad = 1;                       /* keeps the refusal path */
    }
    memset(cmd, 0, sizeof *cmd);
    return COLI_SERVE_READ_IGNORED;
}

typedef struct {
    const char *active;
    Q38ControlAction action;
    int fatal;
    double next_check;
} Q38ServeControl;

static int q38_serve_poll_control(void *context) {
    Q38ServeControl *poll = context;
    double now = now_s();
    if (now < poll->next_check)
        return poll->action != Q38_CONTROL_NONE || poll->fatal;
    poll->next_check = now + Q38_CONTROL_INTERVAL_S;
    while (coli_stdin_readable()) {
        ColiServeCommand command = {0};
        ColiServeReadResult result = q38_serve_read(stdin, &command);
        if (result == COLI_SERVE_READ_IGNORED) {
            q38_pending_clear();
            continue;
        }
        if (result != COLI_SERVE_READ_OK) {
            if (result != COLI_SERVE_READ_EOF)
                coli_serve_write_error(stdout, command.id,
                    result == COLI_SERVE_READ_NOMEM ? "BAD_REQUEST out of memory" :
                    result == COLI_SERVE_READ_BAD_REQUEST ? "BAD_REQUEST" : "BAD_FRAME");
            poll->fatal = 1;
            return 1;
        }
        Q38ControlAction action = q38_control_action(
            command.kind, command.id, poll->active);
        if (action == Q38_CONTROL_NOT_FOUND)
            coli_serve_write_error(stdout, command.id, "NOT_FOUND");
        else if (action == Q38_CONTROL_BUSY)
            coli_serve_write_error(stdout, command.id, "ENGINE_BUSY");
        else if (action == Q38_CONTROL_STOP || action == Q38_CONTROL_CANCEL)
            poll->action = action;
        coli_serve_command_dispose(&command);
    }
    return poll->action != Q38_CONTROL_NONE || poll->fatal;
}

/* --- sampling ----------------------------------------------------------
 * temperature and top-p arrive on the wire; top-k and min-p are engine env
 * knobs (Q38_TOP_K / Q38_MIN_P) because the mux SUBMIT header has no field
 * for them and inventing one would break every other engine's parser.
 * temp <= 0 is greedy argmax, which is what every reference gate runs.
 *
 * The candidate prefilter matters at this vocabulary: a full sort of 248,320
 * entries costs more than a decode step.  Logits more than 24*temp nats below
 * the max carry < 4e-11 of the mass after the temperature divide, so they are
 * dropped before the sort; a few hundred candidates survive in practice. */
static int sample_profile_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("Q38_SAMPLE_PROFILE") ? 1 : 0;
    return enabled;
}
static void sample_profile_add(int candidates, double seconds) {
    static int calls = 0, min_n = 0, max_n = 0;
    static long long total_n = 0;
    static double total_s = 0;
    if (!calls || candidates < min_n) min_n = candidates;
    if (!calls || candidates > max_n) max_n = candidates;
    calls++; total_n += candidates; total_s += seconds;
    if (calls == 64) {
        fprintf(stderr, "[sample] candidates avg %.0f min %d max %d | %.3f ms/token\n",
                (double)total_n / calls, min_n, max_n, 1000.0 * total_s / calls);
        calls = 0; min_n = max_n = 0; total_n = 0; total_s = 0;
    }
}
static int serve_sample(const float *lo, int V, float temp, float top_p) {
    if (!(temp > 0.f)) { int b = 0; for (int i = 1; i < V; i++) if (lo[i] > lo[b]) b = i; return b; }
    double sample_t0 = sample_profile_enabled() ? now_s() : 0.0;
    /* P10 D6: read once, not per token */
    static int   top_k = -1;
    static float min_p = 0.f;
    if (top_k < 0) {
        top_k = getenv("Q38_TOP_K") ? atoi(getenv("Q38_TOP_K")) : 0;
        if (top_k < 0) top_k = 0;
        min_p = getenv("Q38_MIN_P") ? (float)atof(getenv("Q38_MIN_P")) : 0.f;
    }
    float mx = lo[0];
    for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    float floor_logit = mx - 24.f * temp;
    static Q38SampleProb *rank = NULL;
    static int rank_cap = 0;
    if (rank_cap < V) {
        rank = realloc(rank, (size_t)V * sizeof(*rank));
        if (!rank) { fprintf(stderr, "OOM sampling\n"); exit(1); }
        rank_cap = V;
    }
    int n = 0; double sum = 0;
    for (int i = 0; i < V; i++) {
        if (lo[i] < floor_logit) continue;
        float p = expf((lo[i] - mx) / temp);
        sum += p; rank[n++] = (Q38SampleProb){p, i};
    }
    int pick = q38_sample_pick(rank, n, sum, top_k, min_p, top_p,
                               (double)rand() / RAND_MAX, NULL);
    if (sample_t0) sample_profile_add(n, now_s() - sample_t0);
    return pick;
}

/* Chat turns end on <|im_end|>, base completions on <|endoftext|>; both ids
 * come from the tokenizer's added_tokens (this vocabulary is 248,320 tokens,
 * so the 151k-vocab Qwen constants do not apply).  Q38_EOS overrides. */
static int serve_eos_ids(int *ids, int cap) {
    int n = 0;
    if (getenv("Q38_EOS")) { ids[n++] = atoi(getenv("Q38_EOS")); return n; }
    for (int k = 0; k < g_nspecial && n < cap; k++)
        if (!strcmp(g_sp_str[k], "<|im_end|>") || !strcmp(g_sp_str[k], "<|endoftext|>"))
            ids[n++] = g_sp_id[k];
    return n;
}

static int serve_sample_without_eos(float *lo, int V, float temp, float top_p,
                                    const int *eos_ids, int n_eos) {
    float saved[4]; int masked[4], n_masked = 0;
    for (int e = 0; e < n_eos; e++) {
        int id = eos_ids[e], duplicate = 0;
        if (id < 0 || id >= V) continue;
        for (int k = 0; k < n_masked; k++) if (masked[k] == id) duplicate = 1;
        if (duplicate) continue;
        masked[n_masked] = id;
        saved[n_masked++] = lo[id];
        lo[id] = -INFINITY;
    }
    int token = serve_sample(lo, V, temp, top_p);
    for (int e = 0; e < n_masked; e++) lo[masked[e]] = saved[e];
    return token;
}

/* UTF-8 resynchroniser: a token's bytes can end mid-codepoint, and a DATA
 * frame carrying half a character makes the gateway's incremental decoder
 * emit U+FFFD.  Hold the tail until it completes (out_bytes does the same
 * job for the CLI path, straight to stdout). */
static void serve_utf8_drain(unsigned char *buf, int *bn, const unsigned char *in, int n,
                             unsigned char *out, int *on) {
    *on = 0;
    for (int k = 0; k < n; k++) {
        if (*bn < 16) buf[(*bn)++] = in[k];
        int j = 0;
        while (j < *bn) {
            unsigned char lead = buf[j]; int need;
            if (lead < 0x80) need = 1;
            else if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else { out[(*on)++] = buf[j]; memmove(buf+j, buf+j+1, (size_t)(*bn-j-1)); (*bn)--; continue; }
            if (j + need > *bn) break;
            memcpy(out + *on, buf + j, (size_t)need); *on += need;
            memmove(buf+j, buf+j+need, (size_t)(*bn-j-need)); *bn -= need;
        }
    }
}

/* --- dashboard telemetry lines ---------------------------------------- */
static void serve_hwinfo(void) {
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof ln, ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':');
            if (p) { p++; while (*p == ' ') p++;
                     int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
                     snprintf(cpu, sizeof cpu, "%s", p); }
            break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof ln, mi)) {
            /* 1024-based everywhere the dashboard says "GB" (a 24 GB card shows 24, not 25) */
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1048576.0;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1048576.0;
        } fclose(mi); }
    Q38TierStat ts; int on = q38t_stat(&ts);
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n", cores, rt, ra, on ? 1 : 0,
           on ? (ts.vram_total_bytes ? ts.vram_total_bytes : ts.budget_bytes)/1073741824.0 : 0.0,
           cpu[0] ? cpu : "unknown",
           on ? q38t_device_name() : "");
    fflush(stdout);
}

/* TIERS: qwen38 has exactly two tiers.  Experts live as raw GGML blocks in
 * the mmap'd container (page-cache/NVMe = "disk") and the VRAM tier promotes
 * a subset.  There is no third RAM-slot copy, so the RAM column is 0 and the
 * dashboard's middle band stays empty -- that is the truth about this engine,
 * not a gap in the reporting.  ram_gb carries the dense backbone's RSS so the
 * hardware panel still shows where the resident memory went. */
static void serve_tiers(Model *m) {
    Q38TierStat ts; q38t_stat(&ts);
    long long total = (long long)m->c.n_layers * m->c.n_experts;
    printf("TIERS %llu 0 %lld %.2f %.2f\n",
           (unsigned long long)ts.resident, total - (long long)ts.resident,
           ts.resident_bytes/1e9, rss_gb());
    fflush(stdout);
}

static void serve_emap(Model *m) {
    int nl = m->c.n_layers, ne = m->c.n_experts;
    char *hex = malloc((size_t)nl * ne * 2 + 1);
    if (!hex) return;
    int w = 0;
    for (int l = 0; l < nl; l++)
        for (int e = 0; e < ne; e++) {
            uint32_t u = g_tele_cnt ? g_tele_cnt[(size_t)l*ne + e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; }
            if (heat > 63) heat = 63;
            int b = (q38t_resident(l, e) << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", nl, ne, hex);
    fflush(stdout); free(hex);
}

static void serve_hits(Model *m) {
    int nl = m->c.n_layers, ne = m->c.n_experts;
    size_t bits = (size_t)nl * ne, bytes = (bits + 7) / 8;
    unsigned char *bm = calloc(bytes, 1);
    char *hex = malloc(bytes * 2 + 1);
    if (!bm || !hex) { free(bm); free(hex); return; }
    for (size_t i = 0; i < bits; i++)
        if (g_tele_hit[i]) { bm[i >> 3] |= (unsigned char)(1u << (i & 7)); g_tele_hit[i] = 0; }
    for (size_t i = 0; i < bytes; i++) {
        hex[i*2]   = "0123456789abcdef"[bm[i] >> 4];
        hex[i*2+1] = "0123456789abcdef"[bm[i] & 15];
    }
    hex[bytes*2] = 0;
    printf("HITS %d %d %s\n", nl, ne, hex);
    fflush(stdout); free(bm); free(hex);
}

/* PFETCH: the P4 prefetcher's hit split, as a line of its own.  It is a NEW
 * line kind, so it only exists because openai_server.py learned to parse it
 * in the same change -- the dispatcher treats an unknown kind as a protocol
 * error, so "servers ignore what they do not know" is a rule for the
 * dashboard, not for the gateway.  Fields are cumulative counters; the
 * dashboard differences them.
 *   PFETCH <hits> <miss> <pin_hit> <pf_hit> <issued> <done> <evicted> <pinned>
 * where hits-pin_hit-pf_hit is the demand share of the VRAM hit rate. */
static void serve_pfetch(void) {
    Q38TierStat ts;
    if (!q38t_stat(&ts) || !ts.prefetch_on) return;
    printf("PFETCH %llu %llu %llu %llu %llu %llu %llu %d\n",
           (unsigned long long)ts.hits, (unsigned long long)ts.miss,
           (unsigned long long)ts.pin_hit, (unsigned long long)ts.pf_hit,
           (unsigned long long)ts.pf_issued, (unsigned long long)ts.pf_done,
           (unsigned long long)ts.pf_evicted, ts.pinned);
    fflush(stdout);
}

/* Presence penalty (Qwen3.8 model card, instruct mode 1.5; OpenAI/vLLM
 * semantics): every token that appeared in this turn's output so far has
 * the penalty subtracted from its logit before sampling.  Sampled turns
 * only (temperature > 0): the greedy path stays bit-identical. */
static float    g_presence;
static uint8_t *g_seen;        /* [vocab] bitmap of tokens emitted this turn */
static int     *g_seen_list;   /* distinct emitted ids, for the O(distinct) subtraction */
static int      g_seen_n, g_seen_cap;
static void presence_reset(int V, float penalty) {
    g_presence = penalty;
    if (!(penalty > 0.f)) return;
    if (g_seen_cap < V) { free(g_seen); free(g_seen_list); g_seen = calloc((size_t)V, 1);
                          g_seen_list = malloc((size_t)V * sizeof(int)); g_seen_cap = g_seen ? V : 0; }
    else memset(g_seen, 0, (size_t)V);
    g_seen_n = 0;
}
static void presence_mark(int tk) {
    if (!(g_presence > 0.f) || !g_seen || tk < 0 || tk >= g_seen_cap || g_seen[tk]) return;
    g_seen[tk] = 1; g_seen_list[g_seen_n++] = tk;
}
static void presence_apply(float *lo, int V, float temp) {
    if (!(g_presence > 0.f) || !(temp > 0.f) || !g_seen) return;
    for (int i = 0; i < g_seen_n; i++) if (g_seen_list[i] < V) lo[g_seen_list[i]] -= g_presence;
}

static void serve_one(Model *m, ColiServeCommand *cmd, float *logits) {
    const char *id = cmd->id;
    int reasoning_budget = 0;
    int request_mtp = 1;
    float presence = 0.f;
    if (cmd->extension_bytes) {
        const unsigned char *ext = coli_serve_command_extension(cmd);
        /* 4 bytes: controls; 8 bytes: controls + f32 presence penalty */
        if (cmd->extension_bytes != 4 && cmd->extension_bytes != 8) {
            coli_serve_write_error(stdout, id, "BAD_REQUEST invalid thinking budget");
            return;
        }
        if (cmd->extension_bytes == 8) memcpy(&presence, ext + 4, 4);
        if (!(presence >= 0.f && presence <= 10.f)) presence = 0.f;
        uint32_t controls = (uint32_t)ext[0] | ((uint32_t)ext[1] << 8) |
                            ((uint32_t)ext[2] << 16) | ((uint32_t)ext[3] << 24);
        request_mtp = !(controls & 0x80000000u);
        g_req_router = g_gpu_router && (controls & 0x40000000u);   /* P10 D4 toggle */
        reasoning_budget = (int)(controls & 0x3fffffffu);
        if (reasoning_budget > cmd->max_tokens) {
            coli_serve_write_error(stdout, id, "BAD_REQUEST invalid thinking budget");
            return;
        }
    }
    /* P6: an image announced for THIS request; one announced for another id
     * is stale and dropped rather than silently answered about. */
    int has_img = g_pending_img.id[0] && !strcmp(g_pending_img.id, id);
    if (g_pending_img.id[0] && !has_img) q38_pending_clear();
    if (has_img && (g_pending_img.bad || !g_pending_img.n)) {
        q38_pending_clear();
        coli_serve_write_error(stdout, id,
            "VISION_UNSUPPORTED image patch payload did not match its grid");
        return;
    }
    int *ids = NULL, np = 0;
    encode_text((const char*)cmd->payload, &ids, &np);   /* the gateway owns the chat template */
    presence_reset(m->c.vocab, cmd->temperature > 0.f ? presence : 0.f);
    int max_ctx = q38_max_ctx();
    if (np < 1) {
        coli_serve_write_error(stdout, id, "EMPTY_PROMPT");
        free(ids); return;
    }
    if (np + cmd->max_tokens > max_ctx) {
        /* Harnesses send a fixed max_tokens (Prime Agent: 32,000) regardless
         * of the conversation's length; a 99.6K prompt then failed with
         * 131,594 > 131,072 (2026-09-03) although the window had 31K to
         * spare.  Clamp the output budget to the room left when at least
         * Q38_CTX_MIN_OUT tokens (default 64) remain; the reply is then
         * length-limited at the window instead of refused.  A prompt that
         * leaves less room than that is still refused (below). */
        static int min_out = -1;
        if (min_out < 0) { const char *e = getenv("Q38_CTX_MIN_OUT"); min_out = e ? atoi(e) : 64; }
        if (max_ctx - np >= min_out) {
            fprintf(stderr, "[ctx] max_tokens %d clamped to %d (prompt %d of %d)\n",
                    cmd->max_tokens, max_ctx - np, np, max_ctx);
            cmd->max_tokens = max_ctx - np;
        }
    }
    if (np + cmd->max_tokens > max_ctx) {
        /* Format contract: the gateway parses `CONTEXT_EXCEEDED <used>
         * <limit>` into an OpenAI context_length_exceeded 400
         * (openai_server.py _engine_error).  qwen36/kimi emit a key=value
         * variant that renders as "maximum context length is requested=512
         * tokens"; deepseek_v4 documents this positional form, so follow it. */
        char msg[192];
        snprintf(msg, sizeof msg, "CONTEXT_EXCEEDED %d %d", np + cmd->max_tokens, max_ctx);
        coli_serve_write_error(stdout, id, msg);
        free(ids); return;
    }
    /* P6: encode the image and wire it into this prompt before ACCEPT, so a
     * mismatch (grid vs placeholders, missing mmproj) is a clean error and
     * not a half-started turn.  The encode itself is seconds of CPU. */
    q38_vision_clear(m);
    if (has_img) {
        char verr[192];
        int rc = q38_vision_prepare(m, ids, np, np + cmd->max_tokens,
                                    g_pending_img.patches, g_pending_img.gh,
                                    g_pending_img.gw, g_pending_img.n, verr, sizeof verr);
        q38_pending_clear();
        if (rc != 0) {
            char msg[256];
            snprintf(msg, sizeof msg, "VISION_UNSUPPORTED %s", verr);
            coli_serve_write_error(stdout, id, msg);
            free(ids); return;
        }
    }
    coli_serve_write_accept(stdout, id, np);

    /* Per-REQUEST state.  ensure_kv only grows, so a long first turn keeps
     * its buffers for every later one; reset_state clears the GDN recurrence,
     * the PLE history, the indexer block counts and the token history, which
     * is what makes turn N independent of turn N-1.
     * P3: when a stored checkpoint's token ids are a strict prefix of this
     * prompt, q38pool_restore replaces the reset with a byte-copy of that
     * state (the complete state -- KV, GDN, PLE, indexer, toks) and only the
     * tail is prefilled.  Turn independence is preserved: the restored bytes
     * are exactly what a full prefill of the same ids would have produced. */
    ensure_kv(m, np + cmd->max_tokens);
    int pstart = q38pool_restore(m, ids, np, logits);
    if (pstart > 0) { q38g_qsa_kv_invalidate(0); q38g_idx_invalidate(0); }   /* P9: restored host state */
    /* pstart == np: identical prompt, the checkpoint carried the last position's
     * logits -- nothing to prefill, the first token samples from `logits` */
    int prefix_end = pstart ? 0 : q38_prefix_hint_tokens(m, cmd, ids, np);
    int want_spec = request_mtp && m->has_mtp && (!g_gpu_dense || g_gpu_spec) &&
                    !(cmd->temperature > 0.f);      /* P6.4: image turns speculate too */
    /* A non-speculative turn does not advance the MTP KV.  If its pooled
     * checkpoint is later reused with MTP enabled, rebuild once rather than
     * drafting against a hole in the MTP history. */
    if (want_spec && pstart && m->mtp_pend_pos != pstart - 1) {
        reset_state(m);
        pstart = 0;
    }
    if (!pstart) reset_state(m);
    q38t_prefetch_phase(1);

    double t0 = now_s();
    Q38ServeControl control = {id, Q38_CONTROL_NONE, 0,
                               t0 + Q38_CONTROL_INTERVAL_S};
    int gpu_prefill = 0;
    int tail_start = pstart;
    g_sprog.id = id; g_sprog.np = np; g_sprog.restored = pstart;
    g_sprog.prefix_end = prefix_end > pstart ? prefix_end : 0;
    g_sprog.t0 = t0; g_sprog.last = 0;
    if (np > pstart) q38_serve_progress(pstart, 1);
    if (prefix_end > pstart) {
        gpu_prefill = q38_prefill_from(m, ids, prefix_end, pstart, NULL, 0,
                                      q38_serve_poll_control, &control);
        if (!control.fatal && control.action == Q38_CONTROL_NONE) {
            if (gpu_prefill) gpu_state_pull(m);
            q38pool_store(m, 0, NULL, 1);       /* the stable system/tool prefix: pinned */
        }
        tail_start = prefix_end;
    }
    if (!control.fatal && control.action == Q38_CONTROL_NONE && np > tail_start)
        gpu_prefill = q38_prefill_from(m, ids, np, tail_start, logits, 0,
                                      q38_serve_poll_control, &control);
    /* the prompt-end logits, kept for the checkpoint (the decode loop reuses `logits`) */
    float *first_logits = malloc((size_t)m->c.vocab * sizeof(float));
    if (first_logits) memcpy(first_logits, logits, (size_t)m->c.vocab * sizeof(float));
    double tp = now_s();
    g_sprog.id = NULL;
    if (control.fatal || control.action != Q38_CONTROL_NONE) {
        q38_vision_clear(m);
        free(ids); free(first_logits);
        if (control.fatal) return;
        if (control.action == Q38_CONTROL_CANCEL) {
            coli_serve_write_error(stdout, id, "CANCELLED");
        } else {
            ColiServeDone done = {0, 0.0, 0.0, rss_gb(), np, 0};
            coli_serve_write_done(stdout, id, &done);
        }
        return;
    }
    if (!gpu_prefill) gpu_state_push(m);    /* CPU prefill -> device for decode */
    coli_serve_write_prefill(stdout, id, np, pstart, now_s() - t0);
    /* P10 D6: the post-prefill checkpoint store (~60 ms plus the device pull)
     * runs after the FIRST token is emitted, before the state advances past
     * the prompt; pool_pending keeps it exact if the turn ends earlier. */
    int pool_pending = 1;
    #define Q38_POOL_STORE_ONCE() do { if (pool_pending) { pool_pending = 0; q38pool_store(m, gpu_prefill, first_logits, 0); } } while (0)

    int eos_ids[4]; int n_eos = serve_eos_ids(eos_ids, 4);
    Q38ReasoningGuard reasoning;
    q38_reasoning_init(&reasoning, (const char*)cmd->payload);
    int *reasoning_end = NULL, n_reasoning_end = 0, reasoning_tokens = 0;
    if (reasoning_budget)
    {
        /* A budget cut that only injects the marker leaves the model mid-thought:
         * it carries its checklist on in the answer channel.  Hand it over with
         * a sentence first (llama.cpp's --reasoning-budget-message does the
         * same); Q38_BUDGET_MESSAGE overrides, empty keeps the bare marker. */
        const char *bm = getenv("Q38_BUDGET_MESSAGE");
        if (!bm) bm = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n";
        char *cut = malloc(strlen(bm) + 16);
        if (!cut) { fprintf(stderr, "OOM on the budget message\n"); exit(1); }
        snprintf(cut, strlen(bm) + 16, "%s</think>\n\n", bm);
        encode_text(cut, &reasoning_end, &n_reasoning_end);
        free(cut);
    }
    int effective_budget = reasoning_budget;
    if (reasoning_budget && effective_budget > cmd->max_tokens - n_reasoning_end)
        effective_budget = cmd->max_tokens - n_reasoning_end;
    if (reasoning_budget && effective_budget < 1) effective_budget = 1;
    unsigned char sbuf[16]; int sbn = 0;
    int gen = 0, limited = 1, cancelled = 0, next_control_token = 0;
    /* P5: speculate only on greedy turns (temperature > 0 would need
     * rejection sampling -- documented v1 fallback: plain decode); image
     * turns speculate since P6.4 */
    int spec_on = want_spec;
    for (int s = 0; s < cmd->max_tokens; ) {
        if (gen >= next_control_token) {
            next_control_token = gen + 16;
            if (q38_serve_poll_control(&control)) {
                limited = 0;
                cancelled = control.fatal || control.action == Q38_CONTROL_CANCEL;
                break;
            }
        }
        if (q38_reasoning_budget_reached(&reasoning, effective_budget,
                                         reasoning_tokens)) {
            for (int j = 0; j < n_reasoning_end && s < cmd->max_tokens; j++) {
                int tk = reasoning_end[j];
                unsigned char tmp[256]; int tn = 0;
                decode_id_to_bytes(tk, tmp, &tn);
                q38_reasoning_feed(&reasoning, tmp, (size_t)tn);
                unsigned char chunk[288]; int cn = 0;
                serve_utf8_drain(sbuf, &sbn, tmp, tn, chunk, &cn);
                if (cn > 0) coli_serve_write_data(stdout, id, chunk, (size_t)cn);
                gen++; s++;
                Q38_POOL_STORE_ONCE();
                if (s < cmd->max_tokens) q38_decode_step(m, tk, np + s - 1, logits);
            }
            free(reasoning_end); reasoning_end = NULL; n_reasoning_end = 0;
            /* The emitted suffix is authoritative even if an unusual tokenizer
             * did not decode it as one contiguous marker. */
            reasoning.open = 0;
            continue;
        }
        presence_apply(logits, m->c.vocab, cmd->temperature);   /* once per step, before any sampling */
        int tk = serve_sample(logits, m->c.vocab, cmd->temperature, cmd->top_p);
        int is_eos = 0; for (int e = 0; e < n_eos; e++) if (tk == eos_ids[e]) is_eos = 1;
        if (is_eos && q38_reasoning_ignore_eos(&reasoning)) {
            tk = serve_sample_without_eos(logits, m->c.vocab, cmd->temperature,
                                          cmd->top_p, eos_ids, n_eos);
            is_eos = 0;
            for (int e = 0; e < n_eos; e++) if (tk == eos_ids[e]) is_eos = 1;
        }
        presence_mark(tk);
        if (is_eos) { limited = 0; break; }
        unsigned char tmp[256]; int tn = 0; decode_id_to_bytes(tk, tmp, &tn);
        if (reasoning.open) reasoning_tokens++;
        q38_reasoning_feed(&reasoning, tmp, (size_t)tn);
        unsigned char chunk[288]; int cn = 0;
        serve_utf8_drain(sbuf, &sbn, tmp, tn, chunk, &cn);
        if (cn > 0) coli_serve_write_data(stdout, id, chunk, (size_t)cn);
        gen++; s++;
        Q38_POOL_STORE_ONCE();
        if (s >= cmd->max_tokens) break;       /* the next logits would be discarded */
        if (spec_on && !reasoning.open) {
            int extra[MTP_MAX_N + 1];
            int ne = q38_spec_step(m, tk, np + s - 1, cmd->max_tokens - s - 1, logits, extra);
            int hit_eos = 0;
            for (int j = 0; j < ne; j++) {
                int tk2 = extra[j];
                for (int e = 0; e < n_eos; e++) if (tk2 == eos_ids[e]) hit_eos = 1;
                if (hit_eos) { limited = 0; break; }
                tn = 0; decode_id_to_bytes(tk2, tmp, &tn);
                cn = 0; serve_utf8_drain(sbuf, &sbn, tmp, tn, chunk, &cn);
                if (cn > 0) coli_serve_write_data(stdout, id, chunk, (size_t)cn);
                gen++; s++;
            }
            if (hit_eos) break;
        } else {
            q38_decode_step(m, tk, np + s - 1, logits);
        }
    }
    free(reasoning_end);
    Q38_POOL_STORE_ONCE();                  /* no token emitted: store the prompt state now */
    #undef Q38_POOL_STORE_ONCE
    if (cancelled) {
        q38_vision_clear(m);
        free(ids); free(first_logits);
        if (!control.fatal) coli_serve_write_error(stdout, id, "CANCELLED");
        return;
    }
    if (sbn > 0) coli_serve_write_data(stdout, id, sbuf, (size_t)sbn);
    if (reasoning.ignored_eos)
        fprintf(stderr, "[serve] %s: ignored %u EOS token%s inside reasoning\n",
                id, reasoning.ignored_eos, reasoning.ignored_eos == 1 ? "" : "s");
    /* P3: end-of-turn checkpoint.  A GPU decode advanced the GDN state on the
     * device, so pull it from there (the host copy is stale since prefill). */
    q38pool_store(m, g_gpu_dense && gen > 0, NULL, 0);
    free(first_logits);
    q38_vision_clear(m);                 /* next turn starts text-only clean */
    free(ids);

    double dt = now_s() - tp;
    Q38TierStat ts; q38t_stat(&ts);
    double tot = (double)(ts.hits + ts.miss);
    serve_hwinfo(); serve_tiers(m); serve_emap(m); serve_hits(m); serve_pfetch();
    ColiServeDone done = {
        .completion_tokens = gen,
        .tokens_per_second = dt > 0 ? gen / dt : 0.0,
        .cache_hit_percent = tot > 0 ? 100.0 * ts.hits / tot : 0.0,
        .rss_gb = rss_gb(),
        .prompt_tokens = np,
        .length_limited = limited,
    };
    coli_serve_write_done(stdout, id, &done);
    q38pool_flush();      /* P3: SSD spill after DONE, off the TTFT path */
    if (g_presence > 0.f) fprintf(stderr, "[serve] %s: presence penalty %.2f over %d distinct tokens\n", id, g_presence, g_seen_n);
    fprintf(stderr, "[serve] %s: %d prompt + %d gen | prefill %.2fs | decode %.2f tok/s\n",
            id, np, gen, tp - t0, dt > 0 ? gen/dt : 0.0);
    q38rt_sync();
    mtp_print_stats();
}

/* SIGTERM = "shut down", not "die".  The gateway's Engine.close() sends
 * SIGTERM without first closing the engine's stdin, so the read loop below
 * would otherwise never see EOF and the process would be killed where it
 * stands -- taking the learned heat table with it.  That table is the whole
 * reason the SECOND boot is warm (q38t_shutdown writes it, decayed, to
 * HEAT_FILE), so losing it on every `systemctl restart` would silently undo
 * the warmstart the serving profile is built around.
 *
 * The handler only flips a flag: sigaction WITHOUT SA_RESTART means the
 * blocking fgetc in the read loop fails with EINTR, the loop reads that as
 * EOF, serve_loop returns, and main runs the ordinary shutdown path. A signal
 * that lands mid-generation is honored at the next frame boundary, which is
 * what TimeoutStopSec in the unit file budgets for. */
static volatile sig_atomic_t g_serve_stop;
static void serve_on_signal(int sig) { (void)sig; g_serve_stop = 1; }

static void serve_install_signals(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = serve_on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;                       /* no SA_RESTART: interrupt the read */
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
}

static void serve_loop(Model *m) {
    coli_serve_stdio_init();
    serve_install_signals();
    const char *sd = getenv("SEED");
    srand(sd ? (unsigned)strtoul(sd, NULL, 10) : (unsigned)time(NULL));
    float *logits = falloc(m->c.vocab);
    coli_serve_write_ready(stdout, rss_gb());
    serve_hwinfo(); serve_tiers(m); serve_emap(m);
    for (;;) {
        ColiServeCommand cmd;
        ColiServeReadResult r = q38_serve_read(stdin, &cmd);
        if (g_serve_stop) { fprintf(stderr, "[serve] signal: shutting down\n"); break; }
        if (r == COLI_SERVE_READ_EOF) break;              /* graceful shutdown */
        if (r == COLI_SERVE_READ_IGNORED) continue;
        if (r == COLI_SERVE_READ_BAD_REQUEST) { coli_serve_write_error(stdout, cmd.id, "BAD_REQUEST"); continue; }
        if (r == COLI_SERVE_READ_NOMEM)       { coli_serve_write_error(stdout, cmd.id, "BAD_REQUEST out of memory"); continue; }
        if (r != COLI_SERVE_READ_OK)          { coli_serve_write_error(stdout, cmd.id, "BAD_FRAME"); break; }
        /* A control frame can still arrive here if it raced with DONE. */
        if (cmd.kind != COLI_SERVE_COMMAND_SUBMIT) {
            if (cmd.kind == COLI_SERVE_COMMAND_RESET) {
                if (cmd.slot != 0) {
                    coli_serve_write_error(stdout, cmd.id, "INVALID_CACHE_SLOT");
                } else {
                    q38pool_clear();
                    reset_state(m);
                    gpu_state_push(m);
                    q38_vision_clear(m);
                    coli_serve_write_reset(stdout, cmd.id);
                }
            } else if (cmd.kind == COLI_SERVE_COMMAND_CANCEL)
                coli_serve_write_error(stdout, cmd.id, "CANCELLED");
            coli_serve_command_dispose(&cmd);
            continue;
        }
        serve_one(m, &cmd, logits);
        coli_serve_command_dispose(&cmd);
    }
    free(logits);
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) {
        fprintf(stderr, "usage: SNAP=<container> [TOK=<tokenizer.json>] [N_NEW=32] [PROMPT=<text>] %s [prompt.txt]\n", argv[0]);
        return 1;
    }
    g_debug = getenv("QWEN38_DEBUG") ? atoi(getenv("QWEN38_DEBUG")) : 0;

    /* tokenizer */
    {
        const char *tokpath = getenv("TOK");
        char tpb[2048];
        if (!tokpath || !*tokpath) { snprintf(tpb, sizeof tpb, "%s/tokenizer.json", snap); tokpath = tpb; }
        load_tokenizer(tokpath);
        if (!g_tok) { fprintf(stderr, "[tok] tokenizer required\n"); return 1; }
    }

    /* P7: in serve mode the prompts arrive over the wire, so skip the argv
     * prompt entirely -- `coli serve` launches the engine as
     * `qwen38 <cap>`, and reading argv[1] as a file would kill the process
     * before serve_loop is ever reached. */
    int serve_mode = getenv("SERVE") && getenv("SERVE")[0] == '1';

    /* prompt: PROMPT env wins, else argv[1] file */
    char *text = NULL;
    int *prompt = NULL, np = 0;
    if (!serve_mode) {
        if (getenv("PROMPT")) text = strdup(getenv("PROMPT"));
        else if (argc > 1) {
            FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
            fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
            text = malloc(n+1); if (fread(text,1,n,f)!=(size_t)n){} text[n]=0; fclose(f);
            /* strip one trailing newline (editors add it; llama-cli -p has none) */
            if (n > 0 && text[n-1]=='\n') text[n-1]=0;
        } else { fprintf(stderr, "no prompt (argv[1] file or PROMPT env)\n"); return 1; }

        encode_text(text, &prompt, &np);
        fprintf(stderr, "[enc] %d prompt tokens:", np);
        for (int i = 0; i < np && i < 128; i++) fprintf(stderr, " %d", prompt[i]);
        fprintf(stderr, "\n");
        if (getenv("TOKENIZE_ONLY")) { printf("["); for (int i=0;i<np;i++) printf("%s%d", i?",":"", prompt[i]); printf("]\n"); return 0; }
    }

    int n_new = getenv("N_NEW") ? atoi(getenv("N_NEW")) : 32;
    if (n_new < 1) n_new = 1;

    Model m;
    double t0 = now_s();
    g_dense_i8 = getenv("Q38_DENSE_I8") ? atoi(getenv("Q38_DENSE_I8")) : 0;
    if (g_dense_i8) fprintf(stderr, "[qwen38] dense-i8 requant enabled (Q38_DENSE_I8)\n");
    model_init(&m, snap);
    Cfg *c = &m.c;
    int n_qsa = 0; for (int i = 0; i < c->n_layers; i++) n_qsa += c->is_attn[i];
    fprintf(stderr, "[qwen38] loaded: hidden=%d layers=%d experts=%d topk=%d inter=%d "
            "qsa=%d/%d q_heads=%d kv_heads=%d head_dim=%d n_rot=%d theta=%.0f eps=%g\n",
            c->hidden, c->n_layers, c->n_experts, c->topk, c->inter,
            n_qsa, c->n_layers,
            c->q_heads, c->kv_heads, c->head_dim, c->n_rot, c->theta, c->eps);
    fprintf(stderr, "[qwen38] gdn: vh=%d kh=%d kd=%d vd=%d convk=%d conv_dim=%d | hc rank=%d\n",
            c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim, c->hc_rank);
    fprintf(stderr, "[qwen38] ple: layer=%d ngram=%d heads=%d dim=%d rows=%lld eos=%d convk=%d "
            "mult=[%llu,%llu,%llu]\n",
            c->ple_layer, c->ple_ngram, c->ple_n_heads, c->ple_dim, (long long)c->ple_rows,
            c->ple_eos, c->ple_convk,
            (unsigned long long)c->ple_mult[0], (unsigned long long)c->ple_mult[1], (unsigned long long)c->ple_mult[2]);
    fprintf(stderr, "[qwen38] dense loaded in %.1fs | RSS %.2f GB\n", m.dense_load_s, rss_gb());
    g_timers = getenv("COLI_TIMERS") ? atoi(getenv("COLI_TIMERS")) : 0;

    /* S3 env */
    m.kv_q8     = getenv("Q38_KV_Q8") ? atoi(getenv("Q38_KV_Q8")) : 0;
    g_full_attn = getenv("Q38_FULL_ATTN") ? atoi(getenv("Q38_FULL_ATTN")) : 0;
    g_ms_decode = getenv("Q38_MS_DECODE") ? atoi(getenv("Q38_MS_DECODE")) : 0;
    if (m.kv_q8) fprintf(stderr, "[qwen38] q8 KV cache enabled (Q38_KV_Q8)\n");
    if (g_full_attn) fprintf(stderr, "[qwen38] QSA forced to full attention (Q38_FULL_ATTN)\n");
    q38rt_open(c);                               /* Q38_ROUTE_TRACE=<path>; no-op if unset */
    int max_ctx = q38_max_ctx();
    memory_report(&m, max_ctx);

    /* P5: MTP env. */
    { const char *e = getenv("Q38_MTP_N");
      if (e && *e) { g_mtp_n = atoi(e); if (g_mtp_n < 1) g_mtp_n = 1; if (g_mtp_n > MTP_MAX_N) g_mtp_n = MTP_MAX_N; } }
    { const char *e = getenv("Q38_MTP_FOLD");
      if (e && !strcmp(e, "stream")) g_mtp_fold_stream = 1;
      if (e && !strcmp(e, "mean"))   g_mtp_fold_stream = 0; }
    g_mtp_debug = getenv("Q38_MTP_DEBUG") ? atoi(getenv("Q38_MTP_DEBUG")) : 0;
    if (m.has_mtp)
        fprintf(stderr, "[mtp] speculative decode armed: N=%d\n", g_mtp_n);

    /* P4: dense backbone to VRAM BEFORE the expert tier sizes itself -- the
     * tier's auto budget reads cudaMemGetInfo, so these bytes are already out
     * of `free` and no extra reservation is needed (an explicit
     * CUDA_EXPERT_GB must be lowered by ~5 GB by hand). */
    g_gpu_dense = gpu_dense_setup(&m);
    if (!gpu_graph_prime(&m)) return 1;

    /* S2: CUDA VRAM expert tier (COLI_CUDA=1; no-op stubs otherwise).  KV and
     * indexer caches are host allocations and do not reduce its CUDA budget.
     * P5: the MTP module's 512 experts join as one extra layer slot (48). */
    int tier_nl = c->n_layers + m.has_mtp;
    /* P6.1: the GPU vision tower allocates ~1 GB later (lazily, on the
     * first image); keep that out of an auto expert budget when the mmproj
     * is present and the GPU tower is not disabled. */
    if (g_gpu_dense && !getenv("QT_RESERVE_MB") &&
        !(getenv("Q38_GPU_VIT") && !atoi(getenv("Q38_GPU_VIT")))) {
        char pb[2048];
        const char *mp = getenv("Q38_MMPROJ");
        if (!mp || !*mp) { snprintf(pb, sizeof pb, "%s/mmproj-F16.gguf", m.snap); mp = pb; }
        FILE *pf = fopen(mp, "rb");
        if (pf) { fclose(pf); setenv("QT_RESERVE_MB", "1024", 0); }   /* 0.83 GB weights + small-image activations; the 2 GiB margin covers large images */
    }
    if (m.have_mmap &&
        q38t_init(tier_nl, c->n_experts, c->hidden, c->inter, c->topk,
                  m.e_gu_type, m.e_d_type)) {
        for (int i = 0; i < tier_nl; i++)
            for (int e = 0; e < c->n_experts; e++) {
                int64_t si = (int64_t)i * c->n_experts + e;
                q38t_set_src(i, e, m.e_g[si], m.e_u[si], m.e_d[si]);
            }
        /* shared experts ride the GPU chain (same [inter,D]/[D,inter] shapes) */
        if (!getenv("Q38_NO_GPU_SHARED"))
            for (int i = 0; i < tier_nl; i++) {
                Layer *l = &m.L[i];
                if (l->sh_g.type == l->sh_u.type)
                    q38t_set_shared(i, l->sh_g.raw, l->sh_u.raw, l->sh_d.raw,
                                    l->sh_g.type, l->sh_d.type);
            }
        /* P4: arm the async prefetcher.  The serving layout keeps the router
         * artifact next to the model shards; Q38_ROUTER overrides.  A missing
         * artifact degrades to the prev-token-union policy alone. */
        {
            char rp[2048];
            const char *rpath = getenv("Q38_ROUTER");
            if (!rpath || !*rpath) { snprintf(rp, sizeof rp, "%s/router_v1_shallow8.bin", snap); rpath = rp; }
            q38t_prefetch_setup(rpath);
        }
        q38t_warmstart();
    }

    tele_init(c);          /* EMAP/HITS grids: cheap, and CLI runs report them too */

    /* P7: `coli serve` mode speaks the gateway wire protocol instead of doing
     * one argv generation.  It runs AFTER the tier init on purpose -- serve
     * turns must ride the same VRAM experts and the same warmstarted heat
     * table an argv run does, or the first HTTP request would pay a cold tier
     * that the CLI never sees.  serve_loop returns only on stdin EOF. */
    if (serve_mode) {
        fprintf(stderr, "[qwen38] serve mode: ctx %d, gpu_dense=%d, tier=%s\n",
                max_ctx, g_gpu_dense, q38t_ready() ? "on" : "off");
        q38pool_init(&m);          /* P3: prompt/KV cache pool (serve only) */
        serve_loop(&m);
        q38rt_close();
        if (g_router_tok) fprintf(stderr, "[router] device logits: %ld token-layers, %ld host fallbacks (%.3f %%), gap %g\n",
                                  g_router_tok, g_router_fallback, 100.0 * g_router_fallback / g_router_tok, router_gap_env());
        if (g_router_checked) fprintf(stderr, "[router] check: %ld decisions, %ld selection mismatches vs host, max |dev-host logit| %.3e\n",
                                      g_router_checked, g_router_mismatch, g_router_maxerr);
        q38t_stats(); q38t_shutdown(); q38g_shutdown();
        return 0;
    }

    if (np + n_new > max_ctx) { fprintf(stderr, "[ctx] %d+%d exceeds %d (raise Q38_CTX)\n", np, n_new, max_ctx); return 1; }
    ensure_kv(&m, np + n_new);
    reset_state(&m);

    /* P6: Q38_IMAGE=<patches.f32> + Q38_GRID=HxW feed one preprocessed image
     * (tools/qwen38_image.py --out) whose <|image_pad|> placeholders must be
     * in the prompt.  Q38_VIS_DUMP=<path> writes the projected embeddings. */
    if (getenv("Q38_IMAGE")) {
        int gh = 0, gw = 0;
        const char *g = getenv("Q38_GRID");
        if (!g || sscanf(g, "%dx%d", &gh, &gw) != 2) {
            fprintf(stderr, "Q38_IMAGE needs Q38_GRID=HxW (patch grid)\n"); return 1;
        }
        int64_t want = (int64_t)gh * gw * Q38_VIS_PATCH_FLOATS;
        FILE *pf = fopen(getenv("Q38_IMAGE"), "rb");
        if (!pf) { perror(getenv("Q38_IMAGE")); return 1; }
        float *patches = falloc(want);
        if (fread(patches, sizeof(float), (size_t)want, pf) != (size_t)want) {
            fprintf(stderr, "%s: expected %lld floats for grid %dx%d\n",
                    getenv("Q38_IMAGE"), (long long)want, gh, gw); return 1;
        }
        fclose(pf);
        char verr[192];
        float *one[1] = { patches };
        if (q38_vision_prepare(&m, prompt, np, np + n_new, one, &gh, &gw, 1,
                               verr, sizeof verr) != 0) {
            fprintf(stderr, "[vis] %s\n", verr); return 1;
        }
        free(patches);
    }
    if (getenv("Q38_IDX_CHECK") && atoi(getenv("Q38_IDX_CHECK")))
        g_idx_check_pos = np - 1;

    int *out = malloc((size_t)(np + n_new) * sizeof(int));
    memcpy(out, prompt, (size_t)np * sizeof(int));
    float *logits = falloc(c->vocab);

    /* S3 prefill: layer-major chunks (bit-identical to token-at-a-time);
     * logits only for the last prompt token.  Q38_PREFILL_B=1 restores the
     * S1 loop (forced under QWEN38_DEBUG>=2, whose taps are per-token). */
    double tp0 = now_s();
    int gpu_prefill = q38_prefill(&m, prompt, np, logits, 1);
    double tp1 = now_s();
    fprintf(stderr, "[prefill] %d tokens in %.1fs (%.2f tok/s)\n", np, tp1-tp0, np/(tp1-tp0));
    /* P4: hand the GDN recurrent/conv state to the device for the decode loop */
    if (!gpu_prefill) gpu_state_push(&m);

    /* greedy decode */
    int eos_im_end = -1, eos_eot = -1;
    for (int k = 0; k < g_nspecial; k++) {
        if (!strcmp(g_sp_str[k], "<|im_end|>")) eos_im_end = g_sp_id[k];
        if (!strcmp(g_sp_str[k], "<|endoftext|>")) eos_eot = g_sp_id[k];
    }
    int stop_on_eos = getenv("NO_EOS_STOP") ? 0 : 1;
    unsigned char sbuf[16]; int sbn = 0;
    fprintf(stderr, "IDs :");
    int gen = 0;
    int spec_on = m.has_mtp && (!g_gpu_dense || g_gpu_spec);   /* P6.4: image turns speculate too */
    /* Q38_LOGIT_DUMP=<path>: every step's full logits (raw f32 rows), for
     * the router-parity measurements (KL, max error, top-1 margin) */
    FILE *ldump = getenv("Q38_LOGIT_DUMP") ? fopen(getenv("Q38_LOGIT_DUMP"), "wb") : NULL;
    for (int s = 0; s < n_new; ) {
        if (ldump) fwrite(logits, sizeof(float), (size_t)c->vocab, ldump);
        int best = 0; float bv = logits[0];
        for (int i = 1; i < c->vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        out[np + gen++] = best;
        fprintf(stderr, " %d", best);
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(best, tmp, &tn);
        out_bytes(sbuf, &sbn, tmp, tn);
        fflush(stdout);
        if (stop_on_eos && (best == eos_im_end || best == eos_eot)) break;
        s++;
        if (s >= n_new) break;
        if (spec_on) {
            /* P5: draft-verify.  Accepted drafts arrive verified (argmax of
             * the same batched logits a sequential greedy decode produces),
             * so they are emitted exactly like sampled tokens. */
            int extra[MTP_MAX_N + 1];
            int ne = q38_spec_step(&m, best, np + s - 1, n_new - s - 1, logits, extra);
            int hit_eos = 0;
            for (int j = 0; j < ne; j++) {
                int tk2 = extra[j];
                out[np + gen++] = tk2;
                fprintf(stderr, " %d", tk2);
                tn = 0; decode_id_to_bytes(tk2, tmp, &tn);
                out_bytes(sbuf, &sbn, tmp, tn);
                s++;
                if (stop_on_eos && (tk2 == eos_im_end || tk2 == eos_eot)) { hit_eos = 1; break; }
            }
            if (hit_eos) break;
        } else {
            /* P4's bigram prefetch hook lives inside q38_decode_step: the next
             * token's identity is known here, so its shallow-layer experts are
             * queued before its forward pass starts. */
            q38_decode_step(&m, best, np + s - 1, logits);
        }
    }
    if (sbn) fwrite(sbuf, 1, (size_t)sbn, stdout);
    printf("\n");
    fprintf(stderr, "\n");
    double t1 = now_s();
    fprintf(stderr, "[decode] %d tokens in %.1fs (%.2f tok/s) | total %.1fs | PEAK RSS %.2f GB\n",
            gen, t1-tp1, gen/(t1-tp1), t1-t0, rss_gb());
    mtp_print_stats();
    fprintf(stderr, "Text: "); print_decoded(out, np, np+gen); fprintf(stderr, "\n");
    if (g_timers && g_tm.tokens) {
        double n = (double)g_tm.tokens;
        fprintf(stderr, "[timers] ms/token over %ld tokens: hc %.1f | gdn %.1f | qsa %.1f | "
                "ple %.1f | route %.1f | moe_cpu %.1f | moe_gpu_wait %.1f | shared %.1f | head %.1f\n",
                g_tm.tokens, 1e3*g_tm.dense_hc/n, 1e3*g_tm.gdn/n, 1e3*g_tm.qsa/n,
                1e3*g_tm.ple/n, 1e3*g_tm.moe_route/n, 1e3*g_tm.moe_cpu/n,
                1e3*g_tm.moe_gpu/n, 1e3*g_tm.moe_shared/n, 1e3*g_tm.head/n);
    }
    q38rt_close();                               /* flush + fclose at EOS / end of decode */
    if (g_router_tok) fprintf(stderr, "[router] device logits: %ld token-layers, %ld host fallbacks (%.3f %%), gap %g\n",
                              g_router_tok, g_router_fallback, 100.0 * g_router_fallback / g_router_tok, router_gap_env());
    if (g_router_checked) fprintf(stderr, "[router] check: %ld decisions, %ld selection mismatches vs host, max |dev-host logit| %.3e\n",
                                  g_router_checked, g_router_mismatch, g_router_maxerr);
    q38t_stats();
    q38t_shutdown();
    q38g_shutdown();

    if (getenv("DUMP")) {
        FILE *df = fopen(getenv("DUMP"), "wb");
        if (df) { fwrite(logits, sizeof(float), (size_t)c->vocab, df); fclose(df);
                  fprintf(stderr, "[dump] wrote %d logits -> %s\n", c->vocab, getenv("DUMP")); }
    }
    return 0;
}
