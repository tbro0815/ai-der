#!/usr/bin/env python3
"""Convert the extracted Qwen3.8-Flash-Next MTP module -> Colibri MTP container.

Input  : ~/models/qwen38-mtp/mtp.safetensors  (31 BF16 tensors, HF names)
         ~/models/qwen38-mtp/mtp_manifest.json
Output : ~/models/qwen38-colibri/mtp/
           mtp-dense.safetensors     norms/fc/attn/hc/shared-expert/router
           mtp-experts.safetensors   mtp.experts.{E}.gate_raw/.up_raw/.down_raw
           mtp_meta.json             tensor map + geometry
           mtp_quant_report.json     per-tensor BF16->Q8_0 rel-RMSE

Conventions mirror tools/convert_qwen38.py and docs/p2-converter-design.md
(REVISION 2): quantized tensors are stored as RAW GGML blocks in safetensors
dtype U8, one flat 1-D tensor each, with the GGML type id recorded per tensor
in the meta json.  Unlike the main converter (which passes GGUF blocks through
byte-identically) this one has BF16 sources, so it quantizes to Q8_0 itself
using the ggml `quantize_row_q8_0_ref` algorithm (ggml/src/ggml-quants.c:276).

Output format spec: docs/mtp-container.md.

Deps: numpy only.  Python 3.10+.
Self-test (no model needed):  python3 tools/convert_mtp.py --self-test
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_qwen38 import (  # noqa: E402  (shared conventions live in the main converter)
    BLOCK_SIZES,
    GGML_F32,
    GGML_Q8_0,
    QTYPE_NAMES,
    bf16_decode,
    dequant_q8_0,
    rel_rmse,
    write_safetensors,
)

# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

# Anything with <= this many elements (and every 1-D tensor) is kept F32:
# norms, hc block-inject rows, qk-norm vectors, shared-expert gate.
F32_MAX_ELEMS = 1 << 16
# Kept F32 regardless of size: routing decisions are discrete, so quantization
# noise there changes *which* experts fire, not just by how much.
FORCE_F32 = {
    "mtp.layers.0.mlp.gate.weight",              # 512-way router  [512, 2560]
    "mtp.layers.0.mlp.shared_expert_gate.weight",
}
MAX_REL_RMSE = 0.02          # hard fail gate for any quantized matrix

EXPERT_FUSED = "mtp.layers.0.mlp.experts.gate_up_proj"
EXPERT_DOWN = "mtp.layers.0.mlp.experts.down_proj"

DENSE_SHARD = "mtp-dense.safetensors"
EXPERT_SHARD = "mtp-experts.safetensors"


# ---------------------------------------------------------------------------
# Q8_0 encoder — ggml quantize_row_q8_0_ref (ggml/src/ggml-quants.c)
#   d  = amax/127 (stored f16), id = d ? 1/d : 0, qs[j] = roundf(x[j]*id)
#   block layout: 2 B f16 scale + 32 B int8 = 34 B / 32 elems
# ---------------------------------------------------------------------------

def _round_half_away(x: np.ndarray) -> np.ndarray:
    """C roundf(): half away from zero (np.rint is half-to-even)."""
    return np.trunc(x + np.copysign(0.5, x))


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """f32 array (any shape, C-contiguous) -> uint8 raw GGML Q8_0 blocks, flat.

    Rows must be a multiple of 32 elements so that no block straddles two rows
    (true for every MTP matrix: input dims are 320/640/2560/6144/10240)."""
    be, bb = BLOCK_SIZES[GGML_Q8_0]
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if x.size % be:
        raise ValueError(f"Q8_0: {x.size} elems not a multiple of {be}")
    nb = x.size // be
    xb = x.reshape(nb, be)
    amax = np.abs(xb).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    idv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0).astype(np.float32)
    q = _round_half_away(xb * idv[:, None])
    q = np.clip(q, -127.0, 127.0).astype(np.int8)
    out = np.empty((nb, bb), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:bb] = q.view(np.uint8)
    return out.reshape(-1)


def q8_0_nbytes(n_elems: int) -> int:
    be, bb = BLOCK_SIZES[GGML_Q8_0]
    return n_elems // be * bb


# ---------------------------------------------------------------------------
# mmap'd safetensors source (BF16 in, no full-file load)
# ---------------------------------------------------------------------------

ST_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
         "U8": np.uint8, "I8": np.int8}


class STSource:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        (hlen,) = struct.unpack("<Q", self.f.read(8))
        self.header = json.loads(self.f.read(hlen))
        self.base = 8 + hlen
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = np.frombuffer(self.mm, dtype=np.uint8)
        self.names = [k for k in self.header if k != "__metadata__"]

    def meta(self, name):
        m = self.header[name]
        return m["dtype"], tuple(m["shape"])

    def raw(self, name):
        m = self.header[name]
        a, b = m["data_offsets"]
        return self.buf[self.base + a: self.base + b]

    def f32(self, name, index=None):
        """Whole tensor (or index-th slice of dim 0) as f32."""
        dt, shape = self.meta(name)
        raw = self.raw(name)
        arr = raw.view(ST_NP[dt]).reshape(shape)
        if index is not None:
            arr = arr[index]
        if dt == "BF16":
            return bf16_decode(np.ascontiguousarray(arr))
        return np.ascontiguousarray(arr).astype(np.float32)


# ---------------------------------------------------------------------------
# geometry derived from the manifest / source shapes
# ---------------------------------------------------------------------------

def derive_geometry(shapes):
    def s(n):
        return shapes.get(n)

    gate_up = s(EXPERT_FUSED)          # [E, 2*moe_inter, hidden]
    down = s(EXPERT_DOWN)              # [E, hidden, moe_inter]
    hidden = gate_up[2]
    n_experts = gate_up[0]
    moe_inter = gate_up[1] // 2
    q_out = s("mtp.layers.0.self_attn.q_proj.weight")[0]
    k_out = s("mtp.layers.0.self_attn.k_proj.weight")[0]
    o_in = s("mtp.layers.0.self_attn.o_proj.weight")[1]
    head_dim = s("mtp.layers.0.self_attn.q_norm.weight")[0]
    hc_dim = s("mtp.layers.0.mlp_hyper_connection.hc_norm.weight")[0]
    hc_rank = s("mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight")[0]
    geo = {
        "hidden": hidden,
        "n_mtp_layers": 1,
        "num_experts": n_experts,
        "moe_inter": moe_inter,
        "shared_inter": s("mtp.layers.0.mlp.shared_expert.gate_proj.weight")[0],
        "q_proj_out": q_out,
        "k_proj_out": k_out,
        "v_proj_out": s("mtp.layers.0.self_attn.v_proj.weight")[0],
        "o_in": o_in,
        "head_dim": head_dim,
        # o_proj input is heads*head_dim; q_proj out is 2x that (gated attention),
        # matching the main model's q_heads=24 / kv_heads=2 with head_dim 256.
        "q_heads": o_in // head_dim,
        "kv_heads": k_out // head_dim,
        "q_proj_gate_multiple": q_out // o_in,
        "indexer": {
            "qk_out": s("mtp.layers.0.self_attn.indexer.index_qk_proj.weight")[0],
            "norm_dim": s("mtp.layers.0.self_attn.indexer.q_layernorm.weight")[0],
        },
        "hc": {"streams": s("mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight")[0],
               "rank": hc_rank, "dim": hc_dim},
        "fc": {"embedding_in": s("mtp.fc_embedding.weight")[1],
               "embedding_out": s("mtp.fc_embedding.weight")[0],
               "hidden_in": s("mtp.fc_hidden.weight")[1],
               "hidden_out": s("mtp.fc_hidden.weight")[0],
               "pre_fc_norm_embedding": s("mtp.pre_fc_norm_embedding.weight")[0],
               "pre_fc_norm_hidden": s("mtp.pre_fc_norm_hidden.weight")[0]},
        "expert_layout": {
            "gate_up_proj": {"source_shape": list(gate_up),
                             "axes": ["expert", "out(gate|up)", "in"],
                             "gate_rows": [0, moe_inter],
                             "up_rows": [moe_inter, 2 * moe_inter]},
            "down_proj": {"source_shape": list(down), "axes": ["expert", "out", "in"]},
        },
    }
    return geo


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def plan_dense(src):
    """-> list of (name, kind) where kind is 'F32' or 'Q8_0'."""
    plan = []
    for name in sorted(src.names):
        if name in (EXPERT_FUSED, EXPERT_DOWN):
            continue
        _dt, shape = src.meta(name)
        n = int(np.prod(shape))
        if len(shape) == 1 or n <= F32_MAX_ELEMS or name in FORCE_F32:
            plan.append((name, "F32"))
        else:
            plan.append((name, "Q8_0"))
    return plan


def convert(args):
    src_path = Path(args.src).expanduser()
    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    src = STSource(src_path)

    shapes = {n: list(src.meta(n)[1]) for n in src.names}
    man_shapes = {k: v["shape"] for k, v in manifest["tensors"].items()}
    for n, sh in shapes.items():
        if man_shapes.get(n) != sh:
            sys.exit(f"{n}: safetensors shape {sh} != manifest {man_shapes.get(n)}")
    for n in man_shapes:
        if n not in shapes:
            sys.exit(f"manifest tensor {n} missing from {src_path}")
    geo = derive_geometry(shapes)
    n_experts = geo["num_experts"] if args.limit_experts is None else min(
        args.limit_experts, geo["num_experts"])
    moe_inter, hidden = geo["moe_inter"], geo["hidden"]

    report = {}          # container name -> rel-RMSE (quantized tensors only)
    tmap = {}            # container name -> meta entry

    def record(name, ggml_type, st_dtype, shape, shard, nbytes):
        tmap[name] = {"ggml_type": ggml_type, "ggml_type_name": QTYPE_NAMES[ggml_type],
                      "st_dtype": st_dtype, "shape": [int(s) for s in shape],
                      "shard": shard, "nbytes": int(nbytes)}

    def quant_provider(name, get_f32, cname):
        def prov():
            a = get_f32()
            raw = quantize_q8_0(a)
            err = rel_rmse(dequant_q8_0(raw, a.size).reshape(a.shape), a)
            report[cname] = err
            if err > MAX_REL_RMSE:
                sys.exit(f"{cname}: Q8_0 rel-RMSE {err:.4%} > {MAX_REL_RMSE:.0%}")
            return raw.tobytes()
        return prov

    # ---- dense shard -----------------------------------------------------
    dense_entries = []
    for name, kind in plan_dense(src):
        shape = shapes[name]
        if kind == "F32":
            arr = src.f32(name)
            record(name, GGML_F32, "F32", shape, DENSE_SHARD, int(np.prod(shape)) * 4)
            dense_entries.append((name, "F32", tuple(shape), arr))
        else:
            n = int(np.prod(shape))
            record(name, GGML_Q8_0, "U8", shape, DENSE_SHARD, q8_0_nbytes(n))
            dense_entries.append((name, "U8", (q8_0_nbytes(n),),
                                  quant_provider(name, lambda nm=name: src.f32(nm), name)))
    t0 = time.time()
    write_safetensors(out / DENSE_SHARD, dense_entries)
    print(f"[dense] {DENSE_SHARD}: {len(dense_entries)} tensors, "
          f"{(out / DENSE_SHARD).stat().st_size / 2**20:.1f} MiB in {time.time() - t0:.1f}s")

    # ---- expert shard ----------------------------------------------------
    # gate_up_proj [E, 2*I, H] -> per expert rows [0:I] = gate, [I:2I] = up,
    # each [out=I, in=H];  down_proj [E, H, I] -> per expert [out=H, in=I].
    # (llama.cpp conversion/qwen.py Qwen2MoeModel.modify_tensors, see docs.)
    gu_bytes = q8_0_nbytes(moe_inter * hidden)
    dn_bytes = q8_0_nbytes(hidden * moe_inter)
    expert_entries = []
    for e in range(n_experts):
        for part, rows in (("gate", (0, moe_inter)), ("up", (moe_inter, 2 * moe_inter))):
            cname = f"mtp.experts.{e}.{part}_raw"
            record(cname, GGML_Q8_0, "U8", [moe_inter, hidden], EXPERT_SHARD, gu_bytes)

            def get(e=e, rows=rows):
                return src.f32(EXPERT_FUSED, index=e)[rows[0]:rows[1], :]
            expert_entries.append((cname, "U8", (gu_bytes,), quant_provider(cname, get, cname)))
        cname = f"mtp.experts.{e}.down_raw"
        record(cname, GGML_Q8_0, "U8", [hidden, moe_inter], EXPERT_SHARD, dn_bytes)
        expert_entries.append((cname, "U8", (dn_bytes,),
                               quant_provider(cname, lambda e=e: src.f32(EXPERT_DOWN, index=e),
                                              cname)))
    t0 = time.time()
    write_safetensors(out / EXPERT_SHARD, expert_entries)
    print(f"[experts] {EXPERT_SHARD}: {n_experts} experts x3, "
          f"{(out / EXPERT_SHARD).stat().st_size / 2**30:.2f} GiB in {time.time() - t0:.1f}s")

    # ---- meta ------------------------------------------------------------
    meta = {
        "model_type": "qwen38-mtp",
        "container_format": "colibri-mtp-1",
        "expert_format": "ggml_raw",
        "source": {"repo": manifest.get("repo"), "file": src_path.name,
                   "manifest_total_bytes": manifest.get("total_bytes")},
        "ggml_type_names": {str(GGML_F32): QTYPE_NAMES[GGML_F32],
                            str(GGML_Q8_0): QTYPE_NAMES[GGML_Q8_0]},
        "geometry": geo,
        "expert_ggml_types": {"0": {"gate_type": GGML_Q8_0, "up_type": GGML_Q8_0,
                                    "down_type": GGML_Q8_0}},
        "num_experts_written": n_experts,
        "shards": [DENSE_SHARD, EXPERT_SHARD],
        "tensors": tmap,
        "source_shapes": man_shapes,
        "quant_report": "mtp_quant_report.json",
        "max_rel_rmse_gate": MAX_REL_RMSE,
    }
    (out / "mtp_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "mtp_quant_report.json").write_text(json.dumps(
        {k: round(v, 6) for k, v in sorted(report.items())}, indent=1))

    # ---- summary ---------------------------------------------------------
    dense_err = {k: v for k, v in report.items() if not k.startswith("mtp.experts.")}
    exp_err = np.array([v for k, v in report.items() if k.startswith("mtp.experts.")])
    print("\n[rel-RMSE BF16->Q8_0] dense matrices:")
    for k in sorted(dense_err):
        print(f"  {dense_err[k]:.4%}  {k}")
    if exp_err.size:
        worst = max((v, k) for k, v in report.items() if k.startswith("mtp.experts."))
        print(f"[rel-RMSE] experts ({exp_err.size} tensors): min {exp_err.min():.4%} "
              f"p50 {np.percentile(exp_err, 50):.4%} p99 {np.percentile(exp_err, 99):.4%} "
              f"max {worst[0]:.4%} ({worst[1]})")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    worst_all = max(report.values()) if report else 0.0
    print(f"[out] {out}  {total / 2**30:.2f} GiB total, worst rel-RMSE {worst_all:.4%}")


# ---------------------------------------------------------------------------
# self-test: encoder must match a scalar transcription of quantize_row_q8_0_ref
# ---------------------------------------------------------------------------

def _q8_0_ref_scalar(x):
    import math
    out = bytearray()
    for i in range(0, len(x), 32):
        blk = x[i:i + 32]
        amax = max(abs(float(v)) for v in blk)
        d = np.float32(amax / 127.0)
        idv = np.float32(1.0 / d) if d != 0 else np.float32(0.0)
        out += np.float16(d).tobytes()
        for v in blk:
            x0 = float(np.float32(v) * idv)
            out.append(int(math.floor(x0 + 0.5) if x0 >= 0 else math.ceil(x0 - 0.5)) & 0xFF)
    return np.frombuffer(bytes(out), dtype=np.uint8)


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
        ok = ok and cond

    rng = np.random.default_rng(0)
    for tag, x in (("gauss", rng.standard_normal(32 * 40).astype(np.float32)),
                   ("zeros", np.zeros(64, np.float32)),
                   ("spiky", (rng.standard_normal(320) * rng.choice([1, 50], 320)
                              ).astype(np.float32)),
                   ("halves", (np.arange(64, dtype=np.float32) - 32) / 64.0 * 1.0)):
        got, ref = quantize_q8_0(x), _q8_0_ref_scalar(x)
        check(f"q8_0 byte-exact vs ggml ref [{tag}]", bool(np.array_equal(got, ref)))
    x = rng.standard_normal((64, 256)).astype(np.float32)
    err = rel_rmse(dequant_q8_0(quantize_q8_0(x), x.size).reshape(x.shape), x)
    check("q8_0 round-trip rel-RMSE in [0.1%, 1%]", 0.001 < err < 0.01, f"({err:.4%})")
    check("block size 34B/32", BLOCK_SIZES[GGML_Q8_0] == (32, 34))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/models/qwen38-mtp/mtp.safetensors")
    ap.add_argument("--manifest", default="~/models/qwen38-mtp/mtp_manifest.json")
    ap.add_argument("--out", default="~/models/qwen38-colibri/mtp")
    ap.add_argument("--limit-experts", type=int, default=None,
                    help="convert only the first N experts (smoke runs)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    convert(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
