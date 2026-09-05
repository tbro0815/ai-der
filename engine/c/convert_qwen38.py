#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next (arch qwen4exp, UD-IQ4_XS GGUF) -> Colibri container.

AI-DER container converter (raw GGML block storage).
Container layout mirrors the qwen36 conventions (engine/c/tools/convert_qwen36.py,
engine/c/st.h) but weights are NOT requantized: every quantized tensor is stored
as a byte-identical copy of its GGML blocks (safetensors dtype U8), with the
GGML type id recorded per tensor in qwen38_meta.json.  F32/F16/BF16 tensors are
stored typed and byte-identical (BF16 is kept as BF16 — st.h reads it natively).

Per expert E of layer L the three matrices are individually addressable:
  model.layers.{L}.mlp.experts.{E}.gate_raw / .up_raw / .down_raw   (U8, raw blocks)
GGML type ids per layer live in qwen38_meta.json under expert_ggml_types;
all other raw tensors are listed under raw_tensors with type id + logical shape.

Output:
  out/
    model-globals.safetensors      embed_tokens, lm_head, final norm, output_hc_*
    model-00000..000NN.safetensors one shard per layer (dense + 512 experts, raw)
    model-ple.safetensors          ple key/value/conv1d/norm projections
    ple_table.bin                  raw IQ4_NL per_layer_token_embd rows (COLIPLE1 header)
    config.json / qwen38_meta.json / tokenizer.gguf.json (+ chat_template.jinja)

Deps: numpy, gguf (pip).  No torch.  Python 3.12+.
Self-test (no model needed):  python3 engine/c/convert_qwen38.py --self-test
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GGML quant constants (ported from llama.cpp ggml/src/ggml-quants.c / ggml-common.h)
# ---------------------------------------------------------------------------

# GGML type ids (gguf.constants.GGMLQuantizationType values)
GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q6_K = 0, 1, 8, 14
GGML_IQ4_NL, GGML_IQ3_S, GGML_IQ4_XS, GGML_BF16 = 20, 21, 23, 30

# (block_elems, block_bytes) per type we support
BLOCK_SIZES = {
    GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2),
    GGML_Q8_0: (32, 34), GGML_Q6_K: (256, 210),
    GGML_IQ4_NL: (32, 18), GGML_IQ3_S: (256, 110), GGML_IQ4_XS: (256, 136),
}
QTYPE_NAMES = {
    GGML_F32: "F32", GGML_F16: "F16", GGML_BF16: "BF16", GGML_Q8_0: "Q8_0",
    GGML_Q6_K: "Q6_K", GGML_IQ4_NL: "IQ4_NL", GGML_IQ3_S: "IQ3_S", GGML_IQ4_XS: "IQ4_XS",
}
TYPED_ST_DTYPE = {GGML_F32: "F32", GGML_F16: "F16", GGML_BF16: "BF16"}

# kvalues_iq4nl (ggml-common.h)
IQ4NL_LUT = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                      1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)

# iq3s_grid, 512 x uint32 (each entry = 4 packed uint8 magnitudes), hex dump of
# the ggml-common.h table, in order.
_IQ3S_GRID_HEX = """
0101010101010103010101050101010b0101010f010103010101030301010305010103090101030d0101050101010503
0101050b0101070701010901010109050101090b0101090f01010b0301010b0701010d0101010d0501010f0301010f09
01010f0f0103010101030103010301050103010901030301010303030103030b01030501010305070103050f01030703
0103070b0103090901030d0301030d0b01030f0501050101010501030105010b0105010f01050301010503070105030d
010505030105050b0105070101050709010509050105090b0105090f01050b0301050b0701050f0101050f0701070107
010703030107030b010705010107050501070703010707070107070d0107090901070b0101070b0501070d0f01070f03
01070f0b01090101010903070109030f010905030109050901090705010909010109090701090b0301090f01010b0105
010b0109010b0501010b0505010b050d010b0707010b0903010b090b010b090f010b0d0d010b0f07010d010d010d0303
010d0307010d0703010d0b05010d0f03010f0101010f0105010f0109010f0501010f0505010f050d010f0707010f0b01
010f0b09030101010301010303010105030101090301030103010303030103070301030b0301030f0301050103010505
03010703030107090301070d03010b0903010b0d03010d0303010f050303010103030103030301070303010d03030301
030303090303050303030701030307070303090303030b0103030b0503030f0103030f0d03050101030503050305030b
0305030f030505010305050903050705030509010305090703050b0b03050d0103050f0503070103030701090307010f
0307030103070307030705030307050f03070701030707090307090303070d0503070f01030901070309010b03090305
030903090309070303090707030909050309090d03090b0103090b09030b0103030b0301030b0307030b0503030b0701
030b0705030b0b03030d0501030d0509030d050f030d0909030d090d030f0103030f0107030f0301030f0305030f0503
030f070b030f0903030f0d05030f0f010501010105010103050101070501010b0501010f050103010501030505010309
0501030d05010503050105070501050f050107010501070505010903050109070501090b05010b0105010b0505010d0f
05010f0105010f0705010f0b050301010503010505030301050303070503030f050305050503050b0503070305030709
0503090505030b0305050103050501090505010f0505050305050507050507010505070f0505090305050b0705050b0f
05050f0305050f0905070101050701050507010b05070303050705050507050905070703050707070507090505070b01
05070d0d050901030509010f0509050105090507050907050509070b0509090305090f0505090f0b050b0109050b0303
050b0505050b070f050b0901050b0b07050b0f01050d0101050d0105050d010f050d0503050d0b0b050d0d03050f010b
050f0303050f050d050f0701050f0907050f0b010701010507010303070103070701030b0701030f0701050507010703
070107070701070b07010905070109090701090f07010b0307010d0707010f0307030103070301070703010b07030309
07030503070305070703090107030d0107030f0507030f0d070501010705030507050501070507050705070907050b01
07070103070703010707030907070503070705070707050f0707070107070903070709070707090f07070b0b07070f07
07090107070903030709030d070905050709070307090b0507090d0107090d09070b0103070b0301070b0305070b050b
070b0705070b0909070b0b0d070b0f07070d030d070d0903070f0103070f0107070f0501070f0505070f070b09010101
090101090901030509010501090105090901050f090107050901090309010b0109010f01090301050903010f09030303
0903030709030505090307010903070b0903090709030b0309030b0b0905010309050107090503010905030b09050503
090507070905090109050b0f09050d0509050f010907010909070303090703070907050109070505090707030907070b
0909010109090105090905090909070f0909090109090f03090b010b090b010f090b0503090b0d05090d0307090d0709
090d0d01090f0301090f030b090f0701090f0907090f0b030b0101050b0103010b0103090b0105050b0109010b010909
0b01090f0b010b050b010d0d0b010f090b0301030b0301070b03010b0b0303050b0305030b0307050b030f050b050101
0b0503030b0505070b0507010b05070d0b050b070b0701050b07010f0b0703010b07050f0b0709090b070b030b070d0b
0b070f070b0901030b0901090b0905010b0907050b09090d0b0b03050b0b050d0b0b0b030b0b0b070b0d09050b0f0105
0b0f01090b0f05050d0103030d0103070d01030b0d0107030d0107070d010d010d0301010d0305010d03050f0d030d09
0d0503050d0507090d0509050d050b0b0d050d050d050f010d0701010d0703090d0705030d0709010d09050b0d090907
0d090d050d0b01010d0b01070d0b07090d0b0d010d0d010b0d0d09010d0f03030d0f03070f0101010f0101090f01010f
0f0105010f0105050f01070d0f0109010f010b090f010d050f0301050f0303030f0305090f0309070f03090b0f050103
0f0501090f0503010f05030d0f0505030f0507010f050b030f0701050f0707050f07070b0f070b070f0901030f09010b
0f0903070f0905010f090b010f0b05050f0b09050f0d01050f0d07030f0f0101
"""
_grid_u32 = np.array([int(_IQ3S_GRID_HEX.replace("\n", "")[i:i + 8], 16)
                      for i in range(0, 512 * 8, 8)], dtype=np.uint32)
# entry bytes little-endian: grid[idx][j] = byte j of the uint32
IQ3S_GRID = _grid_u32.view(np.uint8).reshape(512, 4).astype(np.float32)
assert IQ3S_GRID[0].tolist() == [1.0, 1.0, 1.0, 1.0]

# ---------------------------------------------------------------------------
# float helpers
# ---------------------------------------------------------------------------

def bf16_encode(a: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round to nearest even), returned as uint16."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_decode(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# GGML block dequantizers (numpy vectorized).  Input: raw bytes, output: f32 1-D.
# Kept for validation tooling (verify_container informational RMSE, engine
# reference oracles); the converter itself no longer dequantizes weights.
# ---------------------------------------------------------------------------

def _blocks(raw: np.ndarray, n: int, qtype: int) -> np.ndarray:
    be, bb = BLOCK_SIZES[qtype]
    if n % be:
        raise ValueError(f"{QTYPE_NAMES[qtype]}: {n} elems not a multiple of block {be}")
    nb = n // be
    raw = np.frombuffer(raw, dtype=np.uint8, count=nb * bb) if not isinstance(raw, np.ndarray) \
        else raw.reshape(-1)[:nb * bb]
    return raw.reshape(nb, bb)


def dequant_q8_0(raw, n):
    b = _blocks(raw, n, GGML_Q8_0)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)          # (nb,1)
    qs = b[:, 2:34].view(np.int8).astype(np.float32)                  # (nb,32)
    return (d * qs).reshape(-1)


def dequant_q6_k(raw, n):
    b = _blocks(raw, n, GGML_Q6_K)
    nb = b.shape[0]
    ql = b[:, 0:128].reshape(nb, 2, 64)          # 2 chunks of 128 values use 64 ql bytes
    qh = b[:, 128:192].reshape(nb, 2, 32)
    sc = b[:, 192:208].view(np.int8).reshape(nb, 2, 8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(nb, 1, 1, 1)
    l = np.arange(32)
    y = np.empty((nb, 2, 4, 32), dtype=np.float32)  # chunk, quarter(q1..q4), l
    q1 = ((ql[:, :, l] & 0xF) | (((qh[:, :, l] >> 0) & 3) << 4)).astype(np.int16) - 32
    q2 = ((ql[:, :, l + 32] & 0xF) | (((qh[:, :, l] >> 2) & 3) << 4)).astype(np.int16) - 32
    q3 = ((ql[:, :, l] >> 4) | (((qh[:, :, l] >> 4) & 3) << 4)).astype(np.int16) - 32
    q4 = ((ql[:, :, l + 32] >> 4) | (((qh[:, :, l] >> 6) & 3) << 4)).astype(np.int16) - 32
    is_ = l // 16                                  # 0 or 1
    y[:, :, 0] = sc[:, :, is_ + 0] * q1
    y[:, :, 1] = sc[:, :, is_ + 2] * q2
    y[:, :, 2] = sc[:, :, is_ + 4] * q3
    y[:, :, 3] = sc[:, :, is_ + 6] * q4
    return (d * y).reshape(-1)


def dequant_iq4_nl(raw, n):
    b = _blocks(raw, n, GGML_IQ4_NL)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)          # (nb,1)
    qs = b[:, 2:18]
    lo = IQ4NL_LUT[qs & 0xF]                                          # (nb,16)
    hi = IQ4NL_LUT[qs >> 4]
    return (d * np.concatenate([lo, hi], axis=1)).reshape(-1)


def dequant_iq3_s(raw, n):
    b = _blocks(raw, n, GGML_IQ3_S)
    nb = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    qs = b[:, 2:66].astype(np.uint16)            # 64: 8 grid-groups per 32-val subblock
    qh = b[:, 66:74]                             # 8: one byte per subblock
    signs = b[:, 74:106]                         # 32: one byte per 8 values
    scales = b[:, 106:110]                       # 4: nibble per subblock
    bit = np.arange(8)
    hi = ((qh[:, :, None] >> bit) & 1).astype(np.uint16).reshape(nb, 64)
    idx = qs | (hi << 8)                         # (nb,64)
    mags = IQ3S_GRID[idx].reshape(nb, 256)       # group order == value order
    sbits = ((signs[:, :, None] >> bit) & 1).reshape(nb, 256)
    sign = 1.0 - 2.0 * sbits.astype(np.float32)
    nib = np.empty((nb, 8), dtype=np.float32)
    nib[:, 0::2] = (scales & 0xF)
    nib[:, 1::2] = (scales >> 4)
    db = d[:, None] * (1.0 + 2.0 * nib)          # (nb,8) per-subblock scale
    return (np.repeat(db, 32, axis=1) * mags * sign).reshape(-1)


def dequant_iq4_xs(raw, n):
    b = _blocks(raw, n, GGML_IQ4_XS)             # 136 B: d(2) scales_h(2) scales_l(4) qs(128)
    nb = b.shape[0]
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    scales_h = b[:, 2:4].copy().view(np.uint16).astype(np.uint32).reshape(nb, 1)
    scales_l = b[:, 4:8]                         # nibble per sub-block pair
    qs = b[:, 8:136].reshape(nb, 8, 16)          # 8 sub-blocks x 16 bytes
    ib = np.arange(8)
    ls_lo = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF
    ls_hi = (scales_h >> (2 * ib)) & 3
    ls = (ls_lo | (ls_hi << 4)).astype(np.float32) - 32.0    # (nb,8)
    lo = IQ4NL_LUT[qs & 0xF]                     # (nb,8,16)
    hi = IQ4NL_LUT[qs >> 4]
    vals = np.concatenate([lo, hi], axis=2)      # (nb,8,32)
    return (d[:, :, None] * ls[:, :, None] * vals).reshape(-1)


def dequant_f32(raw, n):
    return np.frombuffer(raw, dtype=np.float32, count=n).astype(np.float32)


def dequant_f16(raw, n):
    return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)


def dequant_bf16(raw, n):
    return bf16_decode(np.frombuffer(raw, dtype=np.uint16, count=n))


DEQUANT = {
    GGML_F32: dequant_f32, GGML_F16: dequant_f16, GGML_BF16: dequant_bf16,
    GGML_Q8_0: dequant_q8_0, GGML_Q6_K: dequant_q6_k,
    GGML_IQ4_NL: dequant_iq4_nl, GGML_IQ3_S: dequant_iq3_s,
    GGML_IQ4_XS: dequant_iq4_xs,
}


def dequant(qtype: int, raw, n_elems: int) -> np.ndarray:
    fn = DEQUANT.get(qtype)
    if fn is None:
        raise ValueError(f"unsupported GGML qtype {qtype} ({QTYPE_NAMES.get(qtype, '?')})")
    raw = np.asarray(raw).reshape(-1).view(np.uint8)
    return fn(raw, n_elems)


def rel_rmse(approx: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.linalg.norm(ref.astype(np.float64)))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm((approx - ref).astype(np.float64))) / denom


# ---------------------------------------------------------------------------
# minimal safetensors writer (streaming) + reader
# ---------------------------------------------------------------------------

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1}


def encode_tensor(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "F32":
        return np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    if dtype == "BF16":
        return bf16_encode(np.ascontiguousarray(arr, dtype=np.float32)).tobytes()
    if dtype == "F16":
        return np.ascontiguousarray(arr, dtype=np.float16).tobytes()
    if dtype in ("U8", "I8"):
        return np.ascontiguousarray(arr).tobytes()
    raise ValueError(dtype)


def write_safetensors(path, entries):
    """entries: list of (name, dtype_str, shape_tuple, provider).
    provider is an ndarray (encoded per dtype) or a zero-arg callable -> bytes.
    Header offsets are computed up-front from shape/dtype, so tensors are
    streamed one at a time (peak memory = one tensor)."""
    header = {}
    off = 0
    for name, dt, shape, _prov in entries:
        n = 1
        for s in shape:
            n *= s
        nb = n * DTYPE_BYTES[dt]
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hj)) % 8
    hj += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for name, dt, shape, prov in entries:
            data = prov() if callable(prov) else encode_tensor(prov, dt)
            want = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            if len(data) != want:
                raise RuntimeError(f"{name}: produced {len(data)} bytes, header says {want}")
            f.write(data)


def read_safetensors(path):
    """-> dict name -> (dtype_str, shape, raw_bytes)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for name, m in header.items():
            if name == "__metadata__":
                continue
            a, b = m["data_offsets"]
            f.seek(base + a)
            out[name] = (m["dtype"], tuple(m["shape"]), f.read(b - a))
    return out


def st_to_f32(dtype, shape, raw):
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "BF16":
        return bf16_decode(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# GGUF input (all shards merged)
# ---------------------------------------------------------------------------

class GGUFSource:
    def __init__(self, first_shard: str):
        import gguf  # noqa: deferred so --self-test runs without the package
        paths = self._shard_paths(first_shard)
        print(f"[gguf] opening {len(paths)} shard(s)")
        self.readers = [gguf.GGUFReader(p) for p in paths]
        self.tensors = {}          # name -> ReaderTensor
        self.kv = {}               # merged KVs, first shard wins
        for r in self.readers:
            for t in r.tensors:
                self.tensors[t.name] = t
            for k, fld in r.fields.items():
                if k not in self.kv:
                    try:
                        self.kv[k] = fld.contents()
                    except Exception:
                        pass
        print(f"[gguf] {len(self.tensors)} tensors, {len(self.kv)} KV keys")

    @staticmethod
    def _shard_paths(p):
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p)
        if not m:
            return [p]
        stem, _, total = m.group(1), m.group(2), m.group(3)
        paths = [f"{stem}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]
        missing = [q for q in paths if not os.path.isfile(q)]
        if missing:
            sys.exit(f"missing GGUF shard(s): {missing}")
        return paths

    def find(self, name):
        return self.tensors.get(name)

    def ne(self, t):
        """GGUF dims (ne[0] = innermost / input dim)."""
        return [int(x) for x in t.shape]

    def raw_bytes(self, t) -> np.ndarray:
        """The tensor's raw on-disk bytes (GGML blocks or typed data), 1-D u8 view."""
        return np.asarray(t.data).reshape(-1).view(np.uint8)

    def dequant_full(self, t) -> np.ndarray:
        """Full tensor -> f32 array in container orientation (reversed dims)."""
        ne = self.ne(t)
        n = 1
        for x in ne:
            n *= x
        flat = dequant(int(t.tensor_type), np.asarray(t.data), n)
        return flat.reshape(tuple(reversed(ne)))

    def expert_raw(self, t, e):
        """Raw block bytes of expert e of a [ne0, ne1, n_expert] tensor + elem count.
        Expert-major layout: ne[0] is fastest, the expert index is the last axis,
        so expert e is one contiguous byte slab of ne0*ne1 elements."""
        ne = self.ne(t)
        n_e = ne[0] * ne[1]
        be, bb = BLOCK_SIZES[int(t.tensor_type)]
        ebytes = n_e // be * bb
        raw = self.raw_bytes(t)
        return raw[e * ebytes:(e + 1) * ebytes], n_e

    def dequant_expert(self, t, e) -> np.ndarray:
        """Expert e -> f32 [ne1, ne0] (= [O, I], no transpose needed)."""
        ne = self.ne(t)
        raw, n_e = self.expert_raw(t, e)
        return dequant(int(t.tensor_type), raw, n_e).reshape(ne[1], ne[0])


# ---------------------------------------------------------------------------
# name mapping (design §3).  Storage dtype is decided by the SOURCE ggml type
# (raw U8 for quantized, typed passthrough for F32/F16/BF16), not by the map.
# ---------------------------------------------------------------------------

LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_a.weight": "linear_attn.in_proj_a.weight",
    "ssm_a": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_beta": "linear_attn.in_proj_b.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_alpha.weight",
    "ssm_alpha": "linear_attn.in_proj_alpha.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}
EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")

GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}


def map_layer_suffix(suffix: str):
    """-> container suffix or None."""
    if suffix in LAYER_MAP:
        return LAYER_MAP[suffix]
    m = re.search(r"indexer[._]?(q_proj|k_proj|q_norm|k_norm)", suffix)
    if m:
        return f"self_attn.indexer.{m.group(1)}.weight"
    m = re.match(r"hc_(attn|ffn)_(up|down|inject|norm)(?:\.weight)?$", suffix)
    if m:
        return f"hc_{m.group(1)}.{m.group(2)}.weight"
    return None


def map_ple_name(name: str):
    """Global or per-layer ple_* projection tensor -> model.ple.* name, or None."""
    m = re.search(r"(?:^|\.)ple_([a-z0-9_]+?)(?:\.weight)?$", name)
    if not m:
        return None
    return f"model.ple.{m.group(1)}.weight"


def map_global(name: str):
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    m = re.match(r"output_hc_(up|down|inject|norm)(?:\.weight)?$", name)
    if m:
        return f"model.output_hc.{m.group(1)}.weight"
    return None


def container_name_for(src, name):
    """GGUF tensor name -> container name (experts and the PLE table excluded)."""
    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if m:
        if m.group(2) in EXPERT_SUFFIXES:
            return None
        p = map_ple_name(name)
        if p:
            return p
        suf = map_layer_suffix(m.group(2))
        return f"model.layers.{m.group(1)}.{suf}" if suf else None
    if name == "per_layer_token_embd.weight":
        return None
    return map_ple_name(name) or map_global(name)


# ---------------------------------------------------------------------------
# conversion driver
# ---------------------------------------------------------------------------

def kv_get(kv, arch, key, default=None):
    for k in (f"{arch}.{key}", key):
        if k in kv:
            return kv[k]
    return default


def passthrough_entry(src, t, out_name):
    """Byte-identical storage entry for a non-expert tensor.
    F32/F16/BF16 -> typed safetensors (same bytes); quantized -> U8 raw blocks."""
    qt = int(t.tensor_type)
    if qt not in BLOCK_SIZES:
        sys.exit(f"{t.name}: unsupported ggml type {qt}")
    raw = src.raw_bytes(t)

    def prov(raw=raw):
        return raw.tobytes()
    if qt in TYPED_ST_DTYPE:
        shape = tuple(reversed(src.ne(t)))
        return (out_name, TYPED_ST_DTYPE[qt], shape, prov)
    return (out_name, "U8", (int(raw.size),), prov)


def convert_layer(src, layer, out_path):
    """One layer -> model-{layer:05d}.safetensors (dense + experts, all raw)."""
    prefix = f"blk.{layer}."
    entries = []
    names = sorted(n for n in src.tensors if n.startswith(prefix))
    if not names:
        sys.exit(f"layer {layer}: no tensors found in GGUF")
    expert_ts = {}
    for name in names:
        suffix = name[len(prefix):]
        for es in EXPERT_SUFFIXES:
            if suffix == es:
                expert_ts[es.split("_")[1]] = src.find(name)   # gate/up/down
                break
        else:
            if map_ple_name(name):
                continue                                        # ple projections: own shard
            cname = container_name_for(src, name)
            if cname is None:
                sys.exit(f"layer {layer}: unmapped tensor '{name}' — refusing "
                         f"(qtype {QTYPE_NAMES.get(int(src.find(name).tensor_type), '?')})")
            entries.append(passthrough_entry(src, src.find(name), cname))
    if set(expert_ts) != {"gate", "up", "down"}:
        sys.exit(f"layer {layer}: expert tensors incomplete: {sorted(expert_ts)}")
    t0 = time.time()
    n_experts = src.ne(expert_ts["gate"])[2]
    for mat in ("gate", "up", "down"):
        t = expert_ts[mat]
        qt = int(t.tensor_type)
        if qt not in DEQUANT:
            sys.exit(f"layer {layer} ffn_{mat}_exps: unexpected qtype {qt}")
        ne = src.ne(t)
        if ne[2] != n_experts:
            sys.exit(f"layer {layer}: expert count mismatch across gate/up/down")
        for e in range(n_experts):
            raw, _n_e = src.expert_raw(t, e)
            entries.append((f"model.layers.{layer}.mlp.experts.{e}.{mat}_raw",
                            "U8", (int(raw.size),),
                            (lambda raw=raw: raw.tobytes())))
    # order experts after dense but grouped per expert index for readability of
    # the header only; data offsets are what the loader uses, order is free.
    write_safetensors(out_path, entries)
    types = ", ".join(f"{m}={QTYPE_NAMES[int(expert_ts[m].tensor_type)]}"
                      for m in ("gate", "up", "down"))
    print(f"[layer {layer}] {Path(out_path).name}: {len(entries)} tensors, "
          f"{n_experts} experts raw ({types}), {time.time() - t0:.1f}s")


def write_globals(src, out):
    entries = []
    for name, t in src.tensors.items():
        if name.startswith("blk."):
            continue
        if name == "per_layer_token_embd.weight" or map_ple_name(name):
            continue
        cname = map_global(name)
        if cname is None:
            sys.exit(f"unmapped global tensor '{name}' — refusing")
        entries.append(passthrough_entry(src, t, cname))
    path = out / "model-globals.safetensors"
    write_safetensors(path, entries)
    print(f"[globals] {path.name}: {len(entries)} tensors")


def write_ple_projections(src, out):
    entries = []
    for name, t in src.tensors.items():
        if name == "per_layer_token_embd.weight":
            continue
        cname = map_ple_name(name)
        if cname:
            entries.append(passthrough_entry(src, t, cname))
    if not entries:
        print("[ple] WARNING: no ple projection tensors found")
        return
    path = out / "model-ple.safetensors"
    write_safetensors(path, entries)
    print(f"[ple] {path.name}: {len(entries)} tensors")


PLE_HDR_FMT = "<8sIIQIIII24x"          # magic, version, qtype, n_rows, row_dim, row_bytes, ngram, heads
assert struct.calcsize(PLE_HDR_FMT) == 64


def copy_ple_table(src, out, ngram, heads):
    t = src.find("per_layer_token_embd.weight")
    if t is None:
        print("[ple_table] WARNING: per_layer_token_embd.weight not in GGUF, skipping")
        return
    qt = int(t.tensor_type)
    if qt != GGML_IQ4_NL:
        sys.exit(f"per_layer_token_embd is {QTYPE_NAMES.get(qt, qt)}, expected IQ4_NL "
                 f"(design keeps raw IQ4_NL only)")
    ne = src.ne(t)
    row_dim = ne[0]
    n_rows = ne[1]
    be, bb = BLOCK_SIZES[GGML_IQ4_NL]
    row_bytes = row_dim // be * bb
    raw = src.raw_bytes(t)
    total = n_rows * row_bytes
    if raw.size != total:
        sys.exit(f"ple table byte size {raw.size} != rows*row_bytes {total}")
    path = out / "ple_table.bin"
    hdr = struct.pack(PLE_HDR_FMT, b"COLIPLE1", 1, GGML_IQ4_NL,
                      n_rows, row_dim, row_bytes, ngram, heads)
    chunk = 256 << 20
    t0 = time.time()
    with open(path, "wb") as f:
        f.write(hdr)
        for a in range(0, total, chunk):
            f.write(raw[a:a + chunk].tobytes())
    print(f"[ple_table] {path.name}: {n_rows} rows x {row_bytes} B "
          f"({total / 2**30:.2f} GiB) in {time.time() - t0:.1f}s")


def is_json_safe(v):
    return isinstance(v, (int, float, str, bool)) or (
        isinstance(v, list) and len(v) <= 64 and all(is_json_safe(x) for x in v))


def collect_raw_maps(src, n_layers):
    """Walk ALL tensors (independent of --limit-layers) and record ggml types.
    -> (expert_ggml_types {layer: {gate_type,up_type,down_type}},
        raw_tensors {container_name: {"ggml_type", "shape"}})   (quantized only;
    typed F32/F16/BF16 tensors are self-describing in safetensors)."""
    expert_types = {}
    raw_tensors = {}
    for name, t in src.tensors.items():
        qt = int(t.tensor_type)
        m = re.match(r"blk\.(\d+)\.(ffn_(gate|up|down)_exps)\.weight$", name)
        if m:
            expert_types.setdefault(m.group(1), {})[f"{m.group(3)}_type"] = qt
            continue
        if name == "per_layer_token_embd.weight":
            continue
        cname = container_name_for(src, name)
        if cname is None:
            continue                     # convert_layer will refuse loudly
        if qt not in TYPED_ST_DTYPE:
            raw_tensors[cname] = {"ggml_type": qt,
                                  "shape": list(reversed(src.ne(t)))}
    return expert_types, raw_tensors


def write_meta_config(src, out, layer_types, n_layers):
    kv, arch = src.kv, src.kv.get("general.architecture", "qwen4exp")

    def g(key, default=None):
        v = kv_get(kv, arch, key, default)
        if isinstance(v, np.generic):
            v = v.item()
        return v

    hidden = int(g("embedding_length", 2560))
    inter = int(g("expert_feed_forward_length", 640))
    cfg = {
        "model_type": "qwen38",
        "source_arch": arch,
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_experts": int(g("expert_count", 512)),
        "num_experts_per_tok": int(g("expert_used_count", 10)),
        "moe_intermediate_size": inter,
        "shared_expert_intermediate_size": int(g("expert_shared_feed_forward_length", inter)),
        "vocab_size": int(g("vocab_size", 248320)),
        "num_attention_heads": int(g("attention.head_count", 24)),
        "num_key_value_heads": int(g("attention.head_count_kv", 2)),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_theta": float(g("rope.freq_base", 10000000.0)),
        "max_position_embeddings": int(g("context_length", 262144)),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[config] config.json")

    # derived / shape-authoritative dims
    def shape_of(name):
        t = src.find(name)
        return src.ne(t) if t is not None else None

    qsa = next((i for i, lt in enumerate(layer_types) if lt == "sparse_attention"), None)
    gdn = next((i for i, lt in enumerate(layer_types) if lt == "linear_attention"), None)
    meta = {
        "model_type": "qwen38",
        "hidden": hidden,
        "n_layers": n_layers,
        "n_active": n_layers,
        "layer_types": layer_types,
        "num_experts": cfg["num_experts"],
        "topk": cfg["num_experts_per_tok"],
        "moe_inter": inter,
        "shared_inter": cfg["shared_expert_intermediate_size"],
        "rms_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "expert_format": "ggml_raw",     # revision 2: raw GGML blocks, no requant
        "q_heads": cfg["num_attention_heads"],
        "kv_heads": cfg["num_key_value_heads"],
    }
    if qsa is not None:
        kshape = shape_of(f"blk.{qsa}.attn_k.weight")           # [H, kv*head_dim]
        qshape = shape_of(f"blk.{qsa}.attn_q.weight")
        oshape = shape_of(f"blk.{qsa}.attn_output.weight")
        qn = shape_of(f"blk.{qsa}.attn_q_norm.weight")
        if kshape:
            meta["head_dim"] = kshape[1] // meta["kv_heads"]
        if qshape:
            meta["q_head_dim"] = qshape[1] // meta["q_heads"]
        if oshape:
            meta["o_in"] = oshape[0]
        if qn:
            meta["qk_norm_dim"] = qn[0]
    if gdn is not None:
        dt = shape_of(f"blk.{gdn}.ssm_dt.bias")
        nrm = shape_of(f"blk.{gdn}.ssm_norm.weight")
        conv = shape_of(f"blk.{gdn}.ssm_conv1d.weight")
        meta["dn_vheads"] = dt[0] if dt else 48
        meta["dn_vdim"] = nrm[0] if nrm else 128
        meta["dn_convk"] = conv[0] if conv else 4
        meta["dn_conv_dim"] = conv[1] if conv else 10240
        kh = g("ssm.key_head_count") or g("linear_num_key_heads")
        kd = g("ssm.key_head_dim") or g("linear_key_head_dim")
        rem = meta["dn_conv_dim"] - meta["dn_vheads"] * meta["dn_vdim"]
        if kh and kd:
            meta["dn_kheads"], meta["dn_kdim"] = int(kh), int(kd)
        else:
            # derive: 2*kh*kd == conv_dim - vheads*vdim; assume kdim == vdim
            meta["dn_kdim"] = meta["dn_vdim"]
            meta["dn_kheads"] = rem // 2 // meta["dn_kdim"]
            print(f"[meta] WARNING: dn_kheads/kdim not in KVs; derived "
                  f"{meta['dn_kheads']}x{meta['dn_kdim']} assuming kdim==vdim")
        if 2 * meta["dn_kheads"] * meta["dn_kdim"] + meta["dn_vheads"] * meta["dn_vdim"] \
                != meta["dn_conv_dim"]:
            print("[meta] WARNING: 2*kh*kd + vh*vd != conv_dim — check ssm KVs")
    iq = shape_of(f"blk.{qsa}.attn_indexer.q_proj.weight") if qsa is not None else None
    ik = shape_of(f"blk.{qsa}.attn_indexer.k_proj.weight") if qsa is not None else None
    meta["indexer"] = {
        "heads": int(g("attention.indexer.head_count", 4) or 4),
        "q_out": iq[1] if iq else 512,
        "k_out": ik[1] if ik else 128,
    }
    hcup = next((src.ne(t) for n, t in src.tensors.items() if "hc_attn_up" in n), None)
    meta["hc"] = {"streams": 4, "rank": hcup[0] if hcup else 320}
    ple_t = src.find("per_layer_token_embd.weight")
    ple_dim = int(g("embedding_length_per_layer_input", 160))
    meta["ple"] = {
        "inject_layers": [int(x) for x in (g("ple.layers", None) or [1])],
        "ngram_size": int(g("ple.ngram_size", 3) or 3),
        "heads_per_ngram": int(g("ple.heads_per_ngram", 8) or 8),
        "dim": ple_dim,
        "rows": src.ne(ple_t)[1] if ple_t is not None else 320001536,
        "table_file": "ple_table.bin",
        "qtype": "iq4_nl",
    }
    expert_types, raw_tensors = collect_raw_maps(src, n_layers)
    meta["ggml_type_names"] = {str(i): n for i, n in QTYPE_NAMES.items()}
    meta["expert_ggml_types"] = expert_types     # per layer: gate_type/up_type/down_type
    meta["raw_tensors"] = raw_tensors            # container name -> {ggml_type, shape [O,I]}
    meta["gguf_kv"] = {k: (v.item() if isinstance(v, np.generic) else v)
                       for k, v in src.kv.items()
                       if not k.startswith("tokenizer.")
                       and is_json_safe(v.item() if isinstance(v, np.generic) else v)}
    (out / "qwen38_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] qwen38_meta.json ({len(raw_tensors)} raw dense tensors, "
          f"{len(expert_types)} layers of expert types)")
    return meta


def write_tokenizer(src, out, tokenizer_path):
    if tokenizer_path:
        import shutil
        shutil.copy2(tokenizer_path, out / "tokenizer.json")
        print(f"[tokenizer] copied {tokenizer_path} -> tokenizer.json")
        return
    kv = src.kv
    tok = {}
    for k, v in kv.items():
        if not k.startswith("tokenizer."):
            continue
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        tok[k] = v
    if not tok:
        print("[tokenizer] WARNING: no tokenizer KVs in GGUF")
        return
    (out / "tokenizer.gguf.json").write_text(json.dumps(tok))
    n_tok = len(tok.get("tokenizer.ggml.tokens", []))
    n_merges = len(tok.get("tokenizer.ggml.merges", []))
    print(f"[tokenizer] tokenizer.gguf.json ({n_tok} tokens, {n_merges} merges)")
    print("[tokenizer] NOTE: HF-style tokenizer.json NOT reconstructed (byte-level "
          "merge round-trip is not trivially safe); pass --tokenizer <hf tokenizer.json> "
          "to ship one, or set TOK= at engine launch.")
    ct = tok.get("tokenizer.chat_template")
    if ct:
        (out / "chat_template.jinja").write_text(ct)
        print("[tokenizer] chat_template.jinja written")


def parse_layer_range(spec, n_layers):
    if spec is None:
        return list(range(n_layers))
    m = re.match(r"^(\d+)-(\d+)$", spec)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in spec.split(",")]


def run_convert(args):
    src = GGUFSource(args.gguf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layer_ids = sorted({int(m.group(1)) for n in src.tensors
                        for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    layer_types = ["sparse_attention" if src.find(f"blk.{i}.attn_q.weight") is not None
                   else "linear_attention" for i in range(n_layers)]
    n_qsa = layer_types.count("sparse_attention")
    print(f"[layers] {n_layers} total: {n_qsa} sparse_attention, {n_layers - n_qsa} linear_attention")

    todo = parse_layer_range(args.layers, n_layers)
    if args.limit_layers is not None:
        todo = todo[:args.limit_layers]

    write_globals(src, out)
    write_ple_projections(src, out)
    write_meta_config(src, out, layer_types, n_layers)
    write_tokenizer(src, out, args.tokenizer)

    for l in todo:
        convert_layer(src, l, out / f"model-{l:05d}.safetensors")

    if not args.skip_ple_table:
        meta = json.loads((out / "qwen38_meta.json").read_text())
        copy_ple_table(src, out, meta["ple"]["ngram_size"], meta["ple"]["heads_per_ngram"])
    else:
        print("[ple_table] skipped (--skip-ple-table)")

    print(f"\nDone. Container at: {out} ({len(todo)}/{n_layers} layers, raw GGML blocks)")


# ---------------------------------------------------------------------------
# self-test: dequantizer math against hand-built reference vectors
# (requant tests removed with the int4-gs64 path; the dequantizers stay as
#  reference oracles for validation tooling / the engine's decode kernels)
# ---------------------------------------------------------------------------

def _f16b(x):
    return np.float16(x).tobytes()


def _st_scalar_iq3s_ref(raw, n):
    """Literal scalar port of dequantize_row_iq3_s (independent oracle)."""
    kmask = [1, 2, 4, 8, 16, 32, 64, 128]
    grid = IQ3S_GRID
    y = np.zeros(n, np.float32)
    nb = n // 256
    b = np.frombuffer(raw, np.uint8).reshape(nb, 110)
    for i in range(nb):
        d = float(np.frombuffer(b[i, 0:2].tobytes(), np.float16)[0])
        qs, qh = b[i, 2:66], b[i, 66:74]
        signs, scales = b[i, 74:106], b[i, 106:110]
        yo = i * 256
        qso = qho = so = 0
        for ib32 in range(0, 8, 2):
            db1 = d * (1 + 2 * (scales[ib32 // 2] & 0xF))
            db2 = d * (1 + 2 * (scales[ib32 // 2] >> 4))
            for half, db in ((0, db1), (1, db2)):
                for l in range(4):
                    i1 = int(qs[qso + 2 * l]) | ((int(qh[qho + half]) << (8 - 2 * l)) & 256)
                    i2 = int(qs[qso + 2 * l + 1]) | ((int(qh[qho + half]) << (7 - 2 * l)) & 256)
                    for j in range(4):
                        s1 = -1.0 if signs[so + l] & kmask[j] else 1.0
                        s2 = -1.0 if signs[so + l] & kmask[j + 4] else 1.0
                        y[yo + j] = db * grid[i1][j] * s1
                        y[yo + j + 4] = db * grid[i2][j] * s2
                    yo += 8
                qso += 8
                so += 4
            qho += 2
    return y


def _st_scalar_q6k_ref(raw, n):
    """Literal scalar port of dequantize_row_q6_K."""
    nb = n // 256
    b = np.frombuffer(raw, np.uint8).reshape(nb, 210)
    y = np.zeros(n, np.float32)
    for i in range(nb):
        d = float(np.frombuffer(b[i, 208:210].tobytes(), np.float16)[0])
        ql, qh = b[i, 0:128], b[i, 128:192]
        sc = b[i, 192:208].view(np.int8)
        yo = i * 256
        qlo = qho = sco = 0
        for _n in range(0, 256, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((int(ql[qlo + l]) & 0xF) | (((int(qh[qho + l]) >> 0) & 3) << 4)) - 32
                q2 = ((int(ql[qlo + l + 32]) & 0xF) | (((int(qh[qho + l]) >> 2) & 3) << 4)) - 32
                q3 = ((int(ql[qlo + l]) >> 4) | (((int(qh[qho + l]) >> 4) & 3) << 4)) - 32
                q4 = ((int(ql[qlo + l + 32]) >> 4) | (((int(qh[qho + l]) >> 6) & 3) << 4)) - 32
                y[yo + l] = d * sc[sco + is_] * q1
                y[yo + l + 32] = d * sc[sco + is_ + 2] * q2
                y[yo + l + 64] = d * sc[sco + is_ + 4] * q3
                y[yo + l + 96] = d * sc[sco + is_ + 6] * q4
            yo += 128
            qlo += 64
            qho += 32
            sco += 8
    return y


def self_test():
    rng = np.random.default_rng(38)
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            fails.append(name)

    print("=== self-test: dequantizers ===")
    # Q8_0: hand-built block
    qs = np.arange(-16, 16, dtype=np.int8)
    raw = _f16b(0.5) + qs.tobytes()
    got = dequant_q8_0(np.frombuffer(raw, np.uint8), 32)
    check("Q8_0 manual", np.allclose(got, 0.5 * qs.astype(np.float32)))

    # IQ4_NL: encode known values via codebook: pick target = d * LUT[idx]
    d = 0.25
    idx_lo = np.arange(16, dtype=np.uint8)
    idx_hi = (15 - np.arange(16)).astype(np.uint8)
    raw = _f16b(d) + (idx_lo | (idx_hi << 4)).tobytes()
    got = dequant_iq4_nl(np.frombuffer(raw, np.uint8), 32)
    want = np.concatenate([d * IQ4NL_LUT[idx_lo], d * IQ4NL_LUT[idx_hi]])
    check("IQ4_NL manual (codebook)", np.allclose(got, want))

    # IQ3_S: hand-computed spot values + scalar-port oracle on random blocks
    blk = bytearray(110)
    blk[0:2] = _f16b(1.0)
    blk[2:66] = bytes(range(64))              # qs[j] = j
    blk[66:74] = bytes([0x01] + [0] * 7)      # qh: subblock 0 bit0 set -> group 0 idx += 256
    blk[74:106] = bytes([0x81] + [0] * 31)    # signs: values 0 and 7 of subblock 0 negative
    blk[106:110] = bytes([0x21, 0x43, 0x65, 0x87])
    got = dequant_iq3_s(np.frombuffer(bytes(blk), np.uint8), 256)
    db0 = 1.0 * (1 + 2 * 1)                   # scales[0] low nibble = 1 -> db = 3
    # value 0: group 0, idx = 0|256 = 256 -> grid[256] byte0; sign bit0 set -> negative
    want0 = -db0 * IQ3S_GRID[256][0]
    # value 5: group 1 (qs[1]=1, qh bit1=0 -> idx 1), byte 5-4=1, sign bit5=0
    want5 = db0 * IQ3S_GRID[1][1]
    # value 7: sign bit7 of signs[0]=0x81 -> negative; group 1 byte 3
    want7 = -db0 * IQ3S_GRID[1][3]
    check("IQ3_S manual spots",
          np.isclose(got[0], want0) and np.isclose(got[5], want5) and np.isclose(got[7], want7),
          f"y0={got[0]} want {want0}")
    nblk = 8
    raw = rng.integers(0, 256, nblk * 110, dtype=np.uint8)
    raw = raw.reshape(nblk, 110).copy()
    raw[:, 0:2] = np.frombuffer(np.float16(rng.uniform(0.01, 2, nblk)).tobytes(),
                                np.uint8).reshape(nblk, 2)
    got = dequant_iq3_s(raw.reshape(-1), nblk * 256)
    ref = _st_scalar_iq3s_ref(raw.tobytes(), nblk * 256)
    check("IQ3_S vectorized == scalar C-port", np.allclose(got, ref))

    # Q6_K: all-zero block -> every value = d*sc*(-32); + scalar-port oracle
    blk = bytearray(210)
    blk[192:208] = bytes([1] * 16)
    blk[208:210] = _f16b(2.0)
    got = dequant_q6_k(np.frombuffer(bytes(blk), np.uint8), 256)
    check("Q6_K manual (zeros)", np.allclose(got, -64.0))
    raw = rng.integers(0, 256, 4 * 210, dtype=np.uint8).reshape(4, 210).copy()
    raw[:, 208:210] = np.frombuffer(np.float16(rng.uniform(0.01, 1, 4)).tobytes(),
                                    np.uint8).reshape(4, 2)
    got = dequant_q6_k(raw.reshape(-1), 4 * 256)
    ref = _st_scalar_q6k_ref(raw.tobytes(), 4 * 256)
    check("Q6_K vectorized == scalar C-port", np.allclose(got, ref))

    # F16/BF16/F32 passthrough
    v = rng.standard_normal(64).astype(np.float32)
    check("F32", np.array_equal(dequant_f32(v.tobytes(), 64), v))
    check("F16", np.allclose(dequant_f16(v.astype(np.float16).tobytes(), 64), v, atol=1e-2))
    check("BF16 roundtrip", np.allclose(bf16_decode(bf16_encode(v)), v, rtol=1 / 128))

    # cross-check all four block dequantizers against the gguf pip package
    try:
        import gguf.quants as gq
        from gguf.constants import GGMLQuantizationType as QT
        for qt in (QT.Q8_0, QT.Q6_K, QT.IQ4_NL, QT.IQ3_S):
            be, bb = BLOCK_SIZES[int(qt)]
            nb = 16 if bb < 100 else 4
            raw = rng.integers(0, 256, nb * bb, dtype=np.uint8).reshape(nb, bb).copy()
            # keep the f16 scale sane (position differs per format)
            dpos = 208 if int(qt) == GGML_Q6_K else 0
            raw[:, dpos:dpos + 2] = np.frombuffer(
                np.float16(rng.uniform(0.01, 1, nb)).tobytes(), np.uint8).reshape(nb, 2)
            mine = dequant(int(qt), raw.reshape(-1), nb * be)
            theirs = gq.dequantize(raw.reshape(-1), qt).reshape(-1).astype(np.float32)
            check(f"{qt.name} == gguf pkg", np.allclose(mine, theirs, rtol=1e-5, atol=1e-6))
    except ImportError:
        print("  [skip] gguf package not installed; cross-check skipped")

    print("=== self-test: safetensors writer ===")
    a = rng.standard_normal((4, 6)).astype(np.float32)
    u8 = rng.integers(0, 256, 32, dtype=np.uint8)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"convert_qwen38_selftest_{os.getpid()}.safetensors"
    write_safetensors(tmp, [
        ("a.f32", "F32", a.shape, a),
        ("b.bf16", "BF16", a.shape, a),
        ("c.u8", "U8", (u8.size,), u8),
        ("d.streamed", "F32", a.shape, lambda: encode_tensor(a, "F32")),
    ])
    back = read_safetensors(tmp)
    tmp.unlink()
    check("st F32 exact", np.array_equal(st_to_f32(*back["a.f32"]), a))
    check("st BF16", np.allclose(st_to_f32(*back["b.bf16"]), a, rtol=1 / 128))
    check("st U8 raw", back["c.u8"][2] == u8.tobytes() and back["c.u8"][0] == "U8")
    check("st streamed provider", np.array_equal(st_to_f32(*back["d.streamed"]), a))

    # BF16 typed passthrough is byte-identical (raw_bytes -> BF16 entry)
    bf_raw = bf16_encode(a)
    check("BF16 passthrough bytes", bf_raw.tobytes() ==
          encode_tensor(bf16_decode(bf_raw), "BF16"))

    # PLE header sanity
    hdr = struct.pack(PLE_HDR_FMT, b"COLIPLE1", 1, GGML_IQ4_NL, 320001536, 160, 90, 3, 8)
    magic, ver, qt, rows, rd, rb, ngram, heads = struct.unpack(PLE_HDR_FMT, hdr)
    check("PLE header pack/unpack", len(hdr) == 64 and magic == b"COLIPLE1" and rows == 320001536
          and rd == 160 and rb == 90 and qt == 20)

    print(f"\nSELF-TEST {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Qwen3.8 GGUF -> Colibri container converter "
                                             "(raw GGML block storage)")
    ap.add_argument("--gguf", help="path to first GGUF shard (…-00001-of-0000N.gguf) or single file")
    ap.add_argument("--out", help="output container directory")
    ap.add_argument("--layers", help="layer range 'A-B' or comma list (default: all)")
    ap.add_argument("--limit-layers", type=int, help="convert only the first N selected layers")
    ap.add_argument("--skip-ple-table", action="store_true", help="skip the 27 GB ple_table.bin copy")
    ap.add_argument("--tokenizer", help="HF tokenizer.json to copy into the container")
    ap.add_argument("--self-test", "--selftest", action="store_true", dest="self_test",
                    help="run dequantizer/container math self-test on synthetic data and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if not args.gguf or not args.out:
        ap.error("--gguf and --out are required (or --self-test)")
    run_convert(args)


if __name__ == "__main__":
    main()
