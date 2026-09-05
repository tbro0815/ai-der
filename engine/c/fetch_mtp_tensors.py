#!/usr/bin/env python3
"""Extract MTP tensors from a HuggingFace safetensors-sharded repo using HTTP range
requests only (no full shard downloads).

Used by the installer (install_models.sh) and runnable by hand. Reads the local shard index, fetches each relevant shard's
safetensors header via a couple of small Range requests, then pulls exactly the
byte ranges of the MTP tensors and assembles them into one local safetensors file.
"""

import argparse
import json
import os
import struct
import sys
import time

import requests
from huggingface_hub import hf_hub_url

REPO_ID = "Qwen/Qwen3.8-Flash-Next"
CHUNK = 64 * 1024 * 1024  # max bytes per range request
RETRIES = 3

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def session():
    s = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    s.headers["User-Agent"] = "fetch-mtp-tensors/1.0"
    return s


def range_get(sess, url, start, end_inclusive, attempts=RETRIES):
    """GET [start, end_inclusive] bytes. Returns bytes."""
    last = None
    for i in range(attempts):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end_inclusive}"},
                         allow_redirects=True, timeout=120, stream=True)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code} for range {start}-{end_inclusive}")
            if r.status_code == 200:
                raise RuntimeError("server ignored Range header (200 full body)")
            data = r.content
            want = end_inclusive - start + 1
            if len(data) != want:
                raise RuntimeError(f"short read: got {len(data)} want {want}")
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"range_get failed after {attempts} attempts: {last}")


def range_stream_to(sess, url, start, end_inclusive, fh, attempts=RETRIES):
    """Stream [start, end_inclusive] into open file handle fh at its current pos,
    chunked into <= CHUNK sized range requests. Returns bytes written."""
    total = 0
    pos = start
    while pos <= end_inclusive:
        stop = min(pos + CHUNK - 1, end_inclusive)
        buf = range_get(sess, url, pos, stop, attempts)
        fh.write(buf)
        total += len(buf)
        pos = stop + 1
    return total


def read_shard_header(sess, url):
    """Return (header_dict, data_start_offset)."""
    n = struct.unpack("<Q", range_get(sess, url, 0, 7))[0]
    raw = range_get(sess, url, 8, 8 + n - 1)
    return json.loads(raw.decode("utf-8")), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.expanduser(
        "~/models/qwen38-official-tok/model.safetensors.index.json"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/models/qwen38-mtp"))
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--pattern", default="mtp")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mtp.safetensors")
    man_path = os.path.join(args.out_dir, "mtp_manifest.json")

    with open(args.index) as f:
        weight_map = json.load(f)["weight_map"]

    names = sorted(n for n in weight_map if args.pattern in n.lower())
    if not names:
        sys.exit(f"no tensors matching {args.pattern!r} in index")

    by_shard = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)

    print(f"{len(names)} MTP tensors across {len(by_shard)} shards")

    sess = session()

    # Pass 1: shard headers -> metadata for every MTP tensor.
    meta = {}  # name -> dict(dtype, shape, shard, url, abs_start, abs_end, nbytes)
    for shard in sorted(by_shard):
        url = hf_hub_url(args.repo, shard)
        hdr, data_start = read_shard_header(sess, url)
        for n in by_shard[shard]:
            if n not in hdr:
                sys.exit(f"tensor {n} not present in header of {shard}")
            e = hdr[n]
            s, t = e["data_offsets"]
            meta[n] = {
                "dtype": e["dtype"], "shape": e["shape"], "shard": shard,
                "url": url, "abs_start": data_start + s, "abs_end": data_start + t - 1,
                "nbytes": t - s,
            }
        print(f"  header {shard}: {len(by_shard[shard])} tensor(s)")

    # Layout of the output file (safetensors requires header keys we emit sorted).
    out_names = sorted(meta)
    header = {}
    off = 0
    for n in out_names:
        m = meta[n]
        nb = m["nbytes"]
        exp = DTYPE_SIZE[m["dtype"]]
        for d in m["shape"]:
            exp *= d
        if exp != nb:
            sys.exit(f"{n}: byte range {nb} != dtype/shape product {exp}")
        header[n] = {"dtype": m["dtype"], "shape": m["shape"],
                     "data_offsets": [off, off + nb]}
        off += nb
    total = off

    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(len(hb) + 8)) % 8
    hb += b" " * pad

    print(f"payload {total} bytes ({total/2**30:.2f} GiB); downloading...")

    downloaded = 0
    t0 = time.time()
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for n in out_names:
            m = meta[n]
            got = range_stream_to(sess, m["url"], m["abs_start"], m["abs_end"], fh)
            if got != m["nbytes"]:
                sys.exit(f"{n}: wrote {got} expected {m['nbytes']}")
            downloaded += got
            print(f"  {n:55s} {m['dtype']:>5s} {str(m['shape']):>22s} "
                  f"{got/2**20:9.2f} MiB  <- {m['shard']}")

    manifest = {n: {"dtype": meta[n]["dtype"], "shape": meta[n]["shape"],
                    "source_shard": meta[n]["shard"], "nbytes": meta[n]["nbytes"]}
                for n in out_names}
    with open(man_path, "w") as f:
        json.dump({"repo": args.repo, "tensors": manifest,
                   "total_bytes": total}, f, indent=2)

    dt = time.time() - t0
    print(f"\ndownloaded {downloaded} bytes ({downloaded/2**30:.2f} GiB) in {dt:.0f}s "
          f"({downloaded/2**20/max(dt,1):.1f} MiB/s)")
    print(f"wrote {out_path}")
    print(f"wrote {man_path}")

    # Verification.
    from safetensors import safe_open

    print("\nverifying...")
    with safe_open(out_path, framework="np") as f:
        keys = sorted(f.keys())
        if keys != out_names:
            sys.exit("key mismatch between file and manifest")
        for n in keys:
            sl = f.get_slice(n)
            shp = list(sl.get_shape())
            dt_ = sl.get_dtype()
            if shp != list(manifest[n]["shape"]):
                sys.exit(f"{n}: shape {shp} != {manifest[n]['shape']}")
            if dt_ != manifest[n]["dtype"]:
                sys.exit(f"{n}: dtype {dt_} != {manifest[n]['dtype']}")
    print(f"  shapes/dtypes OK for {len(out_names)} tensors")

    # Byte-compare first 1 KB of 3 tensors against a fresh HF fetch.
    sample = [out_names[0], out_names[len(out_names) // 2], out_names[-1]]
    with open(out_path, "rb") as fh:
        base = 8 + len(hb)
        for n in sample:
            k = min(1024, meta[n]["nbytes"])
            fh.seek(base + header[n]["data_offsets"][0])
            local = fh.read(k)
            remote = range_get(sess, meta[n]["url"], meta[n]["abs_start"],
                               meta[n]["abs_start"] + k - 1)
            if local != remote:
                sys.exit(f"BYTE MISMATCH for {n}")
            print(f"  first {k}B match for {n}")
    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
