#!/usr/bin/env bash
# AI-DER: the optional Strata backend (github.com/Niko1221/Strata, MIT), an inference engine for
# Qwen3.8-Flash-Next: built from source at a pinned revision against the private CUDA 13 toolkit
# (engine/c/install_cuda13.sh), plus up to three models, each with its strata-<model>.json:
#   gsq         ISTA-DASLab's GSQ-RCO IQ3_S pack (~84 GB download)
#   uncensored  mradermacher's i1-IQ3_S of OrcaRouter's Qwen3.8-Flash-Next-Uncensored (~89 GB),
#               packed for Strata's kernels with its tools/iq_pack.py --compat-bf16
#   ud          unsloth's UD-IQ4_XS, the GGUF install_models.sh already downloaded (no download)
# Every config computes all routed experts (STRATA_ROUTE_TAIL_SKIP=0); Strata's experimental
# speed projection stays off. AI_DER_STRATA_MODELS picks the models (default: gsq,uncensored,ud).
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_strata.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_STRATA_COMMIT:-011c9e6}       # engine 0.1.47 / app 0.2.2, the reference box (2026-10-22)
want=${AI_DER_STRATA_MODELS:-gsq,uncensored,ud}
dir=$data/strata
sdata=$data/strata-data
models=$data/models/strata
cuda=$data/cuda13
hf=$root/.venv/bin/hf
ud=$data/models/qwen38-flash-next/UD-IQ4_XS
say() { echo "AI-DER Strata: $*"; }
fail() { echo "AI-DER Strata: $*" >&2; exit 1; }
[[ -x $cuda/bin/nvcc ]] || fail "no CUDA 13 toolkit at $cuda; run engine/c/install_cuda13.sh first."
[[ -x $hf ]] || fail "no hf CLI at $hf; run ./install.sh first."
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

if [[ ! -d $dir/.git ]]; then
    say "cloning Niko1221/Strata -> $dir"
    git clone --quiet https://github.com/Niko1221/Strata "$dir"
fi
git -C "$dir" fetch --quiet origin
git -C "$dir" checkout --quiet "$commit"
cd "$dir"
export STRATA_NVCC=$cuda/bin/nvcc CUDA_HOME=$cuda PATH=$cuda/bin:$PATH
# setup.sh: its own .venv, the pinned llama.cpp sources, the engine compiled from this checkout
# (--build), the MTP draft layer and the vision encoder (downloaded once), the start config
common=(--data-dir "$sdata" --context 262144 --kv int8 --vision gpu --mtp on --port 8084
        --build --no-start --no-browser --yes)

if [[ $want == *ud* ]]; then
    if [[ -f strata-unsloth-ud-iq4_xs.json ]]; then say "UD-IQ4_XS configured, skipping"
    elif [[ -f $ud/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf ]]; then
        say "UD-IQ4_XS from $ud"
        ./setup.sh --family unsloth --model UD-IQ4_XS --gguf-dir "$ud" "${common[@]}"
    else say "no UD-IQ4_XS GGUF under $ud (engine/c/install_models.sh), skipping"; fi
fi
if [[ $want == *gsq* ]]; then
    if [[ -f strata-iq3_s.json ]]; then say "GSQ-RCO IQ3_S configured, skipping"
    else
        say "GSQ-RCO IQ3_S (ISTA-DASLab, ~84 GB)"
        "$hf" download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
            IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
            IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --local-dir "$models/gsq-rco"
        ./setup.sh --family qwen --model IQ3_S --gguf-dir "$models/gsq-rco/IQ3_S" "${common[@]}"
    fi
fi
if [[ $want == *uncensored* ]]; then
    uncensored=$models/uncensored-i1/Qwen3.8-Flash-Next-Uncensored.i1-IQ3_S.gguf
    if [[ -f strata-uncensored-i1-iq3_s.json ]]; then say "uncensored i1-IQ3_S configured, skipping"
    else
        say "uncensored i1-IQ3_S (mradermacher, ~89 GB)"
        "$hf" download mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF \
            Qwen3.8-Flash-Next-Uncensored.i1-IQ3_S.gguf --local-dir "$models/uncensored-i1"
        [[ -d third_party/llama.cpp/gguf-py ]] || fail "no gguf-py in $dir/third_party/llama.cpp (a setup.sh run brings it)"
        # the hyper-connection and small attention projections Strata's kernels read as BF16
        STRATA_GGUF_PY=$dir/third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
            --gguf "$uncensored" --out "$sdata/packs/uncensored-i1-iq3_s" --compat-bf16
        # its config: the IQ3_S config (same expert layout) pointed at this pack and file
        .venv/bin/python - "$uncensored" "$sdata" <<'EOF'
import json, os, sys
gguf, sdata = sys.argv[1], sys.argv[2]
template = next(f for f in ("strata-iq3_s.json", "strata-unsloth-ud-iq4_xs.json") if os.path.isfile(f))
cfg = json.load(open(template))
a = cfg["args"]
a[a.index("--pack") + 1] = os.path.join(sdata, "packs", "uncensored-i1-iq3_s")
a[a.index("--native") + 1] = gguf
if "--ple-gguf" in a:
    a[a.index("--ple-gguf") + 1] = gguf
else:
    a += ["--ple-gguf", gguf]
cfg["args"] = cfg["args_written"] = a
env = {k: v for k, v in (cfg.get("env") or {}).items() if k != "STRATA_HC_Q8"}   # unsloth's Q8 hyper-connections only
cfg["env"] = env
cfg["tokenizer"] = os.path.join(sdata, "packs", "uncensored-i1-iq3_s", "tokenizer")
cfg["model_name"] = "qwen3.8-flash-next-uncensored-i1-iq3_s"
cfg["log"] = os.path.join(os.getcwd(), "strata-uncensored-i1-iq3_s.log")
if cfg.get("vision"):
    cfg["vision"]["model"] = gguf
json.dump(cfg, open("strata-uncensored-i1-iq3_s.json", "w"), indent=1)
EOF
    fi
fi
# every model computes all routed experts (Strata's route-tail skip trades accuracy for speed)
for cfg in strata-*.json; do
    [[ -f $cfg ]] || continue
    .venv/bin/python - "$cfg" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg.setdefault("env", {})["STRATA_ROUTE_TAIL_SKIP"] = "0"
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
EOF
done
ls strata-*.json >/dev/null 2>&1 || fail "no model configured (AI_DER_STRATA_MODELS=$want)"
"$dir/engine/strata" --version 2>&1 | head -n 1
say "done: $(ls strata-*.json | tr '\n' ' ')"
