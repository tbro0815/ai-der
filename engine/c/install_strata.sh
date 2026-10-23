#!/usr/bin/env bash
# AI-DER: the optional Strata backend (github.com/Niko1221/Strata, MIT), an inference engine for
# Qwen3.8-Flash-Next: built from source at a pinned revision against the private CUDA 13 toolkit
# (engine/c/install_cuda13.sh), plus up to three models, each with its strata-<model>.json:
#   gsq         ISTA-DASLab's GSQ-RCO IQ3_S pack (~84 GB download)
#   uncensored  DavidAU's abliterated GSQ-RCO IQ3_S (~84 GB; from SC117's abliteration of the
#               GSQ-RCO pack), packed like the GSQ-RCO model with Strata's tools/iq_pack.py
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
    ab=$models/davidau-abliterated
    if [[ -f strata-davidau-abliterated-iq3s.json ]]; then say "abliterated GSQ-RCO IQ3_S configured, skipping"
    else
        say "abliterated GSQ-RCO IQ3_S (DavidAU, ~84 GB)"
        "$hf" download DavidAU/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-GGUF \
            Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf \
            Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf --local-dir "$ab"
        s1=$ab/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf
        s2=$ab/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf
        [[ -d third_party/llama.cpp/gguf-py ]] || fail "no gguf-py in $dir/third_party/llama.cpp (a setup.sh run brings it)"
        # its own dense weights and tokenizer (the abliteration changed the dense side): the
        # same pack step setup runs for the GSQ-RCO model
        STRATA_GGUF_PY=$dir/third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
            --gguf "$s1" --out "$sdata/packs/davidau-abliterated-iq3s"
        [[ -f $sdata/packs/davidau-abliterated-iq3s/tokenizer/vocab.json ]] || \
            STRATA_GGUF_PY=$dir/third_party/llama.cpp/gguf-py .venv/bin/python tools/strata_tokenizer.py \
                --gguf "$s1" --out "$sdata/packs/davidau-abliterated-iq3s"
        # its config: the GSQ-RCO IQ3_S config (same layout) pointed at these shards and this pack
        [[ -f strata-iq3_s.json ]] || fail "the abliterated model's config is made from the GSQ-RCO one (AI_DER_STRATA_MODELS needs gsq)"
        .venv/bin/python - "$s1" "$s2" "$sdata" <<'EOF'
import json, os, sys
s1, s2, sdata = sys.argv[1:4]
pack = os.path.join(sdata, "packs", "davidau-abliterated-iq3s")
cfg = json.load(open("strata-iq3_s.json"))
a = cfg["args"]
a[a.index("--pack") + 1] = pack
a[a.index("--native") + 1] = s1
a[a.index("--ple-gguf") + 1] = s2
cfg["args"] = cfg["args_written"] = a
cfg["tokenizer"] = os.path.join(pack, "tokenizer")
cfg["model_name"] = "qwen3.8-flash-next-davidau-abliterated-iq3_s"
cfg["log"] = os.path.join(os.getcwd(), "strata-davidau-abliterated-iq3s.log")
if cfg.get("vision"):
    cfg["vision"]["model"] = s1
json.dump(cfg, open("strata-davidau-abliterated-iq3s.json", "w"), indent=1)
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
