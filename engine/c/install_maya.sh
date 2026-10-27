#!/usr/bin/env bash
# AI-DER: the optional Maya backend (github.com/mw00/project-maya, MIT), Strata's fork for GLM-5.3-Flash,
# DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash: built from source at a pinned revision against the private
# CUDA 13 toolkit (engine/c/install_cuda13.sh), plus up to five models, each with the maya-<model>.json its
# setup writes:
#   glm-s24         GLM-5.3-Flash Maya-S24 (~96 GB with the vision files), the quant for 24 GB cards
#   glm-m-derisked  GLM-5.3-Flash Maya-M-Derisked (~117 GB), Maya-M with Blackfrost_AI's refusal edit
#   ds4-s           DeepSeek-V4.1-Flash Maya-S (~341 GB with its vision tower; more than RAM + VRAM:
#                   the experts that do not fit are read from the SSD)
#   minimax-xs      MiniMax-M3.1-Flash Maya-XS (~97 GB with the vision tower and DSpark draft), the whole
#                   model in a 128 GB PC's memory. Not offered: Maya-S (122 GB). MiniMax's engine keeps every
#                   expert VRAM does not hold in pinned RAM (no SSD path), ~114 GB with the KV cache on a
#                   121 GiB box, and its first start stalled the reference box until a deadline killed it.
# MiniMax's weights are under the MiniMax Community License (non-commercial use free, commercial use with
# conditions). patches/maya-mm3-indexers.patch hands the MiniMax quants' -maya-indexers.bin to the pack
# (v1.1.31's setup passes only the .gguf shards, and the pack then misses layers 0-2's indexers).
# Maya's setup downloads and verifies every file (sha256), packs the model and writes its config;
# DeepSeek's DSpark drafter stays off (--drafting off). DeepSeek's engine holds one conversation and
# reuses it only for a prompt that continues it: MAYA_DS4_SLOT_KEEP parks the others (an agent's
# interleaved calls) and MAYA_DS4_KEEP_THINKING keeps the reasoning in a tool-less chat's history,
# so its next turn continues the state. AI_DER_MAYA_MODELS picks the models
# (default: glm-s24,glm-m-derisked,ds4-s,minimax-xs).
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_maya.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_MAYA_COMMIT:-73c166e}         # v1.1.32, the reference box (2026-10-27)
want=${AI_DER_MAYA_MODELS:-glm-s24,glm-m-derisked,ds4-s,minimax-xs}
patch=$root/engine/c/patches/maya-mm3-indexers.patch
dir=$data/maya
models=$data/models/maya
cuda=$data/cuda13
say() { echo "AI-DER Maya: $*"; }
fail() { echo "AI-DER Maya: $*" >&2; exit 1; }
[[ -x $cuda/bin/nvcc ]] || fail "no CUDA 13 toolkit at $cuda; run engine/c/install_cuda13.sh first."

if [[ ! -d $dir/.git ]]; then
    say "cloning mw00/project-maya -> $dir"
    git clone --quiet https://github.com/mw00/project-maya "$dir"
fi
git -C "$dir" fetch --quiet origin
git -C "$dir" checkout --quiet -- maya.py            # (an earlier run's patch)
git -C "$dir" checkout --quiet "$commit"
git -C "$dir" apply "$patch"
cd "$dir"
export PATH=$cuda/bin:$PATH CUDA_PATH=$cuda
# maya.sh: its own .venv, the pinned llama.cpp sources, the engine compiled for this GPU, the
# model downloaded (--download-model) and packed, the vision encoder, the start config
common=(--setup --models-dir "$models" --port 8085 --download-model --no-start --yes --plain)

setup_glm() {   # <model> <config>
    if [[ -f $2 ]]; then say "$1 configured, skipping"; return; fi
    say "GLM-5.3-Flash $1"
    ./maya.sh "${common[@]}" --model "$1" --context 131072 --drafting off
    [[ -f $2 ]] || fail "the setup of $1 wrote no $2"
}
[[ $want == *glm-s24* ]] && setup_glm Maya-S24 maya-maya-s24.json
[[ $want == *glm-m-derisked* ]] && setup_glm Maya-M-Derisked maya-maya-m-derisked.json
if [[ $want == *ds4-s* ]]; then
    if [[ -f maya-ds4.json ]]; then say "DeepSeek-V4.1-Flash Maya-S configured, skipping"
    else
        say "DeepSeek-V4.1-Flash Maya-S (~341 GB)"
        ./maya.sh "${common[@]}" --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-s --context 131072 --drafting off \
            --env MAYA_DS4_SLOT_KEEP=1 --env MAYA_DS4_KEEP_THINKING=1
    fi
fi
setup_minimax() {   # <model> <config>
    if [[ -f $2 ]]; then say "$1 configured, skipping"; return; fi
    say "MiniMax-M3.1-Flash $1"
    ./maya.sh "${common[@]}" --model "$1" --context 131072 --drafting on
    [[ -f $2 ]] || fail "the setup of $1 wrote no $2"
}
[[ $want == *minimax-xs* ]] && setup_minimax minimax-maya-xs maya-minimax-maya-xs.json
ls maya-*.json >/dev/null 2>&1 || fail "no model configured (AI_DER_MAYA_MODELS=$want)"
say "done: $(ls maya-*.json | tr '\n' ' ')"
