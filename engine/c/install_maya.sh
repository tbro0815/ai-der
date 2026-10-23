#!/usr/bin/env bash
# AI-DER: the optional Maya backend (github.com/mw00/project-maya, MIT), Strata's fork for GLM-5.3-Flash and
# DeepSeek-V4.1-Flash: built from source at a pinned revision against the private CUDA 13 toolkit
# (engine/c/install_cuda13.sh), plus up to three models, each with the maya-<model>.json its setup writes:
#   glm-s24         GLM-5.3-Flash Maya-S24 (~96 GB with the vision files), the quant for 24 GB cards
#   glm-m-derisked  GLM-5.3-Flash Maya-M-Derisked (~117 GB), Maya-M with Blackfrost_AI's refusal edit
#   ds4-s           DeepSeek-V4.1-Flash Maya-S (~341 GB with its vision tower; more than RAM + VRAM:
#                   the experts that do not fit are read from the SSD)
# Maya's setup downloads and verifies every file (sha256), packs the model and writes its config;
# DeepSeek's DSpark drafter stays off (--drafting off). DeepSeek's engine holds one conversation and
# reuses it only for a prompt that continues it: MAYA_DS4_SLOT_KEEP parks the others (an agent's
# interleaved calls) and MAYA_DS4_KEEP_THINKING keeps the reasoning in a tool-less chat's history,
# so its next turn continues the state. AI_DER_MAYA_MODELS picks the models
# (default: glm-s24,glm-m-derisked,ds4-s).
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_maya.sh
set -euo pipefail
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_MAYA_COMMIT:-8412d5e}         # v1.1.25, the reference box (2026-10-23)
want=${AI_DER_MAYA_MODELS:-glm-s24,glm-m-derisked,ds4-s}
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
git -C "$dir" checkout --quiet "$commit"
cd "$dir"
export PATH=$cuda/bin:$PATH CUDA_PATH=$cuda
# maya.sh: its own .venv, the pinned llama.cpp sources, the engine compiled for this GPU, the
# model downloaded (--download-model) and packed, the vision encoder, the start config
common=(--setup --models-dir "$models" --port 8085 --download-model --no-start --yes --plain
        --drafting off)

setup_glm() {   # <model> <config>
    if [[ -f $2 ]]; then say "$1 configured, skipping"; return; fi
    say "GLM-5.3-Flash $1"
    ./maya.sh "${common[@]}" --model "$1" --context 131072
    [[ -f $2 ]] || fail "the setup of $1 wrote no $2"
}
[[ $want == *glm-s24* ]] && setup_glm Maya-S24 maya-maya-s24.json
[[ $want == *glm-m-derisked* ]] && setup_glm Maya-M-Derisked maya-maya-m-derisked.json
if [[ $want == *ds4-s* ]]; then
    if [[ -f maya-ds4.json ]]; then say "DeepSeek-V4.1-Flash Maya-S configured, skipping"
    else
        say "DeepSeek-V4.1-Flash Maya-S (~341 GB)"
        ./maya.sh "${common[@]}" --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-s --context 131072 \
            --env MAYA_DS4_SLOT_KEEP=1 --env MAYA_DS4_KEEP_THINKING=1
    fi
fi
ls maya-*.json >/dev/null 2>&1 || fail "no model configured (AI_DER_MAYA_MODELS=$want)"
say "done: $(ls maya-*.json | tr '\n' ' ')"
