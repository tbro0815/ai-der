#!/usr/bin/env bash
# AI-DER: install the Qwen3.8-27B vLLM stack (syv-ai/HyperQwen, formerly qwen38-27b-rtx3090) for the optional vLLM backend:
# a Python 3.12 venv with the pinned vLLM, the published W4A16 weights, the one-time CPU
# preparation (int8 heads, int8 MTP draft, draft vocabulary, "fast" variant) and the repo's patches.
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_vllm.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_VLLM_STACK_COMMIT:-95ee3b6}   # the revision the reference box serves (vLLM 0.31.0, 2026-10)
weights_repo=${AI_DER_VLLM_WEIGHTS:-dbirks/Qwen3.8-27B-W4A16-AutoRound}
dir=$data/vllm/qwen38-27b-rtx3090
uv=$root/.venv/bin/uv
say() { echo "AI-DER vLLM: $*"; }
fail() { echo "AI-DER vLLM: $*" >&2; exit 1; }
[[ -x $uv ]] || fail "no uv at $uv; run ./install.sh first."
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

if [[ ! -d $dir/.git ]]; then
    say "cloning syv-ai/HyperQwen -> $dir"
    mkdir -p "$(dirname "$dir")"
    git clone --quiet https://github.com/syv-ai/HyperQwen "$dir"
fi
git -C "$dir" fetch --quiet origin
git -C "$dir" checkout --quiet "$commit"
cd "$dir"

free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
need_gb=40; [[ -d models/Qwen3.8-27B-W4A16-AutoRound-fast ]] && need_gb=0
(( free_gb >= need_gb )) || fail "$free_gb GB free under $data, about $need_gb GB are needed for the vLLM venv and weights."

say "1/5 Python 3.12 venv with the pinned vLLM (docker/requirements.txt)"
[[ -x venv/bin/python ]] || "$uv" venv --quiet --python 3.12 venv
# vLLM is installed fresh on every run: the patch series below applies to a pristine tree
# only, and a reinstall alone keeps the files the series added
rm -rf venv/lib/python3.12/site-packages/vllm
"$uv" pip install --quiet --python venv/bin/python --reinstall-package vllm -r docker/requirements.txt
# prebuilt FlashInfer kernels; not on PyPI past 0.6.13, so from FlashInfer's own index
"$uv" pip install --quiet --python venv/bin/python --extra-index-url https://flashinfer.ai/whl/ \
    flashinfer-cubin==0.7.0.post1
venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

say "2/5 weights $weights_repo (about 20 GB)"
if [[ -f models/Qwen3.8-27B-W4A16-AutoRound/config.json ]]; then
    say "present, skipping the download"
else
    venv/bin/hf download "$weights_repo" --local-dir models/Qwen3.8-27B-W4A16-AutoRound
fi

say "3/5 one-time CPU preparation (int8 heads, int8 MTP draft, draft vocabulary, fast variant)"
V=venv/bin/python; M=models/Qwen3.8-27B-W4A16-AutoRound
# Each step runs while the index still names its input: the scripts complete an
# interrupted run, but stop on a finished one (the tensor they read is gone).
has() { "$V" -c 'import json, sys; sys.exit(sys.argv[2] not in json.load(open(sys.argv[1]))["weight_map"])' \
    "$M/model.safetensors.index.json" "$1"; }
has lm_head.weight && "$V" prepare/quant_lm_head.py "$M"
has model.language_model.embed_tokens.weight && "$V" prepare/quant_embed.py "$M"
has mtp.layers.0.mlp.down_proj.weight && "$V" prepare/quant_mtp.py "$M"
has mtp.draft_lm_head.weight_packed || "$V" prepare/build_draft_vocab.py "$M" --ids prepare/draft_vocab_ids.json
"$V" prepare/fetch_fast_variant.py
[[ -d models/Qwen3.8-27B-W4A16-AutoRound-fast ]] || fail "the fast variant did not appear under models/"

say "4/5 vLLM patches (patches/series, in order)"
site=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
bash patches/apply.sh "$site" > /dev/null || fail "the vLLM patch series does not apply (patches/apply.sh names the patch above)"

say "5/5 verify (no server start)"
bash verify.sh --no-server 2>&1 | tail -n 5
say "done: launcher $dir/single-user/start_qwen.sh, model $dir/models/Qwen3.8-27B-W4A16-AutoRound-fast"
