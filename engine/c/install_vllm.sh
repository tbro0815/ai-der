#!/usr/bin/env bash
# AI-DER: install the Qwen3.8-27B vLLM stack (syv-ai/qwen38-27b-rtx3090) for the optional vLLM backend:
# a Python 3.12 venv with the pinned vLLM, the published W4A16 weights, the one-time CPU
# preparation (int8 heads, int8 MTP draft, draft vocabulary, "fast" variant) and the repo's patches.
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_vllm.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_VLLM_STACK_COMMIT:-2bbd292}   # the revision the reference box serves (2026-09)
weights_repo=${AI_DER_VLLM_WEIGHTS:-dbirks/Qwen3.8-27B-W4A16-AutoRound}
dir=$data/vllm/qwen38-27b-rtx3090
uv=$root/.venv/bin/uv
say() { echo "AI-DER vLLM: $*"; }
fail() { echo "AI-DER vLLM: $*" >&2; exit 1; }
[[ -x $uv ]] || fail "no uv at $uv; run ./install.sh first."
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

if [[ ! -d $dir/.git ]]; then
    say "cloning syv-ai/qwen38-27b-rtx3090 -> $dir"
    mkdir -p "$(dirname "$dir")"
    git clone --quiet https://github.com/syv-ai/qwen38-27b-rtx3090 "$dir"
fi
git -C "$dir" fetch --quiet origin
git -C "$dir" checkout --quiet "$commit"
cd "$dir"

free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
need_gb=40; [[ -d models/Qwen3.8-27B-W4A16-AutoRound-fast ]] && need_gb=0
(( free_gb >= need_gb )) || fail "$free_gb GB free under $data, about $need_gb GB are needed for the vLLM venv and weights."

say "1/5 Python 3.12 venv with the pinned vLLM (docker/requirements.txt)"
[[ -x venv/bin/python ]] || "$uv" venv --quiet --python 3.12 venv
"$uv" pip install --quiet --python venv/bin/python -r docker/requirements.txt
"$uv" pip install --quiet --python venv/bin/python huggingface_hub hf_transfer ninja flashinfer-python flashinfer-cubin==0.6.13
venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

say "2/5 weights $weights_repo (about 20 GB)"
if [[ -f models/Qwen3.8-27B-W4A16-AutoRound/config.json ]]; then
    say "present, skipping the download"
else
    venv/bin/hf download "$weights_repo" --local-dir models/Qwen3.8-27B-W4A16-AutoRound
fi

say "3/5 one-time CPU preparation (int8 heads, int8 MTP draft, draft vocabulary, fast variant)"
V=venv/bin/python; M=models/Qwen3.8-27B-W4A16-AutoRound
"$V" prepare/quant_lm_head.py "$M"
"$V" prepare/quant_embed.py "$M"
"$V" prepare/quant_mtp.py "$M"
"$V" prepare/build_draft_vocab.py "$M" --ids prepare/draft_vocab_ids.json
"$V" prepare/fetch_fast_variant.py
[[ -d models/Qwen3.8-27B-W4A16-AutoRound-fast ]] || fail "the fast variant did not appear under models/"

say "4/5 vLLM patches"
site=venv/lib/python3.12/site-packages/vllm
for p in patches/*.patch; do
    if patch -p1 -d "$site" --dry-run --forward -s < "$p" >/dev/null 2>&1; then
        patch -p1 -d "$site" --forward -s < "$p" >/dev/null
    elif patch -p1 -d "$site" --dry-run --reverse -s < "$p" >/dev/null 2>&1; then
        :   # already applied
    else
        fail "patch $(basename "$p") does not apply to the installed vLLM"
    fi
done

say "5/5 verify (no server start)"
bash verify.sh --no-server 2>&1 | tail -n 5
say "done: launcher $dir/single-user/start_qwen.sh, model $dir/models/Qwen3.8-27B-W4A16-AutoRound-fast"
