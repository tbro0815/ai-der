#!/usr/bin/env bash
# AI-DER: download the Qwen3.8-Flash-Next weights and build the engine container.
# Called by ../../install.sh; runnable alone from the checkout root:
#   AI_DER_DATA=~/ai-der-data engine/c/install_models.sh
# Every step is skipped when its result already exists, so a rerun resumes.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
python=${AI_DER_PYTHON:-$root/.venv/bin/python}
hf=${AI_DER_HF:-$root/.venv/bin/hf}
say() { echo "AI-DER models: $*"; }
fail() { echo "AI-DER models: $*" >&2; exit 1; }
[[ -x $python ]] || fail "no Python at $python; run ./install.sh first."
[[ -x $hf ]] || fail "no hf CLI at $hf; run ./install.sh first."
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

gguf_repo=${AI_DER_GGUF_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}
gguf_dir=$data/models/qwen38-flash-next
shard1=$gguf_dir/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
hf_repo=${AI_DER_HF_REPO:-Qwen/Qwen3.8-Flash-Next}
official=$data/models/qwen38-official
mtp_src=$data/models/qwen38-mtp
container=$data/models/qwen38-container
mkdir -p "$gguf_dir" "$official" "$mtp_src" "$data/models"

# free space: GGUF 94 GB + mmproj 1 GB + container 61 GB + MTP source 5 GB + MTP container 3 GB
need_gb=170
[[ -f $container/qwen38_meta.json && -f $container/mtp/mtp_meta.json ]] && need_gb=0
free_gb=$(df -Pk "$data" | awk 'NR==2 {print int($4 / 1048576)}')
(( free_gb >= need_gb )) || fail "$free_gb GB free under $data, about $need_gb GB are needed for the weights and the container."

say "1/5 GGUF shards + vision projector from $gguf_repo -> $gguf_dir"
if [[ -f $shard1 && -f $gguf_dir/mmproj-F16.gguf ]]; then
    say "present, skipping the download"
else
    "$hf" download "$gguf_repo" --include 'UD-IQ4_XS/*' 'mmproj-F16.gguf' --local-dir "$gguf_dir"
fi
[[ -f $shard1 ]] || fail "shard 1 missing after the download: $shard1"

say "2/5 tokenizer, chat template and weight index from $hf_repo -> $official"
if [[ -f $official/tokenizer.json && -f $official/chat_template.jinja && -f $official/model.safetensors.index.json ]]; then
    say "present, skipping"
else
    "$hf" download "$hf_repo" tokenizer.json chat_template.jinja model.safetensors.index.json --local-dir "$official"
fi

say "3/5 MTP draft tensors (HTTP range requests, about 5 GB) -> $mtp_src"
if [[ -f $mtp_src/mtp.safetensors && -f $mtp_src/mtp_manifest.json ]]; then
    say "present, skipping"
else
    "$python" "$root/engine/c/fetch_mtp_tensors.py" --index "$official/model.safetensors.index.json" \
        --out-dir "$mtp_src" --repo "$hf_repo"
fi

say "4/5 engine container (GGUF -> $container, about 61 GB, tens of minutes)"
if [[ -f $container/qwen38_meta.json ]]; then
    say "present, skipping the conversion"
else
    "$python" "$root/engine/c/convert_qwen38.py" --gguf "$shard1" --tokenizer "$official/tokenizer.json" --out "$container"
fi
[[ -f $container/chat_template.jinja ]] || cp "$official/chat_template.jinja" "$container/chat_template.jinja"

say "5/5 MTP container (BF16 -> Q8_0, $container/mtp)"
if [[ -f $container/mtp/mtp_meta.json ]]; then
    say "present, skipping"
else
    "$python" "$root/engine/c/convert_mtp.py" --src "$mtp_src/mtp.safetensors" --manifest "$mtp_src/mtp_manifest.json" --out "$container/mtp"
fi
say "done: container $container, GGUF $gguf_dir"
