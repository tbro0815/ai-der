#!/bin/sh
set -eu

: "${SNAP:?set SNAP to the qwen38-colibri model directory}"
BIN=${BIN:-"$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)/qwen38"}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
prompt='<|im_start|>user
Write a short paragraph about the sea.<|im_end|>
<|im_start|>assistant
'

for batch in 1 64 256; do
    PROMPT="$prompt" N_NEW=16 NO_EOS_STOP=1 COLI_CUDA=1 CUDA_EXPERT_GB=0.001 \
        Q38_DENSE_I8=1 Q38_GPU_DENSE=1 Q38_MS_DECODE=1 Q38_PREFETCH=0 \
        Q38_PREFILL_B=$batch "$BIN" >"$work/$batch.out" 2>"$work/$batch.err"
    sed -n 's/^IDs *: */IDs: /p' "$work/$batch.err" >"$work/$batch.ids"
done

cmp "$work/1.ids" "$work/64.ids"
cmp "$work/1.ids" "$work/256.ids"
echo "qwen38 GPU prefill parity: ok, batches 64 and 256 ($(cat "$work/64.ids"))"

# P9: q8 KV takes the device QSA walk for batches >= 16 (host walk at B=1)
for batch in 1 256; do
    PROMPT="$prompt" N_NEW=16 NO_EOS_STOP=1 COLI_CUDA=1 CUDA_EXPERT_GB=0.001 \
        Q38_DENSE_I8=1 Q38_GPU_DENSE=1 Q38_MS_DECODE=1 Q38_PREFETCH=0 Q38_KV_Q8=1 \
        Q38_PREFILL_B=$batch "$BIN" >"$work/q8-$batch.out" 2>"$work/q8-$batch.err"
    sed -n 's/^IDs *: */IDs: /p' "$work/q8-$batch.err" >"$work/q8-$batch.ids"
done
cmp "$work/q8-1.ids" "$work/q8-256.ids"
echo "qwen38 GPU prefill parity (q8 KV, device QSA walk): ok ($(cat "$work/q8-256.ids"))"
