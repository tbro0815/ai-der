#!/usr/bin/env bash
# AI-DER: build llama.cpp's llama-server (CUDA) for the optional llama.cpp backend.
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_llamacpp.sh
set -euo pipefail
data=${AI_DER_DATA:-$HOME/ai-der-data}
commit=${AI_DER_LLAMACPP_COMMIT:-f1793c1}     # the revision the reference box serves (2026-09)
arch=${AI_DER_CUDA_ARCH:-86}
dir=$data/llama.cpp
say() { echo "AI-DER llama.cpp: $*"; }
fail() { echo "AI-DER llama.cpp: $*" >&2; exit 1; }
command -v cmake >/dev/null || fail "cmake is missing (the installer's apt step provides it)."
command -v git >/dev/null || fail "git is missing."

if [[ -x $dir/build/bin/llama-server && $(git -C "$dir" rev-parse --short HEAD 2>/dev/null) == "$commit"* ]]; then
    say "llama-server already built at $commit, skipping"
    exit 0
fi
if [[ ! -d $dir/.git ]]; then
    say "cloning ggml-org/llama.cpp -> $dir"
    git clone --quiet https://github.com/ggml-org/llama.cpp "$dir"
fi
git -C "$dir" fetch --quiet --depth 1 origin "$commit" 2>/dev/null || git -C "$dir" fetch --quiet origin
git -C "$dir" checkout --quiet "$commit"
say "configuring (GGML_CUDA=ON, sm_$arch, host compiler g++-13)"
cmake -S "$dir" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-g++-13}" \
    -DLLAMA_CURL=OFF -DGGML_NATIVE=ON >/dev/null
say "building llama-server (10-20 minutes)"
cmake --build "$dir/build" --config Release --target llama-server -j"$(nproc)" >/dev/null
[[ -x $dir/build/bin/llama-server ]] || fail "build finished without $dir/build/bin/llama-server"
"$dir/build/bin/llama-server" --version 2>&1 | head -n 2
say "done: $dir/build/bin/llama-server"
