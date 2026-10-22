#!/usr/bin/env bash
# AI-DER: a private CUDA 13 toolkit for the Strata and llama.cpp builds, assembled from NVIDIA's
# pip wheels into $AI_DER_DATA/cuda13 (bin, include, lib, nvvm). The system toolkit is left alone.
# Ubuntu 26.04's glibc (2.41+) declares rsqrt/rsqrtf noexcept, which CUDA 13.0's math_functions.h
# does not; when a test compile shows that conflict, the view carries a private copy of that one
# header with the two declarations matched.
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/install_cuda13.sh
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
venv=$data/cuda13-venv
view=$data/cuda13
uv=$root/.venv/bin/uv
say() { echo "AI-DER CUDA 13: $*"; }
fail() { echo "AI-DER CUDA 13: $*" >&2; exit 1; }
[[ -x $uv ]] || fail "no uv at $uv; run ./install.sh first."

if [[ -x $view/bin/nvcc && -f $view/.ready ]]; then
    say "toolkit view present at $view, skipping"
    exit 0
fi
say "NVIDIA's CUDA 13 wheels -> $venv"
[[ -x $venv/bin/python ]] || "$uv" venv --quiet --python 3.12 "$venv"
"$uv" pip install --quiet --python "$venv/bin/python" \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-runtime==13.0.96 \
    nvidia-cublas==13.1.1.3 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85
cu=$("$venv/bin/python" -c 'import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], "cu13"))')
[[ -x $cu/bin/nvcc ]] || fail "no nvcc in $cu after the install."

# The view: real directories of links. nvcc finds its headers through bin/.., so bin must be a
# directory here (a link to the wheel's bin would resolve .. into the wheel).
say "toolkit view -> $view"
rm -rf "$view"; mkdir -p "$view/bin" "$view/nvvm" "$view/include/crt" "$view/lib"
for d in bin nvvm; do for e in "$cu/$d"/*; do ln -s "$e" "$view/$d/"; done; done
for e in "$cu/include"/*; do [[ $(basename "$e") == crt ]] || ln -s "$e" "$view/include/"; done
for e in "$cu/include/crt"/*; do ln -s "$e" "$view/include/crt/"; done
for f in "$cu/lib"/*.so.* "$cu/lib"/*.a; do
    [[ -e $f ]] || continue
    b=$(basename "$f"); ln -sf "$f" "$view/lib/$b"
    [[ $b == *.so.* ]] && ln -sf "$f" "$view/lib/${b%%.so.*}.so"
done
ln -s lib "$view/lib64"

probe=$(mktemp -d); trap 'rm -rf -- "$probe"' EXIT
printf '#include <cmath>\n__global__ void k(float *x){x[0]=rsqrtf(x[0]);}\nint main(){return 0;}\n' > "$probe/t.cu"
if ! "$view/bin/nvcc" -arch=sm_86 "$probe/t.cu" -o "$probe/t" 2> "$probe/err"; then
    grep -q "exception specification is incompatible" "$probe/err" || { cat "$probe/err" >&2; fail "nvcc cannot compile a test kernel."; }
    say "glibc declares rsqrt/rsqrtf noexcept: matching CUDA's math_functions.h in the view"
    rm "$view/include/crt/math_functions.h"
    sed -e 's/^\(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  *rsqrt(double x)\);/\1 noexcept (true);/' \
        -e 's/^\(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  *rsqrtf(float x)\);/\1 noexcept (true);/' \
        "$cu/include/crt/math_functions.h" > "$view/include/crt/math_functions.h"
    "$view/bin/nvcc" -arch=sm_86 "$probe/t.cu" -o "$probe/t" || fail "nvcc still cannot compile the test kernel."
fi
touch "$view/.ready"
say "done: $("$view/bin/nvcc" --version | tail -n 1)"
