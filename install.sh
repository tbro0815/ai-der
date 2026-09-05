#!/usr/bin/env bash
# AI-DER: install in this checkout on an Ubuntu/CUDA workstation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
fail() { echo "AI-DER: $*" >&2; exit 1; }
case "${1:-}" in
  --help) echo 'Usage: ./install.sh [--check|--no-apt]  (Ubuntu 24.04/26.04, CUDA 12.4+, RTX 3090/3090 Ti)'; exit 0 ;;
  ''|--check|--no-apt) ;;
  *) fail "Unknown argument: $1" ;;
esac
[[ $# -le 1 ]] || fail 'Expected at most one argument.'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Ubuntu x86_64 is required.'
source /etc/os-release
[[ $ID == ubuntu && ( $VERSION_ID == 24.04 || $VERSION_ID == 26.04 ) ]] || fail 'This installer targets Ubuntu 24.04 or 26.04 LTS.'
# /proc excludes reserved memory: a 128 GiB machine reports slightly less.
awk '/MemTotal:/ {exit ($2 < 120 * 1024 * 1024)}' /proc/meminfo || fail 'At least 128 GB installed RAM is required.'
command -v nvidia-smi >/dev/null || fail 'Install the NVIDIA driver, reboot, then rerun.'
gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader)
grep -Eq 'RTX 3090( Ti)?$' <<< "$gpus" || fail 'An NVIDIA RTX 3090 or RTX 3090 Ti is required.'
if [[ -z ${CUDA_HOME:-} ]]; then
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then CUDA_HOME=/usr/local/cuda
    elif [[ -x /usr/lib/cuda/bin/nvcc ]]; then CUDA_HOME=/usr/lib/cuda
    elif command -v nvcc >/dev/null; then CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    else fail 'Install CUDA Toolkit 12.4+ (see README).'; fi
fi
export CUDA_HOME
[[ -x "$CUDA_HOME/bin/nvcc" ]] || fail 'Install CUDA Toolkit 12.4+ and set CUDA_HOME to its directory (see README).'
cuda_version=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
[[ -n $cuda_version ]] || fail 'Cannot determine CUDA Toolkit version.'
[[ $(printf '%s\n' 12.4 "$cuda_version" | sort -V | head -n 1) == 12.4 ]] || fail 'CUDA Toolkit 12.4+ is required.'
echo "GPU: $gpus; CUDA: $cuda_version"
echo 'Ensure the system drive is a PCIe 4.x or newer NVMe SSD; store model weights on NVMe too.'
[[ ${1:-} != --check ]] || exit 0
[[ $EUID -ne 0 ]] || fail 'Run as your normal user; sudo is used only for OS packages.'
if [[ ${1:-} != --no-apt ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential gcc-13 g++-13 python3 python3-venv python3-dev curl ca-certificates xz-utils
fi
command -v gcc-13 >/dev/null && command -v g++-13 >/dev/null || fail 'Install gcc-13 and g++-13 for the CUDA host compiler.'
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin g++-13"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ./engine numpy Pillow gguf

# Vite needs Node >=22.12. Keep a private Node 22 LTS build in the venv
# when a suitable system installation is unavailable; verify its checksum.
if ! command -v node >/dev/null || ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a<22 || (a===22 && b<12) ? 1 : 0)' || ! command -v npm >/dev/null; then
    scratch=$(mktemp -d)
    trap 'rm -rf -- "$scratch"' EXIT
    curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt -o "$scratch/SHASUMS256.txt"
    node_archive=$(awk '$2 ~ /^node-v22\.[0-9]+\.[0-9]+-linux-x64\.tar\.xz$/ {print $2}' "$scratch/SHASUMS256.txt")
    [[ -n $node_archive && $node_archive != *$'\n'* ]] || fail 'Cannot resolve the Node 22 LTS archive.'
    curl -fsSL "https://nodejs.org/dist/latest-v22.x/$node_archive" -o "$scratch/$node_archive"
    (cd "$scratch"; grep "  $node_archive\$" SHASUMS256.txt | sha256sum --check --strict)
    mkdir -p .venv/node
    tar -xJf "$scratch/$node_archive" -C .venv/node --strip-components=1
    export PATH="$PWD/.venv/node/bin:$PATH"
fi
npm --prefix engine/web ci
npm --prefix engine/web run build
make -C engine/c -j"$(nproc)" qwen38 CC=gcc-13 CUDA=1 CUDA_ARCH=sm_86 CUDA_HOME="$CUDA_HOME"
.venv/bin/ai-der --version
printf '\nInstalled. Convert your model and start the server using the README commands.\n'
