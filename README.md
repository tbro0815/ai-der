# AI-DER 1.0.0

**Artificial Intelligence Distributed Engram Runner** is a local inference runner
for Qwen3.8-Flash-Next quantized mixture-of-experts models. It combines SSD-backed
weights, RAM caching and CUDA GPU tiers, with expert-prefetch experiments,
an OpenAI-compatible API and a browser dashboard with chat.

AI-DER is based on [colibrì](https://github.com/JustVugg/colibri), with a custom
Qwen3.8 engine, container converter, GPU routing and inference kernels.

## System requirements

- PC running **Ubuntu 24.04 or 26.04 LTS, x86_64**, with a CUDA-compatible NVIDIA driver.
- **128 GB RAM**, DDR4 or newer.
- **NVIDIA RTX 3090 or RTX 3090 Ti**, 24 GB VRAM.
- **PCIe 4.x or newer NVMe SSD as the system drive**. Keep model weights on NVMe;
  allow space for both the original GGUF shards and the converted container.
- **CUDA Toolkit 12.4 or newer**, including `nvcc`, CUDA runtime and cuBLAS.
- Internet access and sudo rights for installation.

The installer checks the OS, usable RAM, GPU model and toolkit version. Confirm
SSD interface generation and installed RAM type from your hardware specifications.
The installer (`--no-apt`), CUDA build, dashboard build, converter and CPU/CUDA
numerical tests were verified on Ubuntu 26.04 with CUDA 12.4 and an RTX 3090 Ti.
Ubuntu 24.04 and the sudo/apt provisioning path have not been tested end to end.
Full model inference was not restarted as part of release validation.

## Install

Install a compatible NVIDIA driver and CUDA toolkit following the
[NVIDIA Linux installation guide](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html).
Reboot after driver installation and confirm `nvidia-smi` works. If the toolkit
is not at `/usr/local/cuda`, `/usr/lib/cuda`, or on PATH, set `CUDA_HOME` (for example `/usr/local/cuda-12.4`).

```bash
sudo apt-get update
sudo apt-get install -y git
git clone https://github.com/tbro0815/ai-der.git
cd ai-der
git checkout v1.0.0
./install.sh --check
./install.sh
```

Run the installer as your normal user. It installs GCC/G++ 13, Make, Python 3.12 or newer,
venv/development headers, curl, CA certificates and xz via apt; creates `.venv`;
and installs AI-DER in editable mode plus **NumPy**, **Pillow** and **gguf**.
It uses an existing Node.js >=22.12 and npm, or downloads Node 22 LTS into
`.venv/node` and verifies the official SHA-256 checksum. It installs the locked
frontend dependencies with `npm ci`, builds the dashboard, and compiles the
Qwen3.8 CUDA engine for Ampere `sm_86`. The driver and CUDA toolkit are prerequisites. On an already provisioned host,
use `./install.sh --no-apt` to skip system package installation.
Keep the checkout in place: the editable command runs its engine sources here.

## Prepare a model and run

Model weights are separate downloads and are not included or downloaded by the
installer. Obtain all shards of a compatible **Qwen3.8-Flash-Next UD-IQ4_XS GGUF**
and its matching Hugging Face `tokenizer.json`, subject to the model's license.
Convert the first shard (the converter finds its siblings automatically):

```bash
.venv/bin/python engine/c/convert_qwen38.py \
  --gguf /path/to/model-00001-of-00003.gguf \
  --tokenizer /path/to/tokenizer.json \
  --out /path/to/qwen38-container

COLI_CUDA=1 .venv/bin/ai-der serve \
  --model /path/to/qwen38-container --host 127.0.0.1 --port 8080
```

Open [the dashboard](http://127.0.0.1:8080); the API base URL is
`http://127.0.0.1:8080/v1`. The default backend is AI-DER. Optional llama.cpp and
vLLM backends require separate installations. MTP requires additional draft
weights; the base GGUF conversion does not install them.

For LAN access set `COLI_API_KEY`, configure `COLI_ALLOWED_HOSTS` for your server
hostname/IP, and bind to `0.0.0.0`. Keep credentials outside the repository.

## Versions and development

`engine/c/version.py` is the single application version source, shared by
Python package metadata and `ai-der --version` (`coli` remains an alias).
Use semantic versioning: MAJOR for incompatible changes, MINOR for compatible
features, PATCH for fixes. Releases are tagged `vMAJOR.MINOR.PATCH`; this release
is **v1.0.0**. Internal upstream component metadata is independent.

```bash
.venv/bin/ai-der --version
.venv/bin/python engine/c/convert_qwen38.py --self-test
python3 engine/c/tests/test_release.py
```

## Credits and license

The engine is a modified Apache-2.0 colibrì fork; see [LICENSE](engine/LICENSE),
[upstream provenance](engine/UPSTREAM.txt) and [third-party notices](engine/THIRD_PARTY_NOTICES.md).
GGUF and quantization formats build on llama.cpp/ggml (MIT). Model architecture
and weights are from the Qwen team; UD-IQ4_XS quantization is by Unsloth.
Original copyright and license notices are retained. Model licenses are separate.
