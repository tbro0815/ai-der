# AI-DER 1.1.1

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
- **PCIe 4.x or newer NVMe SSD as the system drive**, with about **230 GB free** for
  the full installation (weights 95 GB, engine container 61 GB, MTP draft 8 GB,
  vLLM stack and weights 40 GB, llama.cpp 3 GB). Keep the data directory on NVMe.
- **CUDA Toolkit 12.4 or newer**, including `nvcc`, CUDA runtime and cuBLAS.
- Internet access and sudo rights for installation. Model weights are downloaded
  from Hugging Face under their own licenses.

`./install.sh --check` verifies the OS, usable RAM, GPU model, toolkit version and
free disk space. Confirm the SSD interface generation and the installed RAM type
from your hardware specifications.

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
git checkout v1.1.1
./install.sh --check
./install.sh
```

Run the installer as your normal user. The default installation does everything:

1. **Engine and dashboard.** GCC/G++ 13, CMake, Make, Python 3.12 or newer,
   venv/development headers, curl, patch and xz via apt; a `.venv` with AI-DER in
   editable mode plus NumPy, Pillow, gguf, the Hugging Face CLI and uv; Node.js
   >=22.12 (system or a checksum-verified Node 22 LTS in `.venv/node`); the locked
   dashboard build; the Qwen3.8 CUDA engine for Ampere `sm_86`.
2. **Weights and container** (`engine/c/install_models.sh`): the Qwen3.8-Flash-Next
   UD-IQ4_XS GGUF shards and vision projector (Unsloth), the tokenizer and chat
   template (Qwen), the MTP draft tensors fetched by HTTP range requests, then the
   engine container conversion and the MTP container (BF16 to Q8_0). Tens of
   minutes after the download; every step resumes on rerun.
3. **llama.cpp backend** (`engine/c/install_llamacpp.sh`): `llama-server` built with
   CUDA at a pinned revision.
4. **vLLM backend** (`engine/c/install_vllm.sh`): the syv-ai Qwen3.8-27B stack at a
   pinned revision in its own Python 3.12 venv, the published W4A16 weights, the
   one-time CPU preparation (int8 heads, int8 MTP draft, draft vocabulary, fast
   variant) and the stack's vLLM patches.
5. **`ai-der.env`** in the checkout: the serving environment for all installed
   backends. `--systemd` also writes `~/.config/systemd/user/ai-der.service`.

Everything lands under `~/ai-der-data` (`--data-dir DIR` changes it). Flags:
`--core-only` installs the engine and dashboard only, `--skip-models`,
`--skip-llamacpp` and `--skip-vllm` drop one component, `--no-apt` skips the sudo
step on a provisioned host. Keep the checkout in place: the editable command runs
its engine sources here. The full download-and-convert path is long (an hour or
more, dominated by the 95 GB download and the conversion) and was assembled from
the steps used on the reference machine; it has not yet been rerun end to end on a
fresh host for this release, so a failing step is rerunnable and reports its name.

## Run

```bash
set -a; source ai-der.env; set +a
.venv/bin/ai-der serve --model ~/ai-der-data/models/qwen38-container \
  --host 127.0.0.1 --port 8080 --ngen 32768
```

Open [the dashboard](http://127.0.0.1:8080); the API base URL is
`http://127.0.0.1:8080/v1`. The default backend is AI-DER; the Extra section of
the dashboard switches to llama.cpp or vLLM at the next restart and selects the
vLLM profile (`long`, 131K, or `fast`, 64K, which locks the backend to vLLM).
With `--systemd`: `systemctl --user enable --now ai-der.service`.

For LAN access set `COLI_API_KEY`, configure `COLI_ALLOWED_HOSTS` for your server
hostname/IP (both have commented placeholders in `ai-der.env`), and bind to
`0.0.0.0`. Keep credentials outside the repository.

### Manual model preparation (`--core-only`)

Obtain all shards of a compatible **Qwen3.8-Flash-Next UD-IQ4_XS GGUF** and the
matching Hugging Face `tokenizer.json`, then convert the first shard (the converter
finds its siblings):

```bash
.venv/bin/python engine/c/convert_qwen38.py \
  --gguf /path/to/model-00001-of-00003.gguf \
  --tokenizer /path/to/tokenizer.json \
  --out /path/to/qwen38-container
```

MTP needs the draft module: `engine/c/fetch_mtp_tensors.py` fetches it from the
Qwen repository and `engine/c/convert_mtp.py` writes `<container>/mtp`.

## Versions and development

`engine/c/version.py` is the single application version source, shared by
Python package metadata and `ai-der --version` (`coli` remains an alias).
Use semantic versioning: MAJOR for incompatible changes, MINOR for compatible
features, PATCH for fixes. Releases are tagged `vMAJOR.MINOR.PATCH`; this release
is **v1.1.1**. Internal upstream component metadata is independent.

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
