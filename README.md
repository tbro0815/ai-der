# AI-DER 1.5.2

**Artificial Intelligence Distributed Engram Runner** serves large
mixture-of-experts models from a single 24 GB GPU. One gateway with OpenAI- and
Anthropic-compatible APIs fronts five inference engines. A browser dashboard
covers chat, engine and model selection, and live telemetry. Models far larger
than VRAM run from RAM and NVMe: up to DeepSeek-V4.1-Flash at 340 GB on a
128 GB machine.

![The AI-DER dashboard: runtime and inference settings, live engine log](engine/web/screenshots/dashboard.webp)

## Capabilities

**One API for every engine.** Clients keep a single provider configuration and
one model id, `aider-omni-flash`. The engine and model behind it are chosen in
the dashboard and take effect at the next restart.

- OpenAI Chat Completions, Completions and Responses; Anthropic Messages with
  `count_tokens`. Streaming, usage with cached prompt tokens, `/v1/models` with
  the served context window.
- Tool calls on every engine, parsed by the gateway from the model's native
  format.
- Thinking on/off with reasoning effort. A thinking budget on every engine
  except Maya (on vLLM for text turns).
- Image input on every engine with a vision tower: up to 32 images per request,
  older ones folded into notes so the prompt stays cache-stable.
- Prompt caching across turns and conversations on every engine, with
  persisted prefixes where the engine allows it.
- Bounded request queue, a stall guard that restarts a hung engine, and the
  dashboard's own web search tool (Serper).
- Tested with Claude Code, Codex CLI and agent harnesses that speak the
  OpenAI or Anthropic APIs.

**Five engines, one at a time.** The selected engine owns the GPU.

| Engine | Models | Context | Strengths |
|---|---|---|---|
| AI-DER | Qwen3.8-Flash-Next UD-IQ4_XS | 131K | Own CUDA engine: experts tiered across VRAM, RAM and NVMe with prefetch, MTP drafting, multi-slot KV with a persistent prompt pool, device router, own vision tower |
| llama.cpp | Qwen3.8-Flash-Next UD-IQ4_XS | 131K, 262K | MoE expert cache in VRAM, prompt cache, persisted stable prefix |
| vLLM | Qwen3.8-27B W4A16 (dense) | 80K, 131K, 190K | Highest decode speed, MTP drafting, prefix cache |
| Strata | Qwen3.8-Flash-Next GSQ-RCO IQ3_S, abliterated GSQ-RCO IQ3_S, UD-IQ4_XS | 131K, 262K | Fastest prefill, flat decode up to 184K tokens, experts in pinned RAM with a ranked VRAM cache |
| Maya | GLM-5.3-Flash Maya-S24 and Maya-M-Derisked, DeepSeek-V4.1-Flash Maya-S, MiniMax-M3.1-Flash Maya-XS | 131K, 262K | GLM-5.3-Flash, DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash on one GPU; DeepSeek's experts beyond RAM read from NVMe |

**Dashboard.** Chat with images and web search, context profile, engine and
model selection with restart, inference defaults, a live engine log, hardware
panel (VRAM, RAM, page cache), a speed meter for dashboard and API runs, and an
expert-routing view for the AI-DER engine. English, German, Italian and
Chinese (simplified and traditional).

## Performance

Reference server: AMD Ryzen 9 3900X (12 cores), 128 GB DDR4 (dual channel,
121 GiB usable), NVIDIA RTX 3090 Ti 24 GB on PCIe 4.0 x16, models on a
PCIe 4.0 x4 NVMe SSD, Ubuntu 26.04.

Prompt processing (pp) and token generation (tg), in tokens per second. Straight
to each engine, greedy, thinking off, 512 output tokens, a cold prompt of the
given length; tg is given for a prose and a coding task. Agent run: one full
pass of a 70-task coding benchmark driven by an agent harness with thinking on,
as output tokens over the time spent in attempts (tool runs included), and the
length of the job. Pauses between attempts and attempts aborted by gateway or
harness failures are left out.

| Engine | Model | Weights | pp 6K | pp 68K | tg 6K | tg 68K | Agent run |
|---|---|---|---|---|---|---|---|
| Strata | Qwen3.8-Flash-Next GSQ-RCO IQ3_S | 84 GB | 1,957 | 2,714 | 66-102 | 74-107 | 94 tok/s, 2.2 h |
| Strata | Qwen3.8-Flash-Next abliterated GSQ-RCO IQ3_S | 84 GB | | | 98-128 ¹ | | 94 tok/s, 2.6 h |
| Strata | Qwen3.8-Flash-Next UD-IQ4_XS | 95 GB | 1,623 | 2,627 | 56-73 | 55-80 | 77 tok/s, 3.2 h |
| llama.cpp | Qwen3.8-Flash-Next UD-IQ4_XS | 95 GB | 205 | 251 | 35 | 30 | |
| AI-DER | Qwen3.8-Flash-Next UD-IQ4_XS | 95 GB | 100 ² | 90 ² | 27-33 ³ | | |
| vLLM | Qwen3.8-27B W4A16 AutoRound | 15 GB | 1,450-3,800 ⁴ | | 62-157 ⁴ | | 73-82 tok/s, 5.2-7.8 h |
| Maya | GLM-5.3-Flash Maya-S24 | 95 GB | 1,196 | 1,296 | 23-27 | 22-26 | 20 tok/s, 16.8 h |
| Maya | GLM-5.3-Flash Maya-M-Derisked | 116 GB | 1,043 | 1,199 | 17-21 | 17-20 | |
| Maya | DeepSeek-V4.1-Flash Maya-S | 340 GB | 352 | 555 | 8-12 | 10-14 | 11 tok/s, 14.4 h |
| Maya | MiniMax-M3.1-Flash Maya-XS | 93 GB | 207 | 210 | 16-17 | 14-17 | |

1. Not benchmarked separately: replies of 500 tokens or more in agent use,
   10th to 90th percentile (median 115).
2. At 2K and 32K tokens.
3. With a warm expert cache. Fresh agent conversations start at 16-18 tok/s
   while the cache adapts.
4. Low end with the fp8 KV cache of the 131K and 190K profiles, high end with
   the bf16 KV cache of the 80K profile. tg by task type: JSON and German
   lowest, code highest.

Long context: Strata reads a 184K-token prompt at 3,755-3,922 tok/s and decodes
as fast as at 6K. Maya-S24 decodes 22-26 tok/s at 181K. llama.cpp reads the
full 262K window at 177 tok/s and decodes 7 tok/s at that depth.

## System requirements

- PC running **Ubuntu 24.04 or 26.04 LTS, x86_64**, with a CUDA-compatible NVIDIA driver.
- **128 GB RAM**, DDR4 or newer.
- **NVIDIA RTX 3090 or RTX 3090 Ti**, 24 GB VRAM.
- **PCIe 4.x or newer NVMe SSD as the system drive**, with about **410 GB free** for
  the default installation (weights 95 GB, engine container 61 GB, MTP draft 8 GB,
  vLLM stack and weights 40 GB, llama.cpp 3 GB, CUDA 13 toolkit 4 GB, Strata and its
  two IQ3_S models 180 GB), plus about 660 GB for the optional Maya backend. Keep
  the data directory on NVMe.
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
git checkout v1.5.2
./install.sh --check
./install.sh
```

Run the installer as your normal user. The default installation does everything
except Maya:

1. **Engine and dashboard.** GCC/G++ 13, CMake, Make, Python 3.12 or newer,
   venv/development headers, curl, patch and xz via apt; a `.venv` with AI-DER in
   editable mode plus NumPy, Pillow, gguf, the Hugging Face CLI and uv; Node.js
   >=22.12 (system or a checksum-verified Node 22 LTS in `.venv/node`); the locked
   dashboard build; the AI-DER CUDA engine for Ampere `sm_86`.
2. **Weights and container** (`engine/c/install_models.sh`): the Qwen3.8-Flash-Next
   UD-IQ4_XS GGUF shards and vision projector (Unsloth), the tokenizer and chat
   template (Qwen), the MTP draft tensors fetched by HTTP range requests, then the
   engine container conversion and the MTP container (BF16 to Q8_0). Tens of
   minutes after the download; every step resumes on rerun.
3. **CUDA 13 toolkit** (`engine/c/install_cuda13.sh`): NVIDIA's pip wheels assembled
   into `~/ai-der-data/cuda13` for the llama.cpp, Strata and Maya builds; the system toolkit
   is left alone. On glibc 2.41+ (Ubuntu 26.04) the view carries a private copy of
   `crt/math_functions.h` whose `rsqrt`/`rsqrtf` declarations match glibc's.
4. **llama.cpp backend** (`engine/c/install_llamacpp.sh`): `llama-server` built with
   CUDA at a pinned revision that carries the MoE expert cache (`-cmoe
   --moe-cache-mib 12000`: every expert in RAM, the hot ones in VRAM).
5. **vLLM backend** (`engine/c/install_vllm.sh`): the syv-ai HyperQwen stack (Qwen3.8-27B
   on vLLM 0.31.0) at a pinned revision in its own Python 3.12 venv, the published W4A16 weights, the
   one-time CPU preparation (int8 heads, int8 MTP draft, draft vocabulary, fast
   variant) and the stack's vLLM patches.
6. **Strata backend** (`engine/c/install_strata.sh`): [Strata](https://github.com/Niko1221/Strata),
   an inference engine for Qwen3.8-Flash-Next, built from source at a pinned revision,
   with three models: ISTA-DASLab's GSQ-RCO IQ3_S pack, DavidAU's abliterated
   (uncensored) GSQ-RCO IQ3_S and the UD-IQ4_XS GGUF from step 2. `AI_DER_STRATA_MODELS=gsq,uncensored,ud`
   picks a subset.
7. **Maya backend** (`engine/c/install_maya.sh`, only with `--with-maya`):
   [Project Maya](https://github.com/mw00/project-maya), Strata's fork for
   GLM-5.3-Flash, DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash, built from source at a
   pinned revision, with its own setup per model (download with sha256 checks, pack,
   vision encoder): GLM-5.3-Flash Maya-S24 and Maya-M-Derisked, DeepSeek-V4.1-Flash
   Maya-S and MiniMax-M3.1-Flash Maya-XS (about 660 GB together).
   `AI_DER_MAYA_MODELS=glm-s24,glm-m-derisked,ds4-s,minimax-xs` picks a subset.
8. **`ai-der.env`** in the checkout: the serving environment for all installed
   backends. `--systemd` also writes `~/.config/systemd/user/ai-der.service`.

Everything lands under `~/ai-der-data` (`--data-dir DIR` changes it). Flags:
`--core-only` installs the engine and dashboard only, `--skip-models`,
`--skip-llamacpp`, `--skip-vllm` and `--skip-strata` drop one component, `--with-maya` adds
the Maya backend, `--no-apt` skips the sudo
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

Open `http://127.0.0.1:8080` for the dashboard. The default engine is AI-DER.
The Extra section of the dashboard selects, for the next restart, the context
profile first: `standard` (131K, every engine), `fast` (80K, vLLM only),
`vllm-190k` (190K, vLLM only) or `long` (262K, llama.cpp, Strata or Maya). The
engine list then offers the engines that serve that profile, and the model list
the models of the chosen engine. With `--systemd`:
`systemctl --user enable --now ai-der.service`.

**API.** Base URL `http://127.0.0.1:8080/v1`, model `aider-omni-flash`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "aider-omni-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

Anthropic clients use the base URL without `/v1`, for example
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080` for Claude Code. `COLI_MODEL_ALIASES`
accepts other model names as aliases (`claude-*` for Claude Code's names). The
running engine and weights are reported in `/health` (`backend`).

**Web search in the chat.** Enter a [Serper](https://serper.dev) API key in the
sidebar's Web Search section and switch the tool on: the dashboard then declares
one function tool, `web_search`, on every request of the chat, so the model can
search from the first turn. The key stays in the browser's local storage and
travels with each search to `/v1/tools/web_search`, where the gateway relays the
query to Serper and returns titles, links and snippets as a tool message; the
server never stores the key.

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
is **v1.5.2**. Internal upstream component metadata is independent.

```bash
.venv/bin/ai-der --version
.venv/bin/python engine/c/convert_qwen38.py --self-test
python3 engine/c/tests/test_release.py
```

## Credits and license

The AI-DER engine began as a fork of [colibrì](https://github.com/JustVugg/colibri)
(Apache-2.0) and carries its own Qwen3.8 engine, container converter, GPU
routing and inference kernels; see [LICENSE](engine/LICENSE),
[upstream provenance](engine/UPSTREAM.txt) and [third-party notices](engine/THIRD_PARTY_NOTICES.md).
The other engines are separate projects built from source by the installer:
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), the syv-ai HyperQwen
vLLM stack, [Strata](https://github.com/Niko1221/Strata) (MIT) and
[Project Maya](https://github.com/mw00/project-maya) (MIT). GGUF and quantization
formats build on llama.cpp/ggml (MIT). Model weights come from their authors
under their own licenses: Qwen (Qwen3.8), Unsloth (UD-IQ4_XS), ISTA-DASLab
(GSQ-RCO IQ3_S), DavidAU (abliterated IQ3_S), Z.ai (GLM-5.3-Flash), DeepSeek
(DeepSeek-V4.1-Flash) and MiniMax (MiniMax-M3.1-Flash, MiniMax Community License:
non-commercial use free, commercial use with conditions), quantized for Maya by its
authors. Original copyright and license notices are retained.
