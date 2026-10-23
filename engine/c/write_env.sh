#!/usr/bin/env bash
# AI-DER: write ai-der.env (the serving environment for all four backends) into the checkout
# and, with --systemd, a user unit at ~/.config/systemd/user/ai-der.service.
# Called by ../../install.sh; runnable alone: AI_DER_DATA=~/ai-der-data engine/c/write_env.sh [--systemd]
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
data=${AI_DER_DATA:-$HOME/ai-der-data}
container=$data/models/qwen38-container
gguf_dir=$data/models/qwen38-flash-next
llamacpp=$data/llama.cpp/build/bin/llama-server
vllm_dir=$data/vllm/qwen38-27b-rtx3090
strata_dir=$data/strata
maya_dir=$data/maya
env_file=$root/ai-der.env
threads=$(( $(nproc) / 2 )); (( threads >= 4 )) || threads=4
say() { echo "AI-DER env: $*"; }

{
cat <<EOF
# AI-DER serving environment, written by install.sh on $(date +%F). Edit freely; rerun
# engine/c/write_env.sh to regenerate. Load it with:  set -a; source ai-der.env; set +a
# (values with spaces are double-quoted: bash's source and systemd's EnvironmentFile both read them)
# ---- AI-DER engine (Qwen3.8-Flash-Next container) ----
COLI_CUDA=1
CUDA_EXPERT_GB=auto
Q38_DENSE_I8=1
Q38_GPU_DENSE=1
Q38_MTP=1
Q38_MS_DECODE=1
Q38_PIN_HOT=16
Q38_PREFETCH=0
Q38_CTX=131072
Q38_KV_Q8=1
Q38_TOP_K=20
Q38_GPU_ROUTER=1
Q38_ROUTER_CHECK=0
HEAT_FILE=$container/qwen38_heat.bin
COLI_QUEUE_TIMEOUT=3600
COLI_SETTINGS_FILE=$HOME/.config/ai-der/settings.json
COLI_ALLOW_RESTART=1
OMP_NUM_THREADS=$threads
OMP_PLACES=cores
OMP_PROC_BIND=close
EOF
if [[ -x $llamacpp ]]; then cat <<EOF
# ---- llama.cpp backend ----
COLI_LLAMACPP_SERVER=$llamacpp
COLI_LLAMACPP_MODEL=$gguf_dir/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
COLI_LLAMACPP_ARGS="-c 131072 -fa on -ctk q8_0 -ctv q8_0 -cmoe --moe-cache-mib 12000 -b 2048 -ub 512 --mmproj $gguf_dir/mmproj-F16.gguf"
COLI_LLAMACPP_SLOTS=$HOME/.cache/ai-der/llamacpp-slots
EOF
else echo "# llama.cpp backend not installed (engine/c/install_llamacpp.sh)"; fi
if [[ -x $vllm_dir/venv/bin/python && -d $vllm_dir/models/Qwen3.8-27B-W4A16-AutoRound-fast ]]; then cat <<EOF
# ---- vLLM backend (Qwen3.8-27B W4A16 stack) ----
COLI_VLLM_CMD="/bin/bash $vllm_dir/single-user/start_qwen.sh"
COLI_VLLM_MODEL=$vllm_dir/models/Qwen3.8-27B-W4A16-AutoRound-fast
COLI_VLLM_MODEL_ID=qwen3.8-27b
COLI_VLLM_PORT=8082
COLI_VLLM_CONTEXT=131072
COLI_VLLM_ENV="PORT=8082 HOST=127.0.0.1 CTX=long MAX_LEN=131072 SPEC=mtp PREFIX_CACHE=1 MAX_SEQS=1 VISION=1 VLLM_SERVER_DEV_MODE=1 'EXTRA_ARGS=--limit-mm-per-prompt {\"image\":{\"count\":4}}'"
COLI_VLLM_IMAGES=4
COLI_VLLM_PREFIXES=$HOME/.cache/ai-der/vllm-prefixes
EOF
else echo "# vLLM backend not installed (engine/c/install_vllm.sh)"; fi
strata_variants=""
for entry in "gsq-iq3s=strata-iq3_s.json|Qwen3.8-Flash-Next GSQ-RCO IQ3_S" \
             "uncensored-iq3s=strata-davidau-abliterated-iq3s.json|Qwen3.8-Flash-Next Abliterated Uncensored GSQ-RCO IQ3_S (DavidAU)" \
             "ud-iq4xs=strata-unsloth-ud-iq4_xs.json|Qwen3.8-Flash-Next UD-IQ4_XS (unsloth)"; do
    id=${entry%%=*}; rest=${entry#*=}; file=${rest%%|*}; label=${rest#*|}
    if [[ -f $strata_dir/$file ]]; then strata_variants+="${strata_variants:+;}$id=$strata_dir/$file|$label"; fi
done
if [[ -x $strata_dir/.venv/bin/python && -n $strata_variants ]]; then cat <<EOF
# ---- Strata backend (Qwen3.8-Flash-Next; models: id=strata-<model>.json|label, the first is the default) ----
COLI_STRATA_CMD="$strata_dir/.venv/bin/python $strata_dir/serve/server.py --engine strata"
COLI_STRATA_PORT=8084
COLI_STRATA_DIR=$strata_dir
COLI_STRATA_RUNTIME=$HOME/.cache/ai-der/strata
COLI_STRATA_VARIANTS="$strata_variants"
EOF
else echo "# Strata backend not installed (engine/c/install_strata.sh)"; fi
maya_variants=""
for entry in "glm-s24=maya-maya-s24.json|GLM-5.3-Flash Maya-S24" \
             "glm-m-derisked=maya-maya-m-derisked.json|GLM-5.3-Flash Maya-M-Derisked" \
             "ds4-s=maya-ds4.json|DeepSeek-V4.1-Flash Maya-S"; do
    id=${entry%%=*}; rest=${entry#*=}; file=${rest%%|*}; label=${rest#*|}
    if [[ -f $maya_dir/$file ]]; then maya_variants+="${maya_variants:+;}$id=$maya_dir/$file|$label"; fi
done
if [[ -x $maya_dir/.venv/bin/python && -n $maya_variants ]]; then cat <<EOF
# ---- Maya backend (GLM-5.3-Flash, DeepSeek-V4.1-Flash; models: id=maya-<model>.json|label) ----
COLI_MAYA_CMD="$maya_dir/.venv/bin/python $maya_dir/serve/server.py --engine strata"
COLI_MAYA_PORT=8085
COLI_MAYA_DIR=$maya_dir
COLI_MAYA_RUNTIME=$HOME/.cache/ai-der/maya
COLI_MAYA_VARIANTS="$maya_variants"
EOF
else echo "# Maya backend not installed (engine/c/install_maya.sh)"; fi
cat <<EOF
# ---- LAN serving (off by default; see README) ----
# COLI_API_KEY=change-me
# COLI_ALLOWED_HOSTS=my-host,192.168.1.10
EOF
} > "$env_file"
mkdir -p "$HOME/.config/ai-der" "$HOME/.cache/ai-der"
say "wrote $env_file"

if [[ ${1:-} == --systemd ]]; then
    unit_dir=$HOME/.config/systemd/user
    mkdir -p "$unit_dir"
    cat > "$unit_dir/ai-der.service" <<EOF
[Unit]
Description=AI-DER inference gateway (Qwen3.8-Flash-Next)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$root
EnvironmentFile=$env_file
ExecStart=$root/.venv/bin/ai-der serve --model $container --host 127.0.0.1 --port 8080 --ngen 32768
Restart=on-failure
RestartSec=5
# the dense backbone load and the VRAM warm start take minutes on a cold page cache
TimeoutStartSec=900
TimeoutStopSec=60

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload 2>/dev/null || true
    say "wrote $unit_dir/ai-der.service; enable with: systemctl --user enable --now ai-der.service"
fi
