#!/usr/bin/env bash
# Start Kairos WAM on H100 (binds 127.0.0.1:$WAM_PORT).
set -euo pipefail
KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
bash "$KAIROS_ROOT/scripts/bootstrap_kairos_torch27.sh"

# Prefer local /tmp models before env defaults that may point at Ceph home.
export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-/tmp/kairos_models}"
export WAM_PRETRAINED_DIT="${WAM_PRETRAINED_DIT:-$KAIROS_MODEL_DIR/Kairos-model/kairos-agi/kairos-4B-robot-LIBERO-plus/kairos-4B-robot-LIBERO-plus.safetensors}"
# Fallback flat layout used on some pods
if [ ! -f "$WAM_PRETRAINED_DIT" ] && [ -f /tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors ]; then
  export WAM_PRETRAINED_DIT=/tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors
fi

# shellcheck disable=SC1091
source "$KAIROS_ROOT/scripts/env_libero_franka.sh"
export PATH="${KAIROS_TORCH27_VENV:-/tmp/kairos-torch27}/bin:$PATH"
export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-/tmp/kairos_models}"
export WAM_CFG_PATH="${WAM_CFG_PATH:-$KAIROS_ROOT/benchmarks/libero_plus/configs/libero_wam_infer_config_h100.py}"
# Default single GPU (current .23 pod). Dual H100: WAM_GPU_IDS=0,1
export WAM_GPU_IDS="${WAM_GPU_IDS:-0}"
export WAM_PORT="${WAM_PORT:-8005}"
export WAM_EAGER_LOAD_ON_STARTUP=1
export WAM_WORKER_STARTUP_TIMEOUT_SEC="${WAM_WORKER_STARTUP_TIMEOUT_SEC:-1200}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$KAIROS_ROOT/logs"

if [ ! -f "$WAM_PRETRAINED_DIT" ]; then
  echo "[start_wam] missing DiT: $WAM_PRETRAINED_DIT" >&2
  exit 1
fi

PIDFILE="${WAM_PIDFILE:-$KAIROS_ROOT/logs/wam_${WAM_PORT}.pid}"
python -c "import torch,triton,torchvision,os; print(f'torch={torch.__version__} triton={triton.__version__} torchvision={torchvision.__version__} MODEL_DIR={os.environ[\"KAIROS_MODEL_DIR\"]} WAM_GPU_IDS={os.environ.get(\"WAM_GPU_IDS\")}', flush=True)"
cd "$KAIROS_ROOT/benchmarks/common"
# Record pid of this shell's python after exec replacement via wrapper
echo $$ > "$PIDFILE"
exec python -m uvicorn wam_service.server_multi_gpu:app --host 127.0.0.1 --port "$WAM_PORT"
