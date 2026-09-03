#!/usr/bin/env bash
# Preflight for H100 WAM start.
set -euo pipefail
KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
WAM_PORT="${WAM_PORT:-8005}"
VENV="${KAIROS_TORCH27_VENV:-/tmp/kairos-torch27}"
export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-/tmp/kairos_models}"
# shellcheck disable=SC1091
source "$KAIROS_ROOT/scripts/env_libero_franka.sh"

echo "[preflight] nvidia-smi"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [ ! -x "$VENV/bin/python" ]; then
  echo "[preflight] missing venv; run bootstrap_kairos_torch27.sh" >&2
  exit 1
fi

"$VENV/bin/python" - <<'PY'
import torch, triton
assert torch.cuda.is_available(), "CUDA not available"
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert triton.__version__ == "3.3.1", triton.__version__
print("[preflight] torch/triton OK", torch.__version__, triton.__version__)
PY

DIT="${WAM_PRETRAINED_DIT}"
if [ ! -f "$DIT" ] && [ -f /tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors ]; then
  DIT=/tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors
fi
if [ ! -f "$DIT" ]; then
  echo "[preflight] missing DiT weights: $DIT" >&2
  exit 1
fi
echo "[preflight] DiT OK: $DIT"

CFG="${WAM_CFG_PATH}"
test -f "$CFG"
echo "[preflight] cfg OK: $CFG"

if curl -sS --noproxy '*' -m 2 "http://127.0.0.1:${WAM_PORT}/health" >/tmp/wam_health.json 2>/dev/null; then
  echo "[preflight] existing /health:"
  cat /tmp/wam_health.json; echo
else
  echo "[preflight] nothing on :$WAM_PORT (ok if starting fresh)"
fi
echo "[preflight] PASS"
