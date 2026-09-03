#!/usr/bin/env bash
# Idempotent H100 WAM venv: /tmp/kairos-torch27 (torch 2.7.1+cu126 / triton 3.3.1).
set -euo pipefail
VENV="${KAIROS_TORCH27_VENV:-/tmp/kairos-torch27}"
BASE="${KAIROS_CONDA_BASE:-$HOME/.conda/envs/kairos}"

if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" - <<'PY'
import torch, triton, torchvision
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert triton.__version__ == "3.3.1", triton.__version__
print("torch27 env ready", torch.__version__, triton.__version__, torchvision.__version__)
PY
  exit 0
fi

if [ ! -x "$BASE/bin/python" ]; then
  echo "[bootstrap] missing base python: $BASE/bin/python" >&2
  exit 1
fi

"$BASE/bin/python" -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install \
  "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
import torch, triton, torchvision
assert torch.__version__.startswith("2.7.1"), torch.__version__
assert triton.__version__ == "3.3.1", triton.__version__
print("torch27 env created", torch.__version__, triton.__version__, torchvision.__version__)
PY
