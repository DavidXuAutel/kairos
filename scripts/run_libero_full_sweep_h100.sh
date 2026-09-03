#!/usr/bin/env bash
# Resumable LIBERO-Plus full sweep on H100.
# libero_mix × all 7 categories × NUM_TRIALS (default 20).
# Requires multi-GPU WAM on 127.0.0.1:8005 (default GPUs 0,1). Eval shares those GPUs.
set -euo pipefail

KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export LIBERO_PKG_ROOT="${LIBERO_PKG_ROOT:-$KAIROS_ROOT/benchmarks/libero_plus/third_party/LIBERO-plus}"
export EVAL_PYTHON="${EVAL_PYTHON:-/tmp/libero-plus-eval/bin/python}"
export NUM_TRIALS="${NUM_TRIALS:-20}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/tmp/egl_vendor.d}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$KAIROS_ROOT/outputs/libero_plus/libero_config}"
export HYDRA_FULL_ERROR=1
export WAM_SKIP_LOAD_ENGINE=1
export PROJECT_ROOT="$KAIROS_ROOT/benchmarks/libero_plus"
export KAIROS_WAM_ROOT="$KAIROS_ROOT"
export KAIROS_WAM_BENCH_ROOT="$PROJECT_ROOT/kairos_wam"
export PYTHONPATH="$KAIROS_WAM_ROOT:$KAIROS_WAM_BENCH_ROOT:$KAIROS_WAM_BENCH_ROOT/src:$LIBERO_PKG_ROOT:${PYTHONPATH:-}"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8005}"
# Dual-GPU eval slots sharing cards with WAM. MuJoCo EGL ~0.6GB/worker; raise to fill HBM.
# Capacity ≈ 2 × MAX_TASKS_PER_GPU (default 2×24=48). Debug: CUDA_VISIBLE_DEVICES=1 MAX_TASKS_PER_GPU=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MULTIRUN_NUM_GPUS="${MULTIRUN_NUM_GPUS:-2}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-24}"
test -f "$LIBERO_CONFIG_PATH/config.yaml"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${1:-$KAIROS_ROOT/outputs/libero_plus/full_sweep_${STAMP}}"
mkdir -p "$OUT_DIR" "$KAIROS_ROOT/logs"
LOG_FILE="$OUT_DIR/sweep_launcher.log"
STATS="$KAIROS_ROOT/benchmarks/libero_plus/libero_plus_dataset_stats.json"
test -f "$STATS"
test -x "$EVAL_PYTHON"

curl -sS --noproxy '*' -m 15 http://127.0.0.1:8005/health | tee "$OUT_DIR/wam_health.json"
python3 - <<PY
import json
p=json.load(open("$OUT_DIR/wam_health.json"))
assert p.get("workers_loaded") is True, p
print("WAM ready", flush=True)
PY

python3 - <<PY
import json, time
from pathlib import Path
out=Path("$OUT_DIR")
payload={
  "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "out_dir": str(out),
  "num_trials": int("$NUM_TRIALS"),
  "suites": ["libero_mix"],
  "multirun_num_gpus": int("$MULTIRUN_NUM_GPUS"),
  "max_tasks_per_gpu": int("$MAX_TASKS_PER_GPU"),
  "model_endpoint": "$MODEL_ENDPOINT",
  "resume": True,
  "command": "run_libero_plus_by_category.py --suite libero_mix",
}
(out/"sweep_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

cd "$KAIROS_WAM_BENCH_ROOT"
exec "$EVAL_PYTHON" experiments/libero/run_libero_plus_by_category.py \
  --output-dir="$OUT_DIR" \
  --suite libero_mix \
  -- \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/nothing/to/load \
  model=external_model_adapter \
  "model.endpoint=${MODEL_ENDPOINT}" \
  model.skip_load_engine=true \
  model.timeout_s=600 \
  "EVALUATION.dataset_stats_path=${STATS}" \
  "EVALUATION.num_trials=${NUM_TRIALS}" \
  "MULTIRUN.num_gpus=${MULTIRUN_NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  2>&1 | tee -a "$LOG_FILE"
