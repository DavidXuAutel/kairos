#!/usr/bin/env bash
# Minimal LIBERO-Plus episode smoke on H100 (WAM must already be up).
set -euo pipefail
KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export EVAL_PYTHON="${EVAL_PYTHON:-/tmp/libero-plus-eval/bin/python}"
export PATH="$(dirname "$EVAL_PYTHON"):$PATH"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$KAIROS_ROOT/outputs/libero_plus/libero_config}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/tmp/egl_vendor.d}"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$KAIROS_ROOT/benchmarks/libero_plus"
export KAIROS_WAM_ROOT="$KAIROS_ROOT"
export KAIROS_WAM_BENCH_ROOT="$PROJECT_ROOT/kairos_wam"
export LIBERO_PKG_ROOT="$PROJECT_ROOT/third_party/LIBERO-plus"
export PYTHONPATH="$KAIROS_WAM_ROOT:$KAIROS_WAM_BENCH_ROOT:$KAIROS_WAM_BENCH_ROOT/src:$LIBERO_PKG_ROOT:${PYTHONPATH:-}"
export WAM_SKIP_LOAD_ENGINE=1

OUT_DIR="${1:-$KAIROS_ROOT/outputs/libero_plus/smoke_h100_manual}"
mkdir -p "$OUT_DIR"
STATS="$KAIROS_ROOT/benchmarks/libero_plus/libero_plus_dataset_stats.json"
test -f "$STATS"
test -x "$EVAL_PYTHON"

cd "$KAIROS_WAM_BENCH_ROOT"
exec "$EVAL_PYTHON" experiments/libero/eval_libero_single.py \
  model=external_model_adapter \
  model.endpoint=http://127.0.0.1:8005 \
  model.timeout_s=600 \
  model.skip_load_engine=true \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.category_value=null \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.env_num=1 \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.output_dir="$OUT_DIR" \
  MULTIRUN.run_all_categories=false \
  2>&1 | tee "$OUT_DIR/eval_smoke.log"
