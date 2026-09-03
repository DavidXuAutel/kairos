#!/usr/bin/env bash
# Phase-2 open-loop pick-up-a-pen on lab PC (yao@10.229.20.125).
# Usage:
#   bash run_phase2_pick_pen.sh --plan-only
#   KAIROS_ARM_TOKEN=... bash run_phase2_pick_pen.sh --i-approve-motion --arm-token ...
set -eo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kairos
set +u
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash 2>/dev/null || true
source ~/kairos/scripts/env_libero_franka.sh
set -u

export KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export WAM_URL="${WAM_URL:-http://127.0.0.1:8005}"
export PYTHONPATH="${KAIROS_ROOT}/benchmarks/common:${KAIROS_ROOT}/scripts/phase2:${PYTHONPATH:-}"

exec python "${KAIROS_ROOT}/scripts/phase2/run_pick_pen_open_loop.py" "$@"
