#!/usr/bin/env bash
# Phase-1B launcher: sensors -> WAM -> logs only (no motion)
set -eo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kairos
# ROS setup.bash references unset vars; disable nounset around sourcing
set +u
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash 2>/dev/null || true
source ~/kairos/scripts/env_libero_franka.sh
set -u

export KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export WAM_URL="${WAM_URL:-http://127.0.0.1:8005}"
export PYTHONPATH="${KAIROS_ROOT}/benchmarks/common:${PYTHONPATH:-}"

exec python "${KAIROS_ROOT}/scripts/dryrun_franka_sensors_to_wam.py" "$@"
