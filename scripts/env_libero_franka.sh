#!/usr/bin/env bash
# Shared Kairos / LIBERO-plus path defaults (host-neutral).
# Override any variable before sourcing.
set +u
export KAIROS_ROOT="${KAIROS_ROOT:-$HOME/kairos}"
export KAIROS_MODEL_DIR="${KAIROS_MODEL_DIR:-$KAIROS_ROOT/models}"
export PYTHONPATH="${KAIROS_ROOT}:${KAIROS_ROOT}/benchmarks/common:${PYTHONPATH:-}"
export WAM_CFG_PATH="${WAM_CFG_PATH:-$KAIROS_ROOT/benchmarks/libero_plus/configs/libero_wam_infer_config_h100.py}"
export WAM_PRETRAINED_DIT="${WAM_PRETRAINED_DIT:-$KAIROS_MODEL_DIR/Kairos-model/kairos-agi/kairos-4B-robot-LIBERO-plus/kairos-4B-robot-LIBERO-plus.safetensors}"
export WAM_PORT="${WAM_PORT:-8005}"
export WAM_GPU_IDS="${WAM_GPU_IDS:-0}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
echo "KAIROS_ROOT=$KAIROS_ROOT"
echo "KAIROS_MODEL_DIR=$KAIROS_MODEL_DIR"
echo "WAM_PRETRAINED_DIT=$WAM_PRETRAINED_DIT"
