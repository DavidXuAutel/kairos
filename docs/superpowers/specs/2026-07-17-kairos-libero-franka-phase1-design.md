# Kairos LIBERO-plus → FR3 Phase 1 Design

**Date:** 2026-07-17  
**Host:** `yao@10.229.20.125` (RTX 4090 D)  
**Robot:** Franka FR3 @ `10.229.20.91` (wired FCI only; no Desk/network changes)  
**Status:** Approved in chat 2026-07-17

## Goal

Prove the LIBERO-plus World-Action Model (WAM) stack end-to-end in **simulation**, and prove **real camera + joint state → WAM → logged actions** on the lab PC, **without commanding the robot**.

## Non-goals (this phase)

- Open-loop or closed-loop motion on FR3 (Phase 2 / 3)
- Changing Franka Desk / shopFloor / robot network (including via `10.229.66.70`)
- Installing optional accelerators (flash-attn / apex) unless WAM fails to load without them
- Full LIBERO-Plus category sweep (one short sim smoke is enough)

## Constraints

- Prefer existing conda env `kairos` for WAM server (`SERVER_PYTHON`)
- New eval env: `libero-plus-eval` for MuJoCo / robosuite client
- Reuse already-running RealSense ROS topics (`/cam1`, `/cam2`) and `/franka/joint_states`
- Disk is tight (~57G free); avoid redundant large downloads
- Safety: `ENABLE_MOTION=0` hard-coded in any real-robot script; no Franka command publishers

## Architecture

```
┌─────────────────┐     HTTP pickle      ┌──────────────────────┐
│ LIBERO-Plus sim │ ───────────────────► │ wam_service :8005    │
│ (eval conda)    │ ◄─────────────────── │ (kairos conda, GPU0) │
└─────────────────┘     video + action   └──────────────────────┘
                                                      ▲
┌─────────────────┐     same /infer API               │
│ Phase-1B dryrun │ ──────────────────────────────────┘
│ cam + joints    │     log only (no motion)
└─────────────────┘
```

## Steps

### 0. Environment preparation

1. Clone `https://github.com/sylvestf/LIBERO-plus.git` → `benchmarks/libero_plus/third_party/LIBERO-plus` on remote (and mirror locally if needed).
2. Create conda env `libero-plus-eval` (Python 3.10), install `benchmarks/libero_plus/requirements-eval.txt` + editable LIBERO-plus package.
3. Verify EGL: `eglQueryDevicesEXT count > 0` inside eval env.
4. Confirm model paths via `scripts/check_libero_resources.sh` (already DONE for weights).

### 1A. WAM smoke

1. Export:
   - `KAIROS_MODEL_DIR=~/kairos/models`
   - `WAM_PRETRAINED_DIT=.../kairos-4B-robot-LIBERO-plus.safetensors`
   - `WAM_CFG_PATH=.../libero_wam_infer_config.py`
   - `WAM_GPU_IDS=0`, `WAM_PORT=8005`, `WAM_EAGER_LOAD_ON_STARTUP=1`
2. Start `uvicorn wam_service.server_multi_gpu:app` from `benchmarks/common`.
3. Wait until `/health` reports `workers_loaded=true`.
4. Dummy `infer_action` (gray PIL image + zero state dim 8) → expect action tensor shape compatible with LIBERO 7-DoF.

**Success:** health OK + infer returns finite actions.

### 1-Sim. LIBERO-Plus simulation smoke

1. Point `SERVER_PYTHON` / `EVAL_PYTHON` / `KAIROS_MODEL_DIR` / `WAM_PRETRAINED_DIT` / `LIBERO_PKG_ROOT`.
2. Run a **minimal** eval (single category or Hydra overrides for 1 task / few episodes), preferably via `benchmarks/libero_plus/run.sh` or the underlying eval entry with tight limits.
3. Prefer headless EGL; do not allocate extra GPUs for render if avoidable (WAM owns GPU0).

**Success:** ≥1 episode completes without crash; artifacts under `outputs/libero_plus/`.

### 1B. Real sensor dry-run (no motion)

1. Confirm topics: `/cam1/cam1/color/image_raw` (or cam2) and `/franka/joint_states` (or `/joint_states`).
2. New script under `scripts/` (remote + local mirror), e.g. `dryrun_franka_sensors_to_wam.py`:
   - Subscribe image + joints
   - Map to WAM payload (resize to model input; build 8-D state best-effort from joints/EEF/gripper)
   - Call WAM `/infer`
   - Write frames + JSONL actions to `~/kairos/dryrun_logs/<timestamp>/`
   - **Never** create publishers for impedance / gripper / trajectory
3. Save a few RGB frames as camera-capture proof.

**Success:** log directory contains images + action rows; `ros2 topic info` shows no new command pubs from the script.

### Camera capture

- Do **not** restart RealSense if topics already publish.
- If topics are dead, use existing lab launcher (`~/gello_desk/lerobot_record/start_dual_realsense.sh`) rather than inventing a new stack.

## Action / state notes

- LIBERO-plus config: `action_dim=7`, `action_state_dim=8`.
- Sim path uses `libero_plus_dataset_stats.json` for denorm inside eval client.
- Real dry-run Phase 1 only needs **raw model output logged**; precise FR3 denorm/mapping is Phase 2.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `kairos` env missing apex/flash-attn | Try load first; install only if import/load fails |
| Disk pressure | Skip optional assets; prune old dryrun logs |
| EGL / MuJoCo fail | Follow README vendor JSON / `__EGL_VENDOR_LIBRARY_FILENAMES` |
| Watchdog / pgrep self-match | Prefer explicit PIDs for process checks |
| Accidental motion | No motion APIs in 1B script; code review for publishers |

## Deliverables

- This design doc
- Remote: `libero-plus-eval` env + `third_party/LIBERO-plus`
- Running notes appended to `DEPLOY.md`
- `scripts/dryrun_franka_sensors_to_wam.py` (+ tiny launcher shell)
- Log sample under `~/kairos/dryrun_logs/`

## Approval

- Approach approved in chat: Phase 1 = official LIBERO-Plus sim + 1A/1B sensor dry-run, no motion (2026-07-17).
