# Kairos LIBERO-plus Phase 1 Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On `yao@10.229.20.125`, run LIBERO-Plus MuJoCo sim smoke + WAM 1A + real-sensor dry-run 1B with cameras; never command FR3.

**Architecture:** Shared FastAPI `wam_service` on GPU0; sim client in `libero-plus-eval`; 1B ROS2 subscriber script logs actions only (`ENABLE_MOTION=0`).

**Tech Stack:** conda `kairos` + `libero-plus-eval`, ROS2 Humble, RealSense topics, MuJoCo/EGL, Kairos WAM HTTP pickle API.

## Global Constraints

- Host: `yao@10.229.20.125`; SSH key `~/.ssh/franka_ros2_ed25519`
- Robot wired only `10.229.20.91`; no Desk / shopFloor / network changes
- Models already verified DONE under `~/kairos/models/`
- No motion publishers in Phase 1B
- Prefer reuse running `/cam1` `/cam2` topics

---

### Task 0: Install LIBERO-Plus eval stack

**Files:**
- Remote create: `~/kairos/benchmarks/libero_plus/third_party/LIBERO-plus/`
- Remote conda: `libero-plus-eval`

- [x] **Step 1:** Clone LIBERO-plus into `third_party` if missing
- [x] **Step 2:** `conda create -n libero-plus-eval python=3.10 -y` and `pip install -r requirements-eval.txt`
- [x] **Step 3:** Editable install LIBERO-plus; verify EGL device count > 0
- [x] **Step 4:** Smoke `import libero` succeeds

---

### Task 1A: Start WAM + dummy infer

**Files:**
- Use: `benchmarks/common/wam_service/server_multi_gpu.py`
- Use: `benchmarks/common/clients/wam_http_client.py`

- [x] **Step 1:** Launch uvicorn on port 8005 with `KAIROS_MODEL_DIR`, `WAM_PRETRAINED_DIT`, `WAM_CFG_PATH`, `WAM_GPU_IDS=0`
- [x] **Step 2:** Wait `/health` `workers_loaded=true` (install apex/flash-attn only if load fails)
- [x] **Step 3:** Dummy infer (448x256 gray image, state zeros 8) → print action shape

---

### Task 1-Sim: Minimal LIBERO-Plus episode

**Files:**
- Use: `benchmarks/libero_plus/run.sh` with tight overrides OR eval entry with 1 task

- [x] **Step 1:** Export `SERVER_PYTHON`, `EVAL_PYTHON`, model paths, `LIBERO_PKG_ROOT`
- [x] **Step 2:** Run one short category/task; capture logs under `outputs/libero_plus/`
- [x] **Step 3:** Confirm episode finished without crash

---

### Task 1B: Sensor dry-run + camera frames

**Files:**
- Create: `scripts/dryrun_franka_sensors_to_wam.py`
- Create: `scripts/run_dryrun_franka.sh`
- Modify: `DEPLOY.md` (status notes)

- [x] **Step 1:** Script subscribes `/cam1/cam1/color/image_raw` + `/joint_states`, calls WAM, writes `~/kairos/dryrun_logs/<ts>/`
- [x] **Step 2:** Hard-code `ENABLE_MOTION=False`; no control publishers
- [x] **Step 3:** Save ≥3 RGB frames; JSONL with actions
- [x] **Step 4:** Verify no motion topics published by the script

---

### Task Docs: Update DEPLOY.md

- [x] Record Phase 1 commands, ports, log paths, remaining Phase 2 gate
