# Kairos Phase2 + H100 WAM + LIBERO Full Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Check in H100 WAM launch/smoke scripts, fix LIBERO category evaluation with a local adapter and start a resumable 20-trial full sweep, and add a Phase-2 FR3 safety skeleton that never commands the robot.

**Architecture:** H100 hosts `wam_service` on `127.0.0.1:8005` from `/tmp/kairos-torch27`. Eval workers use a category adapter over upstream LIBERO-Plus (no `category_value` ctor kwarg). Phase-2 modules map/denorm/clamp into a `FakeController` with arming disabled.

**Tech Stack:** bash, FastAPI/uvicorn WAM, PyTorch 2.7.1+cu126, LIBERO-Plus/MuJoCo eval env, pytest, ROS2 Humble (sensor topics only for Phase-2 offline fixtures).

## Global Constraints

- No FR3 motion publishers/actions/services in this plan.
- No Desk / shopFloor / `10.229.66.70` network changes.
- Robot I/O only on wired lab PC when sensors are involved.
- H100: torch **2.7.1+cu126**, triton **3.3.1**, `TRITON_CACHE_DIR=/tmp/triton_cache`.
- Full sweep: **20 trials/task**, resumable background launch (~200,600 episodes).
- Spec: `docs/superpowers/specs/2026-07-21-kairos-phase2-wam-libero-design.md`.

## File map

| Path | Role |
|------|------|
| `scripts/env_libero_franka.sh` | Shared path defaults (host-neutral) |
| `scripts/bootstrap_kairos_torch27.sh` | Idempotent `/tmp/kairos-torch27` |
| `scripts/start_wam_h100.sh` | Start WAM |
| `scripts/stop_wam.sh` | Stop by port/pidfile |
| `scripts/check_wam_preflight.sh` | Preflight checks |
| `scripts/wam_first_frame_smoke.py` | HTTP first-frame smoke |
| `scripts/run_libero_smoke_h100.sh` | One-episode sim smoke |
| `scripts/run_libero_full_sweep_h100.sh` | Resumable full sweep launcher |
| `benchmarks/libero_plus/kairos_wam/experiments/libero/libero_plus_eval_utils.py` | Category adapter |
| `benchmarks/libero_plus/kairos_wam/experiments/libero/run_libero_manager.py` | Use adapter |
| `benchmarks/libero_plus/kairos_wam/experiments/libero/eval_libero_single.py` | Use adapter |
| `benchmarks/libero_plus/kairos_wam/experiments/libero/run_libero_plus_by_category.py` | Fix `--suite` CLI |
| `benchmarks/libero_plus/kairos_wam/experiments/libero/test_libero_plus_eval_utils.py` | Unit tests |
| `scripts/phase2/*.py` | State/action/limits/fake/arming/offline |
| `scripts/phase2/test_phase2_offline.py` | Offline tests |
| `DEPLOY.md` | Durable procedure + dated status |

---

### Task 1: Check in H100 WAM scripts

**Files:**
- Create: `scripts/env_libero_franka.sh`
- Create: `scripts/bootstrap_kairos_torch27.sh`
- Create: `scripts/start_wam_h100.sh`
- Create: `scripts/stop_wam.sh`
- Create: `scripts/check_wam_preflight.sh`
- Create: `scripts/wam_first_frame_smoke.py`
- Create: `scripts/run_libero_smoke_h100.sh`
- Modify: `DEPLOY.md`

**Interfaces:**
- Produces: startable WAM via `bash scripts/start_wam_h100.sh`; smoke via `python scripts/wam_first_frame_smoke.py`

- [x] **Step 1: Write host-neutral env + bootstrap + start/stop/preflight/smoke scripts** (adapt from remote H100 copies; use `$KAIROS_ROOT` / `$HOME/kairos`, never hardcode `a25689` home in Python).

- [x] **Step 2: Sync scripts to H100 `~/kairos/scripts/`**

- [x] **Step 3: Verify on H100**

```bash
ssh -p 31893 a25689@10.239.121.25 'curl -s --noproxy "*" http://127.0.0.1:8005/health'
# if down: setsid bash ~/kairos/scripts/start_wam_h100.sh > ~/kairos/logs/wam_h100.log 2>&1 < /dev/null &
# then: /tmp/kairos-torch27/bin/python ~/kairos/scripts/wam_first_frame_smoke.py
```

Expected: `workers_loaded=true`; smoke prints action shape.

- [x] **Step 4: Update DEPLOY.md durable H100 procedure**

---

### Task 2: Category adapter + unit tests

**Files:**
- Modify: `benchmarks/libero_plus/kairos_wam/experiments/libero/libero_plus_eval_utils.py`
- Create: `benchmarks/libero_plus/kairos_wam/experiments/libero/test_libero_plus_eval_utils.py`

**Interfaces:**
- Produces:
  - `build_suite(suite_name: str, category_value: str | None) -> SuiteView`
  - `SuiteView.n_tasks`, `.get_task(i)`, `.get_task_init_states(i)`, `.source_task_id(i)`
  - `libero_mix` = deterministic concat of four base suites filtered by category when category set

- [x] **Step 1: Write failing tests** for: no-category base suite; category filter stable IDs; mix aggregate; spaces in category names; unknown suite raises.

- [x] **Step 2: Run tests — expect FAIL**

```bash
cd benchmarks/libero_plus/kairos_wam && python -m pytest experiments/libero/test_libero_plus_eval_utils.py -v
```

- [x] **Step 3: Implement adapter** using upstream ctors without `category_value`; filter via `task_classification.json`; virtual mix.

- [x] **Step 4: Run tests — expect PASS**

---

### Task 3: Wire adapter into manager + worker + CLI

**Files:**
- Modify: `run_libero_manager.py` (`create_task_file`)
- Modify: `eval_libero_single.py` (suite construction)
- Modify: `run_libero_plus_by_category.py` (`--suite` replace default; unknown suite error)

- [x] **Step 1: Replace `benchmark_dict[name](category_value=...)` with `build_suite(...)`**

- [x] **Step 2: Fix CLI `--suite` to overwrite default list**

- [x] **Step 3: On H100, create-only task list for one category**

```bash
# create_only path or by-category --create-lists-only
# Expected: no TypeError; rows look like suite,task_id,category
```

- [x] **Step 4: One-episode smoke with non-null category** (base suite) and one virtual mix episode

---

### Task 4: Resumable full sweep launcher (20 trials)

**Files:**
- Create: `scripts/run_libero_full_sweep_h100.sh`
- Modify: `DEPLOY.md`

**Interfaces:**
- Consumes: WAM healthy; adapter; existing `run_libero_parallel_test.sh` / by-category runner
- Produces: background job + log dir under `outputs/libero_plus/full_sweep_*`

- [x] **Step 1: Write launcher** with `NUM_TRIALS=20`, resume by skipping tasks that already have success result JSON, durable status under `$HOME/kairos/outputs/...`

- [x] **Step 2: Preflight create-all task lists**

- [x] **Step 3: Start background full sweep on H100**; record PID and log path in `DEPLOY.md` / status file

- [x] **Step 4: Confirm progress artifacts appear within first completed tasks**

---

### Task 5: Phase-2 offline skeleton + tests

**Files:**
- Create: `scripts/phase2/state_builder.py`
- Create: `scripts/phase2/action_pipeline.py`
- Create: `scripts/phase2/limits.py`
- Create: `scripts/phase2/fake_controller.py`
- Create: `scripts/phase2/arming.py`
- Create: `scripts/phase2/offline_runner.py`
- Create: `scripts/phase2/test_phase2_offline.py`
- Modify: `DEPLOY.md`

**Interfaces:**
- Produces: `build_proprio(eef_pos, eef_axisangle, gripper) -> (8,) float32`
- Produces: `denormalize_actions(raw, stats) -> actions`
- Produces: `clamp_eef_delta(dx, limits) -> clamped`
- Produces: `ArmingGate` default disarmed; `FakeController` records only
- Hard rule: no `create_publisher` / Franka command imports in these modules

- [x] **Step 1: Write failing offline tests** (shape, non-finite reject, clamp, disarmed, denorm fixture)

- [x] **Step 2: Implement modules minimally**

- [x] **Step 3: `python -m pytest scripts/phase2/test_phase2_offline.py -v` PASS**

- [x] **Step 4: Replay one Phase-1B JSONL if available on 125 into audit log; else synthetic fixture**

---

### Task 6: Final verification notes

- [x] **Step 1: Confirm WAM health + smoke still OK on H100**
- [x] **Step 2: Confirm full sweep running or resumable with documented command**
- [x] **Step 3: Confirm Phase-2 tests pass and DEPLOY.md states hardware motion still gated**

---

## Self-review

1. Spec coverage: A/B/C workstreams each have tasks; full sweep 20 trials + resume covered; no-motion Phase 2 covered.
2. No TBD placeholders in steps.
3. Adapter interface names consistent across Tasks 2–4.
