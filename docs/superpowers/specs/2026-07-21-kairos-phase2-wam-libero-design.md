# Kairos Phase 2 Skeleton + H100 WAM Hardening + LIBERO Full Sweep

**Date:** 2026-07-21  
**Status:** Approved in chat 2026-07-21  
**Builds on:** Phase 1 design (`2026-07-17-kairos-libero-franka-phase1-design.md`) — Phase 1 complete

## Goal

Deliver three coordinated workstreams in this order:

1. **H100 WAM hardening** — reproducible start / preflight / first-frame smoke checked into the repo and verified on the live H100 host.
2. **LIBERO-Plus full evaluation** — fix `category_value` compatibility via a local adapter, then launch a **resumable** full sweep at **20 trials/task** against H100 WAM.
3. **Phase 2 FR3 safety skeleton** — mapping, limits, fake controller, and offline unit tests; **never command the robot** in this phase.

## Non-goals

- Open-loop or closed-loop motion on FR3 (requires a later explicit approval beyond this spec).
- Changing Franka Desk / shopFloor / robot network (including via `10.229.66.70`).
- Controlling the robot from any remote GPU pod; robot I/O stays on the wired lab PC (`yao@10.229.20.125` ↔ `10.229.20.91`).
- A800 recovery as a hard dependency (optional later; this phase targets H100).
- Patching upstream LIBERO-Plus constructors in an unpinned checkout.
- Gripper actuation on hardware (gripper remains disabled in any future first motion test; this phase has no hardware motion at all).

## Constraints

- H100 SSH (current): `a25689@10.239.121.25:31893` — treat ports as ephemeral; scripts must be host/port parameterized.
- WAM binds `127.0.0.1:8005`; lab PC reaches it via SSH tunnel when needed.
- Ceph home is durable; **`/tmp` is local and wiped on pod recreate** — models, torch env, Triton cache, and eval env must be re-materializable from scripts.
- H100 WAM Python: `/tmp/kairos-torch27` (torch **2.7.1+cu126** / matching triton); **not** torch 2.6.0 (known first-frame abort).
- Eval Python: `/tmp/libero-plus-eval`.
- Mandatory: `TRITON_CACHE_DIR=/tmp/triton_cache` (never `~/.triton` on Ceph).
- Full sweep scale: **~10,030 tasks × 20 trials = ~200,600 episodes**; must be background + resumable; may run for weeks.
- Phase 2 code defaults permanently to dry-run / disarmed; no ROS control publishers, action clients, or service clients that can move hardware.

## Architecture

```
┌────────────────────────────┐     HTTP pickle      ┌──────────────────────────┐
│ LIBERO full sweep (eval)   │ ───────────────────► │ H100 wam_service :8005   │
│ category adapter + resume  │ ◄─────────────────── │ /tmp/kairos-torch27      │
└────────────────────────────┘                      └──────────────────────────┘

┌────────────────────────────┐
│ Phase-2 skeleton (lab PC)  │  sensors → map → denorm → clamp → FakeController
│ ENABLE_MOTION=False        │  unit tests only; no FR3 command interfaces
└────────────────────────────┘
```

---

## Workstream A — H100 WAM hardening

### Problem

Operational scripts (`start_wam_h100.sh`, `wam_first_frame_smoke.py`, `run_libero_smoke_h100.sh`, bootstrap helpers) exist primarily on the remote host / `/tmp`. The local tree is untracked and incomplete. Pod recreate currently requires tribal knowledge.

### Design

Check into `scripts/` (host-neutral, env-overridable):

| Script | Responsibility |
|--------|----------------|
| `scripts/env_libero_franka.sh` | Shared path defaults (`KAIROS_MODEL_DIR`, DiT path, cfg path); no host IPs hard-coded as required |
| `scripts/bootstrap_kairos_torch27.sh` | Idempotent create/repair of `/tmp/kairos-torch27` from a pinned requirements manifest |
| `scripts/start_wam_h100.sh` | Export caches, assert DiT exists, `uvicorn` on `127.0.0.1:$WAM_PORT` |
| `scripts/wam_first_frame_smoke.py` | Gray image + batched zero state → print action shape; fail non-zero on error |
| `scripts/check_wam_preflight.sh` | CUDA / torch-triton pair / model files / port free / optional `/health` |
| `scripts/stop_wam.sh` | Graceful kill by port / pidfile |

Also add a short **dated host-status** subsection in `DEPLOY.md` separate from durable procedure.

### Success criteria

1. From a clean process state on H100: preflight → start → `/health` `workers_loaded=true`.
2. First-frame smoke returns a finite action tensor (shape compatible with LIBERO 7-DoF horizon).
3. Scripts live in the repo and run with env overrides only (no hard dependency on A800).

---

## Workstream B — LIBERO category adapter + resumable full sweep

### Problem

Harness calls `BenchmarkClass(category_value=...)`, but installed LIBERO-Plus constructors only accept `task_order_index=0`. Base suites and `libero_mix` both fail. Upstream `LIBERO_MIX` additionally lacks `task_maps['libero_mix']`.

Classification file lists **~10,030** category-tagged variants across four base suites and seven categories. Current successful smokes required `EVALUATION.category_value=null`.

### Design — local category adapter (recommended)

Add harness-owned helpers in `benchmarks/libero_plus/kairos_wam/experiments/libero/libero_plus_eval_utils.py` (or sibling module):

1. Instantiate upstream suites **without** `category_value`.
2. Load `task_classification.json` via `LIBERO_PKG_ROOT`.
3. Match classification entries to `get_task_names()`; filter by category; keep **stable source task indices**.
4. Implement virtual `libero_mix` as a **deterministic aggregate** of the four base suites (do not rely on broken upstream `LIBERO_MIX`).
5. Expose only what workers need: `n_tasks`, `get_task(i)`, `get_task_init_states(i)`.

Replace constructor call sites in:

- `run_libero_manager.py` (`create_task_file`)
- `eval_libero_single.py` (suite construction)

Also fix CLI footguns discovered in exploration (minimal set required for reliable full launch):

- `--suite` must **replace** default, not append to `["libero_mix"]`.
- Unknown suite names must **error**, not silently fall back to all four base suites.
- Task-list generation must emit `suite,task_id,category` rows consumable by the existing parallel scheduler.

### Full sweep policy

| Parameter | Value |
|-----------|-------|
| Suites | Four base suites + virtual `libero_mix` aggregate as configured for official matrix; default launch uses the harness’s intended full category matrix via `run_all_categories` / by-category runner |
| Trials | **20** per task |
| Episodes | ~200,600 |
| Launch | Background, **resumable** (skip completed `gpu*_task*_results.json` or equivalent status files) |
| WAM | Existing H100 `:8005` (GPU0); eval workers on remaining GPU(s) |
| Progress | Timestamped log dir under `outputs/libero_plus/`; periodic summary via existing `summarize_results.py` when enough results exist |

Preflight before long run:

1. Adapter create-only task list for all seven categories without constructor exceptions.
2. Task counts reconcile with classification.
3. One base-suite + one virtual-mix single-episode smoke with non-null category.
4. Then start resumable full launch.

### Success criteria

1. Unit tests cover adapter filtering, stable IDs, suite resolution, and mix aggregation.
2. Create-only full task list succeeds on H100.
3. Single-episode category smoke succeeds for a base suite and for virtual mix.
4. Full sweep process is running (or checkpointed) with documented resume command; progress artifacts visible.

---

## Workstream C — Phase 2 FR3 safety skeleton (no motion)

### Problem

Repository has sensor→WAM logging only. Real dry-run state incorrectly uses the first eight joint positions; actions are logged **raw** (not denormalized). There is no controller, limit stack, watchdog, or arming token.

### Design

New modules under `scripts/phase2/` (or `benchmarks/franka_phase2/`), all importable without ROS when testing pure math:

| Component | Responsibility |
|-----------|----------------|
| `state_builder.py` | Build 8-D proprio `[xyz(3), axis-angle(3), gripper(2)]` from recorded / fake EEF+gripper messages (same layout as `eval_libero_single.py`) |
| `action_pipeline.py` | Normalize state / denormalize actions using `libero_plus_dataset_stats.json`; optional gripper binarize flag (off for HW path) |
| `limits.py` | Per-step Δxyz / Δrot clamps, absolute workspace box, finite checks, stale-age rejection |
| `fake_controller.py` | Records commanded targets; never opens ROS pubs |
| `arming.py` | One-shot arm token; default **disarmed**; consuming token still does not create real pubs in this phase |
| `offline_runner.py` | Replay dry-run JSONL/frames → pipeline → fake controller → audit log |

Hard rules:

- Default `ENABLE_MOTION=False` (or equivalent `armed=False`).
- This phase’s entrypoints **must not** import or construct Franka command publishers / action clients / controller switch services.
- Unit tests prove: wrong state shape rejected; non-finite rejected; clamp applied; disarmed path never calls `FakeController.send` in a way that would map to hardware; denorm matches sim helper within tolerance on a fixture.

### Success criteria

1. Pytest (or lightweight unittest) suite green offline on a developer machine / lab PC without robot motion.
2. Replay of an existing Phase-1B log produces denormalized, clamped action audit JSON.
3. `DEPLOY.md` Phase-2 section states clearly: skeleton complete; **hardware motion still gated**.

---

## Execution order

1. Workstream A (hours) — unblock long eval stability.
2. Workstream B (days to weeks for the sweep itself) — start after A smoke green.
3. Workstream C (parallelizable after A starts; no dependence on full sweep completion) — code + tests only.

## Risks

| Risk | Mitigation |
|------|------------|
| Full sweep runtime / cost | Resumable launcher; monitor H100 uptime; do not require finishing before C |
| Pod wipe mid-sweep | Task status on durable home path; WAM/env rebuild via A scripts |
| Upstream LIBERO drift | Adapter isolates category logic; pin documented `LIBERO_PKG_ROOT` commit when known |
| Accidental FR3 motion | No motion APIs in C; code review for publishers; explicit later approval for armed HW |
| Wrong proprio if someone arms later | Document that joints≠EEF; C builds correct 8-D path from the start |

## Deliverables

- This design doc
- Checked-in H100 WAM scripts + `DEPLOY.md` durable procedure update
- Category adapter + tests + resumable full-sweep launch notes/logs
- Phase-2 skeleton modules + offline tests + dry-run audit sample
- Implementation plan under `docs/superpowers/plans/`

## Approval

- Chat approval 2026-07-21: all three items; Phase 2 = design+code+offline tests only; WAM target = H100; full sweep = 20 trials, resumable background launch.
