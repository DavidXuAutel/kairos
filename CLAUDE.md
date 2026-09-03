# Franka dual-cam mapping (lab)

> Migrated from `.cursor/rules/franka-camera-mapping.mdc` (alwaysApply: true).

After 2026-07-23 serial swap on the lab PC:

| ROS ns | Role | Notes |
|--------|------|--------|
| **cam1** | **腕部 / wrist** (eye-in-hand) | D435 |
| **cam2** | **第三视角 / scene** (third-person) | D435I |

## WAM / LIBERO-plus image layout

Training expects horizontal concat **`[scene | wrist]`** → **`[cam2 | cam1]`**, each center-cropped to 224×224 → **224×448**.

Prefer compressed topics:

- `/cam1/cam1/color/image_raw/compressed` (wrist)
- `/cam2/cam2/color/image_raw/compressed` (scene)

## Do not

- Do not swap cam1/cam2 in scripts, logs, or debug frames without an explicit hardware remount.
- When labeling saved frames: left of WAM concat = cam2 (scene), right = cam1 (wrist).

## Prior Cursor history

Migrated from Cursor `state.vscdb` on 2026-07-28: **49 conversations, 3362 messages** at `~/.cursor-history/kairos/cursor-history.md`. Grep or read that file when you need context on past decisions or work done before this project switched to Claude Code.
