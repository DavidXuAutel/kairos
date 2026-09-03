# Kairos 远端部署

> **2026-07-23**：当前推理主环境迁至 **H100 `10.239.121.23:30987`**（1×H100；`WAM_GPU_IDS=0`）。端到端仍为 H100 `:8005` → 反向隧道 → `yao@10.229.20.125:8005`。旧双卡节点 `10.239.121.25:31893` 可作备用。勿改 Franka Desk / `10.229.66.70`。

## H100 WAM（可复现启动）

| 项 | 值 |
|---|---|
| SSH（当前） | `ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes -p 30987 a25689@10.239.121.23`（端口会变） |
| GPU | 1× H100 80GB；WAM 默认 `WAM_GPU_IDS=0` / `127.0.0.1:8005` |
| 仓库脚本 | `scripts/check_wam_preflight.sh`、`bootstrap_kairos_torch27.sh`、`start_wam_h100.sh`、`stop_wam.sh`、`wam_first_frame_smoke.py`、`run_libero_smoke_h100.sh`、`run_libero_full_sweep_h100.sh` |
| Python | `/tmp/kairos-torch27`（`torch 2.7.1+cu126` + `triton 3.3.1`） |
| 权重 | `/tmp/kairos_models`（禁止直接 mmap Ceph home） |
| Triton 缓存 | **`TRITON_CACHE_DIR=/tmp/triton_cache`** |
| 启动 | `WAM_GPU_IDS=0 setsid bash ~/kairos/scripts/start_wam_h100.sh > ~/kairos/logs/wam_h100.log 2>&1 < /dev/null &` |
| 端到端隧道 | `bash scripts/tunnel_wam_h100_to_franka.sh start`（默认 `.23:30987` → `125:8005`） |
| 预检 / 冒烟 | `bash ~/kairos/scripts/check_wam_preflight.sh`；`/tmp/kairos-torch27/bin/python ~/kairos/scripts/wam_first_frame_smoke.py` → **`(1, 4, 7)`** |
| Eval Python | `/tmp/libero-plus-eval` |
| 仿真冒烟 | `bash ~/kairos/scripts/run_libero_smoke_h100.sh [out_dir]` |
| Full sweep | `bash ~/kairos/scripts/run_libero_full_sweep_h100.sh [out_dir]`（默认 **2×24=48** eval 并行；可 resume） |
| `LIBERO_CONFIG_PATH` | `~/kairos/outputs/libero_plus/libero_config`（缺则 BDDL 落到 `/path/to/...`） |
| 125 可视化 | `~/Desktop/kairos_libero_plus_sim/h100_libero_spatial_task0_success.mp4` |

```bash
# Full sweep（双卡 WAM + 16 路 eval；同一 out_dir resume）
sshh100 'curl -sS --noproxy "*" http://127.0.0.1:8005/health'
sshh100 'nohup bash ~/kairos/scripts/run_libero_full_sweep_h100.sh \
  ~/kairos/outputs/libero_plus/full_sweep_20260721_141412 \
  > ~/kairos/outputs/libero_plus/full_sweep_launch.log 2>&1 < /dev/null &'
# 再加压：MAX_TASKS_PER_GPU=12
sshh100 'MAX_TASKS_PER_GPU=12 bash ~/kairos/scripts/run_libero_full_sweep_h100.sh <out_dir>'
```

根因链：
1. `torch 2.6.0` 固定 `triton 3.2.0` → 首帧 DiT 触发 `LinearLayout.cpp:565` abort
2. 升级到官方配对 `torch 2.7.1+cu126` / `triton 3.3.1` 后 assert 消失
3. 首帧仍会编译 Triton kernel；缓存必须放本地 `/tmp`，否则卡在 `ceph_mdsc_wait_request`

```bash
alias sshh100='ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes -p 30987 a25689@10.239.121.23'
sshh100 'curl -s --noproxy "*" http://127.0.0.1:8005/health'
sshh100 'bash ~/kairos/scripts/run_libero_smoke_h100.sh'
```

---

## 当前主环境（A800 pod）

| 项 | 值 |
|---|---|
| 主机 | `a25689@172.29.0.178`（端口 **31126**） |
| SSH | `ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes -p 31126 a25689@172.29.0.178` |
| GPU | 8× NVIDIA A800-SXM4-80GB（CUDA 12.6 / driver 560.35.03） |
| 代码 | `/home/a25689/kairos`（Ceph home）；运行时建议用 `/tmp/kairos_code` 副本 |
| WAM Python | `/tmp/kairos-env`（torch **2.6.0+cu126**；勿用 Ceph 上的 `~/.conda/envs/kairos` 做首帧 infer） |
| Eval Python | `/tmp/libero-plus-eval`（软链 `~/.conda/envs/libero-plus-eval`） |
| WAM | `127.0.0.1:8005`，`libero_wam_infer_config_h100.py`，`attn_method=torch_compat` |
| 本地权重 | `/tmp/kairos_models/`（DiT 19G + Qwen 扁平目录 + VAE） |

```bash
alias ssha800='ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes -p 31126 a25689@172.29.0.178'
ssha800 'curl -s --noproxy "*" http://127.0.0.1:8005/health'
```

## 旧环境

| 主机 | 状态 |
|---|---|
| `a25689@10.239.121.25:31893`（双卡 H100） | 旧主节点；可作备用 |
| `yao@10.229.20.125`（4090） | 真机相机/关节 / Phase1B·2 端点；WAM 经反向隧道 `:8005` |

## 资源盘点（当前 pod）

| 资源 | 状态 | 路径 / 说明 |
|---|---|---|
| `kairos-4B-robot-LIBERO-plus` | **DONE** | Ceph + `/tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors` |
| `Qwen2.5-VL-7B-Instruct` | **DONE** | `/tmp/kairos_models/Qwen/Qwen2.5-VL-7B-Instruct/`（已校验 safetensors） |
| Wan `Wan2.1_VAE.pth` | **DONE** | `/tmp/kairos_models/Wan-AI/Wan2.2-T2V-A14B/` |
| LIBERO-plus `assets` | **DONE**（9.5G） | `/tmp/libero_assets_extract` → 软链；勿往 Ceph 解压 |
| `libero-plus-eval` | **DONE** | `/tmp/libero-plus-eval`；EGL via conda `libegl/libgl/libglvnd`；`egl_probe` 手动 cmake 编译 |
| `/tmp/kairos-env` | **DONE** | 网络新建；apex FusedRMSNorm shim；modelscope 等 |

## Phase 1 状态（**DONE** 2026-07-21）

| 项 | 状态 |
|---|---|
| 环境补齐（A800 pod） | **DONE** |
| WAM `/health` workers loaded | **DONE**（~27GB VRAM，GPU0；进程可能已停，需 `/tmp/start_wam.sh` 重启） |
| LIBERO-Plus sim smoke（A800） | **DONE**（`libero_spatial` task0 / 1 trial，约 111s；`success=False` 属任务表现） |
| 冒烟产物（A800） | `~/kairos/outputs/libero_plus/smoke_20260721_093732/` |
| 125 可视化 | **DONE**：`~/Desktop/kairos_libero_plus_sim/kairos_libero_smoke_task0.mp4`（DISPLAY=:1） |
| Phase 1B 真机干跑 | **DONE**（见下）；`ENABLE_MOTION=False` 写死 |

### Phase 1B（`yao@10.229.20.125`，无运动）

| 项 | 值 |
|---|---|
| 脚本 | `~/kairos/scripts/run_dryrun_franka.sh` → `dryrun_franka_sensors_to_wam.py` |
| 图像 | **WAM 输入**：水平拼接 `[cam2 场景 \| cam1 腕部]`，各 224×224 → **224×448**（与训练 `libero_2cam` 一致）。订阅 **compressed** 话题（lab 上 raw 常无帧）。串号 2026-07-23 已对调。重启：`bash ~/gello_desk/lerobot_record/start_dual_realsense.sh start` |
| 关节 | **`/joint_states`**（`fr3_joint*`）；`/franka/joint_states` 亦约 1kHz |
| FR3 | FCI **`10.229.66.91`**（经 125；`example_fr3_config.yaml` / `franka_gripper`）；勿改 Desk / `10.229.66.70` |
| WAM | **H100** `127.0.0.1:8005` → **直连反向隧道**到 125（无 Mac 中转） |
| 成功样例 | `~/kairos/dryrun_logs/20260722_073718/`（3× infer，`action (1,8,7)`，`ENABLE_MOTION=False`） |
| 安全 | 不创建 control publisher；`ENABLE_MOTION=False` |

推理端直连真机工位（H100 → 125，本机一键）：

```bash
# 密钥已装在 H100: ~/.ssh/kairos_h100_to_125_ed25519（仅隧道用）
bash scripts/tunnel_wam_h100_to_franka.sh start   # 或 status / stop
# 等价：在 H100 上
#   ssh -fN -i ~/.ssh/kairos_h100_to_125_ed25519 -R 8005:127.0.0.1:8005 yao@10.229.20.125
ssh125 'curl -4 -s --noproxy "*" -m 30 http://127.0.0.1:8005/health'
ssh125 'bash ~/kairos/scripts/run_dryrun_franka.sh --num-infer 3 --num-inference-steps 5'
```

### Phase 2（开环真机，已批准）

| 项 | 值 |
|---|---|
| 入口 | `bash ~/kairos/scripts/run_reliable_pick.sh`（推荐）或 `run_phase2_pick_pen.sh` |
| 控制 | **杀掉** GELLO publisher → 独占 `/gello/joint_states` → 结束后 hold（默认不重启 GELLO） |
| 映射 | WAM approach（默认 clamp ±2.5 cm）→ 脚本下降/抬升；夹爪 **hybrid**（approach 跟模型，合爪兜底脚本） |
| Proprio | `xyz -= origin`(默认 `0.45,0,0`) + 标定 `R_offset` + 夹爪 `[w,-w]`；`--proprio-origin` / 可选 `--flip180` |
| 门控 | `KAIROS_ARM_TOKEN` + `--i-approve-motion --arm-token` |
| 恢复 | FCI `unavailable` 时：`desk_prep --recover` + 重启 `franka_fr3_arm_controllers`（见 `~/gello_desk/restart_teleop.sh`） |
| 注意 | 结束后勿对未对齐的 GELLO 做 SIGCONT，否则会大力矩对抗 |

```bash
export KAIROS_ARM_TOKEN=PICK_PEN_20260722
bash ~/kairos/scripts/run_reliable_pick.sh --i-approve-motion --arm-token "$KAIROS_ARM_TOKEN" \
  --prompt "pick up a pen" --approach-replans 1 --grasp-z 0.16 --lift-z 0.32
# 可选：--flip180  --gripper-mode model|scripted  --max-abs-xyz 0.03
# 无运动对齐检查：
#   python3 ~/kairos/scripts/phase2/dump_proprio_compare.py
#   bash ~/kairos/scripts/run_dryrun_franka.sh --num-infer 1 --num-inference-steps 5
```

### 关键经验（本机必读）

Ceph MDS 会导致：
1. conda/`transformers` 小文件 import 卡死 → WAM 用 **`/tmp/kairos-env`**
2. 首帧 infer 卡在 `~/.triton/cache` → 必须 **`TRITON_CACHE_DIR=/tmp/triton_cache`**
3. 权重 mmap 慢/不完整 → DiT/Qwen 放到 **`/tmp/kairos_models`**

### WAM 启动（A800）

```bash
# 已有 /tmp/start_wam.sh 时直接：
nohup /tmp/start_wam.sh >/dev/null 2>&1 &
# 关键环境变量：
#   PATH=/tmp/kairos-env/bin
#   PYTHONPATH=/tmp/kairos_code:/tmp/kairos_code/benchmarks/common
#   KAIROS_MODEL_DIR=/tmp/kairos_models
#   WAM_PRETRAINED_DIT=/tmp/kairos_models/kairos-4B-robot-LIBERO-plus.safetensors
#   TRITON_CACHE_DIR=/tmp/triton_cache
#   TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
#   attn_method=torch_compat（h100 config）
curl -s --noproxy '*' http://127.0.0.1:8005/health
```

### 最小仿真冒烟（WAM 已就绪）

```bash
# eval 用 GPU1；WAM 占 GPU0
# 脚本参考 /tmp/run_libero_smoke.sh
# suite=libero_spatial task_id=0 num_trials=1
# model.endpoint=http://127.0.0.1:8005
```

## SSH 快捷命令

```bash
alias ssha800='ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes -p 31126 a25689@172.29.0.178'
alias ssh125='ssh -i ~/.ssh/franka_ros2_ed25519 -o IdentitiesOnly=yes yao@10.229.20.125'
```

端口变更后只需改 `-p`；home 数据一般仍在，但 **`/tmp` 内容随 pod 重建会丢**，需重解压 assets / 重拷权重 / 重建 `/tmp/kairos-env`。
