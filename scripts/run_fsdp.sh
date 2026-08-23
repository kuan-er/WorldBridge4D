#!/usr/bin/env bash
# 256×256 三数据集 FSDP 训练的统一启动器。
#
# 本脚本只负责“启动前准备”：解析环境变量、检查数据和缓存、选择断点、
# 将大文件暂存到高速盘，最后用 torchrun 启动训练。模型构建、训练循环、
# loss、反向传播和 checkpoint 内容都在 worldbridge.trainer 中。
#
# 两卡启动示例（配置中的 batch/K 等参数仍由 YAML 决定）：
#   GPUS=0,1 NPROC=2 CONFIG=/path/to/config.yaml OUTPUT=/path/to/output \
#     bash scripts/run_fsdp.sh
#
# 启动顺序：环境变量 -> fresh/resume 选择 -> 数据/缓存检查 -> 输入暂存
#          -> 拼接 Python 参数 -> torchrun。
set -euo pipefail  # 任一命令失败即退出；未定义变量和管道中间错误也视为失败。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 配置、输出与 checkpoint ----------
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml}"  # step-100k source-RGB 主线。
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_source_rgb}"  # 日志、W&B ID 和状态文件目录。
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT}"  # 本次运行读写 checkpoint 的目录，可与 OUTPUT 分开。
DURABLE_CHECKPOINT="${DURABLE_CHECKPOINT:-}"  # 可选：每次保存后异步复制到这个持久化路径。
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # 可选：显式指定恢复文件；优先于 CHECKPOINT_DIR/latest.pt。
WANDB_LOG_AFTER_STEP="${WANDB_LOG_AFTER_STEP:--1}"  # 仅记录大于该 step 的指标；-1 表示从头记录。
POST_RESUME_CHECKSUM_MARKER="${POST_RESUME_CHECKSUM_MARKER:-}"  # 可选：严格恢复后异步计算断点 SHA-256，并写 marker。

# ---------- GPU 拓扑与训练长度 ----------
GPUS="${GPUS:-0,1}"  # 默认物理 GPU 编号，可由环境覆盖。
NPROC="${NPROC:-2}"  # 当前生产训练固定为两个 ranks。
STEPS="${STEPS:-}"  # 可选：覆盖 YAML 中的目标总 step；不是“再训练多少步”。

# ---------- latent 生成模式（两者互斥） ----------
LAZY_VAE_CACHE="${LAZY_VAE_CACHE:-0}"  # 1：启动 FSDP 前补齐本次计划需要的 VAE latent。
LAZY_VAE_PIPELINE="${LAZY_VAE_PIPELINE:-0}"  # 1：训练时由后台流水线按需生成 latent。
PIPELINE_LOOKAHEAD_STEPS="${PIPELINE_LOOKAHEAD_STEPS:-16}"  # 后台流水线最多提前准备多少个 step。

# ---------- 实验模式与启动安全开关 ----------
ALLOW_FOUR_GPU_EXPERIMENT="${ALLOW_FOUR_GPU_EXPERIMENT:-0}"  # 允许四卡实验 batch 模式；不等于正式协议。
ALLOW_ARBITRARY_WORLD="${ALLOW_ARBITRARY_WORLD:-0}"  # 允许已注册的非标准 world size，主要用于容量实验。
FRESH_START="${FRESH_START:-0}"  # 1：强制全新轨迹；发现旧状态或 checkpoint 时拒绝启动。
STAGE_INPUTS="${STAGE_INPUTS:-1}"  # 1：把模型/恢复断点暂存到 STAGING_ROOT，减少慢盘争用。
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"

# EXTRA 收集最终传给 Python 训练入口的可选参数。
EXTRA=()
# 将 shell 环境开关翻译成训练入口的显式命令行参数。两卡模式只用于
# 已注册的 gate/实验协议，Python 侧仍会继续检查 batch、K 和 world size。
if [[ "$NPROC" == "2" ]]; then
  EXTRA+=(--allow-two-gpu-gate)
fi
if [[ "$ALLOW_FOUR_GPU_EXPERIMENT" == "1" ]]; then
  if [[ "$NPROC" != "4" ]]; then
    echo "ALLOW_FOUR_GPU_EXPERIMENT=1 requires NPROC=4" >&2
    exit 2
  fi
  EXTRA+=(--allow-four-gpu-experiment)
fi
if [[ "$ALLOW_ARBITRARY_WORLD" == "1" ]]; then
  EXTRA+=(--allow-arbitrary-world)
fi
if [[ -n "$STEPS" ]]; then
  EXTRA+=(--steps "$STEPS")
fi
if [[ "$LAZY_VAE_CACHE" == "1" && "$LAZY_VAE_PIPELINE" == "1" ]]; then
  echo "LAZY_VAE_CACHE and LAZY_VAE_PIPELINE are mutually exclusive" >&2
  exit 2
fi
if [[ "$LAZY_VAE_CACHE" == "1" ]]; then
  EXTRA+=(--lazy-vae-cache)
fi
if [[ "$LAZY_VAE_PIPELINE" == "1" ]]; then
  EXTRA+=(--lazy-vae-pipeline --pipeline-lookahead-steps "$PIPELINE_LOOKAHEAD_STEPS")
fi
# ---------- 选择 fresh start 或恢复点 ----------
# 优先级：FRESH_START=1 > RESUME_CHECKPOINT > CHECKPOINT_DIR/latest.pt。
# fresh start 会检查旧轨迹标记，避免误覆盖；自动恢复只认 latest.pt。
RESUME=""
if [[ "$FRESH_START" == "1" ]]; then
  for artifact in train_status.json wandb_run_id; do
    if [[ -e "$OUTPUT/$artifact" ]]; then
      echo "FRESH_START=1 refuses existing trajectory artifact: $OUTPUT/$artifact" >&2
      exit 2
    fi
  done
  if [[ -e "$CHECKPOINT_DIR/latest.pt" ]]; then
    echo "FRESH_START=1 refuses existing trajectory artifact: $CHECKPOINT_DIR/latest.pt" >&2
    exit 2
  fi
  if compgen -G "$CHECKPOINT_DIR/checkpoint-*.pt" >/dev/null; then
    echo "FRESH_START=1 refuses existing checkpoints under $CHECKPOINT_DIR" >&2
    exit 2
  fi
elif [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  RESUME="$RESUME_CHECKPOINT"
elif [[ -f "$CHECKPOINT_DIR/latest.pt" ]]; then
  RESUME="$CHECKPOINT_DIR/latest.pt"
fi

# ---------- 原始数据挂载预检 ----------
# 在 torchrun/NCCL 初始化前失败，避免某个只读挂载消失或旧配置仍指向废弃路径时，
# 多卡进程启动后才报错。即使 VAE latent 全部命中缓存，监督几何仍依赖原始数据。
python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit(f"three-dataset config is missing: {config_path}")
config = yaml.safe_load(config_path.read_text())
missing = []
for name in ("kubric", "pointodyssey", "dynamic_replica"):
    root = Path(config["datasets"][name]["raw_root"])
    if not root.is_dir():
        missing.append(f"{name}={root}")
if missing:
    raise SystemExit(
        "required raw dataset mounts are unavailable; not launching FSDP: "
        + ", ".join(missing)
    )
PY

# ---------- 权威缓存根检查 ----------
# 训练只接受可持久化的模型输入缓存；拒绝断开的软链接、软链接缓存根，以及
# 伪装成持久缓存的 /tmp 路径。--create 只创建合法目录，不生成训练数据。
python "$ROOT/scripts/prepare_data.py" cache-roots \
  --config "$CONFIG" --create

# ---------- 可选的高速盘暂存 ----------
# helper 对大模型文件和恢复断点做内容寻址、校验和原子发布，并输出两行：
# 1) 改写为暂存路径后的配置；2) 暂存后的恢复断点（没有则为空）。
# 暂存只改变读取位置，不改变模型权重或训练协议。
if [[ "$STAGE_INPUTS" == "1" ]]; then
  STAGE_ARGS=(--config "$CONFIG" --staging-root "$STAGING_ROOT")
  if [[ -n "$RESUME" ]]; then
    STAGE_ARGS+=(--resume "$RESUME")
  fi
  mapfile -t STAGED < <(python "$ROOT/scripts/prepare_data.py" stage-inputs "${STAGE_ARGS[@]}")
  if [[ "${#STAGED[@]}" -ne 2 ]]; then
    echo "staging helper returned an invalid response" >&2
    exit 2
  fi
  CONFIG="${STAGED[0]}"
  RESUME="${STAGED[1]}"
fi
# ---------- 拼接恢复、保存和审计参数 ----------
if [[ -n "$RESUME" ]]; then
  EXTRA+=(--resume "$RESUME")
fi
EXTRA+=(--checkpoint-dir "$CHECKPOINT_DIR")
EXTRA+=(--wandb-log-after-step "$WANDB_LOG_AFTER_STEP")
if [[ -n "$POST_RESUME_CHECKSUM_MARKER" ]]; then
  [[ -n "$RESUME" ]] || { echo "post-resume checksum requires a resume checkpoint" >&2; exit 2; }
  EXTRA+=(--post-resume-checksum-marker "$POST_RESUME_CHECKSUM_MARKER")
fi
if [[ -n "$DURABLE_CHECKPOINT" ]]; then
  EXTRA+=(--durable-checkpoint "$DURABLE_CHECKPOINT")
fi

# ---------- 启动分布式训练 ----------
# CUDA_VISIBLE_DEVICES 把物理卡映射为每个 worker 看到的本地编号 0..NPROC-1；
# torchrun 注入 RANK/WORLD_SIZE/LOCAL_RANK。exec 让 torchrun 接管当前 shell，
# 因而退出码和终止信号可以直接传递给外层调度器。
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# expandable_segments 可降低动态 decoder batch 带来的 CUDA 内存碎片。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  "$ROOT/scripts/train.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" "${EXTRA[@]}"
