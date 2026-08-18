# 256×256 / 200M 三数据集分布式训练实现

实现配置：`configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml`

训练采用 4 rank `FSDP FULL_SHARD`（不是参数复制式 DDP）。两卡仅用于启动 gate，正式配置仍要求四卡。

K19 长训练中验证的 geometry mmap、进程 LRU、有界并发、depth-2 prefetch、page-cache 诊断及稳定吞吐经验见 [`THREE_DATASET_GEOMETRY_CACHE_PREFETCH_LESSONS.md`](THREE_DATASET_GEOMETRY_CACHE_PREFETCH_LESSONS.md)。

## 1. 先生成三个固定 UMT5 condition

```bash
python scripts/create_wan_text_conditions.py \
  --wan-source "$WAN_SOURCE_ROOT" \
  --checkpoint-dir /data/WorldBridge4D/Wan2.1-T2V-1.3B \
  --output-dir /data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/text_conditions
```

生成器会保存每个 condition 的 prompt、shape 和 SHA-256；训练与推理都会严格校验 metadata，禁止回退到 empty condition。

## 2. 重新生成 256 latent

旧 128 RGB/16×16 latent 不可复用。以下命令总是读取原始 RGB，先构造 256 RGB，再用 frozen Wan VAE posterior mean 生成 `[16,6,32,32]` float32 safetensors。

```bash
# MOVi-F 512 source
python scripts/preprocess_three_dataset_256.py --dataset kubric \
  --raw-root /dataset/MOVi-F/512x512 \
  --cache-root /data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/kubric

# PointOdyssey：复用已验证 v1 scene index/物理 crop，只新建 256 index copy + latent tier
python scripts/preprocess_three_dataset_256.py --dataset pointodyssey \
  --raw-root /dataset/nas0/PointOdyssey \
  --geometry-cache-root /data/WorldBridge4D-persistent/pointodyssey_worldbridge4d_v1 \
  --cache-root /data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/pointodyssey

# Dynamic Replica
python scripts/preprocess_three_dataset_256.py --dataset dynamic_replica \
  --raw-root /dataset/data/Dynamic_dataset/dynamic_stereo \
  --geometry-cache-root /data/WorldBridge4D-persistent/datasets/dynamic_stereo_worldbridge4d_v1 \
  --cache-root /data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/dynamic_replica
```

缓存写入独立 persistent 路径，不写原始数据目录。可用 `--max-clips` 先构造小 gate cache。

## 3. 审计和 mixture normalization

```bash
python scripts/audit_three_dataset_256.py \
  --config configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml \
  --output /data/WorldBridge4D-persistent/worldbridge4d_256_three_dataset_v1/audit/validation_report.json

python scripts/compute_three_dataset_256_stats.py \
  --config configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml
```

stats 脚本只使用 train split，每 clip 最多固定采样 4096 个 valid diagonal 点，然后按 35/30/35 合并 moments，避免 dense 数据集因像素更多而支配 normalization。

## 4. 两卡 optimizer gate

先选择两张空闲卡，执行至少两个 optimizer updates；建议完整 gate 为 100–200 updates。若完整 latent 尚未生成，可启用确定性 lazy VAE warmup：每个 rank 根据 `(seed, step, microstep, rank)` 先计算本次 gate 实际会访问的 clips，多 rank 重叠由确定性 owner 去重，结果以 per-clip safetensors + 文件锁 + 原子 rename 写入 persistent cache。VAE 在 FSDP 模型构建前释放，因此不会与 Wan/Adam 同时常驻显存；resume 和后续 gate 直接命中缓存。

这不是在训练 forward 中反复运行 VAE：训练开始前会 fail-closed 验证本次计划的所有 latent 已可读取。仅推荐 gate 使用；正式四卡训练仍应完成完整 shard cache 和审计。

```bash
LAZY_VAE_CACHE=1 GPUS=1,2 NPROC=2 STEPS=2 \
OUTPUT=/data/WorldBridge4D-runs/worldbridge4d_256_two_gpu_gate_k6 \
  bash scripts/run_three_dataset_256_fsdp.sh
```

lazy 文件位于各数据集 `cache_root/latents/wan2.1_1.3b_fp32_256_lazy/`，metadata 严格绑定 dataset、clip ID、split-local index、VAE SHA-256 和 posterior-mean FP32 contract。完整 gate：

完整 gate：

```bash
LAZY_VAE_CACHE=1 GPUS=1,2 NPROC=2 STEPS=200 \
OUTPUT=/data/WorldBridge4D-runs/worldbridge4d_256_two_gpu_gate_k6 \
  bash scripts/run_three_dataset_256_fsdp.sh
```

若 K=6 OOM、peak allocated 超过 72–74 GiB 或吞吐不可接受，将配置复制为 gate 专用 YAML，把 `targets_per_source` 固定为 `4`，换新 output 重新测试。正式 run 启动后不得改变 K。

> 两卡 gate 只能验证代码路径、第二步 reduction、optimizer state、显存和 checkpoint；方案要求的正式 capacity gate 仍需四卡执行。

## 5. 四卡正式启动

```bash
GPUS=0,1,2,3 NPROC=4 \
OUTPUT=/data/WorldBridge4D-runs/worldbridge4d_256_three_dataset_200m \
  bash scripts/run_three_dataset_256_fsdp.sh
```

launcher 检测到 `OUTPUT/latest.pt` 后自动 exact resume。Checkpoint 包含 full model、AdamW、global step、各数据集 clips seen、每 rank Python/NumPy/Torch/CUDA RNG 和数据集 cycle offset。

## 6. 带对应 prompt 的推理

推理必须显式指定数据集，脚本据此加载同一数据集训练时使用的 condition：

```bash
python scripts/infer_three_dataset_256.py \
  --config configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml \
  --checkpoint /data/WorldBridge4D-runs/worldbridge4d_256_three_dataset_200m/latest.pt \
  --dataset pointodyssey --index 0 --source 0 \
  --targets 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --target-chunk 2 \
  --output /data/WorldBridge4D-runs/inference/pointodyssey_000000.pt
```

checkpoint、配置和 condition metadata 中的 prompt 必须逐字一致；否则推理会直接失败。

## 已实现 gate

- 256 latent contract 与显式 per-forward prompt；
- `[13,14,15,29]` logits 精确初始化为 30/30/30/10；
- 非 Wan 参数量必须精确为 `193,586,693`；
- 20-step schedule 精确为 7/6/7；
- 一个 update 内所有 rank/microstep 同一数据集；
- K-target 仅从 eligible target 无放回均匀抽样，空 pair 不计零损失；
- BF16 FULL_SHARD、accumulation=2、activation checkpointing、cosine 100k horizon；
- 原子 latest 和里程碑 checkpoint、68h graceful stop、W&B/offline logging。

尚未由本次代码编写自动执行的昂贵 gate：真实三数据集完整 256 cache、两/四卡 real-Wan optimizer run、完整 evaluator/验证。训练前必须按训练方案逐项完成。
