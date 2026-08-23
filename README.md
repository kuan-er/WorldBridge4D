# WorldBridge4D

仓库保留 **256×256、Wan2.1-1.3B、三数据集、FSDP FULL_SHARD** 的 step-100k source-RGB 生产主线。旧的 128px Compact/Full baseline、H004–H022 消融、单数据集 DDP 和 Wan-14B capacity 脚本已从当前工作树移除；需要时可从 Git 历史恢复。

## 当前训练契约

- 物理 GPU：`1,4`，2 ranks
- 数据：Kubric MOVi-F、PointOdyssey、Dynamic Replica
- 输入 latent：`[B,16,6,32,32]`
- Wan hidden readout：`[13,14,15,29]`，初始权重 `0.3/0.3/0.3/0.1`
- decoder：32/64/128/256px source-RGB fusion，约 194.4M 非 Wan 参数，输出 256×256 source-grid XYZ
- batch：每 rank microbatch 2、gradient accumulation 2
- 每个 source 采样 19 个有效 targets；每 update 共 152 pairs
- checkpoint：full model、AdamW、计数器及每 rank RNG；严格 exact resume
- 最终生产端点：step 100,000；checkpoint 可按原 schedule provenance 显式 exact resume

精确配置在 `configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml`。历史无 source-RGB 配置保留在 `configs/worldbridge4d_gpu14_k19_150k.yaml`。

## 当前实际执行链

```text
scripts/run_three_dataset_256_fsdp.sh
  -> scripts/train_three_dataset_256_fsdp.py       # thin CLI
  -> worldbridge.trainer.WorldBridgeTrainer
       -> worldbridge.models
       -> worldbridge.data
       -> worldbridge.trainer
       -> worldbridge.evaluation
```

旧的 `dense4d.py`、`training256.py`、`dense4d_runtime.py` 等模块仅保留兼容导入；新代码应使用上述 package。

## 使用

环境检查：

```bash
PYTHONPATH=src python scripts/check_environment.py --require-cuda
```

按当前配置启动/恢复普通两卡训练：

```bash
GPUS=0,1 NPROC=2 LAZY_VAE_CACHE=1 STAGE_INPUTS=0 \
CONFIG=configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml \
OUTPUT=/data/WorldBridge4D-runs/worldbridge4d_256_source_rgb \
  bash scripts/run_three_dataset_256_fsdp.sh
```

需要恢复时显式传入经过校验的完整 checkpoint。推理入口：

```bash
PYTHONPATH=src python scripts/infer_three_dataset_256.py \
  --config configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml \
  --checkpoint /path/to/latest.pt --dataset pointodyssey \
  --index 0 --source 0 --targets 0 1 2 --output /tmp/prediction.pt
```

数据准备、缓存脚本分组见 [`scripts/README.md`](scripts/README.md)，完整实现说明见 [`docs/WORLDBRIDGE4D_256_THREE_DATASET_IMPLEMENTATION.md`](docs/WORLDBRIDGE4D_256_THREE_DATASET_IMPLEMENTATION.md)。

## 目录

- `configs/`：step-100k source-RGB 主线与历史无 source-RGB 配置。
- `src/worldbridge/models/`：模型、Wan wrapper、decoder 与 source-RGB fusion。
- `src/worldbridge/data/`：数据集、geometry、cache、sampling 与 factory。
- `src/worldbridge/trainer/`：训练循环、objective、optimizer、FSDP 与 checkpoint。
- `src/worldbridge/evaluation/`：推理、counterfactual evaluator 与 metrics。
- `scripts/`：训练、handoff、推理、数据准备与当前 cache 构建工具。
- `tests/`：当前三数据集、几何、训练和 exact-resume 回归测试。
- `docs/`：当前数据协议、环境和 K19 cache/prefetch 经验。
- `research/`：历史研究证据；其中提到的旧文件应从对应 Git commit 恢复。

生成数据、模型权重、checkpoint、W&B 凭据和运行输出都不得提交 Git。
