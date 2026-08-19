# WorldBridge4D

仓库现在只保留正在 GPU 1/4 上运行的 **256×256、Wan2.1-1.3B、三数据集、FSDP FULL_SHARD** 主路线。旧的 128px Compact/Full baseline、H004–H022 消融、单数据集 DDP 和 Wan-14B capacity 脚本已从当前工作树移除；需要时可从 Git 历史恢复。

## 当前训练契约

- 物理 GPU：`1,4`，2 ranks
- 数据：Kubric MOVi-F、PointOdyssey、Dynamic Replica
- 输入 latent：`[B,16,6,32,32]`
- Wan hidden readout：`[13,14,15,29]`，初始权重 `0.3/0.3/0.3/0.1`
- decoder：约 193.6M 非 Wan 参数，输出 256×256 source-grid XYZ
- batch：每 rank microbatch 2、gradient accumulation 2
- 每个 source 采样 19 个有效 targets；每 update 共 152 pairs
- checkpoint：full model、AdamW、计数器及每 rank RNG；严格 exact resume
- 当前目标：从 step 56,503 连续 cosine 延长至 step 150,000

精确配置在 `configs/worldbridge4d_gpu14_k19_150k.yaml`。它是当前 live run 的配置快照，包含 resume step 和服务器路径，不是通用模板。

## 当前实际执行链

```text
scripts/wait_resume_three_dataset_256_gpu14_150k.sh
  -> scripts/prepare_three_dataset_256_gpu14_handoff.py
  -> scripts/validate_three_dataset_256_cache_roots.py
  -> scripts/run_three_dataset_256_fsdp.sh
  -> scripts/train_three_dataset_256_fsdp.py
       -> src/worldbridge/{training256,dense4d,dense4d_runtime,wan,text_conditions}.py
       -> src/worldbridge/{data,geometry,pointodyssey,dynamic_replica}.py
       -> scripts/replicate_checkpoint.py
```

当前运行来自 PRL immutable snapshot；整理仓库不会修改正在运行进程的代码或 checkpoint。

## 使用

环境检查：

```bash
PYTHONPATH=src python scripts/check_environment.py --require-cuda
```

按当前配置启动/恢复普通两卡训练：

```bash
GPUS=1,4 NPROC=2 LAZY_VAE_CACHE=1 STAGE_INPUTS=0 \
CONFIG=configs/worldbridge4d_gpu14_k19_150k.yaml \
OUTPUT=/data/WorldBridge4D-runs/worldbridge4d_256_step45005_k19_gpu14_to100k \
  bash scripts/run_three_dataset_256_fsdp.sh
```

生产 handoff 应使用 `scripts/wait_resume_three_dataset_256_gpu14_150k.sh`，它还会检查 marker、checkpoint 身份和 GPU 独占窗口。推理入口：

```bash
PYTHONPATH=src python scripts/infer_three_dataset_256.py \
  --config configs/worldbridge4d_gpu14_k19_150k.yaml \
  --checkpoint /path/to/latest.pt --dataset pointodyssey \
  --index 0 --source 0 --targets 0 1 2 --output /tmp/prediction.pt
```

数据准备、缓存脚本分组见 [`scripts/README.md`](scripts/README.md)，完整实现说明见 [`docs/WORLDBRIDGE4D_256_THREE_DATASET_IMPLEMENTATION.md`](docs/WORLDBRIDGE4D_256_THREE_DATASET_IMPLEMENTATION.md)。

## 目录

- `configs/`：仅当前 GPU1/4 K19/150k 配置。
- `src/worldbridge/`：当前训练的 9 个传递依赖模块及 package init。
- `scripts/`：训练、handoff、推理、数据准备与当前 cache 构建工具。
- `tests/`：当前三数据集、几何、训练和 exact-resume 回归测试。
- `docs/`：当前数据协议、环境和 K19 cache/prefetch 经验。
- `research/`：历史研究证据；其中提到的旧文件应从对应 Git commit 恢复。

生成数据、模型权重、checkpoint、W&B 凭据和运行输出都不得提交 Git。
