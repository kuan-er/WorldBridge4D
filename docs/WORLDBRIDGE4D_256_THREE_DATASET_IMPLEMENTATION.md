# 当前 256×256 三数据集 FSDP 实现

当前唯一配置是 `configs/worldbridge4d_gpu14_k19_150k.yaml`，对应正在物理 GPU 1/4 上运行的 H023 K19/B2/A2 trajectory。

## 模型与训练

1. 三数据集 RGB 被 frozen Wan VAE 编码为 FP32 clean latent `[16,6,32,32]`。
2. Wan2.1-1.3B DiT 读取 blocks `[13,14,15,29]`，初始 gate 为 `0.3/0.3/0.3/0.1`。
3. geometry adapter 构造 21 帧 structured dense features 和 8 个 motion slots。
4. 5 层 cross-attention decoder 输出 source-grid XYZ `[B,K,3,256,256]`。
5. validity-masked SmoothL1 是训练损失；visibility 不参与 loss。
6. 两 rank FSDP FULL_SHARD，每 rank B2、A2、K19，共 8 clips/152 pairs/update。

主入口：

```text
scripts/run_three_dataset_256_fsdp.sh
  -> scripts/train_three_dataset_256_fsdp.py
```

训练脚本只直接导入：

```text
worldbridge.dense4d
worldbridge.dense4d_runtime
worldbridge.training256
worldbridge.wan
worldbridge.text_conditions
```

它们再传递依赖 `data.py`、`geometry.py`、`pointodyssey.py` 和 `dynamic_replica.py`。

## Exact resume

Checkpoint format 3 保存 full model、AdamW、global step、数据集计数和每 rank Python/NumPy/Torch/CUDA RNG。恢复顺序固定为：

1. rank 0 在 FSDP wrap 前严格加载 full model；
2. `sync_module_states=True` 广播一致权重；
3. scatter full optimizer state；
4. 恢复每 rank RNG；
5. 才允许下一次 optimizer update。

GPU1/4 生产 handoff 使用：

```text
scripts/wait_resume_three_dataset_256_gpu14_150k.sh
  -> scripts/prepare_three_dataset_256_gpu14_handoff.py
  -> scripts/run_three_dataset_256_fsdp.sh
```

它要求两 rank、K19、B2/A2、完整 checkpoint/status/RNG、GPU1/4 独占预检及 immutable marker。当前 schedule 在 step 56,503 保持原 100k cosine factor，然后连续衰减到 step 150,000，不产生 LR 跳变。

## 数据和缓存

原始挂载保持只读；当前服务器路径以 YAML 为准。训练依赖：

- 三数据集 train indexes；
- dataset-specific UMT5 conditions；
- train-only mixture coordinate stats；
- Wan clean latent shards或校验过的 per-clip lazy cache；
- Kubric compact metadata + mmap geometry；
- PointOdyssey annotations/depth；
- Dynamic Replica persistent trajectories/depth。

从零准备入口见 `scripts/README.md`。K19 性能基线为 4 geometry workers、prefetch depth 2、Kubric sample LRU 16、90 mmap shards lazy mapped。实测热窗口约 4.82 秒/update；偶发 mmap page-fault tail 仍可能出现。

## 验证

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts tests
bash -n scripts/run_three_dataset_256_fsdp.sh
bash -n scripts/wait_resume_three_dataset_256_gpu14_150k.sh
```

任何训练协议变更都必须使用新的 PRL Run；不要原地修改正在运行的 immutable snapshot 或 checkpoint。
