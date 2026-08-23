# 当前 256×256 三数据集 FSDP 实现

当前生产配置是 `configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml`，对应 H023 step-100k source-RGB K19/B2/A2 endpoint。

## 模型与训练

1. 三数据集 RGB 被 frozen Wan VAE 编码为 FP32 clean latent `[16,6,32,32]`。
2. Wan2.1-1.3B DiT 读取 blocks `[13,14,15,29]`，初始 gate 为 `0.3/0.3/0.3/0.1`。
3. geometry adapter 构造 21 帧 structured dense features 和 8 个 motion slots。
4. 5 层 cross-attention decoder 输出 source-grid XYZ `[B,K,3,256,256]`。
5. validity-masked SmoothL1 是训练损失；visibility 不参与 loss。
6. 两 rank FSDP FULL_SHARD，每 rank B2、A2、K19，共 8 clips/152 pairs/update。

主入口：

```text
scripts/run_fsdp.sh
  -> scripts/train.py
  -> worldbridge.trainer.WorldBridgeTrainer
```

实现按职责分层：

```text
worldbridge.models       模型、Wan backbone、decoder、source-RGB fusion
worldbridge.data         datasets、geometry、cache、sampling、factory
worldbridge.trainer      objective、optimizer、FSDP、checkpoint、训练循环
worldbridge.evaluation   inference、counterfactual evaluation、metrics
```

旧的 `worldbridge.dense4d`、`dense4d_runtime`、`training256`、`wan` 以及顶层数据集兼容路径已经删除；实现只从规范 package 导入。

## Exact resume

Checkpoint format 3 保存 full model、AdamW、global step、数据集计数和每 rank Python/NumPy/Torch/CUDA RNG。恢复顺序固定为：

1. rank 0 在 FSDP wrap 前严格加载 full model；
2. `sync_module_states=True` 广播一致权重；
3. scatter full optimizer state；
4. 恢复每 rank RNG；
5. 才允许下一次 optimizer update。

历史 GPU1/4 handoff watcher 已从正式入口删除，其审计证据保留在 research 和 Git 历史中。当前恢复统一通过 `scripts/run_fsdp.sh` 显式传入完整 checkpoint；它要求两 rank、K19、B2/A2、完整 checkpoint/status/RNG。最终生产端点是 step 100,000；配置保留原连续 150k horizon 仅作为未来显式 exact resume 的 LR provenance。

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
bash -n scripts/run_fsdp.sh
```

任何训练协议变更都必须使用新的 PRL Run；不要原地修改正在运行的 immutable snapshot 或 checkpoint。
