# WorldBridge4D 256px 三数据集主路线

## Goal

使用 Wan2.1-1.3B 作为 feed-forward 4D backbone，从 21 帧单目视频一次性构造 structured 4D representation，再由 `(source,target)` query 输出 256×256 source-grid XYZ，统一 pointmap 与 arbitrary-source 3D tracking。

## Current method

当前 H023 production route 混合 Kubric MOVi-F、PointOdyssey 和 Dynamic Replica。Frozen Wan VAE 输出 `[16,6,32,32]` clean latent；full-trainable Wan DiT 读取 blocks `[13,14,15,29]`；geometry adapter 生成 21 帧 dense features 和 8 motion slots；约 193.6M 参数 decoder 输出 XYZ。训练使用 validity-masked SmoothL1，遮挡有效点不因 visibility 被过滤。

当前 GPU1/4 两-rank FSDP FULL_SHARD trajectory 使用 B2/A2/K19，即每 update 8 clips、152 source-target pairs。Checkpoint 完整保存 model、AdamW、计数器和每 rank RNG。step 56,503 后连续延长 cosine schedule 至 150,000。

## Code map

- `scripts/train_three_dataset_256_fsdp.py`: 唯一训练循环和 exact resume
- `scripts/run_three_dataset_256_fsdp.sh`: 通用 torchrun launcher
- `scripts/wait_resume_three_dataset_256_gpu14_150k.sh`: 当前 GPU1/4 handoff
- `src/worldbridge/training256.py`: 三数据集 schedule、adapter 装配、cache 与采样
- `src/worldbridge/dense4d.py`: structured readout、decoder 和 loss
- `src/worldbridge/dense4d_runtime.py`: real-Wan model/optimizer 构建
- `src/worldbridge/wan.py`: Wan VAE/DiT adapter
- `src/worldbridge/text_conditions.py`: dataset prompt condition 校验
- `src/worldbridge/data.py`, `geometry.py`: Kubric native data/geometry
- `src/worldbridge/pointodyssey.py`, `dynamic_replica.py`: 外部数据集 geometry
- `configs/worldbridge4d_gpu14_k19_150k.yaml`: 当前 live trajectory 配置快照

旧 baseline、消融和被否决 launcher 从当前工作树移除，研究结论保留在 `research/` 和 Git 历史。
