# WorldBridge4D 256px 三数据集主路线

## Goal

使用 Wan2.1-1.3B 作为 feed-forward 4D backbone，从 21 帧单目视频一次性构造 structured 4D representation，再由 `(source,target)` query 输出 256×256 source-grid XYZ，统一 pointmap 与 arbitrary-source 3D tracking。

## Current method

当前 H023 production route 混合 Kubric MOVi-F、PointOdyssey 和 Dynamic Replica。Frozen Wan VAE 输出 `[16,6,32,32]` clean latent；trainable Wan DiT 读取 blocks `[13,14,15,29]`；冻结的 geometry adapter 生成 21 帧 dense features 和 8 motion slots；source RGB pyramid 在 32/64/128/256px 注入 decoder，约 194.4M 非 Wan 参数输出 XYZ。训练使用 validity-masked SmoothL1，遮挡有效点不因 visibility 被过滤。

当前保留的 step-100,000 主线使用两-rank FSDP FULL_SHARD、B2/A2/K19，即每 update 8 clips、152 source-target pairs。Source-RGB 参数使用 10× multiplier，Wan/旧 decoder 使用旧组学习率的 0.1× cap，geometry adapter 冻结。Checkpoint 完整保存 model、AdamW、计数器和每 rank RNG，并沿连续 cosine schedule 计划延长至 150,000。

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
- `configs/worldbridge4d_256_source_rgb_fusion32_cap0p1_gpu01_to150000.yaml`: step-100k source-RGB production trajectory 配置快照
- `configs/worldbridge4d_gpu14_k19_150k.yaml`: 历史无 source-RGB trajectory 配置快照

旧 baseline、消融和被否决 launcher 从当前工作树移除，研究结论保留在 `research/` 和 Git 历史。
