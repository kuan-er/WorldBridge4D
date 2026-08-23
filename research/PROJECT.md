# WorldBridge4D 256px 三数据集主路线

## Goal

使用 Wan2.1-1.3B 作为 feed-forward 4D backbone，从 21 帧单目视频一次性构造 structured 4D representation，再由 `(source,target)` query 输出 256×256 source-grid XYZ，统一 pointmap 与 arbitrary-source 3D tracking。

## Current method

当前 H023 production route 混合 Kubric MOVi-F、PointOdyssey 和 Dynamic Replica。Frozen Wan VAE 输出 `[16,6,32,32]` clean latent；trainable Wan DiT 读取 blocks `[13,14,15,29]`；冻结的 geometry adapter 生成 21 帧 dense features 和 8 motion slots；source RGB pyramid 在 32/64/128/256px 注入 decoder，约 194.4M 非 Wan 参数输出 XYZ。训练使用 validity-masked SmoothL1，遮挡有效点不因 visibility 被过滤。

当前最终生产端点是 step-100,000，使用两-rank FSDP FULL_SHARD、B2/A2/K19，即每 update 8 clips、152 source-target pairs。Source-RGB 参数使用 10× multiplier，Wan/旧 decoder 使用旧组学习率的 0.1× cap，geometry adapter 冻结。Checkpoint 完整保存 model、AdamW、计数器和每 rank RNG。默认训练在 100,000 停止；配置保留生成该 checkpoint 时使用的连续 150k cosine horizon，仅用于未来显式 `--resume ... --steps N` 微调时无 LR 跳变，不代表继续训练计划。

## Code map

- `scripts/train.py`: 薄训练 CLI
- `scripts/infer.py`, `evaluate.py`: 薄推理与评测 CLI
- `scripts/prepare_data.py`: 数据、cache 与 checkpoint 维护的统一子命令入口
- `scripts/run_fsdp.sh`: 通用 torchrun launcher
- `src/worldbridge/models/`: structured representation、Wan backbone、decoder 与 source-RGB fusion
- `src/worldbridge/data/`: 三数据集 adapter、geometry、cache、sampling 与 factory
- `src/worldbridge/trainer/`: `WorldBridgeTrainer`、objective、optimizer、FSDP、checkpoint 与 exact resume
- `src/worldbridge/evaluation/`: inference、counterfactual evaluator 与 metrics
- `src/worldbridge/utils/`: atomic I/O 与 checksum helper
- 顶层旧 import facade 已删除；当前实现仅使用上述规范 package
- `src/worldbridge/text_conditions.py`: dataset prompt condition 校验
- `configs/worldbridge4d_256_source_rgb_fusion32_step100000.yaml`: 最终 step-100k source-RGB production 配置及可恢复 schedule provenance
- `configs/worldbridge4d_gpu14_k19_150k.yaml`: 历史无 source-RGB trajectory 配置快照

旧 baseline、消融和被否决 launcher 从当前工作树移除，研究结论保留在 `research/` 和 Git 历史。
