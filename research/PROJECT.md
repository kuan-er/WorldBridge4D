# MOVi-F 四维世界潜变量

## Goal
在真实 Kubric MOVi-F 128×128 数据上，验证预训练 Wan2.1-1.3B 能否以最小改动作为 feed-forward 4D perception backbone：RGB video 经 frozen Wan VAE 和 flow-time-zero 单次 DiT forward 得到无直接监督的 reusable `Z4D`，再由 dense `(source,target)` query 输出完整 source-grid XYZ map，从而统一 pointmap 与 arbitrary-source 3D tracking。早期 Compact/Full geometry-autoencoder 结果保留为基线，不是 H004 的输入路径。

## Method
数据适配器直接解析只读 TFRecord `tf.train.Example`。pointmap 由 uint16 radial depth、相机内外参反投影到 Kubric world，再转换到 clip 第一帧相机坐标。刚体轨迹按 segmentation 实例、实例局部坐标及目标姿态构造；`M` 是 visibility，`A` 是数据/深度/变换 validity，遮挡有效点仍进入 loss。

H004 将 RGB `[B,21,3,128,128]` 编为 Wan clean latent `[B,16,6,16,16]`，对 clean endpoint 使用实际 Wan timestep 0，并按已验证的 `v=epsilon-x0` 约定取 `Z4D=-v`。完整 1,536-token memory 供独立 source/target embedding 所构造的 16x16 dense queries cross-attend；Q/K 使用 spatial-only 2D RoPE，无 query self-attention。低分辨率 query feature 经三次 bilinear 2D residual upsampling 输出 `[B,K,3,128,128]`。唯一主损失是 train-only normalization 下按 pair 平均的 validity-masked SmoothL1。

默认固定 128×128、连续 21/24 帧；Wan 原生 temporal compression 给出 6 个 latent frames，不 padding、pooling 或插值。几何和评估按 source/pair chunk 处理。

## Code map

- `src/worldbridge/data.py`: MOVi-F native TFRecord adapter
- `src/worldbridge/geometry.py`: camera, pointmap, rigid trajectories, visibility/validity
- `src/worldbridge/models.py`: retained Compact/Full baselines
- `src/worldbridge/wan.py`: strict Wan VAE/DiT checkpoint adapters
- `src/worldbridge/dense4d.py`: feed-forward readout, 2D-RoPE cross-attention and dense upsampler
- `src/worldbridge/dense4d_data.py`, `dense4d_runtime.py`: pair maps, normalization and real-Wan runtime
- `src/worldbridge/pipeline.py`: retained blockwise baseline pipeline
- `src/worldbridge/losses.py`, `metrics.py`: objectives and metrics
- `scripts/`: audit, geometry visualization, train, evaluate, model smoke
- `tests/`: geometry, shape, coordinate mapping and decoder-gradient tests
- `configs/`: smoke, tiny-overfit and bounded Compact/Full configs
- `research/`: decisions, state, hypothesis and reproducibility notes
