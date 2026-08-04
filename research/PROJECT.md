# MOVi-F 四维世界潜变量

## Goal
在真实 Kubric MOVi-F 128×128 数据上，以真值三维几何和刚体轨迹为唯一 encoder 输入，验证两种确定性 latent autoencoder 是否能够从源帧/像素查询解码目标时刻三维位置：Compact（第一帧锚定轨迹）与 Full（每个源像素完整轨迹）。主要科学比较是 Full 在 arbitrary-source、后出现物体和遮挡点上是否优于 Compact，同时保持 reconstruction 与 first-frame tracking 的公平 query 口径。

## Method
数据适配器直接解析只读 TFRecord `tf.train.Example`。pointmap 由 uint16 depth、相机内外参反投影到 Kubric world，再转换到 clip 第一帧相机坐标。刚体轨迹按 segmentation 实例、实例局部坐标及目标姿态构造；`M` 是目标投影/实例/深度的 visibility，`V_valid` 是数据/深度/变换有效性，二者不混用。Compact 与 Full 使用同类 temporal trajectory encoder 和匹配 3D residual context encoder，输出可配置 `Z4D [B,Cz,Tz,Hz,Wz]`；统一 decoder 先对源时间/像素进行可微三线性采样，再用 Fourier-coordinate MLP 查询目标时间。坐标统计量只取 train split，loss 是 validity-masked Smooth-L1，评估按 visible/occluded、重建/跟踪和目标相机重投影误差分组。

默认基线为 128×128、clip 21/24、Cz 16、空间 8×、Tz 6；不做 padding。Full 的 O(T²HW) 几何和查询均按源帧、像素块或 query chunk 计算。

## Code map

- `src/worldbridge/data.py`: MOVi-F native TFRecord adapter
- `src/worldbridge/geometry.py`: camera, pointmap, rigid trajectories, visibility/validity
- `src/worldbridge/models.py`: Compact/Full and unified query decoder
- `src/worldbridge/pipeline.py`: blockwise geometry-to-model conversion and queries
- `src/worldbridge/losses.py`, `metrics.py`: objectives and metrics
- `scripts/`: audit, geometry visualization, train, evaluate, model smoke
- `tests/`: geometry, shape, coordinate mapping and decoder-gradient tests
- `configs/`: smoke, tiny-overfit and bounded Compact/Full configs
- `research/`: decisions, state, hypothesis and reproducibility notes
