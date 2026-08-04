# MOVi-F 四维世界潜变量第一版

本仓库实现两个只使用 MOVi-F 真值几何/刚体轨迹的确定性 autoencoder：

- **Compact**：每个源像素使用第一帧锚定的完整目标轨迹 `q[p]`，只保留每帧 pointmap reconstruction 与第一帧 anchor；它不宣称任意源帧 tracking。
- **Full**：每个源帧、源像素编码完整 `X[s,t,p]` 轨迹，支持任意合法 `(s,t)` 查询，包括后出现物体。

RGB 仅用于诊断图；没有 RGB encoder、KL、扩散模型或 Wan 组件。输出 latent 始终为
`Z4D [B,Cz,Tz,Hz,Wz]`，其中时间轴是源帧时间。

## 数据审计得到的约定

数据目录只读：`/dataset/MOVi-F`。实际版本是 `128x128/1.0.0`，原生格式是 TFDS `tf.train.Example` TFRecords；数据字段包括 `video`、`depth`、`segmentations`、`camera/{positions,quaternions,focal_length,sensor_width,field_of_view}` 和 `instances/{positions,quaternions,is_dynamic,visibility,...}`，每条记录是 24 帧、128×128。实际存在 `train` 和 `validation` split。

审计/独立元数据比较解析出：

- depth 是 uint16 PNG，按 `metadata/depth_range` 线性解码；用静态前景光流的一致性比较选择相机径向距离（首样本 median world discrepancy 0.00914，z-depth 为 0.00959）。
- 相机四元数顺序为 **wxyz**，是 camera-local 到 Kubric world 的旋转；相机看向 local **-Z**，local Y 向上。物体四元数同为 wxyz、local-to-world；使用 `bboxes_3d` 的跨帧局部坐标残差验证（约 `1e-7`）。
- 焦距是 `35 / 32 * 128 = 140` 像素。数组像素使用中心坐标 `(W-1)/2,(H-1)/2`；native `instances/image_positions` 使用 edge-normalized 坐标，因此独立审计中会有约 0.5 像素的约定偏移。
- `segmentation=0` 是背景，`1..num_instances` 是实例。`V_valid` 来自源深度和刚体状态；`M` 单独由目标投影、正深度、边界、目标实例 ID 和目标 radial depth 一致性得到。遮挡点仍保持 `V_valid=1` 并进入坐标损失。
- 24 帧中取连续 `clip_length=21`，默认 `clip_start=0`，不 padding；这个决定避免引入额外 mask，已经记录在 `research/DECISIONS.md`。
- 目标深度遮挡阈值是 `0.05 + 0.01 * max(depth,1)` 米，均为配置项。

## 从零运行

依赖建议见 `requirements.txt`。所有研究运行应由 PRL Task 启动；下面的 `<TASK>` 是 `prl task start` 返回的 Task ID。数据审计命令本身不会写数据目录：

```bash
prl context
prl task start --hypothesis H001 --name four-d-world-latents
# 在返回的 worktree 中修改；从项目根启动时给 events 使用该 worktree 的绝对路径
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/inspect_dataset.py --data-root /dataset/MOVi-F
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/audit_conventions.py --data-root /dataset/MOVi-F
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/audit_depth.py --data-root /dataset/MOVi-F
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/validate_geometry.py --data-root /dataset/MOVi-F --output-dir /tmp/worldbridge-geometry
```

`validate_geometry.py` 检查 pixel→3D→pixel、`X[s,s,p]=P[s,p]`、投影后的实例/深度一致性，并保存 pointmap、轨迹和 segmentation 诊断图。`visualize.py` 是相同诊断的显式入口。

模型形状/反向 smoke 和单元测试：

```bash
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/model_shape_smoke.py
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python -m pytest -q
```

训练与评估：

```bash
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/train.py --config configs/compact.yaml
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/train.py --config configs/full.yaml
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/evaluate.py --checkpoint artifacts/compact_baseline/checkpoint.pt --split validation
prl run launch --task <TASK> --events <WORKTREE>/.pi-research/events.yaml -- \
  python scripts/evaluate.py --checkpoint artifacts/full_baseline/checkpoint.pt --split validation
```

建议顺序是 `configs/smoke.yaml` → `configs/tiny_compact.yaml`/`tiny_full.yaml` → bounded `compact.yaml`/`full.yaml`。训练会从 train split 计算坐标统计量，记录配置、seed、commit、硬件、峰值显存、吞吐，保存并立即恢复 checkpoint。`--resume` 支持恢复。`evaluate.py` 以空间 stride 和 query chunk 分块，不物化完整 `O(T²HW)` float32 查询；Full 的 `arbitrary_all_st` 表示所有源/目标时间组合（空间为可复现 stride 网格）。

若环境有 `MLFLOW_TRACKING_URI`，将 `.pi-research/config.yaml` 的 MLflow `enabled` 设为 true 即由 PRL 自动创建可选 run；没有该变量时本地 PRL 运行不受影响。

## 代码结构

- `src/worldbridge/data.py`：原生 TFRecord 适配器，只读解析。
- `src/worldbridge/geometry.py`：相机、pointmap、rigid trajectory、`M`/`V_valid`。
- `src/worldbridge/models.py`：共享 temporal encoder 类型、Compact/Full、matched 3D context、统一可微 query decoder。
- `src/worldbridge/pipeline.py`：坐标统计、按源帧/像素块的 Full 编码和 query 采样。
- `src/worldbridge/losses.py`, `metrics.py`：validity masked Smooth-L1、visible/occluded 与重投影指标。
- `scripts/`：审计、几何诊断、训练、评估入口；`tests/`：几何、shape、latent mapping 和 decoder gradient。
- `research/`：PROJECT、STATE、DECISIONS 和 H001 hypothesis。

## 第一版已知限制

bounded 预算非常小（每个模型 20 steps、4 train clips），因此 validation 误差不是论文结果；它用于验证完整链路和公平的 query 口径。Full 目前参数少于 Compact 是因为 Compact 额外包含 reconstruction/fuse 分支；两者的 trajectory/context 层和超参数匹配。数据适配器的随机访问索引适合 tiny/bounded 实验，不是大规模生产输入管线。下一步最有价值的是在更多 train clips 上增加相同优化预算，并分别报告后出现物体、遮挡和静态背景。
