# MOVi-F 四维世界潜变量研究代码

当前 H004 路线实现 **Dense-Query 4D Latent with Feed-forward Wan**：

```text
RGB [B,21,3,128,128]
 -> frozen Wan VAE clean latent [B,16,6,16,16]
 -> Wan DiT once at rectified-flow time 0
 -> negative raw velocity Z4D [B,16,6,16,16]
 -> full-memory dense (source,target) cross-attention query
 -> XYZ [B,K,3,128,128]
```

`Z4D` 没有真值或直接损失。模型只以 validity-masked XYZ SmoothL1 监督，同一全局 latent 回答 diagonal pointmap、前向/后向 tracking 和任意源帧 query。v1 使用独立 source/target embedding、spatial-only 2D RoPE、无 query self-attention 的 2-layer cross-attention，以及 bilinear+2D ResBlock upsampler；不输入相机、局部 RGB 或 visibility。

仓库也保留早期两个只使用 MOVi-F 真值几何/刚体轨迹的确定性 autoencoder 基线：

- **Compact**：第一帧锚定轨迹；不宣称任意源帧 tracking。
- **Full**：每个源帧、源像素编码完整 `X[s,t,p]` 轨迹。

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

H004/H005 的入口为 `scripts/dense4d_wan_audit.py`、`scripts/dense4d_decoder_smoke.py`、`train_dense4d.py` 和 `evaluate_dense4d.py`；配置顺序为 `configs/dense4d_smoke.yaml` → `dense4d_tiny_overfit.yaml` → `dense4d_arbitrary_overfit.yaml` → bounded `dense4d_train.yaml`。当前 canonical dense4d profiles 统一使用 source-frame 坐标、H005 triad hidden readout `[13,14,15]`、8 个 motion slots、每步一个 source 的全部 21 个 target 监督，以及 `trainable_mode: full` 的 DiT 微调。Wan backbone learning rate 为 `5e-5`，canonical profiles 使用配置中的 linear warmup；geometry adapter/decoder 使用各自的 `learning_rate` 或 `geometry_learning_rate`。坐标统计量必须由 `scripts/compute_coordinate_stats.py` 从 train split 生成，例如：`python scripts/compute_coordinate_stats.py --data-root /dataset/MOVi-F --output /tmp/worldbridge_dense4d/coordinate_stats_train_source.npz --coordinate-frame source`；Wan empty-UMT5 condition 由 `scripts/create_wan_empty_text_condition.py` 生成在仓库外。

原 Compact/Full geometry-autoencoder 路线仍保留为独立基线，使用上一段命令。历史 H004 source-centric velocity-readout、H005/H007 matched、layer-selection 和 motion-slot ablation 配置不再作为默认运行入口。

本项目没有自有 MLflow server，因此 `.pi-research/config.yaml` 中 MLflow 保持关闭，W&B tracking 保持开启。迁移到其他机器时，复制模板并在本地填写非敏感配置：

```bash
cp docs/tracking-env.example .env
# 编辑 .env；不要提交 .env
set -a; source .env; set +a
```

W&B API key 推荐使用 `wandb login` 写入机器本地凭据，或只在 shell 环境中设置 `WANDB_API_KEY`。`.env` 已被 Git 忽略；仓库只提交不含密钥的 `docs/tracking-env.example`。PRL 会为 W&B 注入 run ID、group 和 tags。当前训练脚本尚未调用 `wandb.init()`，因此若要上传训练曲线，还需要显式接入 W&B SDK。

## 其他服务器上的 Pi / PRL

仓库包含 `.pi/settings.json`，会在项目被 Pi 信任后自动安装并固定 `pi-research-loop` Git commit。也可以手动安装：

```bash
pi install git:github.com/kuan-er/pi-research-loop@1124d244b4b7df624c8ddb9b85da1b3864dd66dd
pi list
```

如果需要 shell 中的 `prl` 命令，使用 Pi 自带 Node 安装 Git 版本（该包目前未发布到 npm）：

```bash
PI_BIN="$(dirname "$(command -v pi)")"
"$PI_BIN/npm" install -g "git+https://github.com/kuan-er/pi-research-loop.git#1124d244b4b7df624c8ddb9b85da1b3864dd66dd"
export PATH="$PI_BIN:$PATH"
prl doctor
```

## 代码结构

- `src/worldbridge/data.py`：原生 TFRecord 适配器，只读解析。
- `src/worldbridge/geometry.py`：相机、pointmap、rigid trajectory、`M`/`V_valid`。
- `src/worldbridge/models.py`：早期 Compact/Full geometry-autoencoder 基线。
- `src/worldbridge/wan.py`：严格的 Wan2.1 VAE/DiT 原生 checkpoint adapter。
- `src/worldbridge/dense4d.py`：feed-forward Wan readout、global memory、2D RoPE cross-attention、dense upsampler 和 XYZ loss。
- `src/worldbridge/dense4d_data.py`、`dense4d_runtime.py`：balanced pair targets、train-only normalization 与 real-Wan 构建。
- `src/worldbridge/pipeline.py`：坐标统计、按源帧/像素块的 Full 编码和 query 采样。
- `src/worldbridge/losses.py`, `metrics.py`：validity masked Smooth-L1、visible/occluded 与重投影指标。
- `scripts/`：审计、几何诊断、训练、评估入口；`tests/`：几何、shape、latent mapping 和 decoder gradient。
- `research/`：PROJECT、STATE、DECISIONS 和 H001 hypothesis。

## 第一版已知限制

bounded 预算非常小（每个模型 20 steps、4 train clips），因此 validation 误差不是论文结果；它用于验证完整链路和公平的 query 口径。Full 目前参数少于 Compact 是因为 Compact 额外包含 reconstruction/fuse 分支；两者的 trajectory/context 层和超参数匹配。数据适配器的随机访问索引适合 tiny/bounded 实验，不是大规模生产输入管线。下一步最有价值的是在更多 train clips 上增加相同优化预算，并分别报告后出现物体、遮挡和静态背景。
