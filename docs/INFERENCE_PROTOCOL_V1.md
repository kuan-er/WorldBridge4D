# 三数据集外部方法推理评估规范 V1

## 1. 目的与原则

本规范用于在 Kubric/MOVi-F、PointOdyssey 和 Dynamic Replica 上运行外部方法（例如 VDPM），并保留可复现、可审计的推理结果。

第一版固定采用：

> **Official-native input + fixed dataset benchmark + unified evaluator + separate input-class reporting**

也就是说：

- 每个方法使用其官方推荐的输入分辨率、帧数、resize/crop、normalization、checkpoint 和后处理；
- 不强制所有方法使用 `256×256`；
- 不强制所有方法使用 21 帧；
- 只统一 benchmark clip 列表、数据划分、GT 评测、禁止 GT 泄漏、结果格式和审计记录；
- RGB-only、RGB+calibration、RGB+geometry 方法分开报告。

Official-native 结果用于回答“方法按照官方设置的实际能力是多少”，不等同于严格相同计算预算下的公平比较。如需要计算预算受控的比较，应另立 Common-input protocol，不得覆盖本规范的结果。

## 2. 外部方法分类

### Group A：RGB-only

只允许输入 RGB 视频或 RGB 帧。例如只接收视频的 point tracking、pointmap 方法。

### Group B：RGB + calibration

除 RGB 外，允许使用方法官方要求的 intrinsics 或 camera poses。

### Group C：RGB + geometry

允许使用 depth、segmentation、GT tracks 或其他 GT 几何信息。

Group C 结果不能与 Group A/B 合并为同一主榜单，只能作为独立参考或 upper-bound。第一阶段优先运行 Group A。

## 3. 外部方法目录

所有外部方法、checkpoint、配置和输出统一放在独立目录，不写入 WorldBridge4D 主代码仓库：

```text
/data/WorldBridge4D-inference/
├── repos/
│   ├── vdpm/
│   ├── <method_x>/
│   └── <method_y>/
├── checkpoints/
│   ├── vdpm/
│   ├── <method_x>/
│   └── <method_y>/
├── configs/
│   ├── vdpm.yaml
│   └── <method_x>.yaml
├── benchmark/
│   ├── kubric.jsonl
│   ├── pointodyssey.jsonl
│   └── dynamic_replica.jsonl
└── results/
    └── <method>/<run_id>/
```

每次运行的结果目录至少包含：

```text
<method>/<run_id>/
├── run_manifest.json
├── resolved_config.yaml
├── environment.txt
├── stdout.log
├── predictions/
├── metrics.json
└── failures.jsonl
```

模型权重、外部 repo、推理结果和生成中间文件不得提交到 `/data/WorldBridge4D` 的 Git 仓库。

## 4. 数据集与 benchmark manifest

三个数据集分别使用固定 manifest：

```text
benchmark/
├── kubric.jsonl
├── pointodyssey.jsonl
└── dynamic_replica.jsonl
```

每条记录至少包含：

```json
{
  "dataset": "pointodyssey",
  "clip_id": "...",
  "parent_id": "...",
  "split": "validation",
  "frame_indices": [...],
  "rgb_source": "...",
  "gt_source": "..."
}
```

要求：

- benchmark clip 列表冻结后不能因为某方法失败而更换；
- train 和 validation/test 必须按 parent/scene 隔离；
- 不使用训练 GT；
- 方法无法处理某个 clip 时记录 failure，不得静默跳过；
- 先使用 validation 做适配和比较；
- 方法、配置和 checkpoint 冻结后再运行 test；
- 结果按三个数据集分别报告，再计算三个数据集等权的 macro average；
- 不能按 clip 数量直接合并平均，避免 PointOdyssey 的 clip 数量主导总结果；
- Dynamic Replica 的 validation 必须标记为 temporal holdout；当前它不是独立 validation scene holdout。

## 5. 当前 RGB 数据源

当前机器上三个数据集的 RGB 均可访问，但格式不同：

| 数据集 | RGB 数据源 | 输入读取方式 |
|---|---|---|
| Kubric/MOVi-F | `/dataset/nas0/yejun/MOVi-F/512x512` | 从 TFRecord 的 `video` 字段解码 |
| PointOdyssey | `/dataset/nas0/PointOdyssey` | 从 MP4 解码 |
| Dynamic Replica | `/dataset/data/Dynamic_dataset/dynamic_stereo` | 按 stream 顺序读取 PNG |

各外部方法使用自己的官方 RGB loader。Dynamic Replica 的 PNG 可能带 alpha 通道，RGB-only 方法输入前须将 RGBA 转成 RGB，并在配置中记录该处理。

## 6. Official-native 输入协议

每个方法允许使用其官方推荐的：

- 输入帧数和 temporal sampling；
- 输入分辨率；
- resize、crop 或 padding；
- RGB normalization；
- frame ordering；
- checkpoint 和官方后处理。

但是 benchmark 的 clip/scene 身份和可使用的原始 RGB 帧必须由固定 manifest 指定。方法不能自行替换 benchmark clip 或使用其他 split 的数据。

## 7. 固定预算 arbitrary tracking

第一阶段不进行 exhaustive arbitrary audit，不要求每个方法运行全部 `21×21=441` 个 `(source,target)` 组合。

主评测固定使用以下 query 集合：

```text
pointmap:
  source=target=0..20                         # 21 pairs

first_frame_tracking:
  source=0, target=0..20                      # 21 pairs

arbitrary_tracking:
  source ∈ [5, 10, 15, 20]
  target ∈ [0, 1, ..., 20] 且 target != source # 4×20=80 pairs
```

因此每个 clip 的主评测包含 122 个逻辑 query 条目，其中 arbitrary tracking 固定为 80 个 source-positive pairs。`(source=0,target=0)` 同时属于 pointmap 和 first-frame tracking，分别进入两组汇总，因此 122 个逻辑条目对应 121 个唯一 `(source,target)` 推理结果；实现不得为这个重叠条目重复运行模型。固定 source 覆盖中间帧、forward/backward、短间隔和长间隔；`source=0` 单独作为 first-frame tracking，不混入 arbitrary 指标。

所有方法使用相同的 source/target manifest。一次性输出完整视频轨迹的方法只需由 evaluator 抽取这 122 个逻辑条目；source-conditioned 方法最多执行 4 个 arbitrary source inference。只能处理 `source=0` 的方法可以参加 pointmap 和 first-frame tracking，但 arbitrary tracking 记为 `N/A`。WorldBridge4D 自身的 validation evaluator 也必须使用同一 fixed-budget query manifest，不再将 exhaustive `21×21` 结果作为主诊断。

方法必须能将输出对齐到 benchmark 的 canonical timestamps，才能参加这组 arbitrary 指标；不能对缺失的 source/target 结果静默插值或伪造。若官方 temporal protocol 不支持这些时间点，应记录为 unsupported/failure，并报告覆盖率。

每次运行必须记录实际：

- 输入帧数；
- 输入 frame indices/timestamps；
- 输入分辨率；
- crop/resize/pad 规则；
- normalization；
- 是否使用额外输入；
- 是否执行 test-time optimization。

禁止向 RGB-only 方法提供：

- GT depth；
- GT XYZ；
- GT segmentation；
- GT visibility；
- GT camera pose；
- GT tracks。

如果某方法官方要求 intrinsics、camera pose 或 depth，必须归入相应输入类别，不能和 RGB-only 结果混合比较。

## 8. 输出与 adapter

推理阶段保留外部方法的原始官方输出。评测阶段通过独立 adapter 转换为统一评测接口，不在 evaluator 中隐式修改方法输出。

统一元数据格式：

```json
{
  "method": "vdpm",
  "dataset": "kubric",
  "clip_id": "...",
  "coordinate_frame": "...",
  "unit": "meters",
  "output_type": "dense_3d_pointmap",
  "source_frames": [...],
  "target_frames": [...],
  "resolution": ["H", "W"]
}
```

外部方法可能输出：

- 2D point tracks；
- 3D point tracks；
- per-frame depth；
- dense pointmaps；
- world-coordinate points；
- camera-coordinate points；
- inverse depth。

adapter 必须显式声明输出类型、坐标系、单位和转换方法。不能把不同坐标定义隐式视为相同。

对于能输出 3D pointmap/3D tracking 的方法，评测接口应能表达：

```text
xyz[target, xyz, source_pixel_y, source_pixel_x]
```

如果方法只能输出 2D tracks，则只进入 2D tracking 评测，不得伪造 3D EPE。

## 9. 评测指标

### 9.1 3D pointmap / 3D tracking

统一报告：

- 3D endpoint error（EPE，单位米）；
- XYZ MAE；
- visible EPE；
- occluded-valid EPE；
- `source=0` tracking EPE；
- fixed-budget arbitrary source-target EPE；
- short-gap / long-gap EPE；
- late-appearing EPE；
- target-camera reprojection error（适用时）。

其中：

- pointmap 使用 `source=target`；
- first-frame tracking 使用 `source=0`；
- arbitrary tracking 使用固定 80 个合法的 `(source,target)` 查询；
- visibility 只用于分组统计；
- GT validity/occlusion mask 只在 evaluator 中使用。

### 9.2 默认 Sim(3) 对齐

对于 RGB-only 单目 3D/4D 重建和 3D tracking，**默认主指标必须使用 Sim(3) 对齐**。Sim(3) 包含统一尺度、旋转和平移，用于消除单目方法输出坐标系与 GT 坐标系之间的 gauge ambiguity；它不消除几何形状或运动误差。

固定规则如下：

- tracking：每个 clip、每个 source 单独拟合一个 Sim(3)，使用该 source 的全部 21 个 target 和全部有效 GT 点；
- pointmap：每个 clip 拟合一个 Sim(3)，使用 21 个 diagonal pointmaps 的全部有效 GT 点；
- 不允许逐 frame、逐 point 或逐 trajectory 单独拟合；
- GT 只允许在 evaluator 中用于拟合和评分，绝不能作为 RGB-only 方法的输入；
- 原始未对齐预测必须保留，作为 raw metric-scale diagnostic；
- 主表报告 Sim(3)-aligned EPE，另报告 raw EPE；
- 对齐变换、拟合点数、拟合范围和算法必须写入 metrics/metadata；
- 2D tracking 不进行 Sim(3) 对齐。

默认 evaluator 使用确定性的 closed-form Umeyama Sim(3) 拟合；若复现外部方法官方 RANSAC/对齐实现，必须显式记录实现、随机种子和拟合采样规则，不能与默认结果混称。

因此，当前 V1 的 3D 主指标定义为：

```text
primary = Sim(3)-aligned EPE
secondary = raw EPE
```

### 9.3 2D tracking

只能输出 2D track 的方法单独报告：

- average pixel error；
- PCK；
- visible/occluded tracking；
- long-term tracking；
- failure rate。

2D 指标不能与 3D EPE 排在同一指标列中。

### 9.4 汇总格式

```text
Method | Input Type | Dataset | Pointmap EPE | Tracking EPE | Occluded EPE | Late EPE | Failure Rate
```

每个方法必须有：

```text
Kubric
PointOdyssey
Dynamic Replica
Macro Average
```

## 10. 可复现性记录

每个方法、每个数据集、每次运行必须保存：

```json
{
  "method": "vdpm",
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "code_repo": "...",
  "code_commit": "...",
  "dataset_manifest_sha256": "...",
  "input_protocol": "official_native",
  "seed": 2026,
  "device": "cuda:0",
  "dtype": "float16",
  "extra_inputs": [],
  "postprocess": "...",
  "status": "succeeded"
}
```

同时记录：

主 benchmark 的 tracking 推理结果必须保存，不仅保存汇总指标。每个 clip 至少保存：

```text
predictions/<clip_id>.safetensors
```

其中包含四个 arbitrary source（`5/10/15/20`）对应的全部 80 个 target 预测，以及 pointmap/first-frame 结果（如果方法支持）。canonical prediction 使用 `float32`，并记录 source、targets、坐标系、单位、输出分辨率和 SHA-256。推理过程中仍可按 chunk 计算，但不得因 chunk 而丢弃主 benchmark 预测。

第一阶段不运行和保存 exhaustive 441-pair 结果；相关目录和指标不作为 V1 必需产物。

同时记录：

- inference command；
- resolved method config；
- Python/package/CUDA 环境；
- checkpoint SHA-256；
- 每个 clip 的推理耗时；
- 峰值显存；
- 成功和失败 clip 数；
- 每个失败的具体原因；
- 是否为确定性运行；
- 随机方法使用的 seed。

确定性方法默认运行一次；随机方法至少运行 3 个 seed，并报告 `mean ± sample std`。

## 11. 运行阶段

### Stage 1：Smoke test

每个方法在每个数据集上先运行 3–5 个 clip，检查：

- repo 和依赖可运行；
- checkpoint 可加载；
- RGB loader 正常；
- 输出非空且 shape 正确；
- 坐标系和单位明确；
- 输出可被 evaluator 读取；
- 不发生 GT 输入泄漏。

### Stage 2：Validation benchmark

配置冻结后运行完整 validation split，并保存每个 clip 的固定预算 prediction：

```text
Kubric validation
PointOdyssey validation
Dynamic Replica validation
```

此阶段用于正式方法比较，但不再因单个方法表现调整 benchmark clip。

### Stage 3：Final test

所有方法、配置、checkpoint 和 adapter 冻结后运行 test split。test 结果只用于最终报告，不再进行调参或更改预处理。test 也只运行固定预算 query，不进行 exhaustive audit。

## 12. 推荐统一入口

未来统一入口可以采用：

```bash
python scripts/inference/run_method.py \
  --method vdpm \
  --config /data/WorldBridge4D-inference/configs/vdpm.yaml \
  --benchmark /data/WorldBridge4D-inference/configs/benchmark_three_dataset_v1.yaml \
  --output /data/WorldBridge4D-inference/results/vdpm/<run_id>
```

第一阶段先接入 RGB-only 方法。当前机器尚未确认 VDPM 的 repo 地址和官方 checkpoint 地址；在获得这两个地址后，先执行 Stage 1 smoke test，再进入完整 validation。
