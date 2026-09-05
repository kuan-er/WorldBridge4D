# 三数据集外部方法推理评估规范 V1

协议版本：`v1-three-sim3-metrics-diagonal-source-macro-20260905`

**核心约定：official-native 输入、固定 benchmark、统一 evaluator；tracking 拟合和评分均包含同帧项；永久保存指标，校验完成后删除大型预测。** 本修订替代旧版永久保留预测和强制报告 raw EPE 的要求，不代表旧结果已自动升级。

## 1. 输入与数据集

- 使用各方法官方推荐的帧数、分辨率、resize/crop/pad、normalization、checkpoint 和后处理，不强制统一成 256×256 或 21 帧输入。评测输出必须对应固定的 21 个 canonical timestamps；缺失结果不得静默插值、伪造或替换 clip。
- 输入类别分别报告：**A：RGB-only；B：RGB+calibration；C：RGB+geometry**。A 禁止输入 GT depth/XYZ/segmentation/visibility/poses/tracks；C 只能单独作为参考，不能混入 A/B 主榜。第一阶段优先 A。
- Official-native 不等于相同计算预算的公平比较；Common-input 需另立协议。
- 三个数据集分别冻结 `benchmark/<dataset>.jsonl`，每条含 `dataset, clip_id, parent_id, split, frame_indices, rgb_source, gt_source`，并保存 manifest SHA-256。
- train/validation/test 按 parent/scene 隔离，不使用训练 GT 评 validation。方法失败时保留原 clip，记录原因和覆盖率，不得静默跳过。

| 数据集 | RGB 根目录 | 读取方式 |
|---|---|---|
| Kubric/MOVi-F | `/dataset/nas0/yejun/MOVi-F/512x512` | TFRecord `video` |
| PointOdyssey | `/dataset/nas0/PointOdyssey` | MP4 |
| Dynamic Replica 官方 `valid` | `/dataset/data/Dynamic_dataset/dynamic_stereo/validation` | sequence PNG |

Dynamic Replica 必须使用上述官方 `validation` 根目录的 `frame_annotations_valid.jgz`、`<sequence>/images/` 及同 sequence annotations/geometry；RGBA 输入转 RGB 并记录。**禁止回退到 `dynamic_stereo/train` 或其本地 temporal split**；训练源仅用于训练或明确标注的辅助诊断。文件不可读时记录失败，不替换 split。

独立 adapter 必须声明输出类型、坐标系、单位、分辨率、source/target、原生预处理逆映射及 GT 对应规则，不能隐式修改方法输出或混用不同 head。3D 接口为 `xyz[target, xyz, source_pixel_y, source_pixel_x]`，临时 canonical 预测使用 float32。真实 `source=0` 轨迹不能用 `source=5` 冒充。

## 2. 三个必需主指标与 query

所有 EPE 单位为 **米，越低越好**。WorldBridge4D 自身和外部方法使用同一 query manifest，不运行 exhaustive 441-pair 主评测。

| 主指标 | 固定 JSON 字段 | 拟合与评分 query | 数量 |
|---|---|---|---:|
| Pointmap Sim(3) EPE：每帧三维重建精度 | `pointmap_sim3_epe_m` | `source=target=0..20` | 21 |
| Source=0 tracking Sim(3) EPE：首帧出发的跟踪精度，含首帧重建项 | `source0_tracking_sim3_epe_m` | `source=0, target=0..20` | 21 |
| Arbitrary tracking Sim(3) EPE：指定源帧出发的跟踪精度，含同帧重建项 | `arbitrary_tracking_sim3_epe_m` | `source∈[5,10,15,20], target=0..20` | 84 |

每个 clip 保留 **126 个逻辑评分条目**，对应 **121 个唯一 `(source,target)`**。五个同帧 query `(0,0)/(5,5)/(10,10)/(15,15)/(20,20)` 同时进入 pointmap 和相应 tracking 组，按各组变换独立评分，数值不必相同。能复用原始输出时不得为统计重叠重复运行模型；不同 head 必须记录实际来源。

Tracking **不排除 `source=target`**，因此不能称为纯跨帧 EPE。Arbitrary 仅使用四个 source，不混入 source=0；source-conditioned 方法最多执行四个 arbitrary source inference。不支持的指标记 `unsupported`/`null`，不能伪造。

## 3. Sim(3)、评分与汇总

### 3.1 对齐

默认在 evaluator 中用确定性 closed-form Umeyama 将预测对齐到 GT：

```text
aligned = scale * prediction @ rotation.T + translation
```

- Pointmap：每个 clip 的 21 个 diagonal pointmap 联合拟合 **一个**变换。
- Tracking：每个 clip、每个 source `0/5/10/15/20` 各拟合 **一个**变换，使用该 source 全部 21 个 target 的有效点，包括同帧项。完整 clip 共六个变换。
- 使用 proper rotation（`det(R)=+1`，尺度与反射修正一致）。不得逐 frame/point/trajectory 单独拟合，或跨 source 复用变换。
- 少于 3 个有效对应点、退化点集等记录具体失败状态，不能用单位变换冒充成功。其他算法（如官方 RANSAC）须记录实现、种子和采样规则，并与默认结果分开标注。
- GT 仅用于 evaluator 拟合/评分，不得泄漏到 RGB-only 推理输入。

### 3.2 EPE 与聚合

1. 每点误差为 `||aligned_prediction - GT||₂`。使用 GT-valid 且 GT 有限的点，target 遮挡但有效的点仍参加评分；这些点上的非有限预测记失败，不能通过过滤改善分数。
2. 每个 query 保存 `error_sum_m`、`valid_points` 和 `epe_m = error_sum_m / valid_points`。GT 空项记 `empty_gt`，误差和/点数为 0，EPE 为 `null`；缺预测、缺 GT 文件等不能当作空 GT。
3. **Pointmap / source=0**：分别在各自 21 个 query 上求 `sum(error_sum_m) / sum(valid_points)`。
4. **Arbitrary**：先在每个 source 的 21 个 target 内按有效点加权求 EPE，再求 `(EPE_5 + EPE_10 + EPE_15 + EPE_20) / 4`。保持历史四 source 等权口径，不将全部 84 对的点合并加权。
5. 某组没有有效评分点，或必需 source 无法拟合/评分时，该组主指标为 `null`；部分均值只能另存为明确标注的诊断。
6. **数据集**：对每项成功评测的 clip EPE 等权平均，同时报告覆盖率。**Macro Average**：三个数据集指标等权平均；不足三个可用数据集时为 `null`，不得冒充完整平均。

Raw EPE、XYZ MAE、visible/occluded-valid、short/long-gap、late-appearing、重投影误差均为可选诊断；若配置启用，须在清理预测前保存。Visibility 仅用于诊断分组。仅输出 2D tracks 的方法单独报告 pixel error、PCK、visible/occluded、long-term 和 failure rate，不做 Sim(3)，不与 3D EPE 混列。

## 4. 最终输出格式

所有外部 repo、权重、配置、benchmark 和运行输出置于 `/data/WorldBridge4D-inference/`，不得提交到主代码 Git。每次运行保存：

```text
results/<method>/<run_id>/
├── run_manifest.json          # 配置/代码/数据来源、校验值、命令和运行元信息
├── resolved_config.yaml
├── environment.txt
├── stdout.log
├── predictions/               # 临时 .npz/.safetensors；清理后可为空
├── metrics/<clip_id>.json     # 永久保存逐 query / clip 指标和对齐变换
├── metrics.json               # 数据集及 Macro Average 汇总
├── failures.jsonl
└── cleanup.jsonl              # 永久保存删除审计
```

### 4.1 逐 clip JSON

下面是 **pending 模板**，不是实际结果。三个主指标字段必须始终存在；无值用 `null`，禁止 NaN/Infinity。

```json
{
  "metric_protocol_version": "v1-three-sim3-metrics-diagonal-source-macro-20260905",
  "method": "vdpm",
  "dataset": "pointodyssey",
  "clip_id": "pointodyssey/val/example/start000000",
  "unit": "meters",
  "status": "pending",
  "metrics": {
    "pointmap_sim3_epe_m": null,
    "source0_tracking_sim3_epe_m": null,
    "arbitrary_tracking_sim3_epe_m": null
  },
  "metric_status": {
    "pointmap_sim3_epe_m": "pending",
    "source0_tracking_sim3_epe_m": "pending",
    "arbitrary_tracking_sim3_epe_m": "pending"
  },
  "aggregation": {
    "pointmap": "valid_point_weighted_over_21_diagonal_queries",
    "source0_tracking": "valid_point_weighted_over_21_targets_including_diagonal",
    "arbitrary_tracking": "valid_point_weighted_per_source_over_21_targets__equal_mean_of_4_sources",
    "dataset": "equal_mean_of_successful_clip_metrics",
    "macro_average": "equal_mean_of_3_dataset_metrics"
  },
  "include_diagonal_in_tracking_score": true,
  "arbitrary_tracking_sim3_epe_m_per_source": {"5": null, "10": null, "15": null, "20": null},
  "expected_scoring_queries": {"pointmap": 21, "source0_tracking": 21, "arbitrary_tracking": 84},
  "query_metrics": [],
  "alignment": {
    "algorithm": "closed_form_umeyama",
    "direction": "prediction_to_gt",
    "formula": "aligned = scale * prediction @ rotation.T + translation",
    "transforms": []
  },
  "provenance": {
    "run_manifest": "run_manifest.json",
    "run_manifest_sha256": null,
    "evaluator_commit": null,
    "dataset_manifest_sha256": null,
    "prediction_files": []
  },
  "cleanup": {"state": "retained", "audit_file": "cleanup.jsonl"}
}
```

完整输出必须满足：

| 内容 | 必需字段/约束 |
|---|---|
| `status` | 三项均成功才为 `succeeded`；否则 `pending/partial/failed` |
| `metric_status` | 每项为 `succeeded/pending/unsupported/insufficient_valid_points/degenerate_fit/failed`；非成功项附原因 |
| `query_metrics` | **126 条**；每条含 `group`（`pointmap/source0_tracking/arbitrary_tracking`）、`source, target, error_sum_m, valid_points, epe_m, status`；GT 空项按 §3.2，其他失败误差值为 `null` |
| `arbitrary_tracking_sim3_epe_m_per_source` | 四个 source EPE；任一缺失则主 arbitrary EPE 为 `null` |
| `alignment.transforms` | **六条**；每条含 `group, source, fit_scope, fit_targets, fit_points, scale, rotation, translation, status`；pointmap 的 source 为 `null`，fit_targets 为完整 `0..20`，rotation 为 3×3、translation 为 3；失败变换值为 `null` 并附原因 |
| `provenance` | 填入实际校验值；`prediction_files` 保存原始/对齐预测的路径、字节数和 SHA-256 |

必须保留逐 query 统计和变换，不能只保留三个均值。

### 4.2 汇总与复现记录

`metrics.json` 含 `metric_protocol_version, method, input_type, unit, aggregation`，以及 `datasets`（键 `kubric/pointodyssey/dynamic_replica`）和 `macro_average`：

- 每个数据集保存同名三项 `metrics`；保存 `expected_clips, evaluated_clips, succeeded_clips, partial_clips, failed_clips, pending_clips`。四种状态之和等于 manifest clip 数，`evaluated = expected - pending`。
- `metric_coverage` 按三个指标键保存 `succeeded_clips, expected_clips, coverage`（两者之比）和失败原因计数，包括 unsupported/有效点不足。
- `failure_rate = failed / expected`；`incomplete_rate = (partial + failed + pending) / expected`。分母为 0 时为 `null`。
- `macro_average.metrics` 使用同样三个键，逐项记录 `contributing_datasets`，按 §3.2 聚合。任何缺项都须标注覆盖率，不能称为全量完成。

主表列为：`Method | Input Type | Dataset | Pointmap Sim(3) EPE | Source=0 tracking Sim(3) EPE | Arbitrary tracking Sim(3) EPE | Coverage(each) | Failure Rate | Incomplete Rate`。每方法有三个数据集及 Macro Average 四行，无值填 `N/A` 并说明原因。

运行 manifest/配置/环境须保存：代码与 evaluator commit、checkpoint/benchmark SHA-256、命令、seed、设备/dtype、包/CUDA 版本、实际输入帧索引/timestamps/分辨率、crop/resize/pad/normalization、坐标单位及 GT 映射、额外输入/后处理/test-time optimization，以及逐 clip 耗时、峰值显存、成功/失败原因和确定性信息。确定性方法运行一次，随机方法至少三个 seed，报告 `mean ± sample std`。

## 5. 指标完成后删除预测

**按 clip 校验后删除原始及对齐的大型预测，释放磁盘空间；推荐只在内存中对齐，不再落盘第二份大数组。** 预测在评测完成前须完整保留，可分 chunk 推理但不能丢失必需数据。

1. 完成三个主指标和已启用的可选诊断。原子持久化逐 clip JSON（临时文件、flush/fsync、rename、同步目录），再读回校验。
2. 校验三个主指标均成功、有限非负，126 条 query 和六个变换完整，source/target/GT 对应正确，逐 query 统计能按 §3.2 重建汇总；验证 provenance、预测和 JSON 校验值。**不能仅凭 JSON 存在或旧 `status=succeeded` 就删除。**
3. 更新并持久化汇总及覆盖率。同步写入 `cleanup.jsonl` 的 `authorized` 记录：run/clip、指标 JSON SHA-256、待删准确路径/SHA-256/字节数、时间。
4. 仅删除清单中该 clip、该 run 的预测；记录 `deleted`、释放字节数、时间/错误，更新 `cleanup.state`。若 JSON 随清理状态更新，另记新 SHA-256，保留清理前校验值。允许幂等重试，不得通配删除未评测文件。
5. 永久保留指标、变换、配置、manifest、日志、校验和失败/删除审计。不得删除数据集、GT、checkpoint、repo 或其他运行文件。

`pending/partial/failed`、有效点不足或必需组 unsupported 时**禁止自动删除**，需另行明确确认才能放弃预测。恢复时先检查指标及清理记录，不因“预测文件已不存在”就重复推理。

删除后不能重新可视化、修改评分协议或补算未保存指标；需要这些功能时必须重新推理。

## 6. 历史兼容与执行流程

旧 `tracking_sim3_epe_m_mean` 的 84-pair、四 source 等权口径与本协议一致。核对 `sources=[5,10,15,20]`、`targets=0..20`、四个有限 EPE 及均值后，可保留其数值，标记 `origin="legacy_json"`、原始协议/evaluator、源 JSON 路径和 SHA-256。但 GT 映射、拟合器及非有限预测处理仍须审计后才能混入同一主榜。

执行顺序：每方法每数据集先用 3–5 clips 做 smoke test（依赖、权重、RGB/shape、坐标单位、evaluator、无 GT 泄漏）→ 冻结配置跑完整 validation → 冻结方法/checkpoint/adapter 后跑 test，test 不再调参。两阶段均按固定 query 输出指标并按 §5 清理，不以 exhaustive 441-pair 替代。
