# WorldBridge4D 256×256 三数据集三天训练方案

状态：**训练前方案，尚未完成 256 路线的工程验证或质量验证**  
日期：2026-08-12

## 1. 目标与边界

本轮目标是在 4×A100 80GB、约 72 小时 wall-clock 预算内，直接检验扩大输入信息、有效 Wan 深度、动态数据覆盖和训练预算后，WorldBridge4D 的 4D 几何与跟踪质量能达到什么水平。

这不是单变量消融。本轮同时改变分辨率、readout、文本条件和训练数据，因此结果用于筛选一条强配置，不能把收益单独归因于其中某个组件。

保持以下方法边界：

- 单次 Wan 视频编码，独立 `(source,target)` dense query；
- 不加入 RGB local patch；
- 不加入 query self-attention；
- 不加入 visibility、normal、confidence 等辅助 head 或辅助损失；
- 唯一训练目标仍为 validity-masked XYZ SmoothL1；
- visibility 只用于分桶评估，不从 XYZ loss 中移除 occluded-valid 点。

## 2. 模型配置

### 2.1 输入和 Wan latent

- clip：连续 21 帧；
- RGB 输入：`256×256`；
- frozen Wan VAE clean latent：`[B,16,6,32,32]`；
- Wan DiT：Wan2.1-T2V-1.3B，flow time 固定为 0；
- hidden readout：zero-based blocks `[13,14,15,29]`；
- 跳过生成用 `norm_out`、`proj_out` 和 `scale_shift_table`。

由于最大 readout 是 block 29，全部 30 个 DiT blocks 都进入 forward；在 full-finetuning 模式下，除显式绕过的输出参数外，全部 block 都可接收几何监督梯度。

### 2.2 Readout 融合初始化

四层初始融合概率固定为：

```text
block 13: 0.30
block 14: 0.30
block 15: 0.30
block 29: 0.10
```

对应 softmax logits 可设为：

```text
[0.0, 0.0, 0.0, -1.0986122887]
```

不使用 entropy regularization 或 top-k gate。训练中记录四个有效融合权重。该初始化保留已经验证的 `[13,14,15]` 中层 triad 为主路径，同时允许 block 29 的贡献随训练增加。

### 2.3 Geometry adapter 和 decoder

采用当前约 100M 非 Wan readout 配置：

```yaml
geometry_dim: 512
geometry_spatial_size: 32
geometry_num_heads: 8
geometry_clean_skip: true
motion_slots: 8
structured_local_queries: true

query_dim: 1024
embedding_dim: 512
num_cross_attn_layers: 5
num_heads: 8
query_grid_size: 32
upsample_channels: [1024, 512, 256, 128]
output_size: 256
```

空间解码路径为：

```text
32 → 64 → 128 → 256
```

这里的 `structured_local_queries` 指 source-aligned Wan geometry plane，不是额外的高分辨率 RGB local branch。

## 3. Wan 文本条件

### 3.1 固定 prompts

三个数据集使用固定、caption-style、结构匹配的英文描述。描述只包含视频中可观察的场景和运动语义，不包含数据集名称、标注密度、监督类型或任务答案。

**Kubric MOVi-F**

```text
A rendered monocular video of multiple rigid objects moving independently in a three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.
```

**PointOdyssey**

```text
A rendered monocular video of articulated characters and objects undergoing diverse rigid and non-rigid motion in a three-dimensional scene. The camera viewpoint may change over time, and objects may become occluded and reappear.
```

**Dynamic Replica**

```text
A rendered monocular video of articulated people moving through a furnished indoor three-dimensional scene. The camera viewpoint may change over time, and people may become occluded and reappear.
```

本轮不使用 prompt dropout、prompt augmentation、CFG 或可学习 prompt token。

### 3.2 编码和注入

- 使用 Wan 原生 UMT5-XXL、相同 tokenizer 和官方 checkpoint 离线编码；
- 每个 condition padding 为 `[1,512,4096]`；
- T5 不进入训练，不更新参数；
- 保存精确 prompt、未 padding token 数、tokenizer 路径/版本、T5 checkpoint SHA-256 和 Wan 源码 commit；
- condition 必须作为每次 forward 的显式输入，不能通过逐步修改全局 buffer 实现；
- 一个 optimizer update 内所有 rank 和 accumulation microsteps 使用同一个数据集及其 prompt。

正式评估使用对应数据集的正确 prompt。训练后可在固定小 validation subset 上额外做 correct/wrong/empty prompt permutation 诊断，但它不替代主结果。

## 4. 256×256 数据预处理

现有 `128×128` RGB 或 `16×16` latent 不能上采样复用。三个数据集都必须从原始数据重新构造 256 输入和 `32×32` clean latent cache。

### 4.1 RGB

- 从原始分辨率按版本化 crop/resize 规则生成 256 RGB；
- 使用 bicubic + antialias 或严格 area downsampling；
- 不使用隔点抽样；
- PointOdyssey 和 Dynamic Replica 复用 128 路线的物理 crop/FOV，只提高输出栅格分辨率。

### 4.2 相机内参

对缩放比例 `r`，按 pixel-center 一致规则更新：

\[
f'_x=r f_x,\quad f'_y=r f_y,
\]

\[
c'_x=r(c_x+0.5)-0.5,\quad c'_y=r(c_y+0.5)-0.5.
\]

如果包含 crop，先将 principal point 减去 crop origin，再进行缩放。

### 4.3 几何 GT

不能对 XYZ、instance identity 或跨帧轨迹进行普通 bilinear/area averaging。不同表面在边界处的平均会生成不存在的 3D 点。

- Kubric：由 256 栅格的 depth、segmentation、camera 和 object transforms 重新生成 source point 与轨迹；
- PointOdyssey：按 256 crop/intrinsics 重新投影原始 track identity，解决像素碰撞后保留同一物理点的完整 target trajectory；
- Dynamic Replica：在 256 source grid 上重新生成 diagonal geometry，并把 persistent mesh tracks 映射到相同 source grid；
- validity 与 visibility 分离；occluded-valid XYZ 继续进入 loss。

训练前必须通过：

1. pixel→3D→pixel round trip；
2. `X[s,s,p] = P_s[p]` diagonal identity；
3. Kubric rigid motion consistency；
4. source/target track identity consistency；
5. 至少一个 occluded-valid 样本；
6. RGB、depth、segmentation、tracks 和 intrinsics 使用完全相同的 256 pixel grid；
7. VAE latent 严格为 `[16,6,32,32]`，无 padding、pooling 或 latent interpolation。

## 5. 训练 query：一个 source，K 个 targets

### 5.1 为什么不一次训练全部 21 个 targets

K 只降低 target-dependent decoder 开销，不降低 Wan 视频编码开销。256 路线中：

```text
Wan tokens:          6×32×32 = 6,144
Dense memory tokens: 21×32×32 = 21,504（另加 motion slots）
Queries per target:  32×32 = 1,024
```

cross-attention 主计算近似正比于 `K × N_query × N_memory`。相对 K=21：

- K=6 的 target-dependent decoder 开销约为 `6/21 = 28.6%`；
- K=4 约为 `4/21 = 19.0%`。

K 是显存和吞吐控制，不限制推理时可查询的 target 数。

### 5.2 正式采样

对每个 clip：

1. 从 `0..20` 均匀采样一个 source `s`；
2. 预先确定至少含一个 valid point 的 eligible target set `V_s`；
3. 从 `V_s` 无放回均匀采样 K 个 target；若 `|V_s|<K`，使用全部 eligible targets 并按实际数量归一化；
4. 对选中 `(s,t)` pair 的全部 valid source-grid points 计算 XYZ loss。

不强制包含 diagonal，也不额外分层。由于每个 eligible target 的包含概率相同，普通 K-pair 均值是全 eligible-target 均值的无偏估计：

\[
\mathbb{E}\left[\frac{1}{K}\sum_{t\in S_K}L_{s,t}\right]
=\frac{1}{|V_s|}\sum_{t\in V_s}L_{s,t},\qquad |V_s|\ge K.
\]

Kubric/DR 的 dense clips 通常有 `V_s={0,…,20}`，此时右侧就是 21-target 平均。PointOdyssey 只在确有监督的 target 上定义该目标。每个 pair 先对自身 valid pixels 求均值，再对实际采样的 pairs 和 batch 求均值；完全无 valid point 的 pair 不能作为零损失。

### 5.3 K 的选择

首选 K=6。在正式训练前运行 100–200 个真实 optimizer updates 的系统 gate：

- 4×A100 80GB；
- BF16；
- FSDP FULL_SHARD；
- microbatch 1/GPU；
- gradient accumulation 2；
- 包含 optimizer state、backward、clip 和 checkpoint smoke。

若 K=6 OOM、peak allocated 超过约 72–74GiB，或吞吐明显不可接受，则固定退到 K=4。该 gate 只选择可运行配置，不构成质量消融。正式 run 启动后不改变 K。

## 6. 损失

唯一训练目标为现有 normalized XYZ SmoothL1：

\[
L_{s,t}=\frac{1}{|A_{s,t}|}
\sum_{p:A_{s,t,p}=1}
\operatorname{SmoothL1}(\hat X_{s,t,p},X_{s,t,p};\beta=0.05).
\]

规则：

- 使用 validity `A`，不使用 visibility `M` mask XYZ；
- 每个 pair 独立归一化，避免 dense pair 因点数多而压倒 sparse pair；
- 不加入 visibility、normal、confidence、reprojection 或 motion auxiliary loss；
- 三个数据集共用 source-camera、meter 单位和一组 train-only mixture coordinate normalization；
- mixture statistics 按训练采样概率和每 clip 固定数量的 valid point 估计，不能让 dense 数据仅凭像素数量主导统计量；
- validation/test 不得参与 normalization。

## 7. 三数据集分配

### 7.1 Update 采样比例

```text
Kubric MOVi-F:   35%
PointOdyssey:    30%
Dynamic Replica: 35%
```

选择依据：

- Kubric 提供高覆盖 dense geometry 和几乎 dense 的刚体运动；
- PointOdyssey 提供复杂 articulated/non-rigid、长期 identity tracks，但 source-grid 覆盖较稀疏；
- Dynamic Replica 的 diagonal geometry 近 dense，off-diagonal motion 主要来自 persistent sparse mesh tracks，补充室内人物动态。

比例不按原始 clip 数量决定。采用 per-pair valid mean 后，PointOdyssey/DR 的 sparse motion pair 不会仅因有效点少而自动获得较小 loss。

### 7.2 确定性调度

每 20 个 optimizer updates 构造并按 seed 确定性打乱：

```text
7 Kubric + 6 PointOdyssey + 7 Dynamic Replica
```

一个 optimizer update 的两个 accumulation microsteps使用同一个数据集；四个 rank 也使用同一数据集，但采样不同 clips。这样有效 global batch 为 8 clips，且文本 condition 一致。

数据集内部：

- Kubric：clip 均匀采样；
- PointOdyssey、Dynamic Replica：优先 scene/parent-balanced block sampling，再在 parent 内采样 clip；
- PointOdyssey 保持 scene-block locality，避免随机访问大型压缩 scene archive；
- 所有采样由 `(seed, global_step, microstep, rank)` 唯一确定并可从 checkpoint 精确恢复。

## 8. 优化和三天预算

### 8.1 建议配置

```yaml
precision: bf16
trainable_mode: full
fsdp: full_shard
microbatch_per_gpu: 1
gradient_accumulation: 2
effective_global_batch_clips: 8

optimizer: AdamW
decoder_learning_rate: 3.0e-4
geometry_learning_rate: 3.0e-4
backbone_learning_rate: 5.0e-5
weight_decay: 1.0e-4
warmup_steps: 1000
lr_schedule: cosine
schedule_horizon_steps: 100000
gradient_clip: 1.0
```

- Wan blocks 使用 activation checkpointing；
- VAE frozen，训练读取离线 256 clean latent cache；
- 不使用 CPU parameter offload，除非实测表明是唯一可运行方案；
- `max_steps=100000` 作为可续训目标，不把 LR schedule 压缩到三天结束。

### 8.2 时间预算

总预算为 72 wall-clock hours，即 288 A100 GPU-hours：

- 最多约 68 小时训练；
- 预留约 4 小时做 checkpoint、三个数据集固定验证和最终状态保存。

现有 128 实测不能精确外推 256 + block29 + FSDP。规划区间为约 10k–30k optimizer updates/三天。正式速度必须由最初 200 个稳定 optimizer updates 的 median step time重新估计，记录而不是假定。

有效 global batch 为 8 时：

| optimizer updates | clips seen | Kubric 35% | PO 30% | DR 35% |
|---:|---:|---:|---:|---:|
| 10k | 80k | 28k | 24k | 28k |
| 20k | 160k | 56k | 48k | 56k |
| 30k | 240k | 84k | 72k | 84k |

以当前 train clips `5737/9746/6090` 粗略换算，20k updates 约为 Kubric 9.8、PO 4.9、DR 9.2 个 dataset passes。

三天后如果 48h→68h 的 held-out Sim(3)-aligned EPE 仍稳定改善，则从完整 model/optimizer/scheduler/RNG 状态继续训练；否则先分析数据集负迁移、吞吐和子集误差，不机械跑满 100k。

## 9. Checkpoint 和在线诊断

- checkpoint：2k、5k、10k，之后每 5k updates；另保留原子覆盖的 latest resume checkpoint；
- 固定小 validation：约 12h、24h、48h、68h；
- 完整验证：三天结束时执行；
- evaluation target 以 chunks 推理全部 21 targets；训练 K 不限制完整评价；
- checkpoint 必须包含 model、optimizer、scheduler、global step、clips seen、Python/NumPy/Torch/CUDA RNG、dataset scheduler state 和 prompt metadata。

训练日志至少记录：

- normalized train loss 和 raw train EPE，仅作优化诊断；
- clips/s、pairs/s、step time、GPU-hours、peak memory；
- 三个数据集分别的 validation 指标；
- 四层 readout 融合权重；
- 每个数据集、source、target gap 和 valid count 的实际采样直方图。

## 10. 评估协议：以全局 Sim(3) 对齐结果为主

### 10.1 主指标

与 4RC tracking 评估保持一致，**主报告指标为 global Sim(3)-aligned EPE，越低越好**。对每个独立 clip/video，拟合一个且仅一个：

\[
X^{aligned}=sR\hat X+t,
\]

其中 `s>0`、`R∈SO(3)`、`t∈R³`。禁止按 target frame、轨迹、物体、可见性子集或 temporal-gap bin 分别拟合。

拟合前要求预测和 GT 已在同一坐标基准：

- benchmark 固定单一 source 时，可直接在该 source-camera frame 中拟合；
- 内部 all-source/all-target 评价时，先用相同的已知刚体坐标变换把 prediction 和 GT 转换到共同 frame-0 anchor，再在整段 clip 上拟合一个 Sim(3)。

### 10.2 Benchmark-comparable RANSAC Sim(3)

用于和 4RC 数字直接比较时，复现其公开 tracking evaluator 的主设置：

- flatten 该 clip/video 的有效对应点；
- 超过 16,384 个点时，用固定 seed 均匀采样 16,384 对用于拟合；
- RANSAC 1,000 iterations；
- inlier threshold `0.05 m`；
- 每次最小采样 3 对点；
- 最优模型用全部 inliers 做 closed-form SVD refinement；
- 将同一个最终 Sim(3) 应用于该 clip/video 的全部 predictions；
- EPE 在全部 evaluation-valid points 上计算，而不只在拟合样本或 inliers 上计算。

RANSAC 随机 seed、采样索引、失败回退和实际参数必须写入结果文件。RANSAC 失败时才允许回退到所有有效点的 closed-form Sim(3)，并单独计数。

为了延续 H017 内部分析，还可以同时报告 per-clip all-valid closed-form Sim(3)，但它必须标为 `clip_sim3_closed`，不能与 4RC 的 RANSAC Sim(3) 混称为同一指标。

### 10.3 指标分桶

每个数据集分别报告，不直接平均三者 raw EPE：

- Sim(3)-aligned arbitrary/benchmark tracking EPE（主指标）；
- aligned pointmap/diagonal EPE；
- aligned visible EPE；
- aligned occluded-valid EPE；
- aligned late-appearing EPE；
- aligned EPE by temporal gap；
- APD/threshold metrics（当且仅当完全匹配目标 benchmark 定义）；
- 未对齐 raw metric EPE（次要诊断）。

所有子集指标应复用该 clip/video 的主全局 Sim(3)，不能为每个子集重新拟合，以免增加 oracle 自由度。比较 4RC 时还必须匹配其 clip、帧数、source frame、点采样、validity 和 test split；只匹配“Sim(3)”但不匹配其余协议的数字不可直接横比。

### 10.4 解释限制

Sim(3) 参数使用 evaluation GT 拟合，因此是 oracle gauge alignment。它适合作为 benchmark-comparable 几何/运动诊断，但不代表无需 GT 的部署精度。每次正式结果必须同时保存未对齐 EPE，不能用 Sim(3) 结果掩盖绝对尺度、旋转或平移错误。

checkpoint 选择可使用三个数据集各自的 aligned EPE relative-to-baseline ratio 的 macro average，同时设置单数据集退化 guardrail；不能直接对三个数据集的 normalized loss 或 raw EPE 求平均。

## 11. 启动 gate

三天正式计时前必须全部通过：

1. 三个 256 数据集 manifest、几何审计和 latent cache 完成；
2. 三个 UMT5 condition 的 shape、token metadata 和 checksum 完成；
3. `[13,14,15,29]` 初始权重精确为 `0.30/0.30/0.30/0.10`；
4. block 16–29、geometry adapter 和 decoder 梯度均非零且有限；
5. K=6 或回退 K=4 的完整 optimizer-step capacity gate；
6. 4-rank FSDP、gradient accumulation 和 mixed-dataset schedule 可精确 resume；
7. K-chunk evaluation 与一次性相同 K evaluation 在容差内一致；
8. per-clip RANSAC Sim(3) evaluator 在合成已知变换上恢复正确 `s,R,t`；
9. correct prompt condition 确实进入所有执行的 Wan blocks；
10. W&B/offline logging、checkpoint 原子替换和 68h graceful-stop 已验证。

## 12. 本轮最终冻结项

```text
Resolution:             256×256
Clip length:            21
Wan latent:             16×6×32×32
Readout blocks:         [13,14,15,29]
Initial layer weights:  [0.30,0.30,0.30,0.10]
Text:                   三个固定语义 prompts，无 dropout
Decoder:                约 100M
Training targets:       K=6；OOM/吞吐 gate 失败则固定 K=4
Loss:                   validity-masked XYZ SmoothL1 only
Data mix:               Kubric 35% / PO 30% / DR 35%
Hardware:               4×A100 80GB
Initial budget:         72 wall-clock hours
Primary evaluation:     per-clip/video global RANSAC Sim(3)-aligned EPE
Secondary evaluation:   raw metric EPE 和各困难子集
```
