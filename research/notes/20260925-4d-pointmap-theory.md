# 4D Pointmap 的理论基础：Gauge、可辨识性与 Cocycle

**日期**：2026-09-25
**性质**：理论/定位笔记（不是实验记录，不含新结果）
**目的**：把 WorldBridge4D 相对 4RC 的差异从"工程更好"提升为**可证伪的理论命题**，并给出每条命题对应的判据、现有代码资产与缺口。

---

## 0. 摘要

我们和 4RC 属于**同一个函数类**：

```
video → encoder → 共享潜表征 → query decoder → 每个 (source, target) 的 3D 图
```

因此任何"token 更少 / 层数更浅 / 加了 RGB 旁路"都属于同类实例的工程差异，不构成核心贡献。理论贡献只能来自两类问题：

1. **输出空间的代数结构**（对称群、参数化是否冗余、可辨识性）；
2. **信息在生成模型里被放在哪**（σ=0 的输出头是否退化）。

本笔记给出 6 条命题（T1–T6），其中 **T1 + T2 + T3 构成骨架**，T4/T5/T6 是推论与应用。每条都配：命题 → 证明思路 → 可检验预言 → 现有资产。

| 编号 | 命题 | 类型 | 需要新训练？ | 强度 |
|---|---|---|---|---|
| T1 | RF σ=0 输出退化定理 | 定理 | 探针需要（便宜） | ★★★ |
| T2 | Pointmap 充分性 + Lipschitz 上界 | 定理 | 不需要 | ★★★ |
| T3 | Gauge / 最小参数化 | 命题组 | C 需要四臂消融 | ★★★ |
| T4 | Amodal 可表示性 | 定理 | 不需要 | ★★☆ |
| T5 | Groupoid / cocycle 恒等式 | 定理 | 部分 | ★★☆ |
| T6 | Metric 可辨识性（1 DoF） | 命题 | 不需要 | ★★☆ |

---

## 1. 前提：当前方法的两条路线与一个必须说清的现状

**H023/H031 production 路线（本文档写作时的主叙事）**
- frozen Wan VAE → `[B,16,6,32,32]`（256px）或 `[6,64,64]`（native 512）
- Wan DiT 读 blocks `[13,14,15,29]` 的 hidden states
- geometry adapter → `Z_dense[B,128,21,32,32]` + `Z_motion[B,21,8,128]`
- 5 层 cross-attention decoder + source-RGB pyramid → `source-grid XYZ [B,K,3,256,256]`
- 训练损失是 **validity-masked SmoothL1 on XYZ（米）**

**H032/H033 路线（正在进行，必须纳入考虑）**
- H032：在 decoder 上加 **camera head**（per-pair pose + FoV）→ 重新引入 camera 参数化
- H032 结果：**pose 没有学到** —— 与 "identity rotation + zero translation" 基线统计上不可区分
  （Kubric 3.178°/0.4959m vs 平凡基线 3.074°/0.4722m），且 loss 在该处几乎平坦
  （identity 0.02898012 vs best-constant 0.02897891）
- H033：换成 decoder 内的 camera token + **per-frame ray field**（对角 pair 的 pixel token 出单位射线），
  删掉 7 个模块；camera 参数 17 tensors / 926,218 params
- H033 `open_items` 明确列出 **"linear probe from memory to GT pose"** 与
  **"fixed-clip camera evaluation"** 是决定性测量

> **重要**：H032 的失败不是反例，而是 T3/T6 的**实证支持**——显式 pose 参数化引入了弱监督/不可辨识的自由度，
> 而直接回归 pointmap 不需要这个自由度。H033 的 ray field 是更良态的部分重引入（per-frame、source-independent）。
> 下文凡涉及"我们不参数化相机"，指的是 H023/H031 路线；H032/H033 是在做对照实验。

---

## 2. T1：RF time-zero 输出退化定理（最强）

### 命题

设 rectified-flow / flow-matching 的训练目标为

```
x_σ = (1 − σ)·x₀ + σ·ε,      ε ⊥ (x₀, c),      E[ε] = 0
v_θ(x_σ, σ, c)  ≈  ε − x₀
```

则最优解满足

```
v*(x, 0, c) = E[ε − x₀ | x_σ = x, c] = E[ε] − x = −x
⟹  −v*(x₀, 0, c) = x₀
```

即 **σ=0 时输出头被代数地钉死在输入的恒等副本上。**

### 证明思路

σ=0 时 `x_σ = x₀` 是确定性等式，所以"给定 x_σ"等价于"给定 x₀"；而 ε 由训练过程独立采样，
故 `E[ε | x₀, c] = 0`。条件期望立刻给出 `v* = −x₀`。

### 三个推论

1. 输出头在 σ=0 处对"几何读出"**没有额外表达能力**——它被训练目标固定在 `−x₀`。
2. 想让这个头输出 pointmap（H004 的做法），等于让一个被拉向恒等的头去做另一件事。
3. 要读 4D 结构，只能读被**去噪目标塑形过**的 hidden states（H005）。

### ⚠️ 必须避免的错误表述

**不要**写成"hidden states 比 x₀ 含更多关于世界的信息"。从 data processing inequality 看，
hidden states 是 `(x₀, c)` 的确定性函数，**不可能**含更多信息。
正确的命题是**表征退化 / 可解码性**：输出空间的取值在 σ=0 被代数固定，而 hidden states 被生成目标塑形。
这个区分决定这条能不能过 review。

### 附加推论（较软，需先测）

Wan VAE 是因果卷积、接受野有限，因此 `x₀` 中长程时序对应的信息是**局部**的；
长程 4D 结构只能由 DiT 的全局注意力整合出来。→ 这是"H005 应该赢"的第二个理由，但必须先量出 VAE 接受野。

### 可检验

- (a) 在真实 Wan 上测 `‖(−v(x₀,0)) − x₀‖ / ‖x₀‖`；接近 0 则命题的经验形式成立。
- (b) **三臂 matched 探针**：同容量 readout、同数据、同步数，比较
  `FeedForwardWanBackbone`（velocity 读出）/ `WanHiddenGeometryBackbone`（hidden 读出）/ `CleanLatentBackbone`（纯 latent）。

### 资产与缺口

- 三个 backbone 类在主树 `src/worldbridge/models/backbones/geometry.py` 中存在。
- **H033 分支已删除 `CleanLatentBackbone` 与 `FeedForwardWanBackbone`**
  （refactor slice5: `readout_must_be_wan_hidden_structured`）。
  要跑 T1 必须从 H031/main 恢复这两个类。这是一条明确的施工项。

---

## 3. T2：Pointmap 的充分性与 Lipschitz 上界

### 命题

记 4D pointmap 场 `X = X[s,t,p]`（source 相机系）。以下派生量都是 `X` 的 Lipschitz 投影：

| 派生量 | 表达式 | Lipschitz 常数 |
|---|---|---|
| depth | `‖X[s,s,p]‖` | 1 |
| scene flow | `X[s,t,p] − X[s,s,p]` | √2 |
| 2D 对应 | `π_t(X[s,t,p])` | 在 depth 有正下界处局部有限 |
| 相对位姿 | 两个 pointmap 对齐 | 有限 |

于是对任意派生量 `g`：

```
‖ĝ − g‖ ≤ L_g · ‖X̂ − X‖
⟹  E‖ĝ − g‖ ≤ L_g · E‖X̂ − X‖
```

### 推论

以 pointmap 为唯一监督目标不是"任务选择"，而是**唯一能同时给所有下游任务提供误差上界**的输出空间。
depth / flow / track 空间的损失**不** bound pointmap（信息更少）。
这把"4D pointmap as sole output space"从经验主张升格为理论命题。

### 可检验

用同一个 checkpoint 同时报 pointmap / depth / flow / 2D correspondence / relative pose，
检查它们是否被 `L_g · EPE_pointmap` 统一 bound；以及**不额外训练**就从 pointmap 解码这些量。

### 与 4RC 的正面冲突

4RC 论文 Table 4(b) 正好做了反向消融：

| Motion 输出形式 | Kubric APD ↑ / EPE ↓ | Waymo APD ↑ / EPE ↓ |
|---|---|---|
| 4RC（factorized displacement） | **85.44 / 1.022** | **56.63 / 1.611** |
| (i) Points (World) | 74.64 / 1.412 | 37.08 / 2.359 |
| (ii) Points (Local) | 70.70 / 1.547 | 19.55 / 3.226 |

他们的解释是 "optimization difficulty"（学习难度），并明确说 direct point prediction
"entangles geometry and motion in a single output space"。

**为什么这不能推翻 T2/T3：**

1. 他们的 "Points" 变体**保留了 depth + ray + camera geometry head**，只替换 motion head 的输出。
   那本质上是"参数化 base + 直接位移"，**不是**"直接绝对 pointmap"。
2. 他们在 **unit-scale 归一化**下训练（§3.4：把场景尺度归一化到平均半径 1），尺度信息被删掉。
3. 他们仍保留 geometry / motion 两条通道，`s=t` 与 `s≠t` 走不同代码路径。
4. 他们是端到端从头训 encoder；我们是从冻结的生成式视频先验里读出。
   他们的"学习难度"论证强度直接依赖于 backbone 先验强度。

→ **必须在我们框架内做 matched 消融**（见 §7）。

---

## 4. T3：Gauge 与可辨识性

### 命题 A：评测群 = 输出空间的对称群

视频只确定 `X_world` 的 `Sim(3)` 商（7 DoF 不可辨识）。于是：

| 输出空间 | 自然对称群 | 对应评测 |
|---|---|---|
| 第一帧世界系（4RC、V-DPM） | **单个** `Sim(3)` | per-clip 全局 Sim(3) ✅ |
| 逐 source 相机系（本方法） | `∏_s Sim(3)` | per-(clip, source) Sim(3) ✅ |

我们的评测协议（`evaluation/benchmark.py`：`one_joint_transform_per_clip_source_over_all_21_targets`）
**恰好**是这个商。所以：

- per-source Sim(3) **不是宽松化，而是与输出空间匹配的正确指标**；
- 4RC 的 per-clip 全局 Sim(3) 对它的输出空间同样正确。

这条同时为我们的协议提供理论辩护（审稿人一定会问"是不是在放水"）。

### 命题 B：metric 收缩恰好 1 个自由度

加入 metric 监督后，对称群从 `Sim(3)`（7 DoF）收缩到 `SE(3)`（6 DoF）——
**正好去掉尺度那一个 DoF**。预言 `ŝ → 1`。

**实测**（`/data/WorldBridge4D-runs/evaluation-step100000/`，模型 denormalize 用 `checkpoint` 里的全局 train `mean/scale`）：

| 数据集 | 拟合 Sim(3) scale 均值 | 中位数 | std | p5 / p95 | n |
|---|---:|---:|---:|---:|---:|
| Kubric | **0.9958** | 0.9937 | 0.050 | 0.919 / 1.095 | 3087 |
| PointOdyssey | **1.0103** | 1.0117 | 0.100 | 0.861 / 1.160 | 15099 |
| Dynamic Replica | **1.0015** | 1.0044 | 0.052 | 0.909 / 1.088 | 8526 |

Kubric 上 **93.6%** 的 `(clip, source)` 拟合尺度落在 ±10% 内，100% 落在 ±25% 内。

**但要注意**：Kubric raw point-weighted EPE = **0.7239 m**，sim3 = **0.4803 m**
（per-clip 相对提升中位数 0.298，49% 的 clip 提升 >30%）。
既然 `ŝ ≈ 1`，这 0.24 m 的差距只能来自旋转/平移，而 Kubric 平均深度约 **13.5 m**
（H007 训练统计量 z 均值 `−13.47`）→ **1° 角度误差在该深度就是 ≈0.24 m**，数值完全吻合。

> 准确表述：**尺度已经解决，剩余误差主要是角度/朝向。**

### 命题 C：过参数化 ⇒ 损失不对齐

`Φ: (depth, ray, camera) → X` 与 `Ψ: (P_base, ΔP) → X` 都是**非单射**，纤维非平凡。
作用在分量上的损失不是 `X` 的函数，因此

```
argmin L_components  ⊄  argmin L_X
```

即分量监督**不**最小化被评测的量。直接回归 `X` 是**最小（无冗余）参数化**，
也是唯一与评测目标对齐的损失。

**实证支持（H032）**：显式 pose head（`fc_t` 米 + 四元数）在所有三个数据集上
与 identity/zero 基线不可区分，translation loss 在 identity 和 best-constant 之间几乎相等
（Kubric 0.02898012 vs 0.02897891）。这是"分量参数化引入了不可辨识/弱可学的自由度"的直接证据。

---

## 5. T4：Amodal 可表示性定理

### 命题

对应场 `φ_{s→t}: Ω_s → Ω_t ∪ {⊥}` 在遮挡处未定义（`⊥`）。
而 pointmap `X[s,t,·]` 在遮挡处**仍有定义**（它是一个物理位置）。
并且 `φ` 是 `(X, P_t)` 的**部分**函数，反向不成立。

### 推论

flow / 2D track / sparse trajectory 的输出空间**严格小于** pointmap 空间；
**amodal 4D 在这些空间里不可表示**。这是"统一 pointmap 与 tracking 应以 pointmap 为原语"的理论依据，
也是 TraceAnything / CoTracker 类方法在遮挡上失败的结构性原因。

### 可检验

`occluded_valid` / `late_appearing` 分组。
导出包 `/tmp/4rc_export/.../src/worldbridge/evaluation512.py:121-152` 有现成实现：
`visible`、`occluded_valid`、`late_appearing`、`raw_epe_by_temporal_gap`、
`clip_sim3_closed_epe_by_temporal_gap`。

**预言**：flow/track 空间的方法在 `occluded_valid` 上有**不可约**误差；直接 pointmap 没有这个下界。

---

## 6. T5：Groupoid / Cocycle —— cycle consistency 是定理，不是启发式

> 这是本次讨论中最需要写清楚的一节。以下内容把"cycle consistency"从经验技巧变成恒等式。

### 6.1 "cocycle" 的字面意思

**cocycle（余循环 / 上闭链）** 是描述一族转移量必须满足的**扭复合律**。

群上同调的定义式：

```
a_{gh} = a_g · (g · a_h)
```

注意右边中间那个 `g · a_h`：第二个因子被 `g` **作用了一次**，所以复合律是"扭的"，
不是普通的同态 `a_{gh} = a_g a_h`。这个"扭"就是 cocycle 与 representation 的区别。

纤维丛的转移函数（Čech cocycle）是同一个东西：

```
g_{αγ} = g_{αβ} · g_{βγ}
```

SLAM 里的 **loop closure**（走一圈回来必须抵消）是它的特例。

**直观一句话**：一族"换坐标 / 搬运"的量，绕一圈回到原地时必须复合成恒等。

对照概念：形如 `g_{αβ} = h_α h_β^{-1}` 的叫 **coboundary（平凡的 cocycle）**——
它意味着存在一个**全局**坐标系，所有转移都能由一个统一的 `h` 生成。这个区分在 §6.4 是关键。

### 6.2 在我们的张量里，谁是 cocycle

输出 `X[s, t, p]` 的三个索引角色**完全不同**：

- `s` = **参考系**（用哪个相机的坐标写出来）
- `t` = **物理时间**
- `p` = source 像素

再加隐含的对应场 `φ_{s→t}: Ω_s → Ω_t ∪ {⊥}`。

**恒等式 1（参考系更换，Sim(3)-valued cocycle）**

`X[s,u,p]`（frame s 坐标）与 `X[t,u,φ_{s→t}(p)]`（frame t 坐标）描述**同一个物质点、同一个时间 u**，
只是参考系不同。所以两者只差一个**与点无关**的刚体变换 `T_{t←s}`：

```
X[t, u, φ_{s→t}(p)] = T_{t←s} · X[s, u, p]

T_{u←t} · T_{t←s} = T_{u←s}
T_{s←s} = I
```

`T` 就是 cocycle：`(s→t) ↦ T_{t←s}`，系数取自 `Sim(3)`（带 metric 监督时是 `SE(3)`，见 T3-B）。

**恒等式 2（像素对应，那个"扭"）**

```
φ_{t→u} ∘ φ_{s→t} = φ_{s→u}
φ_{s→s} = id
φ_{s→t}^{-1} = φ_{t→s}
```

**恒等式 3（换时间没有恒等式）**

`X[s,t,p]` 与 `X[s,u,p]` 是同一物质点在不同**时间**的位置，差的是它自己的运动。
**没有任何约束。**

> **(s, t) 的不对称就是 cocycle 的全部内容**：
> 改第一个索引是换坐标（受 cocycle 约束），改第二个索引是走时间（无约束）。
> 这也解释了为什么我们的 source/target 用**两套独立 embedding**——它们在代数上就是不同种类的对象，
> 模型结构上**不应该**对称化它们。

### 6.3 一个具体例子

一辆开动的车上的点，在帧 0/10/20 分别是 `p`、`q`、`r`：

- `φ_{0→20}(p)` **必须**等于 `φ_{10→20}(q)` —— 恒等式 2
- `X[0,20,p]` 与 `X[20,20,r]` 描述同一个点在同一时刻，只差 `T_{20←0}` —— 恒等式 1
- `X[0,20,p]` 与 `X[0,10,p]` 差的是**车的运动**，不是坐标变换 —— 无约束

### 6.4 为什么叫 cocycle 而不是 representation

因为 `φ_{s→t}` 的定义域是 `Ω_s`、值域是 `Ω_t`，**定义域随 s 变**，
所以 `φ_{t→u} ∘ φ_{s→t}` 里的第二步必须作用在第一步的像上——这就是那个"扭"。

范畴语言：`(s→t) ↦ φ_{s→t}` 是从"时间对 groupoid"到"像素集合范畴"的**函子**，
恒等式 2 就是函子性（保持复合、保持单位）。

几何语言：**4D pointmap 场是一个纤维丛截面**，`φ` 是像素丛上的平行移动，
`T` 是以 `Sim(3)` 为结构群的转移函数。这两条恒等式不是经验规律，是这套结构的定义。

### 6.5 coboundary vs cocycle：我们与 4RC 差别的最精确表述

如果存在全局世界系 `W`，每帧有定位姿 `g_s`（`W → frame s`），那么

```
X_world[s,t,p] = g_s · X[s,t,p]     ⟹     T_{t←s} = g_t · g_s^{-1}
```

即 **`T` 是一个 coboundary**——平凡的 cocycle，恒等式 1 **自动满足**。于是：

| | 输出空间 | `T` 的地位 | 代价 |
|---|---|---|---|
| 4RC / V-DPM | 第一帧世界系 | **coboundary**（必须平凡化） | 必须显式估计每帧位姿 `g_s`（geometry head 的工作）；任一帧姿态错就破坏全局一致性 |
| 本方法 | 逐 source 相机系 | **允许非平凡的 cocycle** | 不需要任何相机位姿；代价是跨 source 比较必须先估计 `T` |

**"输出系的选择" = "把 cocycle 放在哪里"。** 4RC 把它平凡化，我们把它吸收进表征。
这解释了为什么 H023 路线不需要 geometry head 就能成立，
也解释了 H007 里 source 坐标对 arbitrary tracking 的 12% 收益与 late-appearing 的退化——
非平凡的 `T` 意味着每个 source 是独立的一张局部 4D 图，跨 source 的"晚期点"没有被任何全局约束串起来。

### 6.6 它买到什么

**(a) cycle consistency 从技巧变成定理。**
H030 的 A→B→A pixel cycle（`src/worldbridge/trainer/cycle.py`）正是恒等式 2 在 `s=0,t=10,u=0` 的特例。
cocycle 语言立刻给出推广：**k 帧闭环（3 帧、k 帧）都应该受约束**，不只是 2 帧往返。

**(b) 它预言"哪里会坏"。**
恒等式 2 在遮挡处**根本无法陈述**（`φ = ⊥`），在边界处最容易错（`φ` 歧义）。
→ 可证伪预言：`φ` 的 cycle 残差图与 boundary / occluded EPE 强相关。

**(c) 它把 T4 讲清楚。**
遮挡点上 `φ` 未定义，但 `X[s,t,p]` 仍有定义。
所以 **pointmap 场正是 cocycle 破裂之后仍然存在的东西**——这就是"pointmap 严格大于 correspondence/flow"的精确含义。

**(d) 它给 latent 级测试时优化一个正确的目标函数**（配合 T3-C）：
在共享 latent 上优化，使 cocycle 残差最小。

### 6.7 诚实说明：检验条件与当前资产状态

- cocycle 恒等式是精确的，但从**我们自己的输出**里直接验证它需要一个相机模型或一个匹配
  （因为 `φ` 要把 3D 投到 frame t 的像素网格上）。H030 用的是 Kubric 已知内参。
- 这是 T3 "不参数化相机"的代价之一：**我们把 cocycle 吸收进表征，就同时把它变成了从输出端不可直接观测的东西。**
  必须写进 limitation。
- **资产状态**：`src/worldbridge/trainer/cycle.py` 在主树存在；
  但 H033 refactor 已删除 `cycle/boundary/contrast/extension_keys` 与
  `tracking_cycle_metric_definitions`（fail-closed）。
  要重启 T5 需要从 H030/H031 恢复。

---

## 7. T6：Metric 可辨识性

### 命题

单目视频的 metric scale **无先验不可辨识**。因此任何 metric 声称都**等价于声明了一个 prior**。

- 本方法：全局 train 统计（`data/commands/compute_stats.py`，35/30/35 mixture，**跨 clip 共享**）
  + 内容驱动的尺寸先验（**不喂内参**，尺度线索只能来自物体尺寸）
- 4RC：尺度归一化训练 = 声明 prior 不存在 → 退化为 `Sim(3)`（§3.4）

### 推论

metric 不是"更准"，而是"**少一个不可辨识自由度**"。
4RC 的 `Sim(3)` 不是缺陷，而是它 prior 选择的必然结果。
这条把 metric 从工程卖点变成信息层论断，并且是 T3-B 的直接推论。

### 边界（必须写进 limitation）

- 这是**训练分布内**的 metric。三个数据集都是合成数据；Kubric 程序化生成、物体尺度随机、
  绝对尺寸线索弱；PO 的尺度 std（0.100）是 Kubric 的 2 倍，说明 PO 的尺度线索更不可靠。
- **跨域（真实 in-the-wild、非常规尺度）不能声称**，必须单独测。
- 混合统计量是 3 个数据集尺度分布的折中；加入尺度分布差异很大的数据集需重新验证。

---

## 8. 判决性实验：四臂 matched 消融

同 backbone / latent / 数据 / 预算 / seed：

| 臂 | base geometry | 输出 | 隔离的因子 |
|---|---|---|---|
| **A（现状）** | 无 | 直接绝对 `X[s,t]` | 基线 |
| **B** | depth + ray + camera 参数化 | 直接绝对 `X[s,t]` | "参数化" |
| **C** | 直接回归 | `X[s,s] + ΔP` 分解 | "分解" |
| **D** | depth + ray + camera | `ΔP`（≈ 4RC 完整形式） | 两者 |

报告分组：pointmap(diagonal) / arbitrary / **occluded-at-source** / **late-appearing** /
长间隔 / **`|ŝ − 1|`**。

**预言必须是差异化的**：如果 A 只在遮挡 / 大位移 / 尺度上赢，而在普通可见点上与 B/C/D 持平，
那结论不是"我们的方法更好"，而是"**分解法在它该失败的地方失败了**"——
这是由偏置分析**预言**的结构性结果，比"全面更好"可信得多，也更容易发表。

---

## 9. 可检验预言汇总

| # | 预言 | 判据 | 数据/代码 |
|---|---|---|---|
| P1 | `−v(x₀,0)` 接近 `x₀` | `‖(−v)−x₀‖/‖x₀‖ → 0` | Wan forward（`models/wan.py`） |
| P2 | hidden 读出 > velocity 读出 > clean-latent | 同容量探针分组 EPE | 需恢复 3 个 backbone 类 |
| P3 | 派生任务误差被 `L_g · EPE_pointmap` bound | 同 checkpoint 解码 depth/flow/pose | evaluator + 后处理 |
| P4 | 拟合尺度 `ŝ → 1` | `|ŝ−1|` | ✅ 已实测（0.996/1.010/1.002） |
| P5 | 残差主要是角度而非尺度 | raw−sim3 差 ≈ 1° × 平均深度 | ✅ 已实测（0.24m ≈ 1° × 13.5m） |
| P6 | occluded_valid 上 flow/track 有不可约误差 | 分组 EPE | `evaluation512.py` 分组 |
| P7 | cycle 残差集中在边界/遮挡 | 残差图 × EPE 相关 | `trainer/cycle.py`（需恢复） |
| P8 | 显式 pose 参数化不可学 | 与 identity/zero 基线比较 | ✅ H032 已证 |

---

## 10. 现有资产索引

| 内容 | 路径 | 状态 |
|---|---|---|
| 全局 train 统计（mixture mean/scale） | `src/worldbridge/data/commands/compute_stats.py` | 在用 |
| raw 反归一化（`raw = normalized*scale + mean`） | `src/worldbridge/evaluation/benchmark.py` | 在用 |
| per-(clip,source) Sim(3) 协议 | 同上（`one_joint_transform_per_clip_source...`） | 在用 |
| raw / sim3 双指标 | `src/worldbridge/evaluation/benchmark.py` | 在用（默认 `sim3_enabled: true`） |
| 分组指标（visible / occluded_valid / late_appearing / gap） | 导出包 `.../src/worldbridge/evaluation512.py:121-152` | 历史实现，需恢复 |
| cycle 正则（A→B→A） | `src/worldbridge/trainer/cycle.py` | 主树在，H033 分支已移除相关开关 |
| velocity / hidden / clean-latent 三个 backbone | `src/worldbridge/models/backbones/geometry.py` | 主树在，**H033 分支已删前两个** |
| Wan forward 与 hidden 读出（任意 tau） | `src/worldbridge/models/wan.py`（`forward` / `forward_hidden_layers`） | 在用 |
| step-100k 评测结果（raw + sim3 + 拟合 transforms） | `/data/WorldBridge4D-runs/evaluation-step100000/` | 已有 |
| cross-res 256/512 对比 | `/data/WorldBridge4D-runs/h031-cross-res-256-512-eval-20260914-full/summary.json` | 已有 |
| 4RC 原生 512 评估协议与 0.3397 m | `docs/NATIVE_512_CONTEMPORARY_EVALUATION.md` + 导出包 | 已有 |
| 4RC 代码 / 论文 | `/data/4RC`、`/data/tmp/4rc_arxiv.txt` | 已有 |
| H032/H033 相机头结果 | `research/hypotheses/H032.md`、`H033.md`（worktree 版本） | 进行中 |

---

## 11. 建议的 theory section 结构

**骨架 = T1 + T2 + T3**：

1. **T1** —— 一条关于生成模型内部表征定位的定理（σ=0 输出退化 ⇒ 必须读 hidden states）
2. **T2 / T3** —— 由它推出的最小输出空间（充分性 + Lipschitz 上界 + 最小参数化）
3. **T4 / T5 / T6** —— 推论与应用（amodal 可表示性、cocycle、metric 可辨识性）

叙事顺序：**"4D 表征在哪里" → "唯一不冗余的输出空间" → "一组可证伪的结构性预言"**。
这比"我们的架构更好"高一个层级，而且每一条都落在已有代码与指标上。

---

## 12. 术语表

| 术语 | 含义 |
|---|---|
| gauge（规范） | 观测不变但输出可自由变换的自由度 |
| Sim(3) / SE(3) 商 | 把 `X_world` 模掉 7 / 6 个自由度；对应评测用的对齐群 |
| cocycle（余循环） | 一族满足扭复合律的转移量 `T_{u←t}T_{t←s}=T_{u←s}` |
| coboundary | 形如 `T_{t←s}=g_t g_s^{-1}` 的平凡 cocycle（存在全局坐标系） |
| groupoid | 只有部分复合的"群"；时间对 `(s→t)` 就是它的态射 |
| 充分统计量 | 足以推出所有下游任务的量；本文中指 4D pointmap 场 |
| amodal | 被遮挡但物理上仍有确定 3D 位置的（相对 visible） |
| aleatoric uncertainty | 数据固有不确定性（4RC 用它给损失加权） |
