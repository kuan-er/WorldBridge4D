# WorldBridge4D 256px 三数据集预处理合同

## 固定样本接口

当前训练 consumer 对每个 clip/source 需要：

```text
clean_latent: float32 [16,6,32,32]
xyz:          float32 [21,3,256,256]
valid:        bool    [21,256,256]
rgb:          uint8   [21,256,256,3]  # 仅 VAE 编码时需要
```

`xyz[t,:,v,u]` 表示 source 帧像素 `(u,v)` 所见物理点在 target `t` 时刻的位置，所有 target 均表达在固定的 source-camera 坐标系。`valid` 与 visibility 分离；target 时刻被遮挡但对应关系有效的点仍参与 loss。

## 通用约定

- 每段恰好 21 帧，不 padding、不插帧。
- 坐标单位为米；相机局部 `+X` 向右、`+Y` 向上、`-Z` 向前。
- 先按 parent scene/video 划分 split，再提取 clip；clip ID 和顺序必须稳定。
- RGB、depth、轨迹和 intrinsics 必须使用同一个 crop/resize 到 256×256。
- 原始数据只读，所有 index/cache/stats 写入独立 persistent root。
- Wan latent 必须使用配置指定的 VAE posterior mean，FP32，禁止从旧 16×16 latent 插值。

## 三个 adapter

- Kubric MOVi-F：TFRecord depth/segmentation、相机和实例刚体状态；dense source-grid supervision。
- PointOdyssey：persistent sparse tracks rasterize 到 source grid；同像素冲突采用确定性最近轨迹。
- Dynamic Replica：persistent mesh trajectories rasterize 到 source grid；source diagonal 由 depth backprojection 覆盖。

对应代码：`src/worldbridge/{training256,data,geometry,pointodyssey,dynamic_replica}.py`。

## 必需产物

```text
CACHE_ROOT/
  splits/train.jsonl
  latents/wan2.1_1.3b_fp32_256/          # immutable shard，可选
  latents/wan2.1_1.3b_fp32_256_lazy/     # per-clip cache
  latents/wan2.1_1.3b_fp32_256_lazy_backup/
```

此外全局需要三个带 prompt/checksum metadata 的 UMT5 conditions，以及仅由 train split 计算的 source-frame coordinate mean/scale。

## 验证门槛

1. shape/dtype/finite/clip order 合同正确；
2. parent split 无泄漏；
3. camera round-trip 和 diagonal identity 正确；
4. occluded-valid 不被 visibility 错误过滤；
5. VAE 重编码确定且 checksum 与配置一致；
6. cache roots、原子临时文件、文件数量审计通过；
7. 两 rank real-Wan 至少完成两个有限 optimizer updates；
8. full checkpoint 能严格恢复 model、AdamW 和每 rank RNG。

入口见 `scripts/README.md`，数据位置见 `docs/DATASET_LOCATIONS.md`。生成数据、latent 和权重不得提交 Git。
