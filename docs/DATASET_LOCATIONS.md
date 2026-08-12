# 训练数据集位置

本文记录当前训练服务器上已确认存在的原始数据集挂载路径，供后续预处理、配置迁移和训练排障参考。检查日期：2026-08-12。

| 数据集 | 原始数据路径 | 已观察到的顶层内容 |
|---|---|---|
| PointOdyssey | `/dataset/nas0/PointOdyssey` | `train/`、`val/`、`test/` 及对应压缩包 |
| Dynamic Replica / dynamic_stereo | `/dataset/data/Dynamic_dataset/dynamic_stereo` | `train/`、`valid/` |
| MOVi-F 512×512 | `/dataset/MOVi-F/512x512` | `1.0.0/` |

## 使用注意事项

- 以上目录是共享原始数据，只读使用；不要在其中写入缓存、latent、统计量或训练输出。
- 这些是服务器挂载位置，不代表已经满足 WorldBridge4D 的训练缓存协议。正式训练前仍需遵循 [`DATASET_PREPROCESSING_PROTOCOL_V1.md`](DATASET_PREPROCESSING_PROTOCOL_V1.md) 完成预处理与验证。
- PointOdyssey 的现有脚本/配置中可能仍出现旧路径 `/dataset/PointOdyssey`。运行 `scripts/preprocess_pointodyssey.py` 时应通过 `--data-root /dataset/nas0/PointOdyssey` 显式指定当前路径，并参考 [`POINTODYSSEY_INTEGRATION_LESSONS.md`](POINTODYSSEY_INTEGRATION_LESSONS.md)。
- Dynamic Replica 的部分配置仍使用 `/dataset/Dynamic_dataset/dynamic_stereo`；当前实际路径多了一层 `data/`，应覆盖为 `/dataset/data/Dynamic_dataset/dynamic_stereo`。`scripts/preprocess_dynamic_replica.py` 可通过 `--raw-root` 指定。
- MOVi-F 此处是 `/dataset/MOVi-F/512x512` 的 **512×512** 版本，而旧 canonical 模型契约是 21 帧、128×128。H023 会一致地将 RGB、深度、分割和相机几何变换到 256×256；其他路线不得把它直接当作 128×128 配置的等价替换。
- 训练配置应记录实际原始数据路径和独立的不可变缓存路径，以避免把原始数据挂载与训练缓存混淆。
