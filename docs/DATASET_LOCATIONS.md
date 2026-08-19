# 当前训练服务器数据位置

以 `configs/worldbridge4d_gpu14_k19_150k.yaml` 为最终准绳：

| 数据集 | 只读原始数据 |
|---|---|
| Kubric MOVi-F 512×512 | `/dataset/nas0/yejun/MOVi-F/512x512` |
| PointOdyssey | `/dataset/nas0/PointOdyssey` |
| Dynamic Replica | `/dataset/data/Dynamic_dataset/dynamic_stereo` |

训练缓存位于 `/data/WorldBridge4D-persistent/`，运行输出位于 `/data/WorldBridge4D-runs/`，Kubric mmap 热缓存位于 `/tmp/worldbridge4d-cache-v2/kubric_geometry_mmap`。

注意：

- 原始挂载只读，禁止写入 latent、统计量、checkpoint 或临时文件。
- `/tmp` 内容可丢失，只能作为由 persistent/raw 数据重建的加速层。
- PointOdyssey 和 Dynamic Replica geometry cache 路径也由 YAML 显式记录。
- 路径迁移时必须同时修改 YAML，并在启动 torchrun 前通过 raw mount/cache-root 预检。
- 预处理合同见 `DATASET_PREPROCESSING_PROTOCOL_V1.md`。
