# scripts

只保留当前 GPU1/4 三数据集路线需要的入口。

## 训练与恢复

- `train_three_dataset_256_fsdp.py`：薄 CLI；实际训练、FSDP、checkpoint/exact resume 位于 `worldbridge.trainer`。
- `run_three_dataset_256_fsdp.sh`：统一 torchrun launcher。
- `wait_resume_three_dataset_256_gpu14_150k.sh`：GPU1/4 K19/150k audited handoff。
- `prepare_three_dataset_256_gpu14_handoff.py`：冻结并验证 handoff checkpoint/config/marker。
- `replicate_checkpoint.py`：异步、原子、单调的 durable checkpoint replica。
- `stage_three_dataset_256_inputs.py`：大模型输入的校验暂存。
- `validate_three_dataset_256_cache_roots.py`：latent cache fail-closed 预检。
- `recover_three_dataset_256_train_status.py`：从完整 checkpoint 恢复缺失 sidecar。

## 推理与条件

- `infer_three_dataset_256.py`：薄 CLI；三数据集 prompt-conditioned 推理位于 `worldbridge.evaluation.inference`。
- `eval_source_rgb_counterfactual_256.py`：薄 CLI；source-RGB 因果评测位于 `worldbridge.evaluation.counterfactual`。
- `create_wan_text_conditions.py`：生成并校验三个固定 UMT5 conditions。

## 数据准备与审计

- `preprocess_{pointodyssey,dynamic_replica}.py`：外部数据集 geometry/index。
- `preprocess_three_dataset_256.py`：256px index 与 immutable latent shards。
- `precompute_latents_256.py`、`run_precompute_latents_5gpu.sh`：当前 deterministic lazy latent 预计算。
- `compute_three_dataset_256_stats.py`：train-only 35/30/35 坐标统计。
- `audit_three_dataset_256.py`：三数据集合同审计。

## 当前训练缓存构建

- `compact_kubric_geometry.py`、`convert_kubric_geometry_to_mmap.py`
- `compact_dynamic_replica_trajectories.py`
- `convert_anno_to_npy.py`、`copy_anno_to_tmp.py`、`copy_depth_to_tmp.py`

这些缓存工具只创建可重建的加速层；原始数据保持只读。历史 benchmark、旧 DDP launcher、H005/H017 消融和 Wan-14B 工具已删除，可从 Git 历史恢复。
