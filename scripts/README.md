# scripts

生产界面只保留五个薄入口；实现均位于 `src/worldbridge/`。

- `train.py`：FSDP 训练与 exact resume。
- `infer.py`：三数据集 prompt-conditioned 推理。
- `evaluate.py`：source-RGB counterfactual 评测。
- `prepare_data.py`：数据、cache 与 checkpoint 维护的统一子命令入口。
- `run_fsdp.sh`：两卡/多卡 `torchrun` 启动和输入预检。

查看统一维护命令：

```bash
python scripts/prepare_data.py --help
python scripts/prepare_data.py <command> --help
```

常用子命令包括 `preprocess-{pointodyssey,dynamic-replica,three-dataset}`、
`precompute-latents`、`compute-stats`、`text-conditions`、`audit`、
`cache-roots`、`stage-inputs`、`recover-status` 和 `validate-checkpoint`。
数据与 cache 实现在 `worldbridge.data.commands`，checkpoint 运维实现在
`worldbridge.trainer.commands`。历史 GPU1/4 handoff watcher 已移除；其结果保留
在 research 记录和 Git 历史中。
