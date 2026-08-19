# WorldBridge4D environment

Validated baseline: Linux x86_64、Python 3.10/3.11、CUDA-enabled PyTorch 2.10.0+cu126、TensorFlow CPU 2.16.x。TensorFlow 只用于只读解析 MOVi-F TFRecord，禁止安装 GPU TensorFlow 与训练争抢显存。

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.10.0'
python -m pip install -r requirements.txt -c constraints-known-good-cu126.txt

PYTHONPATH=src python scripts/check_environment.py --require-cuda
PYTHONPATH=src python -m pytest -q
```

若主机不能使用 CUDA 12.6，应先选择匹配驱动的官方 PyTorch wheel，不要直接复用 CUDA constraint。

## 外部输入

Python package 不包含模型、数据或 condition。当前路径均记录在 `configs/worldbridge4d_gpu14_k19_150k.yaml`：

- Wan2.1-1.3B DiT/VAE；
- Kubric、PointOdyssey、Dynamic Replica 只读挂载；
- 三数据集 persistent geometry/latent cache；
- 三个 dataset-specific UMT5 conditions；
- train-only mixture coordinate stats。

Condition 由 `scripts/create_wan_text_conditions.py` 生成，需要 native Wan source。模型、生成 tensor 和 API key 均不得提交 Git。W&B 使用 `wandb login` 的机器本地凭据或进程环境变量 `WANDB_API_KEY`。

`check_environment.py` 检查直接依赖、Diffusers Wan API、TFRecord、manifest schema 和当前 9 个项目模块；`--require-cuda` 额外检查 CUDA 与 BF16，不加载模型或数据。
