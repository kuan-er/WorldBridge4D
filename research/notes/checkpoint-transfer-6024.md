# Checkpoint 传到 6024 推理机：内网 + Bridge4D_yj

最后实测：2026-09-23。用户明确指定此机器、目标目录，并授权通过 `Bridge4D_yj` 容器写入。本文记录方法，不代表自动传输以后所有 checkpoint 或启动推理。

## 固定连接信息

| 项目 | 值 |
|---|---|
| 对外入口（辨认机器用，不作为默认大文件通道） | `yejun@sy.irmv.top:6024` |
| **实测内网传输入口** | **`yejun@10.129.22.20:22`** |
| 主机名 | `admin1-H3C-UniServer-R5300-G6` |
| 实测 SSH 连接源地址 | `10.129.22.30` |
| 另一可达内网地址 | `192.168.7.49:22`（未用它做此次传输） |
| 不可达地址（此次环境） | `172.31.200.24:22`，连接超时 |
| 写入容器 | **`Bridge4D_yj`**，实测默认 UID 0 |
| 宿主机目标目录 | `/home/yejun/data0/WorldBridge4D-inference/checkpoints` |
| 宿主机解析后目录 | `/data0/yejun/WorldBridge4D-inference/checkpoints` |
| **容器内目标目录** | **`/data/WorldBridge4D-inference/checkpoints`** |
| RW bind mount | 宿主机 `/home/yejun/data0` → 容器 `/data` |

**不要直接以宿主机 yejun 用户 rsync 到这个目录。** 实测目录为 `root:root 0755`，yejun 无写权限，`sudo -n` 也不可用。使用用户授权的容器写入，不 chmod/chown 宿主机目录，不另行绕过权限。

## 连接与预检

使用已有可信 6024 主机密钥校验内网端点，不关闭主机密钥检查：

```bash
ssh -T -p 22 \
  -o 'HostKeyAlias=[sy.irmv.top]:6024' \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=15 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  yejun@10.129.22.20 \
  'docker inspect --format "{{.State.Running}} {{json .Mounts}}" Bridge4D_yj;
   docker exec Bridge4D_yj sh -c "id; test -w /data/WorldBridge4D-inference/checkpoints && echo WRITABLE; df -B1 --output=avail /data/WorldBridge4D-inference/checkpoints"'
```

实测容器有 `/opt/conda/bin/python3` 和 `sha256sum`，**没有 rsync**。因此最终成功方案不是容器内 rsync，而是 **SSH 二进制流 → `docker exec -i` → Python 接收器**。不要加 TTY (`-t`)，避免破坏二进制流。

## 安全传输流程

1. 选择最新**已发布的编号文件** `checkpoint-XXXXXXX.pt`，不要用会变化的 `latest.pt`。
2. 在 worktree 外、同一文件系统的 transfer 目录创建硬链接，固定源文件，避免训练 keep-last3 删除期间丢失；不改训练输出。
3. 记录并核对源文件字节数、SHA256、step；传送原始完整 checkpoint，不擅自转换/删除 optimizer。
4. 检查远端容器目录存在且可写、空间满足待传字节数 + 64 GiB 余量。
5. 若正式目标已存在，只在其大小和 SHA256 一致时视为完成；不同则报错，禁止覆盖。
6. 接收至同目录隐藏文件 `.<filename>.<SHA256>.container.partial`。记录已有长度，发送端 seek 到该 offset；接收端确认 offset 未变化再续写。
7. 收齐预期字节后 flush/fsync，计算远端 SHA256；一致才以 `os.link(partial, final)` 原子、不覆盖发布，删除 partial 并 fsync 目录。新文件设为 0644，宿主机 yejun 可读。异常保留 partial，不把残缺文件暴露为正式 checkpoint。
8. 本地保存 manifest 和 complete.json，完整记录源/目标/大小/hash/耗时。通过 PRL 的退出事件确认成功；传输完整性不等于已通过推理模型加载验证。

训练不停止、不改配置；传输不使用 GPU、不自动启动远程推理。

## 可复用实现与 PRL 执行

成功脚本：`research/analysis/h031_transfer193k_container.py`。

它是 **193000 的固定实例**，不是通用 CLI：`PIN`、`OUT`、`DIGEST`、`SIZE`、`DEST` 以及 manifest 中宿主机文件名均固定。下一次必须在 Task worktree 中同步更新或复制为新 step 的实例，使用新输出目录；不要原样重跑并误以为会自动选择最新 checkpoint。`OUT.mkdir(exist_ok=False)` 防止覆盖旧报告。重试时可沿用相同源/远端 partial，但也需新的本地报告目录。不要同时启动两个 receiver 写同一 partial。

由 `prl_run_launch` 提交 CPU-only Run（先冻结代码、后执行），argv 模式：

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 ionice -c 2 -n 7
python research/analysis/h031_transfer193k_container.py
```

使用 PRL 自动 `process.exit` 通知，随后 `prl_run_inspect` 确认 `TRANSFER_COMPLETE`。若取消/换通道，只停止传输 Run，不操作训练 Run。所有文件/报告在 worktree 外。

## 成功实例

- PRL Run：`R-20260923072233-8e2fe6`，exit 0，2026-09-23 07:26 UTC 完成。
- 文件：`checkpoint-0193000.pt`，**19,363,914,955 bytes**。
- SHA256：`98e3c6bea3bb35e9ba630e31bec180402d829b14064d099f73c236e068d9d40f`。
- 远端宿主机：`/home/yejun/data0/WorldBridge4D-inference/checkpoints/checkpoint-0193000.pt`。
- 远端容器：`/data/WorldBridge4D-inference/checkpoints/checkpoint-0193000.pt`。
- 本地固定源：`/data/WorldBridge4D-runs/transfers/h031-step193000-to6024-20260923/checkpoint-0193000.pt`。
- 完成报告：`/data/WorldBridge4D-runs/transfers/h031-step193000-to6024-container-20260923/complete.json`。
- 实测接收+fsync **174.05 秒**（约 111 MB/s）；接收+远端校验 **191.64 秒**，另有本地 hash 时间。**不能承诺几秒完成**，此次也未证明限速具体来自网络、SSH 加密或磁盘。
- 旧公网入口尝试 `R-20260923070932-9e0a41` 已终止，未验证成功；旧 `h031_transfer193k_6024.py` 宿主机写入方案不应作为默认方法。
