# 三数据集 Geometry Cache 与 Prefetch 性能经验

状态：H023 两卡 K19/B2/A2 长训练的生产性能记录。当前训练协议与入口见 `WORLDBRIDGE4D_256_THREE_DATASET_IMPLEMENTATION.md`。

## 1. 结论

不要以“把整个数据集锁进 RAM”作为保持 GPU 利用率的首选方案。对本项目更有效的组合是：

1. 在本地 SSD 上保存只包含训练所需字段的 geometry/latent cache；
2. 大数组使用 mmap，让 Linux page cache 按需保留物理页面；
3. 使用小型进程内 LRU 保留最近解压或重建的 sample/scene/stream；
4. 用有界 CPU 并发提前计算后续 update 的 geometry；
5. 让 prefetch 生产速度略快于 GPU 消费速度，但不要用过多 worker 制造 page-fault 和内存调度风暴。

验证配置为：

- 两个 FSDP ranks；
- `microbatch_per_gpu=2`、`gradient_accumulation=2`；
- `targets_per_source=19`，即每 update 152 pairs；
- `geometry_prefetch_workers=4`；
- `geometry_prefetch_depth=2`，对应每 rank 八个有界 planned clip tasks；
- Kubric reconstructed-sample LRU 为 16；
- Kubric mmap open-shard LRU 为 90，即保持全部 90 shards 的惰性映射。

这套配置在不改变 deterministic sample/source/target 顺序、XYZ、validity、loss、optimizer 或 RNG 的前提下，把稳定热路径推进到约 `4.82 s/update`。

## 2. 原始故障特征

早期长训练不是 CUDA OOM，也不是数值发散，而是 rank straggler：

- step median 从约 4.8 秒上升到 13.2 秒；
- 单个 geometry task 达到 60--94 秒；
- 两张 GPU 在高低利用率之间交替；
- 一张卡有时显示 100%，但功耗只有约 67 W，另一张卡为 0%。

最后一种现象不能解释为有效满载。FSDP 是锁步执行；当一个 rank 尚未准备好输入时，另一个 rank 可能已经启动 CUDA/NCCL 工作并等待。此时 NVML 会把等待 kernel 计为 busy，但低功耗、低显存带宽和另一 rank 空闲会暴露真实状态。

现场线程检查还观察到慢 rank 位于 `do_user_addr_fault` 和 `rwsem_down_write_slowpath`。这说明 mmap 首次触页、页表建立或大 NumPy 数组 first-touch 可能造成 rank 不对称长尾。它是性能事件，不是训练错误。

## 3. 缓存分层

### 3.1 SSD 上的持久或可恢复缓存

训练热路径不读取 RGB。当前 geometry cache 包含：

- Kubric：固定形状 depth、depth-valid、segmentation mmap，加 compact camera/instance metadata；
- PointOdyssey：scene-level trajectory/intrinsics/extrinsics `.npy` mmap；
- Dynamic Replica：按 stream 保存的 compact trajectory archives；
- 三数据集 Wan clean latents：per-clip safetensors，训练前要求全部命中。

部署时不要根据路径名字推断介质。使用：

```bash
realpath <cache-path>
findmnt -T <cache-path> -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS
```

当前机器的 hot cache 位于 `/tmp/worldbridge4d-cache*`，实际由 `/ssd` backing store 承载。lazy-latent live 路径可以是该 hot tier 的 symlink，但同级 `<lazy>_backup` 必须是非 symlink 的 durable authoritative copy；`scripts/validate_three_dataset_256_cache_roots.py` 对此 fail closed。不要把 scratch symlink 本身描述成 durable cache。

### 3.2 Linux page cache

`mmap` 只建立虚拟地址映射，不保证页面已经驻留。第一次访问仍可能触发 page fault。大 RAM 只表示页面被触碰后有空间保留，不会自动预读全部文件。

使用 `fincore` 做只读驻留检查：

```bash
find <cache-roots...> -type f -print0 \
  | xargs -0 -r -n 256 fincore -b -n -o RES,SIZE \
  | awk '{resident+=$1; size+=$2} END {
      printf "resident_GiB=%.3f size_GiB=%.3f pct=%.2f\n", \
             resident/1073741824, size/1073741824, 100*resident/size
    }'
```

稳定窗口中，geometry 文件只有 `59.56/223.20 GiB`（26.69%）驻留。全量 residency 不是高吞吐的必要条件；真正需要的是当前与近期 future updates 的 working set 已经热，并且 prefetch 队列不被耗空。

### 3.3 进程内 LRU

Page cache 保存文件页面；进程 LRU 保存已经解析、解压或重建的对象，两者不能相互替代。

- Kubric sample LRU 避免同一 clip 的 metadata/sample 重建；
- PointOdyssey scene cache 让 geometry threads 共享 immutable scene arrays；
- Dynamic Replica stream/clip cache 复用解压后的 trajectory arrays；
- Kubric open-shard cache 保留 mmap objects，但不宣称其全部物理页面驻留。

LRU 必须有界。两个 ranks 各自保留所有解压数组会复制大量匿名内存，并可能把有用的共享 file pages 逐出 page cache。

## 4. 为什么选择 4 workers / depth 2

匹配 benchmark `R-20260818100323-232501` 对同一组 16 个真实 K19 clips 做了三轮 Latin-square 比较，并对完整 XYZ、validity 和 source selection 做 SHA-256 exactness 检查。

| 路线 | 16-clip wall time | task p50/p95 | 最大 task |
|---|---|---|---|
| 8 workers / 16 in-flight / cache 2 / shards 8 | 1.181 / 63.643 / 9.912 s | 1.094 / 39.842 s | 60.773 s |
| 4 workers / 8 in-flight / cache 16 / shards 90 | 0.854 / 1.175 / 0.723 s | 0.104 / 0.324 s | 0.509 s |
| 单 loader + 4 compute workers / ring 16 | 0.662 / 5.317 / 29.719 s | 有 head-of-line blocking | 28.578 s |

关键经验：

- worker 越多不等于吞吐越高；
- 过多随机 first-touch 会争抢 mmap/page-table lock、内存带宽和 CPU 调度；
- 单 loader 又会把一个冷文件放大成整个队列的 head-of-line blocking；
- 四个 worker 足以并行独立 clip，同时保留稳定的内存调度余量。

基准中的异常轮次没有对应的 physical read bytes，说明长尾不应简单归因于“SSD 太慢”；并发内存调度和大数组 first-touch 同样重要。

## 5. Prefetch 如何隐藏长任务

训练循环先确定 deterministic sample plan，再把独立 clip geometry 提交为 Futures。GPU 消费当前 update 时，CPU pool 继续准备近期 future updates。主线程只在真正需要当前 clip 时调用 `future.result()`。

因此需要同时记录：

- `geometry_task_seconds_max_rank`：后台任务自身耗时；
- `geometry_wait_seconds_max_rank`：该任务对 GPU 主路径暴露的等待；
- `latent_load_seconds_max_rank`：latent 读取暴露的等待；
- `step_seconds`：完整 update wall time。

一个任务耗时很长并不必然拖慢训练。例如 step 56,505 的 Kubric geometry task 用时 `91.66 s`，但它提前开始，最终暴露给训练的 wait 只有 `0.00014 s`。

Prefetch 的成功判据不是“后台任务永远很快”，而是：

```text
exposed geometry wait ≈ 0
并且
prefetch producer throughput >= GPU update consumption rate
```

## 6. 冷启动、热路径与长尾

精确恢复后的前几步不能代表稳态：

- step 56,504：251.24 s；其中 geometry wait 17.97 s、latent load 0.045 s，其余主要是新进程第一次 FSDP/CUDA forward-backward 路径；
- step 56,505：65.59 s；geometry wait 0.00014 s；
- 后续曾出现 30.61/69.09 s 的 rank-asymmetric tail；
- step 56,550--56,575：25 updates 共 120.49 s，即平均 `4.82 s/update`，覆盖完整 20-step mixture cycle。

稳定窗口中：

- 最近 exposed geometry wait 小于 `0.000074 s`；
- latent load 约 `0.001--0.002 s`；
- 两张 GPU 连续采样为 97--100%，并处于正常计算功耗；
- loss、EPE 和 gradient norm 均 finite。

因此性能报告至少应覆盖一个完整 20-step 三数据集周期，并同时给出 median/p95 或持续窗口平均。只报告一个热 step 或一个冷 step 都会误导。

## 7. 观测命令

### GPU 利用率必须与功耗一起看

```bash
for _ in $(seq 1 10); do
  date -u +%H:%M:%S
  nvidia-smi --id=1,4 \
    --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv,noheader
  sleep 1
done
```

### 检查 rank 是否在 fault/lock 路径

```bash
ps -L -p <rank0-pid>,<rank1-pid> \
  -o pid,tid,psr,stat,pcpu,wchan:36,comm --sort=-pcpu
```

### 检查系统是否存在 I/O、swap 或 blocked-task 压力

```bash
vmstat 1 5
free -h
```

诊断时只做只读检查；不要在训练活跃时用全量 `cat`、`vmtouch` 或自制 warmer 突然读取 200+ GiB，这会与训练争抢 SSD、CPU 和内存带宽。

## 8. 不推荐的捷径

1. **全量 `mlock`**：当前容器 memlock 很小；在共享 1 TiB 主机上锁定数百 GiB 也不安全。
2. **盲目全量预热**：可能改善 file residency，但不能消除 Dynamic Replica 解压或 geometry CPU 计算，且会造成启动停机和资源冲击。
3. **把 worker/depth 调到最大**：已被真实 benchmark 否定，会增加长尾。
4. **把所有 source 的最终 dense XYZ 持久化**：存储会膨胀到数 TiB，而且破坏当前 compact-cache 的工程收益。
5. **仅看 NVML utilization**：低功耗 100% 可能是同步等待，不是有效计算。
6. **在正式 trajectory 中直接调 cache 参数**：performance-only 变更仍应做 deterministic exactness 检查和独立 A/B，再通过 checkpoint boundary handoff。

## 9. 生产检查清单

1. 验证 hot cache 的真实 backing device，而不是只看路径名。
2. 验证 durable backup、hot symlink target 和预期 latent 数量。
3. 对所有 latent 执行 fail-closed warmup；必须为零 miss 后才构建 FSDP。
4. 对候选 pipeline 比较完整 XYZ/validity/source-selection exactness。
5. 每个独立 clip 一个 Future；不要把整个 batch 包进一个 Future。
6. 以 4 workers / depth 2 / sample LRU 16 / open shards 90 作为当前 K19 两卡生产基线。
7. 同时记录 task time、exposed wait、latent load、step time、GPU power。
8. 至少观察一个 20-step mixture cycle，再判断稳态。
9. 保留偶发 cold-shard tail 的预期，不承诺每个 step 都等时。
10. 不为追求短期利用率修改 loss、采样、K、B/A、RNG 或 optimizer 语义。

## 10. 复现信息

- Hypothesis：`research/hypotheses/H023.md`
- Pipeline benchmark：`R-20260818100323-232501`
- 150k optimized resume：`R-20260818180553-c5f105`
- Resume step：56,503
- 稳定窗口：56,550--56,575
- 代码入口：
  - `scripts/train_three_dataset_256_fsdp.py`
  - `src/worldbridge/training256.py`
  - `src/worldbridge/pointodyssey.py`
  - `src/worldbridge/dynamic_replica.py`
  - 历史 matched benchmark（已从当前工作树移除，可由 Git 恢复）

上述数值来自共享主机上的一次生产运行，应视为该硬件和缓存状态下的可复现实证，而不是对所有机器的绝对吞吐承诺。
