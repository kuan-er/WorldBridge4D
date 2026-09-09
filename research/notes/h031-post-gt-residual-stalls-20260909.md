# Post-full-GT residual long tails — read-only diagnosis 2026-09-09 ~14:35Z

Owned training R-20260909074722-72f540 remains unchanged. User asks where remaining time goes. No signal, ptrace, stack-dump enabling, cache re-generation, process control or training code/config modification.

## Established timing scopes

`trainer.py`588–722: geometry wait measures only `future.result()`, NOT all input work. After it come target selection/array indexing, strict `dataset.clean_latent` + `np.stack` (separately timed latent load), normalization, main source RGB assembly, host→device transfers, and cycle reverse `dataset.source_rgb` reads/assembly/transfers. Cycle weight0 still retains this path. Input readiness is AFTER these transfers and reverse RGB preparation, despite CPU_INPUT_READY event name. Then forward, reverse forward, objective/backward, optimizer and distributed diagnostics. `step_seconds`959 includes these costs up to payload construction; geometry/latent timers are max-rank reductions, not mutually exhaustive additive timers. No CUDA-event compute-only breakdown exists for these two updates. Console receipt times can be buffered/batched (not safe as precise barrier durations).

## Observed long updates

-153679 PO:509.522s step, geometry wait0.000149s, latent39.725s; preparation/compute gaps occur across microbatches rather than a missing final checkpoint.
-153680 K:638.386s step, geometry wait0.000128s, latent60.289s. Background geometry task295.312s was already ready at future retrieval: that is overlap, not295s additional blocking.
-PRL input-readiness records:153678 invocation (=completed153679), ALL micro0 receipt14:13:41, ENTER micro1~14:17:11/41; ALL micro2~14:18:11, ENTER micro3~14:20:41/21:11.153679 invocation (=completed153680) also spreads across microbatches. These localize broadly to inter-micro work (compute plus next preparation/synchronization), not pure final optimizer/checkpoint save; receipt timestamps are not exact operation timings.

## Bounded /proc and host sample

Verified exact worker command output path and starttime143931438 for PIDs1236602/1236603 before read-only5.013s sample. At sample end BOTH main threads D-state in `rwsem_down_write_slowpath` (kernel reader/writer semaphore contention, lock object/caller unknown). CPU seconds4.11/4.09, major faults+14/+18, syscall rchar/syscr+0/+0, read_bytes+1576960/+1216512. VmSwap1321372/1287656KiB. This supports ongoing mapped-page/swap-related I/O and kernel lock waits as suspects; it does NOT prove swap thrashing or explain the prior two long updates causally. Attempts to read `/proc/PID/stack` returned EACCES; no workaround or attachment attempted. Open-FD sample includes local PO annotation and DR RGB NPY files; bounded12-path listing is not exhaustive.

Host14:35:29Z:857GiB available RAM,37GiB swap used; interval swap-in68/4KiB/s, swap-out0, CPU idle69/70%, iowait1%. GPUs2/3 utilization100% at instant (shared GPUs; can include NCCL wait/other processes, not proof our forward is compute-bound). Host free memory does not remove already-swapped pages. Requested worker cgroup paths were not present under visible cgroup mounts; separately inspected mounted cgroup root has unlimited memory/CPU and no throttling, but mapping to worker's actual cgroup is unverified, so no worker-specific quota conclusion.

## Follow-up: local does not mean NVMe; latent timer includes RGB

User correctly questions whether local latent reads can take40–60s. Read-only mount/device checks confirm cache root resolves under `/data`, ext4 `/dev/mapper/hdd1--vg-hdd1--lvm[/yejun]`, major:minor253:1 =dm-1. `/sys/devices/virtual/block/dm-1/{dm/name,slaves}` maps to sdf/sdg, both ROTA1 HGST HUS728T8TAL disks, not NVMe. Container `/dev/mapper/...` node unavailable to lsblk -s, but findmnt plus sysfs identifies the underlying devices.

One1s iostat interval:dm-1 read736IOPS/~7362KiB/s,~10KiB/request,14ms r_await,queue10.30,util85.94%; sdf~7409.9KiB/s,14.03ms,util83.96%. This supports current small-I/O/queue pressure on local HDD, not a measurement of the historic60s interval or attribution to our process alone. No storage migration or device tuning.

Important correction to shorthand 'latent read time': `native_kubric.py:clean_latent` first calls `self.local_rgb.read(index)` (bypasses rgb LRU accessor), reading/checking full21-frame RGB identity, then native latent metadata/finite/SHA validation. `NativeRGBCache.read` materializes RGB and computes `rgb_identity`; trainer timer additionally includes `np.stack`, four microbatches summed per rank then max-reduced. Native512 latent FP32 [16,6,64,64] is1.5MiB/clip; full uint8 RGB [21,512,512,3] is15.75MiB/clip. Thus the timed native path touches at least17.25MiB/clip,69MiB/rank-update (not necessarily physical disk bytes because page cache), plus metadata/copies/checks. It is NOT pure latent-device read latency or a single1.5MiB read taking60seconds. PO uses a separate256 reader, so full native RGB revalidation is not automatically an explanation of the PO40s measurement.

Local HDD contention, page faults/mapping/lock waits, CPU integrity validation and copies may all inflate this wall-clock interval. Existing sample values often sub-second establish40–60s as long tails, not expected normal file access. Earlier labels meaning pure latent I/O should be read with this correction. Do not weaken integrity checks to claim an optimization.

## Conclusion / decision

Confirmed tens of seconds in latent retrieval, plus observed kernel-lock/page-fault activity in current workers. Untimed reverse RGB reads, array materialization/H2D, GPU forward/backward, input readiness and FSDP/metric synchronization remain candidates; cannot attribute the remaining hundreds of seconds precisely or call all of it disk I/O/GPU compute. Full GT cache fixed an important path, not every input/memory/synchronization stall. Keep current training unchanged. More precise attribution would require lightweight phase timers/CUDA events in a separately checkpointed future snapshot, not hotpatching/re-enabling the suspect periodic traceback diagnostic.
