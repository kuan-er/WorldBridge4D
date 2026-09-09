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

## Conclusion / decision

Confirmed tens of seconds in latent retrieval, plus observed kernel-lock/page-fault activity in current workers. Untimed reverse RGB reads, array materialization/H2D, GPU forward/backward, input readiness and FSDP/metric synchronization remain candidates; cannot attribute the remaining hundreds of seconds precisely or call all of it disk I/O/GPU compute. Full GT cache fixed an important path, not every input/memory/synchronization stall. Keep current training unchanged. More precise attribution would require lightweight phase timers/CUDA events in a separately checkpointed future snapshot, not hotpatching/re-enabling the suspect periodic traceback diagnostic.
