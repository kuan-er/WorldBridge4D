# User-authorized checkpoint retirement, 2026-09-09

User explicitly confirmed: “好的，可释放的都删除吧”, following the exact proposal of H03046 + H03112 unique checkpoint inodes (384.540199GiB /412.896895GB). This supersedes previous historical protection of the listed old phase origins and ablation weights, NOT protection of current training dependencies. Scope does not include H023 production100000, H027/H029/other early experiments, source data, native caches, logs/configs/reviews, or any process termination.

## Exact scope / provenance

- Manifest: `/data/WorldBridge4D-runs/h031-cleanup-keep155k-and-active-proposal-20260909.json`.
- SHA256: `d134fdaa764412e07c5a3af42b85b1ae15af17aaf0c6580e83675e9fc106bbc7`.
- 58 unique inodes /93 regular `.pt` paths, including hardlink aliases, allocated412896894976bytes. No recursive deletion.
- Executor `research/analysis/h031_execute_checkpoint_cleanup.py`, immutable `d1e814fa0b1d6147fd0a7e0a6b9f803b04dbf41a`.
- CPU dry-run `R-20260909070749-6b863a`: passed07:07:49.709Z; read-only443 visible processes (FDs, mappings, direct path args including resolved handoff aliases), zero blockers/unreadable processes. Exact file identities, sizes, all hardlinks and protected inode exclusions verified. No payload reading/hash or GPU use; seed not applicable.
- Success-gated execution `R-20260909070758-91be97` reruns these checks before unlinking. Metadata-only audit journal fsynced after each unlink; parent dirs opened O_NOFOLLOW, identity rechecked immediately before unlink. Unique output prevents automatic re-execution after a partial operation.
- Audit directory: `/data/WorldBridge4D-runs/h031-checkpoint-cleanup-execution-20260909` with original manifest copy, pre_delete.json, unlink_audit.jsonl and final complete.json upon successful independent absence/preservation checks.

## Retained

- Plain RGB1x155000 `/data/WorldBridge4D-runs/h030-fp32-cycle0-rgb1x-step155000-reference/checkpoint-0155000.pt`, recorded SHA2648be240ca7b0d852cdda0c9b7d7cd47bb3d532ab3acbf06c30de33c55d0490 (all existing aliases of this inode also preserved).
- H031 phase origin152768 `/data/WorldBridge4D-runs/h031-k512-k5-original-prefetch-150012-to160010-gpu23-20260908/checkpoint-0152768.pt`, required by immutable170000 endpoint reviewer.
- H031152774 `/data/WorldBridge4D-runs/h031-k512-k9-mix50-prefix5-to170000-gpu23-20260909/checkpoint-0152774.pt`, current K11 resume/authorized K9 fallback; active handoff alias preserved.
- Entire current K11 output directory, including any new periodic checkpoints. Original Wan/VAE pretrained weights and runtime shim remain untouched. All noncandidate files in touched directories are checked unchanged after deletion.

## Consequence / retired references

Historical checkpoint artifact registrations, original review reports, symlinks and source code may still name retired files. They are historical evidence, not available bytes; do not retry old chains or claim those artifacts remain usable. Among removed weights are H030150000,151500,154059, boundary2x155000 and contrast0.1 endpoints; H031150001/150010, both150012 variants,152000/152500 and152769. Reproducing old evaluations or handoff/numerical gates from these retired weights is no longer directly possible. Current152768-based endpoint review and152774 fallback remain intact; training protocol/quality claims are unchanged.

## Completed / stop cleanup

Execution91be97 succeeded07:10:35.255Z (15:10 Beijing), exit0. All93 exact regular PT paths/58 unique inodes absent;412896894976 allocated bytes unlinked. Free space increased from108402921472 to521295142912bytes, measured delta412892221440bytes (~384.5GiB /412.9GB); concurrent live writes explain the4.67MB difference from unlinked allocation. Retained155000/152768/152774 identity/size/mtime checks passed;94 noncandidate regular files unchanged; current K11 directory untouched. Post-delete442-process audit found zero FD/mmap/direct-path references to deleted weights and zero unreadable processes.

Completion and full per-path fsynced journal are in the audit directory. Cleanup is finished: no retry, additional deletion, cache producer restart or training intervention. Current K11 continues toward170000. Earlier inventory/cleanup proposals are historical; never reuse their removed paths as existing files.
