# User-authorized checkpoint retirement, 2026-09-20

The user explicitly authorized both the conservative and aggressive cleanup proposals in the current conversation. The cleanup retired only superseded H031 checkpoint bytes and their `latest.pt`/`resume.pt` references. No directory was recursively deleted, and logs, configs, reports, caches, source data, current training outputs, and protected reference checkpoints were retained.

## Execution

- Audit manifest: `/data/WorldBridge4D-runs/h031-checkpoint-cleanup-execution-20260920/manifest.json`
- Manifest SHA256: `8ce6ec070389426582e93f5135ede51c46858f3f9cc4b248b4af6f26f763b2f9`
- Pre-delete dry run found 11 unique checkpoint inodes and 28 file/symlink paths.
- All regular hard links were accounted for by `st_nlink`; candidate and protected inode sets were disjoint.
- `/proc` FD and mmap audit covered 645 readable processes and found no open candidate. One unrelated process was unreadable; no process was controlled or terminated.
- Execution unlinked 98,010,202,112 allocated bytes (98.010 GB / 91.279 GiB).
- Observed free-space increase: 98,010,230,784 bytes.
- `/data` free space after cleanup: 586,919,571,456 bytes.
- Post-delete scan found no remaining candidate inode references.

## Retired checkpoints

- Rejected high-LR FULL-unfreeze checkpoint `183238`.
- Superseded pre-source checkpoints `182000` and `182500` from the old K11 directory.
- Superseded H031 recovery/phase checkpoints `175374`, `175361`, `175000`, `169500`, `156000`, old-K11 `155000`, `152774`, and `152768`.
- Their `latest.pt` hard links and old handoff `resume.pt` symlinks were also removed so historical handoffs cannot be mistaken for live recovery inputs.

Historical STATE entries, review reports, artifact IDs, and handoff metadata may still name these retired paths. They remain provenance only and are no longer usable checkpoint bytes.

## Retained and verified

- Active low-LR Run `R-20260919154805-1f6b19` and its entire rolling checkpoint directory.
- Protected source `checkpoint-0183000.pt`, size `7,864,620,263` bytes, including its current handoff aliases.
- H030 plain RGB1x step-155000 reference.
- H027 step-130000 reference and H023 production assets.
- All Kubric512, Dynamic Replica512, and PointOdyssey256 data/cache content.
- All PRL snapshots, logs, W&B records, configs, reports, and non-checkpoint metadata.

After cleanup, the active training Run remained `running`; no training process, configuration, optimizer state, RNG state, or dataset input was modified.
