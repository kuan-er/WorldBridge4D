# Read-only checkpoint cleanup proposal (2026-09-09)

User asks which pre512 checkpoint to preserve and potential savings; **no deletion authorized or performed in this inventory**. Earlier authorization applied only to the already-deleted K15 replay directory, not these weights.

## Quality / keep recommendation

There is no established universal best among recent256 endpoints. Recommend keeping the plain RGB1x155000 reference as the recent unweighted256 baseline, plus boundary2x155000 (small-screen DR gains with Kubric/global/displacement tradeoffs). H023100000 remains the selected production checkpoint, not superseded by a formal promotion. Do not label interrupted contrast0.01 step154059 the best; its final quality is unresolved. Contrast0.1 failed the boundary screen.

- Plain155000: `/data/WorldBridge4D-runs/h030-fp32-cycle0-rgb1x-step155000-reference/checkpoint-0155000.pt`, recorded SHA2648be240ca7b0d852cdda0c9b7d7cd47bb3d532ab3acbf06c30de33c55d0490.
- Boundary155000: `/data/WorldBridge4D-runs/h030-fp32-cycle0-rgb1x-boundary2x-step155000-reference/checkpoint-0155000.pt`.
- Production100000: `/data/WorldBridge4D-persistent/checkpoints/worldbridge4d-source-rgb-fusion32-cap0p1-step100000/checkpoint-0100000.pt`, recorded SHA3181a255d48687f1634fe62372355815a61f9aba5fef40bca50572459145d0f2.

Evidence: H023 hypothesis production promotion; H030 hypothesis endpoint crossed screen at155000. Boundary2x vs plain raw/all-target parent-macro EPE: Kubric+0.177155%, DR−1.532149%; boundary tracking−0.462463%/−1.161188%, but relative displacement+0.101611%/+0.853464%. Four parents per ready dataset; PO RGB-blocked, so not definitive ranking.

## Space snapshot

CPU metadata-only inventory `R-20260909063814-06a975`, immutable c41a7e92054f1a4e613ec3a0bc8781281df3f20c, completed06:38:14.599Z. JSON manifest `/data/WorldBridge4D-runs/h031-checkpoint-cleanup-inventory-20260909.json` lists every candidate path/alias and all keep reasons. No checkpoint tensor loading or hashing; only stat/links/catalog operations.

- 68 unique checkpoint inodes:448.008GiB.
- Conservative keep13 inodes:93.180GiB, including selected256 references, protected historical origins, unresolved0.01 endpoints and current active dependencies.
- Candidate55 inodes:354.828GiB (~381.0 decimalGB): H03042/271.010GiB, H0317/51.272GiB, other old experiments6/32.546GiB.
- Candidate means **subject to user approval and final active/dependency/open-file audit**, not safe to delete wholesale immediately. Preserve small config, train_status, logs, validation reports and manifests. No raw or active cache deletion proposed. Account all hardlinks; removing only a `latest.pt` alias does not reclaim the inode.

Smallest first cleanup proposal: seven H031 intermediate/capacity weights,51.272GiB. K5 main152000/152500; async/quiescent first2150011 each; K5 capacity150001; K9 timeout900150001 and K15 timeout900150001 including their `latest.pt` aliases. Keep original150000, native150010, both150012,152768,152769,152774 and current K11 directory. In particular current170000 endpoint reviewer reads152768, and current K11 resume/authorized K9 fallback requires152774; neither is disposable.
