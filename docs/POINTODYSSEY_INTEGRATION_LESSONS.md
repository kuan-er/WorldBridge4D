# PointOdyssey Integration and Training Lessons

Status: operational record for the PointOdyssey Dense4D route (H020).

This document records the failures, measurements, and fixes encountered while adapting PointOdyssey to the WorldBridge4D 21-frame, 128x128, source-frame XYZ contract. It complements the normative requirements in `DATASET_PREPROCESSING_PROTOCOL_V1.md`.

## 1. Final data contract

The PointOdyssey adapter uses:

- 21 consecutive frames per clip;
- 128x128 RGB, depth, and geometry;
- one source frame per clip and all ordered targets `0..20`;
- XYZ expressed in the selected source camera frame for every target;
- validity distinct from visibility, so occluded-valid tracks remain supervised;
- source-to-source geometry forced to the canonical source-depth backprojection;
- train-only source-frame coordinate statistics;
- deterministic manifest and latent order.

The prepared training cache contains 9,746 clips from 109 scenes. Wan clean latents are stored in canonical contiguous shards independent of DDP world size.

## 2. Coordinate and annotation pitfalls

PointOdyssey annotations cannot be passed directly to the MOVi-F geometry path. The adapter must explicitly reconcile:

- the fixed PointOdyssey crop with the 128x128 output grid;
- the corresponding intrinsic-matrix update;
- image axes, camera axes, and the WorldBridge4D local `+X`, `+Y`, `-Z` convention;
- scene-level sparse 2D/3D tracks and clip-relative frame offsets;
- target validity versus target visibility;
- source-depth diagonal geometry versus off-diagonal track geometry.

The diagonal is generated from the source depth and transformed intrinsics rather than trusted as an incidental sparse-track result. This keeps the interchange identity exact while retaining occluded-valid PointOdyssey tracks off diagonal.

A new dataset route must audit round trips, diagonal identity, finite values, and at least one occluded-valid correspondence before training. Shape checks alone are insufficient.

## 3. Why the first geometry pipeline was extremely slow

PointOdyssey stores annotation at scene granularity in compressed `anno.npz` files. A training clip needs only 21 frames, but the original loader decompressed full-scene arrays including tracks, validity, visibility, intrinsics, and extrinsics.

The original batch-12 capacity sample selected 12 clips from 10 scenes. Their compressed annotation files totalled 4.29 GiB; individual files reached 1.55 GiB. This exposed three compounding problems:

1. Random clip sampling caused concurrent random reads of many large files from NAS.
2. Thread-local dataset instances independently decompressed the same scene when several clips shared it.
3. Track collision resolution, camera transforms, and source-depth replacement used Python loops and became GIL-bound after decompression.

There was also an earlier prefetch bug: one Future wrapped a complete batch, so increasing the thread-pool size did not actually parallelize clips.

The observed signature was high CPU during decompression, approximately one CPU core during Python rasterization, near-zero GPU utilization, and no optimizer step after more than 39 minutes. This was a data-pipeline failure, not a CUDA-memory failure.

## 4. Adopted data-pipeline design

### 4.1 Scene-local sequential clip order

Training clip indices are consumed in contiguous manifest order. Because the manifest is ordered by scene, a batch normally stays within one scene instead of issuing random NAS reads across many scenes. Source frames remain random and deterministic from `(seed, global_step)`.

For DDP, each rank receives a contiguous slice of the step's global clip range. This preserves scene locality as world size grows.

Do not globally shuffle individual clips when the storage unit is a multi-hundred-MiB compressed scene. If epoch-level shuffling is required later, shuffle scene blocks and retain contiguous clips inside each block.

### 4.2 Shared read-only scene LRU

Geometry threads share one `PointOdysseyDataset` and a lock-protected scene LRU. The first thread loads and decompresses a scene; subsequent workers reuse the same immutable NumPy arrays. Four scenes are retained, which is sufficient for sequential batches and bounded prefetch without unbounded RAM growth.

The cache lock protects lookup, insertion, and eviction. Loaded arrays are read-only by convention; per-clip calculations use new outputs and must not mutate cached annotation.

Thread-local caches remain appropriate for small independently decoded samples, but are harmful for large scene-level archives because they multiply NAS traffic and decompressed RAM.

### 4.3 One Future per clip

The prefetcher submits each clip as an independent Future. Queue depth remains bounded, but up to eight clips can perform geometry work concurrently. Never wrap a Python loop over the entire batch in one Future and call it parallel prefetch.

### 4.4 Vectorized sparse-track rasterization

The PointOdyssey adapter now performs the following as NumPy batch operations:

- projection into the cropped 128x128 grid;
- deterministic nearest-track selection for pixel collisions using lexicographic sorting;
- all 21 target camera transforms for selected tracks;
- validity and visibility writes;
- source-depth diagonal replacement.

This removes the per-track and per-pixel Python loops that were limited by the GIL.

### 4.5 Measured result

On the same route, batch-12 geometry changed from more than 39 minutes without completion to 2.88 seconds for 12 contiguous clips from one scene. The optimized pipeline reached model forward promptly, making CUDA capacity—not geometry—the binding limit.

## 5. Persistent geometry and mmap decision

The existing `geometry_cache_entries` setting originated in the MOVi-F path. It did not automatically create a persistent PointOdyssey geometry cache. PointOdyssey initially had only:

- persistent Wan latent shards;
- split and statistics artifacts;
- process-local annotation caching;
- operating-system page cache.

A full uncompressed mmap conversion was not adopted for this run. The raw PointOdyssey training annotations are about 39 GiB compressed and about 81 GiB uncompressed; the selected arrays alone are about 77 GiB. The available data volume did not have enough safe headroom, and dense all-source/all-target XYZ would be much larger.

If persistent acceleration is needed in a future deployment, use one of these designs:

1. contiguous `.npy`/mmap scene arrays when sufficient local SSD space exists;
2. compact source-pixel/track-index shards rather than dense XYZ;
3. chunked compressed arrays with frame/track-aware slicing;
4. scene-block shuffling plus the shared in-memory LRU.

Always measure storage expansion before building a cache. Do not assume that a configuration field used by another dataset creates the required tier.

## 6. DDP failures and restart protocol

### 6.1 Unused Wan parameters

The structured Wan readout consumes selected hidden layers, so not every trainable Wan parameter contributes to every loss. A one-step DDP smoke passed, but the second step failed with:

```text
Expected to have finished reduction in the prior iteration
```

The DDP wrapper therefore uses `find_unused_parameters=True`. A real DDP gate must execute at least two optimizer steps; a one-step gate cannot detect this reduction-state failure.

### 6.2 World-size expansion

PyTorch DDP cannot add ranks to a live process group. Progressive use of physical GPUs 4, 2, and 6 is implemented as an optimizer-boundary handoff:

1. request a stop file;
2. finish the current optimizer step;
3. save model, AdamW state, global step, and `clips_seen`;
4. exit the current `torchrun`;
5. restart with the larger `CUDA_VISIBLE_DEVICES` list and world size;
6. restore the checkpoint instead of initializing from scratch.

Canonical latent shards must not depend on world size. Rank-local latent caches would invalidate this handoff.

A successful handoff in this route saved a 6.1 GiB checkpoint at global step 64 and resumed the optimized run from step 64.

## 7. Capacity probing lessons

Capacity probes must include geometry, Wan forward, decoder forward, backward, and optimizer state where relevant. Data-preparation latency and CUDA OOM must be distinguished from one another.

After fixing geometry:

- batch 12 reached decoder forward quickly but OOMed after the process used about 77.35 GiB and requested another 1.97 GiB;
- batch 8 completed with a one-step peak of 55.99 GiB;
- resumed formal training with batch 8 reached a reported PyTorch peak of about 59.30 GiB.

The selected physical batch is therefore 8 per GPU for the current BF16 full-finetuning configuration. Do not infer capacity from a stalled geometry probe, and do not treat the largest forward-only batch as an optimizer-safe training batch.

## 8. Model and tracking setup pitfalls

The PointOdyssey route also required:

- correcting the configured Wan model path;
- generating and validating the Wan empty text condition;
- checking that VAE and DiT weights exist before scheduling GPUs;
- using canonical latent shard order matching the manifest;
- preserving the full-finetuning BF16 configuration across restart;
- logging explicit global steps to W&B.

When no W&B API credential is present, runs are recorded offline. The resulting directories must later be uploaded with `wandb sync`; lack of a run URL does not mean metrics were discarded.

## 9. Recommended bring-up checklist for the next dataset

1. Freeze the clip, crop, temporal, coordinate-frame, validity, and visibility contract.
2. Split by parent scene before extracting clips.
3. Audit one diagonal and multiple off-diagonal correspondences numerically.
4. Measure the storage unit actually read per sample: frame, clip, or scene.
5. Keep batches local to that storage unit; shuffle blocks rather than individual records if necessary.
6. Decide explicitly whether caches are persistent, process-local, worker-local, or shared.
7. Benchmark cold and warm geometry separately before running a CUDA capacity sweep.
8. Vectorize correspondence collision resolution and coordinate transforms.
9. Submit one prefetch Future per independently executable item.
10. Run at least two DDP optimizer steps with the real model.
11. Test checkpoint restoration of model, optimizer, global step, and sample accounting.
12. Probe physical batch size only after the data pipeline can keep the GPU fed.
13. Verify W&B credentials or record the exact offline sync path.

## 10. Reproduction pointers

Primary implementation files:

- `scripts/preprocess_pointodyssey.py`
- `src/worldbridge/pointodyssey.py`
- `scripts/train_pointodyssey_ddp.py`
- `scripts/run_pointodyssey_progressive_ddp.sh`
- `configs/pointodyssey_dense4d_100m_ddp_30k.yaml`

Representative runs:

- two-step DDP gate: `R-20260811061109-6c46f4`;
- pre-optimization batch-12 stall: `R-20260811061312-fe5954`;
- safe step-64 checkpoint handoff: `R-20260811062558-1ca703`;
- optimized batch-12 CUDA OOM: `R-20260811063827-811e1e`;
- optimized batch-8 resumed training: `R-20260811064034-c9416d`.

The configuration filename retains `30k` for historical reasons, but its formal target is 100,000 global optimizer steps.
