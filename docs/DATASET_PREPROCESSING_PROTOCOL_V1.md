# WorldBridge4D Dataset Preprocessing Protocol v1

Status: normative handoff specification for external datasets.

## 1. Scope and non-negotiable model contract

A training-ready dataset must let the consumer return, for one clip index `i` and one source frame `s`:

```text
clean_latent:   float32 [16,6,16,16]
source:         int64   [21]       # all entries equal s
target:         int64   [21]       # exactly 0..20
xyz_normalized: float32 [21,3,128,128]
valid:          bool    [21,128,128]
```

`xyz[s,t,v,u]` is the 3D position at target time `t` of the physical surface observed at source pixel `(u,v)` at time `s`. XYZ is expressed in the camera coordinate system of frame `s`, held fixed for every target. It is not expressed in the target camera frame.

`valid` means that source depth, correspondence and target state are valid. It is not visibility. A valid point remains supervised when occluded at the target. Visibility may be stored separately for evaluation, but the canonical XYZ loss must not mask occluded-valid points.

The current route is intentionally fixed to 21 consecutive frames and 128x128. Temporal padding, temporal interpolation, silent frame duplication and hidden spatial resizing in the model loader are forbidden.

## 2. Dataset split and clip extraction

1. Split by parent video/scene before extracting clips. Parent IDs must be disjoint across train, validation and test.
2. Assign every clip a stable UTF-8 `clip_id`, plus `parent_id`, original start frame, frame stride, source FPS and timestamps.
3. Extract exactly 21 ordered frames. Record the temporal stride; do not interpolate missing frames.
4. Apply one shared spatial crop/resize to RGB, depth, segmentation and other pixel-aligned annotations. Update intrinsics exactly.
5. Preserve a deterministic lexicographic clip order in every artifact. All caches use this order.

Required hot RGB representation before VAE encoding:

```text
rgb: uint8 [21,128,128,3], sRGB, values 0..255
```

Float RGB is allowed only in `[0,1]`. Do not apply ImageNet normalization. The Wan adapter performs `[0,1] -> [-1,1]` itself.

## 3. Camera and geometry conventions

The canonical interchange representation is:

```text
intrinsics:       float64 [21,3,3]
camera_to_world:  float64 [21,4,4]
depth:            float32 [21,128,128], metres
depth_valid:      bool    [21,128,128]
```

Conventions:

- camera optical axis is local `-Z`;
- local `+X` projects right and local `+Y` projects up;
- image `u` grows right and `v` grows down;
- integer `(u,v)` denotes the pixel centre;
- `camera_to_world` maps camera coordinates to metric world coordinates;
- manifest `depth_convention` must be exactly `radial_meters` or `z_meters`;
- depth conversion to the canonical backprojection must be explicit and tested;
- NaN, infinity, non-positive depth and missing transforms set validity false.

An adapter targeting the current `MOViSample` implementation must either provide centred principal point with `fx=fy`, or extend `CameraModel` to consume the full intrinsic matrix. Silently approximating general intrinsics with scalar focal length is forbidden.

## 4. Off-diagonal supervision modes

The manifest declares one of two modes.

### 4.1 `rigid_instances`

Required annotations:

```text
segmentation:          integer [21,128,128], 0=background, object IDs 1..N
instance_to_world:     float64 [N,21,4,4]
instance_state_valid:  bool    [N,21]
```

For a source object pixel, convert its source world point to object-local coordinates with the source instance pose, transform it with each target instance pose, then express the result in the source camera basis. Background remains fixed in world coordinates unless the dataset explicitly provides a moving-background model.

### 4.2 `dense_xyz`

When rigid instance poses do not exist, the producer directly supplies source-grid trajectories and validity under the contract in section 1. Optical flow alone is insufficient unless it is accompanied by target 3D position/depth and identity-valid correspondence through occlusion.

A dataset containing only RGB, camera and per-frame depth can supervise diagonal pointmaps but cannot claim arbitrary-source 4D tracking. The preprocessor must fail instead of inventing off-diagonal labels.

## 5. Required cache tiers

The cache root is immutable after validation:

```text
CACHE_ROOT/
  manifest.json
  splits/
    train.jsonl
    validation.jsonl
    test.jsonl                 # optional
  samples/                     # geometry metadata, sharded
  latents/
    wan2.1_1.3b_fp32/          # mandatory
  geometry/                    # optional persistent acceleration tier
  stats/
    coordinate_stats_train_source.npz
  audit/
    validation_report.json
```

### 5.1 Manifest and split indexes

`manifest.json` follows `docs/dataset_manifest_v1.schema.json`. Every JSONL row contains at least:

```json
{"index": 0, "clip_id": "...", "parent_id": "...", "start": 0, "stride": 1, "timestamps": [0.0]}
```

The actual timestamps array has exactly 21 finite, strictly increasing values.

### 5.2 Frozen Wan clean latent cache

Use the exact `Wan2.1_VAE.pth` referenced by the manifest and the repository `WanVAEEncoder`:

- posterior mean, never posterior sampling;
- Wan channel mean/std normalization performed by the adapter;
- output shape exactly `[16,6,16,16]` for each 21-frame clip;
- canonical cache dtype float32;
- no noise, padding, pooling or interpolation.

A clip uses 98,304 bytes (96 KiB), so this cache is mandatory even for modest datasets. Store sharded tensor-only files that support safe mmap loading; recommended shard size is 128 or 256 clips. Do not use one giant Python pickle. Record Wan checkpoint SHA-256, preprocessor Git commit and ordered clip IDs in the manifest.

### 5.3 Train-only coordinate statistics

Compute mean and standard deviation only from valid diagonal pointmaps of the training split, after expressing each diagonal pointmap in its own source-camera frame. Accumulate `sum` and `sum_of_squares` in float64; write float32 values:

```text
mean:          float32 [3]
scale:         float32 [3]
examples:      integer
point_count:   integer
coordinate_frame: "source"
stats_source: "diagonal_pointmaps_in_selected_coordinate_frame"
```

Validation/test data must never affect these statistics. Normalize all trajectory targets channel-wise with this file.

### 5.4 Optional persistent geometry acceleration

The default consumer computes one source and all 21 targets using a bounded multi-thread prefetcher. If CPU geometry remains slower than GPU execution, add one of these explicitly versioned tiers:

1. `compact_source_geometry`: source backprojection/object-local coordinates, instance ID and validity. Expected MOVi-F scale is roughly 30 GB for all clips and sources.
2. `dense_xyz_fp16`: all source-target maps in mmap-capable shards. This is fastest but approximately 250 GB for 5,737 clips before masks. Quantization must pass the metric-error audit below.

For dense storage, shard by contiguous clip range and keep source as a directly sliceable dimension. Recommended shape is `[clips,21,21,3,128,128]`; valid masks may be bit-packed. The loader must read only the selected source slice.

## 6. Consumer and performance contract

The dataset adapter exposes:

```python
len(dataset)
dataset.clip_id(index)
dataset.clean_latent(index)                  # [16,6,16,16]
dataset.source_all_targets(index, source)    # xyz [21,3,H,W], valid [21,H,W]
```

Canonical training behavior:

- one random source per clip and all ordered targets 0..20;
- source plans determined by `(seed, global_step)` so prefetch does not alter exact resume;
- geometry prefetch enabled by default;
- 8 CPU workers and bounded queue depth 2 initially;
- each worker owns its geometry cache; no mutable cache is shared across threads;
- visibility disabled for XYZ-only training;
- pinned host tensors and non-blocking host-to-device copies;
- train EPE computed on GPU;
- detailed diagnostics and W&B logging at step 1, every 20 steps, and final step;
- every W&B payload includes explicit `global_step`.

A producer is not complete until a consumer can run a two-step real-Wan forward/backward/Adam smoke with finite gradients in Wan, geometry adapter and decoder.

## 7. Mandatory validation gates

Write all results to `audit/validation_report.json`; any failed gate blocks formal training.

1. **Schema:** manifest, split rows, shapes and dtypes match this protocol.
2. **Split leakage:** parent-ID intersections across splits are empty.
3. **Temporal:** exactly 21 strictly ordered frames; no hidden padding/interpolation.
4. **Alignment:** RGB/depth/segmentation use the same transformed pixel grid.
5. **Camera round trip:** backproject then project valid pixels; max error <= `1e-3` pixel.
6. **Diagonal identity:** `xyz[s,s]` equals source depth backprojection in source coordinates; max error <= `1e-4` metre for float32 labels.
7. **Rigid dynamics:** reconstructed dynamic trajectories agree with direct instance transforms; max error <= `1e-4` metre.
8. **Static background:** valid static world points remain invariant in world coordinates; max error <= `1e-4` metre.
9. **Validity semantics:** at least one audited occluded point remains valid; visibility and validity are not conflated.
10. **Finite values:** every valid XYZ and every transform used by it is finite.
11. **Coordinate statistics:** train-only population/count and source-frame metadata are recorded.
12. **Wan determinism:** re-encode at least 16 sampled clips twice; float32 latent max difference <= `1e-6` and shape is exact.
13. **Cache order:** sampled clip IDs match across index, latent and geometry shards.
14. **Quantization:** if FP16 dense XYZ is used, max valid metric error versus float32 <= `0.02` metre and mean <= `0.002` metre, or stricter dataset-specific thresholds.
15. **Tiny overfit:** loss decreases on one clip with off-diagonal targets.
16. **Real gradient smoke:** two optimizer steps complete with finite Wan/adapter/decoder gradients and correct W&B global steps.

## 8. Reproducibility and handoff record

The producer must report:

- exact argv and environment;
- preprocessing seed;
- source dataset release/checksum;
- preprocessor Git commit;
- Wan VAE checkpoint SHA-256;
- all artifact paths, byte sizes and SHA-256 values;
- clip counts per split and rejected-clip reasons;
- wall time, peak CPU RAM, GPU and disk usage;
- complete validation report;
- a ready-to-run training YAML referencing only immutable cache paths.

Generated datasets, latents and weights are never committed to Git. Only code, manifests without secrets, schemas, audit summaries and reproduction commands are committed.
