# Research Decisions

## 2026-08-04 — native MOVi-F schema and conventions

- The read-only root is `/dataset/MOVi-F/128x128/1.0.0`; native records are TFDS `tf.train.Example` TFRecords, not a guessed directory of per-frame files. The audit found 512 train shards and an available validation split. Relevant keys and encoded shapes are recorded by `scripts/inspect_dataset.py`.
- Native sequence length is 24. We use a continuous 21-frame clip with `clip_start=0`, rather than padding. Padding would add a mask and a second source-time convention without being needed by this dataset. This is explicit in every config.
- Depth PNG is uint16 and is decoded linearly with `metadata/depth_range`; independent static-background/forward-flow consistency favored radial camera distance over z-depth. Camera convention was resolved against `instances/image_positions`: quaternion `wxyz`, camera-local to Kubric-world, local forward `-Z`, local `Y` up. Object convention was independently resolved against the temporal `bboxes_3d` corners: `wxyz`, local-to-world. Integer image pixels use center coordinates `(W-1)/2,(H-1)/2`; native normalized object image positions have an edge-coordinate convention.
- Focal pixels are computed as `focal_length / sensor_width * width` (observed 140 px), not inferred from a guessed field of view. Occlusion tolerance is `0.05 + 0.01*max(target_depth,1)` and is configurable.

## 2026-08-04 — geometry semantics

- `X[s,t,p]` is computed on demand from source depth and source segmentation. Object points are transformed world→object-local at `s`, object-local→world at `t`; background remains in world coordinates. `M` checks target positive depth, bounds, target segmentation ID and target radial depth. `V_valid` depends on source depth and finite transform state only. Thus an occluded but geometrically valid object point has `M=0,V_valid=1` and still enters coordinate loss.
- `P[t,p]=X[t,t,p]` is generated in the first clip camera coordinates. Geometry validation Run `R-20260804060621-1ebf71` reported pixel/depth round-trip max below `1.1e-13`, `X[s,s,p]-P[s,p]` exactly 0 for the tested sample, and zero bad instance/depth checks among sampled visible tracks. A diagnostic figure was written outside the repository.
- No full float32 `[source,target,H,W,3]` cache is retained. `GeometryBuilder.trajectory_block`, `pipeline.encode_sample`, and `evaluate.py` use bounded source/pixel/query blocks.

## 2026-08-04 — model and optimization

- Both methods use the same temporal encoder class and hyperparameters and the same 3D residual context structure. Compact has the required shared 2D reconstruction encoder and explicit MLP fuser; Full directly contexts per-source trajectory features. Both use deterministic latents, no KL or random sampling.
- Reconstruction and tracking queries are balanced by group. Compact trains/evaluates only `s=t` and `s=0`; Full additionally trains arbitrary legal `(s,t)` and evaluates both random and all `s,t` on a deterministic spatial stride. Compact is not advertised as arbitrary-source tracking.
- Coordinate mean/scale are computed only from train pointmaps and stored in checkpoints. The default baseline is deliberately bounded (4 train examples, 20 steps, seed 2026); tiny configs use one fixed clip and fixed queries to verify overfit rather than claim generalization.
- CUDA was available. The runs selected CUDA AMP, batch size 1, no gradient accumulation, and 16,384-pixel trajectory blocks. These selections and peak memory/throughput are saved in run summaries. `MLFLOW_TRACKING_URI` was absent, so local PRL tracking remained enabled without a remote backend.

## PRL records

- Task: `T-20260804055436-2bc3c7`, branch `prl/t-20260804055436-2bc3c7-four-d-world-latents`.
- Dataset audit: `R-20260804055700-fc6af7` (schema), `R-20260804055813-33e2db` (initial geometry candidates), `R-20260804060743-f4b48e` (camera/object conventions), `R-20260804060842-8c6610` (depth-flow comparison).
- Geometry validation: `R-20260804060621-1ebf71`.
- Shape smoke: `R-20260804060902-744786`; both exact `[1,16,6,16,16]`, forward/backward passed on CUDA.
- Unit tests: `R-20260804061842-5f297c`, 6 passed. The preceding failed test Run `R-20260804061755-0f1d96` was modified and retried, not hidden.
- Tiny overfit: Compact `R-20260804062043-059151`, loss ratio 0.345; Full `R-20260804062439-810fb1`, ratio 0.431. Both saved/restored checkpoints.
- Bounded training: Compact `R-20260804062809-41bf71` (20 steps, 5.55 s, 153,449 params, 195 MiB peak, 692 query/s); Full `R-20260804063028-a4fe07` (20 steps, 33.45 s, 137,097 params, 2,174 MiB peak, 115 query/s). Validation evaluation: Compact `R-20260804063321-e1b7f8`; Full `R-20260804063717-1d3757`, with all source/target time pairs on stride-16 pixels.

The bounded comparison is a plumbing baseline, not a conclusive quality result: Compact validation reconstruction endpoint error was 7.04 and first-frame tracking 6.37; Full was 11.29 and 10.17 respectively. Full did provide the requested arbitrary-source random/all-time metrics, but did not improve them under this very small budget. The hypothesis therefore remains a testable design prediction, not supported by this first bounded result.

## 2026-08-04 — remote distribution

- The final research worktree was published to the private GitHub repository `https://github.com/kuan-er/WorldBridge4D`, with `main` pointing to the packaging commit after the research commit `52de66a5e89affdf1c26cf6ea0d7501cfa1064b9`. No dataset, checkpoint, cache, or generated output was pushed.
- `.pi/settings.json` pins `pi-research-loop` to Git commit `1124d244b4b7df624c8ddb9b85da1b3864dd66dd`, so another trusted Pi project can install the extension/skill automatically. The public npm package `pi-research-loop@0.1.0` was tested (`7/7` tests) but publication was blocked by npm's requirement for 2FA or a granular token with 2FA bypass; no npm package was published from this session.

## 2026-08-04 — tracking deployment without MLflow

- This project has no MLflow server, so MLflow tracking is disabled rather than pointing at a nonexistent endpoint. W&B remains enabled with entity `zhaigong2023-sjtu-hpc-center`, project `worldbridge4d`, group `worldbridge4d-full-vs-compact`, and the existing research tags.
- `docs/tracking-env.example` contains only placeholders; real `.env` files, API keys, and credentials remain ignored and must be configured separately on each machine. A private GitHub repository is not a safe secret store because credentials persist in history and may be exposed through clones, logs, backups, or collaborators.
- W&B credentials can be provisioned with `wandb login` or environment variables. PRL injects W&B run metadata, while actual scalar logging still requires the training process to call `wandb.init()`.
