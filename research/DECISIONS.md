# Research Decisions

## 2026-08-08 — H004 source-centric common implementation validated

- Implementation Task `T-20260808075825-d25087` was created from protocol commit `c522637b0c70b90e2da7c0e34e9ddf2922ce2a25`; the outer worktree and finished protocol Task remain untouched.
- Sampling plans are deterministic main-thread values. For global clip position `p`, `clip=p mod N`, `visit=floor(p/N)`, and `source=(clip+visit) mod 21`; every selected source enumerates target `0..20`. Worker completion order therefore cannot affect data order.
- Eight threads each own their `DynamicPointmapCache`; a depth-2 queue bounds pending batches. XYZ, `M`, and `A` are pinned before non-blocking H2D. Run `R-20260808081437-3a97cb` established exact synchronous/asynchronous target and weighted-loss equality on real MOVi-F clips. Formal B0/E3/E5/E6 runs remain pending a real-Wan bounded smoke and release of GPUs 5/6.
- E5 uses the exact fixed 1,000-step schedule statistic from `R-20260808081551-b0c0db`: positive visible-valid count `4,316,774,751`, negative occluded-valid count `926,081,989`, and `pos_weight=0.21453099650044724`. Decoder construction was ordered so every overlapping initialized parameter is bit-identical across B0/E3/E5/E6 under `decoder_seed=424242`; only each arm's declared module differs.

## 2026-08-08 — H004 source-centric ablation screen protocol

- Task `T-20260808072920-bd4011` records the approved follow-up design; no experiment was launched in this phase. All formal arms will use one deterministic source per clip, enumerate all 21 target times, and share a bounded asynchronous CPU geometry prefetch pipeline. Sampling plans are created in the main process; eight shared-memory threads and queue depth 2 are fixed for every formal arm. Synchronous/asynchronous target and loss equality must pass before the screen. These common changes define a new optimized baseline and do not permit direct attribution against the old pair-sampled 2,400-step run.
- The screen is one seed (`2029`), `batch_size=16`, no gradient accumulation, BF16 full-Wan training, exactly one timestep-zero DiT forward, 1,000 optimizer updates, and a fixed 32-clip exhaustive validation subset. XYZ uses `A`, never predicted visibility, with fixed `1/3` diagonal and `2/3` off-diagonal pair weights. Every arm starts from the same Wan checkpoint and decoder seed. Primary ranking is off-diagonal all-valid metric EPE; visible/occluded-valid and direction/gap groups plus throughput, prefetch wait and CUDA memory are mandatory diagnostics.
- `B0` is 2D spatial RoPE, two decoder cross-attention blocks and XYZ-only loss. `E3` changes only to 3D temporal/spatial RoPE with memory time `0..5`, query time `target*5/20`, 8 rotary dimensions per axis and 8 pass-through dimensions per head. `E5` changes only by adding a full-resolution visibility-logit head with `A`-masked off-diagonal BCE, a once-computed fixed train-schedule `pos_weight`, and initial weight `0.1`; its predictions never mask XYZ. `E6` changes only decoder depth from two to four blocks. Depth and reprojection losses are explicitly removed from this screen, as are iterative/multi-step DiT variants. The 1,000-update results are screening evidence only; individual arms are not combined until single-factor results exist, and winners require a longer multi-seed confirmation.

## 2026-08-07 — H004 feed-forward Wan and dense-query v1 conventions

- Task `T-20260807120421-6da6ff` tests exactly one route: frozen native Wan VAE mean latent, one Wan2.1-1.3B DiT final-output forward, and a lightweight dense `(s,t)` XYZ decoder. It reuses the audited MOVi-F adapter/rigid geometry and the strict Wan checkpoint conversion from finished H003; it does not reuse H001/H003 target-latent supervision or define a GT `Z4D`.
- GenCeption arXiv `2607.09024` was read from the actual PDF (SHA256 `26bd3358e5d33da3f221c3dce11878af6635d478bc9d5986d5f7535542218dac`). Section 3.2 feeds a clean video latent directly to DiT, sets Rectified Flow time to zero (the clean endpoint/termination of generation), executes one forward, and negates the final raw velocity `v=epsilon-x0`; it explicitly uses final rather than intermediate features. D4RT arXiv `2512.08924` was also read (SHA256 `b8bb0a667a7c4e6d4206585adaca6711e1d900678e8fa0a80061e4c2535bc452`); its relevant transferable choices are global memory cross-attention, independent queries, and no query self-attention. H004 deliberately omits D4RT's point query, camera query, RGB patch, auxiliary heads, and large decoder.
- Official Wan source and Diffusers 0.36 agree on the convention. The forward corruption is `x_sigma=(1-sigma)x0+sigma*epsilon`, timestep passed to Wan is `1000*sigma`, generation integrates sigma from 1 (noise) to 0 (clean), and the scheduler update is `x_next=x+delta_sigma*model_output`. Therefore the trained raw model output is `d x_sigma/d sigma=epsilon-x0`. H004 passes an exact all-zero timestep tensor to Diffusers Wan and defines `Z4D=-raw_velocity`; the sign is not inferred from a variable name. A scheduler algebra test and a real checkpoint execution are required before training.
- Wan tensors are channel-first video `[B,C,T,H,W]`. MOVi RGB `[B,T,3,H,W]` is mapped from `[0,1]` to `[-1,1]`; VAE channel normalization is `(posterior_mean-latents_mean)/latents_std`. The causal VAE stride is spatial 8 and temporal 4 with first-frame retention, so exactly 21 frames at 128x128 produce `[B,16,6,16,16]`, not 5 temporal tokens. H004 does not resize or interpolate this latent.
- Decoder source and target indices are zero-based in code (`0..20`) but correspond to the paper notation (`1..21`). They use separate embedding tables. The full native final output is flattened in `(latent_time,row,column)` order to 1,536 memory tokens; all six temporal tokens at one spatial location receive the same spatial-only 2D RoPE. There is no temporal/delta-time RoPE in v1.

## 2026-08-07 — H004 staged gates passed before training

- Dataset/source audit `R-20260807120649-ce3667` reconfirmed the native 24-frame TFRecord schema, radial uint16 depth, `wxyz` local-to-world camera/object rotations, camera local `-Z`, and train/validation splits. Geometry Run `R-20260807121556-0d3599` then tested all 21 sources on two real clips: reprojection max `1.08e-13` px, diagonal error exactly zero, dynamic rigid error at most `5.02e-7` m, static-background error at most `2.06e-6` m, and explicit occluded-valid populations. Therefore training XYZ is masked by `A`, never `M`.
- Native empty UMT5 conditioning was produced outside the worktree in `R-20260807121652-27b9ac`: shape `[1,512,4096]`, one unpadded empty-prompt token, from vendored official Wan source commit `a648340f9798f173227a0626fde66a6e9b65879a`. Training does not use the zero-condition smoke fallback.
- Real checkpoint audit `R-20260807121940-5cafa5` encoded the same MOVi clip twice with zero difference and obtained clean latent `[1,16,6,16,16]`. A hook observed the exact Diffusers Wan timestep `[0.0]`; raw output and `Z4D=-raw` both had `[1,16,6,16,16]`; scheduler clean-recovery/sign error and negate error were exactly zero; an XYZ-style scalar backward reached an actual DiT parameter.
- Decoder smoke `R-20260807122056-d99561` used full memory `[1,1536,16]`, four parallel pair maps, low-resolution features `[1,4,256,16,16]`, and output `[1,4,3,128,128]`. Spatial RoPE changed broadcast content across positions (mean absolute diagnostic difference `1.439`), source/target embeddings and swapped query content differed, memory round-trip was exact, gradients reached both backbone and upsampler, and checkpoint restore passed. Unit Run `R-20260807122126-c56ffb` passed the existing and H004 tests.

## 2026-08-07 — H004 feasibility result and stopping decision

- Train-only moments from all 5,737 clips (`1,973,886,172` valid pointmap coordinates) are `mu=[-0.004756,1.984044,-13.474529]`, `sigma=[4.819901,5.335501,11.895373]`, Run `R-20260807122258-659248`. The earlier serial attempt `R-20260807121646-1b8e66` was explicitly terminated and replaced by a reproducible 16-worker reduction; it did not produce statistics.
- Full-DiT fixed-pair overfit succeeded in `R-20260807123135-3e0f39`: four maps on one clip, 24 updates, normalized loss ratio `0.7176`, EPE `15.449 -> 10.828` m, actual Wan and decoder gradients, 13.27 GiB peak, and weights-only checkpoint save/load. The two failed parent attempts are retained (`R-20260807122910-ddca60`: device index; `R-20260807122942-7814b1`: unsafe NumPy metadata in weights-only checkpoint) and their fixes are in the successful commit.
- Random arbitrary-query overfit was stronger (`R-20260807123252-527d39`): 32 updates, evaluation loss ratio `0.5765`, EPE `15.916 -> 8.022` m, with diagonal, forward, backward, source-zero and source-positive groups all improving. This is the main feasibility evidence that pair content is not ignored.
- The deliberately bounded 8-clip/32-update run `R-20260807123356-35e147` and exhaustive two-clip validation `R-20260807123513-8d40a0` are plumbing, not quality evidence. Pointmap/tracking EPE is `11.457/11.419` m; visible/occluded-valid tracking is `11.980/8.670` m; source-zero/source-positive tracking is `10.694/11.455` m. Gap EPE is almost flat (`11.436` at gap 1 and `11.216` at gap 20), so the first obvious limitation is undertrained cross-clip Z4D/Wan adaptation rather than a demonstrated temporal-gap or query-direction failure. The coarse diagnostic was implemented but disabled, so this run does not directly separate global-memory quality from final upsampling detail.
- Stop at v1 as requested. Do not add local RGB, camera queries, visibility heads, delta-time/temporal RoPE, or a larger decoder. The single highest-value next experiment is to hold the architecture fixed and increase the full-DiT training budget/data exposure; this tests the diagnosed Z4D adaptation bottleneck before changing the query or upsampler.

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
- `.pi/settings.json` now installs `npm:pi-research-loop@0.2.0` after the local extension update. On the migrated server, `pi list` recognizes the project and user installations, and `prl doctor` passes. The old npm-publication blocker is therefore no longer the next research task.
- The migrated Python environment has NumPy, PyYAML and CUDA PyTorch, and `python -m compileall` passes. PRL runner smoke Run `R-20260804162323-4cf2cc` imported NumPy `1.26.4` and CUDA PyTorch `2.10.0+cu126` and exited successfully. `pytest` and TensorFlow are not installed, so the full test suite and native TFRecord dataset audit still require dependency installation. Pi reports version `0.82.0`, while the package declares peer compatibility starting at `0.82.1`; upgrading Pi is recommended before long runs.

## 2026-08-04 — post-migration research priority

- Next priority is a matched-budget Full-vs-Compact comparison on this server. Keep seed, train/validation clips, optimizer, query counts, coordinate statistics, and block sizes identical; report reconstruction, first-frame tracking, arbitrary-source tracking, late-appearing objects, occluded points, and target-camera reprojection error. The existing 20-step result is only a plumbing baseline and should not decide H001.

## 2026-08-04 — tracking deployment without MLflow

- This project has no MLflow server, so MLflow tracking is disabled rather than pointing at a nonexistent endpoint. W&B remains enabled with entity `zhaigong2023-sjtu-hpc-center`, project `worldbridge4d`, group `worldbridge4d-full-vs-compact`, and the existing research tags.
- `docs/tracking-env.example` contains only placeholders; real `.env` files, API keys, and credentials remain ignored and must be configured separately on each machine. A private GitHub repository is not a safe secret store because credentials persist in history and may be exposed through clones, logs, backups, or collaborators.
- W&B credentials can be provisioned with `wandb login` or environment variables. PRL injects W&B run metadata, while actual scalar logging still requires the training process to call `wandb.init()`.

## 2026-08-04 — reliability follow-up plan

- After the full-scale one-epoch comparison, run Compact and Full on MOVi-F 128x128 for seeds `2026`, `2027`, and `2028`, with three complete train epochs (`17211` global clip updates) and identical optimizer/query/model settings. Compact uses GPU 6; Full uses two-card DDP on GPUs 0 and 1 when available.
- Reuse the train-only coordinate statistics from the completed all-train Compact pass via `artifacts/reliability/coordinate_stats_train.npz`; this cache contains only mean/scale metadata and is not a geometry or model-weight cache. The cache is valid for the fixed 21-frame, clip-start-0, depth-tolerance configuration.
- Evaluate every seed on all 147 validation clips at pixel stride 16, then aggregate endpoint error, per-coordinate MAE, target-camera reprojection, visible/occluded groups, and Full-only arbitrary-source/late-appearing groups with mean and sample standard deviation using `scripts/aggregate_evaluations.py`.

## 2026-08-04 — PRL extension update

- Project `.pi/settings.json` now pins `npm:pi-research-loop@0.2.0`; reload the local Pi window if the current session still shows the old package version.

## 2026-08-05 — three-seed reliability matrix completed

- The unattended scheduler completed all six matched-budget training runs, six validations, and both three-seed aggregations without retries or errors. Completion marker: `RELIABILITY_MATRIX_OK` from `R-20260804195945-61e4b5`; Full aggregation run: `R-20260805072305-ca927b`.
- Configuration was held fixed across seeds: MOVi-F 128x128, 21-frame clip from the 24-frame native sequence, 17,211 global clip updates (three epochs), train-only coordinate-stat cache, validation on 147 clips, pixel stride 16, query chunk 8192, Compact on GPU 6, and Full with two-card DDP on GPUs 0/1. W&B project was `zhaigong2023-sjtu-hpc-center/worldbridge4d`.
- Final endpoint-error means and sample standard deviations are in `artifacts/reliability/compact_mean_std.json` and `artifacts/reliability/full_mean_std.json` (generated outputs are ignored and not committed). Compact reconstruction / first-frame EPE were `1.2043 ± 0.0273` / `1.0114 ± 0.0405`; Full reconstruction / first-frame EPE were `1.1603 ± 0.0313` / `1.0771 ± 0.0680`.
- Full's arbitrary all-source/all-target EPE was `1.2348 ± 0.0335` (`1.0576 ± 0.0274` visible; `1.9912 ± 0.0604` occluded). Late-appearing EPE was `3.1070 ± 0.1168` (`2.5626 ± 0.1084` visible; `3.8113 ± 0.1279` occluded). These are capability measurements, not a direct Compact comparison because Compact does not expose arbitrary-source queries.
- Decision: retain the Full query capability, but do not claim Full dominates Compact. Full trades a 3.7% reconstruction improvement for a 6.5% first-frame EPE regression overall (8.2% on occluded points). Any follow-up should use an explicit arbitrary-source Compact ablation and investigate late-appearing/occluded errors.

## 2026-08-08 — latent-alignment screen after source-centric arms

- Complete B0/E3/E5/E6 before launching D1/D2/D3; the new arms are independent follow-up screens and must not be used to rewrite the original architecture-ablation protocol.
- Operationalize the latent-distribution hypothesis through downstream XYZ generalization rather than an ungrounded raw latent distance: D1 uses fixed pre-update per-channel whitening, D2 an identity-initialized learned channel affine, and D3 an identity-initialized residual nonlinear 1x1x1 adapter.
- Hold the source-centric schedule, 1,000 updates, seed 2029, full Wan trainability, XYZ weighting, and complete 32-clip/441-pair validation fixed. Use no direct Z4D target or latent-regression loss.
- Explicitly run W&B online and log `global_step`; save D1's deterministic 64-clip feature statistics in the resolved config and checkpoint. Treat all results as screening evidence only.
