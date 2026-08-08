# Full vs Compact MOVi-F Reliability Experiment

> This is the research-level experiment report. `prl experiments rebuild` generates a Run ledger and may overwrite this file; the authoritative scientific interpretation is also recorded in `research/STATE.yaml`, `research/DECISIONS.md`, and `research/hypotheses/H001.md`.

## 1. Question

Does the deterministic Full 4D latent, which retains source-pixel trajectories for every source time, improve source-conditioned 3D decoding over Compact, which uses a first-frame trajectory anchor plus per-frame reconstruction features?

The primary expected advantage of Full was arbitrary-source tracking, especially for late-appearing and occluded objects, while retaining comparable reconstruction and first-frame tracking.

## 2. Dataset and method

- Dataset: read-only `/dataset/MOVi-F`, native MOVi-F 128x128 split.
- Native sequence length: 24 frames; continuous clip: frames `0..20` (`clip_length=21`, `clip_start=0`).
- Training: all available train clips; validation: all 147 validation clips.
- Models: deterministic Compact and Full latent autoencoders, latent shape `[B,16,6,16,16]`.
- Coordinate statistics: computed from train pointmaps only and reused by both models.
- Loss: validity-masked Smooth-L1 coordinate loss with balanced query groups.
- Compact query groups: reconstruction (`s=t`) and first-frame tracking (`s=0`).
- Full query groups: reconstruction, first-frame tracking, and arbitrary legal `(s,t)` tracking.
- Evaluation: pixel stride 16; query chunk 8192; visible/occluded groups; Full-only arbitrary-source and late-appearing groups.
- Optimizer settings: learning rate `0.001`, weight decay `0.0001`, AMP enabled, batch size 1, gradient accumulation 1, gradient clipping 1.0, trajectory block size 16384.
- Training budget: `17211` global clip updates per run, corresponding to three full train epochs.
- Seeds: `2026`, `2027`, `2028`.
- Hardware: Compact on GPU 6; Full with two-card DDP on GPUs 0 and 1.
- Environment: PyTorch `2.10.0+cu126`, NumPy `1.26.4`, W&B enabled; MLflow disabled because no MLflow server was configured.
- Tracking project: `zhaigong2023-sjtu-hpc-center/worldbridge4d`, group `worldbridge4d-full-vs-compact`.

## 3. PRL execution record

The unattended scheduler was `R-20260804195945-61e4b5`. It completed with marker `RELIABILITY_MATRIX_OK` and no errors.

### Compact runs

| Seed | Train Run | Validation Run | Final train loss | Loss ratio |
|---:|---|---|---:|---:|
| 2026 | `R-20260804195953-bcd6da` | `R-20260804212045-a6034d` | 0.6819 | 0.0862 |
| 2027 | `R-20260804212150-886fbf` | `R-20260804224322-52f946` | 0.9303 | 0.1346 |
| 2028 | `R-20260804224548-707970` | `R-20260805000920-ccae29` | 0.5050 | 0.0643 |

Compact aggregation: `R-20260805001025-f69797`.

### Full runs

| Seed | Train Run | Validation Run | Final train loss | Loss ratio |
|---:|---|---|---:|---:|
| 2026 | `R-20260804201840-c2ca56` | `R-20260804231119-ab4916` | 0.5487 | 0.0720 |
| 2027 | `R-20260804231545-10d045` | `R-20260805025748-e653bb` | 0.5928 | 0.0812 |
| 2028 | `R-20260805030354-84a44b` | `R-20260805070939-dba32e` | 0.6631 | 0.0819 |

Full aggregation: `R-20260805072305-ca927b`.

All six formal training runs, six validations, and two aggregations exited successfully. Four old queued smoke runs from an earlier scheduling phase were intentionally terminated during cleanup; they were not part of the formal matrix.

## 4. Final validation results

Values are endpoint error (EPE), reported as mean ± sample standard deviation over the three seeds. Lower is better.

### Common query groups

| Metric | Compact | Full |
|---|---:|---:|
| Reconstruction | 1.2043 ± 0.0273 | **1.1603 ± 0.0313** |
| First-frame, all | **1.0114 ± 0.0405** | 1.0771 ± 0.0680 |
| First-frame, visible | **0.8520 ± 0.0361** | 0.8994 ± 0.0634 |
| First-frame, occluded | **1.5415 ± 0.0571** | 1.6682 ± 0.0840 |

### Full-only query groups

| Metric | Full EPE |
|---|---:|
| Arbitrary all `(s,t)` | 1.2348 ± 0.0335 |
| Arbitrary visible | 1.0576 ± 0.0274 |
| Arbitrary occluded | 1.9912 ± 0.0604 |
| Arbitrary random-source, all | 1.2351 ± 0.0364 |
| Late-appearing, all | 3.1070 ± 0.1168 |
| Late-appearing, visible | 2.5626 ± 0.1084 |
| Late-appearing, occluded | 3.8113 ± 0.1279 |

The close agreement between exhaustive arbitrary `(s,t)` and random-source results supports the stability of the arbitrary-source evaluation: `1.2348` versus `1.2351` overall.

## 5. Analysis

### Main comparison

Full improves common-task reconstruction by approximately 3.7% (`1.2043` to `1.1603`), but Compact is better on first-frame tracking:

- Full first-frame EPE is approximately 6.5% higher overall.
- Full visible first-frame EPE is approximately 5.6% higher.
- Full occluded first-frame EPE is approximately 8.2% higher.

Thus Full is not a uniformly stronger replacement for Compact.

### Interpretation

1. **Task competition:** Compact optimizes two groups, while Full optimizes three equally weighted groups. Arbitrary-source tracking therefore competes directly with reconstruction and first-frame tracking.
2. **Canonical-anchor bias:** Compact has an explicit source-frame-0 path, making first-frame queries particularly well conditioned. Full must preserve and retrieve information for every source time.
3. **Temporal compression:** Full encodes 21 source times into a latent with temporal size 6. Adaptive temporal pooling can mix source-time information, making source identity harder to recover through the decoder.
4. **Capacity and inductive-bias difference:** Compact has approximately 153k parameters and a dedicated 2D reconstruction encoder/fuser; Full has approximately 137k parameters and directly contexts per-source trajectory features. The trajectory/context blocks are matched, but total model capacity is not identical.
5. **Hard query semantics:** Occluded and late-appearing queries have limited direct visual/visibility evidence. Late-appearing occluded EPE (`3.8113`) remains the hardest group and should not be interpreted as a simple capacity failure without auditing source-valid/object-consistent query semantics.
6. **Optimization variance:** Full first-frame EPE has larger across-seed variation (`std=0.0680`) than Compact (`std=0.0405`), consistent with stronger multi-task optimization interference.

Full successfully provides arbitrary-source and late-appearing query capability, but Compact has no corresponding arbitrary-source output in this comparison. Therefore the experiment does not establish that Full is quantitatively better than Compact on arbitrary-source tracking; it establishes that Full enables that capability with a measurable reconstruction benefit and a first-frame accuracy trade-off.

## 6. Conclusion on H001

- **Engineering-feasibility component:** supported. The geometry pipeline, deterministic latents, DDP training, checkpointing, W&B monitoring, arbitrary-source evaluation, and three-seed reliability matrix all completed successfully.
- **Full-quality-dominance component:** not supported. Full is better for reconstruction but worse for first-frame tracking, including visible and occluded points.
- **Scientific claim:** Full should be presented as a capability/accuracy trade-off, not as a universally superior model.

## 7. Follow-up experiments

1. Reweight or temporarily remove Full's arbitrary-source loss to measure first-frame gradient interference.
2. Increase Full latent temporal resolution above 6 or add a source-time-preserving mechanism.
3. Match total parameter counts, not only trajectory/context hyperparameters.
4. Audit arbitrary-source queries to require source-valid and object-consistent correspondences.
5. Implement an explicit source-specific Compact ablation before making a direct arbitrary-source superiority claim.
6. Investigate late-appearing and occluded groups separately, including longer training and visibility-stratified sampling.

Checkpoints, model weights, W&B local directories, and generated evaluation artifacts were removed before the research checkpoint; no secrets, dataset files, or generated artifacts were committed.

## H004 — Dense-query feed-forward Wan feasibility (Task T-20260807120421-6da6ff)

The staged audit and v1 result are authoritative in `research/DECISIONS.md`, `research/STATE.yaml`, and `research/hypotheses/H004.md`. In brief: native clean Wan latent and final negative RF velocity output were both `[1,16,6,16,16]`; decoder memory/output were `[B,1536,16]` and `[B,K,3,128,128]`; 21 tests passed. Fixed-pair tiny EPE fell `15.449 -> 10.828` m, arbitrary-query tiny EPE fell `15.916 -> 8.022` m, and bounded two-clip exhaustive validation measured pointmap/tracking `11.457/11.419` m, visible/occluded-valid `11.980/8.670` m, source-zero/source-positive `10.694/11.455` m. These are feasibility/plumbing measurements under 24–32 updates, not full MOVi-F quality claims.
