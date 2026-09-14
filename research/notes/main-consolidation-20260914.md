# Owner-authorized local main consolidation — 2026-09-14

The repository owner requested preserving both H030/H031, merging to local main,
and deleting unused worktrees while leaving current training dependencies alone.
GitHub metadata confirmed this is the owner's private repository. No push occurred.

## Preservation and review

- H030 pre-operation HEAD: `9e3d7c4a4db189453f6b3d833f8f5dc0c393a8de`.
  Its two dirty research files were preserved in commit `6f259a9de9a7f7e5413f36074f51ebde35457839`
  without changing its working directory. PRL subsequently finished H030 and
  checkpointed the same file tree as `898efe00450c81b0de9d7025fe931a40947cc324`.
- H031 pre-operation HEAD: `5c2dc6d5b0aa19395c037bbfb9f844b17f1f2fd3`.
- Main's pre-existing AGENTS/STATE edits were committed, not discarded.
- Source history remains reachable through main and the original/archive branches.
  All tracked paths in both source trees were checked present in main.
- Exact H030/H031/main STATE originals are in `research/archive/main-consolidation-20260914/`.
  Canonical state keeps current H031 values and adds H030-only recovery information;
  differing historical current_task/active_hypotheses/next_step/updated_at remain archived.
- Merge resolution retains H030256 `cycle_b2_a2_k9`, all H031 native profiles, and
  the H031 runtime/preflight/readiness/worker-audit implementation. A regression
  test rejects legacy/native profile combinations. No model/data execution code
  was changed relative to pre-merge H031; only the configuration validator differs.
- CPU Run `R-20260914043257-a7e72d`, commit `4ccee6687dc32c99943d2461d517d0faffa4c361`:
  `519 passed in 41.99s`. Command: `CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/WorldBridge4D-runs/diagnostics-h030-150k-20260906/runtime:src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 /opt/conda/bin/python -m pytest -q tests`.
  Final src/configs/tests were verified identical to this tested commit.

## Cleanup scope and recovery

Backup root: `/data/WorldBridge4D-runs/pre-main-consolidation-20260914T032858Z`.
It contains verified Git bundles, whole-directory archives, original dirty patches,
per-file checksums, `cleanup-report.json`, and `post-cleanup-verification.json`.

Removed **9 registered worktrees**: finished H030 plus eight explicitly inspected,
terminal, current-session snapshots. Each HEAD is reachable from main. Every
regular/symlink file (including ignored local files) was archived and verified
before removal; a process cwd/executable/fd/argument reference scan found no use.
No `--force`, process signals, or branch/Run/log/checkpoint/cache deletion was used.
The allowlist and reproducible command are in
`research/maintenance/cleanup_unused_worktrees_20260914.py --apply` (one-off;
its existing-bundle guard deliberately prevents blind reruns).

332 worktrees remain at this boundary. This was a conservative allowlist cleanup,
not a claim that every historical snapshot was proved necessary or disposable.
Other-session/unverified snapshots and all recovery dependencies remain untouched.
To restore a removed checkout, use its recorded HEAD with `git worktree add
--detach <path> <HEAD>`, then restore the corresponding archive without replacing
Git's newly created `.git` file. No history pruning/GC was performed.

## Active training protection

H031 remains active, not task-finished merely to merge main. Training
`R-20260913031335-9eddf0` remained `running`, and its200k review
`R-20260913031349-3f2dae` remained `queued`; both termination audits remain empty.
Their immutable src/configs/analysis files were checked unchanged. Current config
SHA remains `6f8836acc0a70864e2961f5056b724af06ae3a791fc43f1939794a5974cb10c7`.
Keep the H031 development worktree, both execution snapshots, gate/recovery
snapshots, external runtime, handoff, checkpoint directories and caches. No stop,
restart, GPU change, training parameter change or automatic retry was requested.
