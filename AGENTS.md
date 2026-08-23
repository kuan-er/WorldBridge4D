# Research Agent Operating Rules

- Read research/PROJECT.md, research/STATE.yaml, and the relevant hypothesis before changing code.
- Work only in the task worktree created by PRL. Do not push, merge, or rewrite shared branches except for GitHub private repositories explicitly requested by the repository owner.
- Keep changes reproducible and record command, configuration, seed, and environment.
- Remove secrets, model weights, and generated outputs before checkpointing.
- Use prl_run_inspect for run details; declare event listeners instead of polling.
- Physical GPUs 0 and 1 are the complete project GPU allowlist; never use or fall back to another GPU.
- Every GPU PRL launch, enqueue, or fork must declare `resources.gpu_ids` as a subset of `["0", "1"]`; two-rank training must request both.
- After an event, decide whether to inspect, modify, retry, or stop.
