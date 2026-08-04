# Research Agent Operating Rules

- Read research/PROJECT.md, research/STATE.yaml, and the relevant hypothesis before changing code.
- Work only in the task worktree created by PRL. Do not push, merge, or rewrite shared branches except for GitHub private repositories explicitly requested by the repository owner.
- Keep changes reproducible and record command, configuration, seed, and environment.
- Remove secrets, model weights, and generated outputs before checkpointing.
- Use prl_run_inspect for run details; declare event listeners instead of polling.
- After an event, decide whether to inspect, modify, retry, or stop.
