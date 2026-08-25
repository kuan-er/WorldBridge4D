# Research Agent Operating Rules

- Read research/PROJECT.md, research/STATE.yaml, and the relevant hypothesis before changing code.
- Work only in the task worktree created by PRL. Do not push, merge, or rewrite shared branches except for GitHub private repositories explicitly requested by the repository owner.
- Keep changes reproducible and record command, configuration, seed, and environment.
- Remove secrets, model weights, and generated outputs before checkpointing.
- Use prl_run_inspect for run details; declare event listeners instead of polling.
- Physical GPUs 0 and 1 are the default GPU set.                                                                                            
- Other physical GPU IDs may be used only when the user explicitly authorizes the exact IDs in the current conversation. Never infer        
 authorization from GPU availability and never silently fall back to or substitute another GPU.                                                
- Every GPU PRL launch, enqueue, or fork must declare `resources.gpu_ids` with the exact selected physical GPU IDs. Two-rank training must  
 request exactly two explicitly selected GPUs. 
- After an event, decide whether to inspect, modify, retry, or stop.
