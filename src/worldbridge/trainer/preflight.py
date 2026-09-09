"""Bound startup payload reads, not the training cohort or execution horizon.

Dataset construction retains its complete index/manifest/publication checks.
The training loop still calls each reader on the *actual* selected index before
forward (including eligibility replacements), so later misses/corruption fail
at use time. No cache generation, skip-on-error or validation bypass is added.
"""
from __future__ import annotations

import json
import time

from .lazy_vae import required_latent_indices

DEFAULT_PREFLIGHT_UPDATES = 5


def preflight_end_step(start: int, execution_end: int, updates: int = DEFAULT_PREFLIGHT_UPDATES) -> int:
    """Zero explicitly requests legacy full-invocation validation, never no checks."""
    if type(updates) is not int or updates < 0:
        raise ValueError('startup-preflight-updates must be a nonnegative integer (0 = full)')
    if start < 0 or execution_end <= start:
        raise ValueError('preflight requires a nonempty execution interval')
    return execution_end if updates == 0 else min(execution_end, start + updates)


def validate_startup_latents(datasets, seed, start, execution_end, rank, accumulation,
                             microbatch_per_gpu=1, *, updates=DEFAULT_PREFLIGHT_UPDATES,
                             dataset_mix_counts=None):
    end = preflight_end_step(start, execution_end, updates)
    report = dict(rank=rank, start=start, end=end, execution_end=execution_end,
                  requested_updates=updates, scope='full' if updates == 0 else 'prefix',
                  all_planned_payloads_checked=end == execution_end,
                  remaining_payload_policy='strict_read_before_forward')
    began = time.perf_counter()
    print(json.dumps(dict(event='startup_latent_payload_validation_begin', **report)), flush=True)
    required = required_latent_indices(datasets, seed, start, end, rank, accumulation,
                                       microbatch_per_gpu, dataset_mix_counts=dataset_mix_counts)
    counts = {name: len(indices) for name, indices in required.items()}
    checked = 0
    for name, indices in required.items():
        for index in indices:
            datasets[name].clean_latent(index)  # exceptions propagate; never substitute or generate
            checked += 1
            if checked % 16 == 0:
                print(json.dumps(dict(event='startup_latent_payload_validation_progress', rank=rank,
                    checked=checked, total=sum(counts.values()), dataset=name, index=index,
                    elapsed_seconds=time.perf_counter()-began)), flush=True)
    report.update(counts=counts, elapsed_seconds=time.perf_counter()-began)
    print(json.dumps(dict(event='startup_latent_payload_validation_complete', **report)), flush=True)
    return report
