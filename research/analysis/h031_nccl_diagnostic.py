"""Synthetic GPU2/3 FSDP FFN diagnostic; not a training continuation.

Matches the failed 1536->6144 FFN all-gather shape, FP32 masters/BF16 compute,
world2/NCCL900, but deliberately excludes checkpoint/data/whole-model effects.
No checkpoint is read or written. Success cannot establish real-train stability.
"""
import json
import os
from pathlib import Path
import resource
import time
from datetime import timedelta

START = time.perf_counter()
RANK = int(os.environ['LOCAL_RANK'])


def event(stage, **extra):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(event='H031_NCCL_DIAGNOSTIC_PHASE', stage=stage,
                          rank=RANK, pid=os.getpid(), seconds=time.perf_counter()-START,
                          major_faults=usage.ru_majflt, maxrss_KiB=usage.ru_maxrss,
                          **extra)), flush=True)


event('import_start')
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from worldbridge.trainer.trainer import fsdp_auto_wrap_policy

event('torch_imported', version=torch.__version__)
# Reproduce demand-reader's CPU-only TF initialization before model construction.
from worldbridge.data.movif import MOViFDataset
tf = MOViFDataset._tf()
tf.config.set_visible_devices([], 'GPU')
tf.constant(0)
event('tensorflow_CPU_initialized', version=tf.__version__)
torch.manual_seed(20260812)
torch.cuda.set_device(RANK)
dist.init_process_group('nccl', timeout=timedelta(seconds=900))
assert dist.get_world_size() == 2
model = torch.nn.Sequential(torch.nn.Linear(1536, 6144), torch.nn.GELU(),
                            torch.nn.Linear(6144, 1536)).cuda()
wrapped = FSDP(model, device_id=RANK, use_orig_params=True,
               sync_module_states=True, limit_all_gathers=True, forward_prefetch=False,
               auto_wrap_policy=lambda module, recurse, nonwrapped_numel:
                   fsdp_auto_wrap_policy(module, recurse, nonwrapped_numel,
                                         min_num_params=5_000_000),
               mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                   reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16))
optimizer = torch.optim.AdamW(wrapped.parameters(), lr=3e-6)
x = torch.randn(1, 5, 1024, 1536, device='cuda', dtype=torch.bfloat16)
event('model_ready')
for iteration in range(5):
    optimizer.zero_grad(set_to_none=True)
    event('forward_start', iteration=iteration)
    loss = wrapped(x).float().square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    value = loss.detach().clone()
    dist.all_reduce(value)
    assert torch.isfinite(value).item()
    event('synthetic_update_done', iteration=iteration, loss=loss.item(),
          peak_GiB=torch.cuda.max_memory_allocated()/2**30)
dist.barrier()
dist.destroy_process_group()
event('H031_NCCL_DIAGNOSTIC_OK')
