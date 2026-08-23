"""Executable rectified-flow convention checks."""
from __future__ import annotations

import torch

def verify_flow_velocity_algebra(device: torch.device | str = "cpu") -> dict[str, float | str]:
    """Tie the Wan sign decision to the scheduler's executable RF equations."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = torch.device(device)
    clean = torch.tensor([[[[2.0, -1.0]]]], device=device)
    noise = torch.tensor([[[[-3.0, 4.0]]]], device=device)
    raw_velocity = noise - clean
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(sigmas=[1.0], device=device)
    timestep = scheduler.timesteps[0]
    recovered = scheduler.step(raw_velocity, timestep, noise, return_dict=False)[0]
    maximum_error = float((recovered - clean).abs().max())
    if maximum_error > 1e-6:
        raise AssertionError(f"Wan RF velocity algebra failed: {maximum_error}")
    return {
        "forward_path": "x_sigma=(1-sigma)*clean+sigma*noise",
        "raw_velocity": "noise-clean",
        "perception_readout": "negative_raw_velocity",
        "maximum_clean_recovery_error": maximum_error,
    }
