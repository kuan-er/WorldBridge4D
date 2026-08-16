from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from scripts.train_three_dataset_256_fsdp import wan_block_auto_wrap_policy
from worldbridge.wan import LoRALinear, WanDiTMapping, inject_wan_lora


class _ToyWan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.unmatched = nn.Linear(8, 8)
        self.ffn = nn.Module()
        self.ffn.net = nn.Sequential(nn.Module())
        self.ffn.net[0].proj = nn.Linear(8, 12)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_q(value) + self.ffn.net[0].proj(value)[..., :8]


def test_lora_injection_preserves_initial_output_and_freezes_base() -> None:
    torch.manual_seed(4)
    model = _ToyWan()
    value = torch.randn(2, 3, 8)
    expected = model(value).detach().clone()

    matched = inject_wan_lora(model, rank=2, alpha=2, targets=("to_q", "ffn.net.0.proj"))

    assert matched == ["to_q", "ffn.net.0.proj"]
    assert isinstance(model.to_q, LoRALinear)
    assert isinstance(model.ffn.net[0].proj, LoRALinear)
    torch.testing.assert_close(model(value), expected)
    assert not model.to_q.base.weight.requires_grad
    assert model.to_q.lora_A.weight.requires_grad
    assert model.to_q.lora_B.weight.requires_grad
    assert model.unmatched.weight.requires_grad

    model(value).sum().backward()
    assert model.to_q.base.weight.grad is None
    assert model.to_q.lora_B.weight.grad is not None
    assert torch.isfinite(model.to_q.lora_B.weight.grad).all()


def test_wan14b_architecture_is_discovered_from_native_config(tmp_path: Path) -> None:
    config = {
        "dim": 5120,
        "num_heads": 40,
        "num_layers": 40,
        "ffn_dim": 13824,
        "in_dim": 16,
        "out_dim": 16,
        "freq_dim": 256,
        "eps": 1e-6,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    checkpoint.write_text("{}")

    discovered = WanDiTMapping._architecture(checkpoint)

    assert discovered["num_layers"] == 40
    assert discovered["num_attention_heads"] == 40
    assert discovered["attention_head_dim"] == 128
    assert discovered["ffn_dim"] == 13824


def test_lora_fsdp_policy_wraps_blocks_but_not_bypassed_parent() -> None:
    WanTransformerBlock = type("WanTransformerBlock", (nn.Module,), {})
    block = WanTransformerBlock()
    parent = nn.Module()

    assert wan_block_auto_wrap_policy(parent, recurse=True, nonwrapped_numel=10)
    assert wan_block_auto_wrap_policy(block, recurse=False, nonwrapped_numel=10)
    assert not wan_block_auto_wrap_policy(parent, recurse=False, nonwrapped_numel=10)


def test_native_14b_checkpoint_keys_convert_to_diffusers_names() -> None:
    convert = WanDiTMapping._convert_native_key
    assert convert("blocks.0.self_attn.q.weight") == "blocks.0.attn1.to_q.weight"
    assert convert("blocks.29.cross_attn.o.bias") == "blocks.29.attn2.to_out.0.bias"
    assert convert("blocks.3.ffn.0.weight") == "blocks.3.ffn.net.0.proj.weight"
