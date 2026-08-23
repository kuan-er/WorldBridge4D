"""Top-level WorldBridge4D model composition."""
from __future__ import annotations

import torch
from torch import nn

from .decoder.query_decoder import DenseQueryDecoder
from .outputs import DenseQueryOutput, StructuredZ4D

class DenseQueryWanModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: DenseQueryDecoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, clean_video_latent: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
                source_rgb: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | StructuredZ4D, DenseQueryOutput]:
        if encoder_hidden_states is None:
            z4d = self.backbone(clean_video_latent)
        else:
            z4d = self.backbone(clean_video_latent, encoder_hidden_states)
        output = self.decoder(z4d, source, target, source_rgb=source_rgb)
        return output.normalized_xyz, z4d, output

    def configure_trainable(self, mode: str = "full", last_blocks: int = 2) -> None:
        mode = str(mode)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        bypassed = list(getattr(self.backbone, "bypassed_parameters", []))
        for parameter in bypassed:
            parameter.requires_grad_(False)
        if mode == "full":
            return
        if mode == "decoder_only":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            return
        if mode == "source_rgb_only":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            encoder = self.decoder.upsampler.source_rgb_encoder
            fusions = self.decoder.upsampler.source_fusions
            if encoder is None or not fusions:
                raise ValueError("source_rgb_only requires an enabled source RGB pyramid")
            for parameter in encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in fusions.parameters():
                parameter.requires_grad_(True)
            return
        if mode == "source_rgb_plus_wan_decoder":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)
            for parameter in getattr(self.backbone, "adapter_parameters", []):
                parameter.requires_grad_(False)
            for parameter in bypassed:
                parameter.requires_grad_(False)
            for parameter in self.decoder.parameters():
                parameter.requires_grad_(True)
            if self.decoder.upsampler.source_rgb_encoder is None:
                raise ValueError(
                    "source_rgb_plus_wan_decoder requires an enabled source RGB pyramid"
                )
            return
        if mode == "geometry_adapter":
            adapter = list(getattr(self.backbone, "adapter_parameters", []))
            if not adapter:
                raise ValueError("geometry_adapter mode requires a structured geometry backbone")
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in adapter:
                parameter.requires_grad_(True)
            return
        if mode == "last_blocks":
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            dit = getattr(self.backbone, "dit", None)
            if dit is None or not hasattr(dit, "blocks"):
                raise ValueError("last_blocks mode requires a Wan-like backbone.dit.blocks")
            for parameter in getattr(self.backbone, "adapter_parameters", []):
                parameter.requires_grad_(True)
            for block in dit.blocks[-int(last_blocks):]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            if not bypassed:
                for name in ("norm_out", "proj_out", "scale_shift_table"):
                    module_or_parameter = getattr(dit, name, None)
                    if isinstance(module_or_parameter, nn.Parameter):
                        module_or_parameter.requires_grad_(True)
                    elif isinstance(module_or_parameter, nn.Module):
                        for parameter in module_or_parameter.parameters():
                            parameter.requires_grad_(True)
            return
        raise ValueError(f"unknown trainable_mode={mode!r}")
