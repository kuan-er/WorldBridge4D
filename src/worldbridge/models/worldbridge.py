"""Top-level WorldBridge4D model composition."""
from __future__ import annotations

import torch
from torch import nn

from .decoder.query_decoder import DenseQueryDecoder
from .outputs import DenseQueryOutput, StructuredZ4D

class DenseQueryWanModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: DenseQueryDecoder,
                 camera_head: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.camera_head = camera_head

    def forward(self, clean_video_latent: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
                source_rgb: torch.Tensor | None = None,
                z4d_override: torch.Tensor | StructuredZ4D | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | StructuredZ4D, DenseQueryOutput]:
        if z4d_override is not None:
            z4d = z4d_override
        elif encoder_hidden_states is None:
            z4d = self.backbone(clean_video_latent)
        else:
            z4d = self.backbone(clean_video_latent, encoder_hidden_states)
        output = self.decoder(z4d, source, target, source_rgb=source_rgb)
        if self.camera_head is not None and z4d_override is None:
            if source.ndim != 2 or not torch.all(source == source[:, :1]):
                raise ValueError('camera forward requires one shared source per batch item')
            output.camera = self.camera_head(z4d, source[:, 0])
        return output.normalized_xyz, z4d, output

    def configure_trainable(self, mode: str = "full") -> None:
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
        raise ValueError(f"unknown trainable_mode={mode!r}")
