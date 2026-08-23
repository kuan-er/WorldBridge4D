"""Compatibility imports for the refactored model package.

New code should import from :mod:`worldbridge.models` and
:mod:`worldbridge.trainer.objective`.
"""
from .models.backbones import CleanLatentBackbone, FeedForwardWanBackbone, WanHiddenGeometryBackbone
from .models.decoder.blocks import CrossAttentionBlock, DenseCrossAttention, ResidualBlock2D, RotaryEmbedding2D
from .models.decoder.query_decoder import DenseQueryDecoder
from .models.decoder.source_rgb import GatedSourceFusion, SourceRGBPyramid
from .models.decoder.upsampler import DenseUpsampler2D
from .models.flow import verify_flow_velocity_algebra
from .models.outputs import (
    DenseQueryOutput, StructuredZ4D, flatten_structured_z4d, flatten_z4d, unflatten_z4d,
)
from .models.worldbridge import DenseQueryWanModel
from .trainer.objective import masked_pair_smooth_l1

__all__ = [
    "CleanLatentBackbone", "CrossAttentionBlock", "DenseCrossAttention",
    "DenseQueryDecoder", "DenseQueryOutput", "DenseQueryWanModel",
    "DenseUpsampler2D", "FeedForwardWanBackbone", "GatedSourceFusion",
    "ResidualBlock2D", "RotaryEmbedding2D", "SourceRGBPyramid",
    "StructuredZ4D", "WanHiddenGeometryBackbone", "flatten_structured_z4d",
    "flatten_z4d", "masked_pair_smooth_l1", "unflatten_z4d",
    "verify_flow_velocity_algebra",
]
