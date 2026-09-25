"""Independent dense source-target query decoder."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..outputs import DenseQueryOutput, StructuredZ4D, flatten_structured_z4d
from ..camera import (
    CameraOutput, CameraQueryHead, RayFieldHead, normalised_grid_coordinates,
)
from ..wan import WAN_LATENT_SHAPE
from .blocks import CrossAttentionBlock, _group_count
from .upsampler import DenseUpsampler2D

# Decoder-relative names of the H033 camera readout. The trainer prefixes them
# with ``decoder.`` for model-level migration prefixes.
CAMERA_MODULE_PREFIXES = ('camera_pose.', 'camera_rays.')

class DenseQueryDecoder(nn.Module):
    """Map global native Z4D and K dense `(s,t)` queries to K XYZ maps."""

    def __init__(self, num_frames: int = 21, latent_shape: tuple[int, int, int, int] = WAN_LATENT_SHAPE,
                 query_dim: int = 256, embedding_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 8, upsample_channels: Sequence[int] = (256, 128, 64, 32),
                 output_size: tuple[int, int] = (128, 128), query_grid_size: int | None = None,
                 structured_motion_slots: int = 0, structured_local_queries: bool = False,
                 source_rgb_pyramid: bool = False,
                 source_rgb_channels: Sequence[int] = (32, 64, 128),
                 source_rgb_fusion_32: bool = False,
                 pre_attention_rgb_query: bool = False, native_512: bool = False,
                 camera_supervision: dict | None = None):
        super().__init__()
        channels, latent_time, latent_height, latent_width = map(int, latent_shape)
        self.native_512 = bool(native_512)
        if self.native_512 and (latent_height != 32 or latent_width != 32
                                or query_grid_size not in (None, 32) or output_size != (256, 256)):
            raise ValueError('native512 requires the original32-grid/256 checkpoint architecture')
        self.num_frames = int(num_frames)
        self.latent_shape = (channels, latent_time, latent_height, latent_width)
        self.query_dim = int(query_dim)
        self.source_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.target_embedding = nn.Embedding(self.num_frames, embedding_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, query_dim), nn.SiLU(), nn.Linear(query_dim, query_dim)
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(query_dim, channels, num_heads) for _ in range(int(num_layers))
        ])
        self.structured_motion_slots = int(structured_motion_slots)
        self.structured_local_queries = bool(structured_local_queries)
        if self.structured_motion_slots < 0:
            raise ValueError("structured motion slot count cannot be negative")
        self.source_local_projection = nn.Conv2d(channels, query_dim, 1) \
            if self.structured_local_queries else None
        query_grid_size = int(query_grid_size or latent_height)
        if query_grid_size < latent_height:
            raise ValueError("query_grid_size cannot be smaller than the Wan latent grid")
        # The upsampler is built on the configured grid (32) and derives 512
        # outputs from a 64-grid feature. The runtime query grid is NOT fixed
        # here: a native512 run mixes 64x64 Kubric/DR clips with 32x32 PO clips,
        # so each forward derives it from the actual dense tensor.
        self.query_grid_shape = (query_grid_size, query_grid_size)
        axis = torch.linspace(0, latent_height - 1, query_grid_size)
        v, u = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("query_coordinates", torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1), persistent=False)
        self.upsampler = DenseUpsampler2D(
            query_dim, upsample_channels, self.query_grid_shape, output_size,
            source_rgb_pyramid=source_rgb_pyramid,
            source_rgb_channels=source_rgb_channels,
            source_rgb_fusion_32=source_rgb_fusion_32, native_512=self.native_512,
        )
        # Constructed after all baseline modules so enabling this ablation does
        # not shift the seeded initialization of any shared decoder parameter.
        self.pre_attention_rgb_query = bool(pre_attention_rgb_query)
        if self.pre_attention_rgb_query:
            if not source_rgb_pyramid:
                raise ValueError("pre-attention RGB query requires the source RGB pyramid")
            if self.query_grid_shape != (32, 32):
                raise ValueError("pre-attention RGB query requires a 32x32 source grid")
            rgb_channels = int(tuple(source_rgb_channels)[-1])
            self.query_rgb_projection = nn.Sequential(
                nn.GroupNorm(_group_count(rgb_channels), rgb_channels, affine=False),
                nn.SiLU(),
                nn.Conv2d(rgb_channels, self.query_dim, 1, bias=False),
            )
            # The migrated step-100k function is initially exact. Unlike the
            # upsampler fusions this is an unconditional additive query
            # residual: the projection itself learns when RGB should matter.
            nn.init.zeros_(self.query_rgb_projection[-1].weight)
        else:
            self.query_rgb_projection = None
        # H033 camera readout. Constructed last so enabling or disabling it does
        # not shift the seeded initialization of any shared decoder parameter.
        # A forked RNG keeps camera init independent of the ambient stream.
        self.camera_supervision = dict(camera_supervision) if camera_supervision else None
        if self.camera_supervision is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.camera_supervision['seed']))
                self.camera_pose = CameraQueryHead(
                    self.query_dim, hidden=int(self.camera_supervision.get('pose_hidden', 256)),
                )
                self.camera_rays = RayFieldHead(
                    self.query_dim, hidden=int(self.camera_supervision.get('ray_hidden', 256)),
                )
        else:
            self.camera_pose = None
            self.camera_rays = None

    @property
    def camera_enabled(self) -> bool:
        return self.camera_pose is not None

    def camera_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.camera_pose is None:
            return ()
        return tuple(self.camera_pose.parameters()) + tuple(self.camera_rays.parameters())

    def pair_indices(self, source: torch.Tensor, target: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise `(s,t)` inputs to [B,K] long tensors, once per forward."""
        source = torch.as_tensor(source, dtype=torch.long, device=self.source_embedding.weight.device)
        target = torch.as_tensor(target, dtype=torch.long, device=self.target_embedding.weight.device)
        if source.shape != target.shape:
            raise ValueError("source and target shapes differ")
        if source.ndim == 1:
            source, target = source[None], target[None]
        if source.ndim != 2:
            raise ValueError("source and target must be [B,K] or [K]")
        if (source < 0).any() or (source >= self.num_frames).any():
            raise ValueError("source index outside clip")
        if (target < 0).any() or (target >= self.num_frames).any():
            raise ValueError("target index outside clip")
        return source, target

    def query_content(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source, target = self.pair_indices(source, target)
        return self.query_mlp(torch.cat((self.source_embedding(source), self.target_embedding(target)), dim=-1))

    def _structured_source_query(self, z4d: StructuredZ4D, source: torch.Tensor, pairs: int,
                                 query_grid_shape: tuple[int, int]) -> torch.Tensor:
        """Project the shared source frame's local plane into every pair query."""
        if self.source_local_projection is None:
            raise RuntimeError("structured local projection was not constructed")
        batch, channels, frames, height, width = z4d.dense.shape
        source = torch.as_tensor(source, device=z4d.dense.device, dtype=torch.long)
        if source.ndim == 1:
            source = source[None]
        if source.shape[0] == 1:
            source = source.expand(batch, -1)
        if source.shape != (batch, pairs) or (source[:, :1] < 0).any() or (source[:, :1] >= frames).any():
            raise ValueError("structured source indices do not match dense Z4D")
        # Canonical source-all-target supervision repeats one source for all K
        # targets: project its local plane once and let autograd sum the
        # gradients through the expanded pair view instead of recomputing the
        # identical 1x1 convolution K times.
        if not bool(torch.all(source == source[:, :1])):
            raise ValueError("pair queries share one source frame per clip")
        row = torch.arange(batch, device=z4d.dense.device)
        local = self.source_local_projection(z4d.dense.permute(0, 2, 1, 3, 4)[row, source[:, 0]])
        if local.shape[-2:] != query_grid_shape:
            local = F.interpolate(local, size=query_grid_shape, mode="bilinear", align_corners=False)
        return local.flatten(2).transpose(1, 2)[:, None].expand(-1, pairs, -1, -1)

    def encode_source_rgb(self, source_rgb: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.upsampler.encode_source_rgb(source_rgb)

    def forward(self, z4d: StructuredZ4D, source: torch.Tensor,
                target: torch.Tensor, source_rgb: torch.Tensor | None = None,
                source_pyramid: dict[int, torch.Tensor] | None = None
                ) -> DenseQueryOutput:
        z4d.validate()
        dense = z4d.dense
        expected_shapes = {self.latent_shape}
        if self.native_512:
            expected_shapes.add((*self.latent_shape[:2], 64, 64))
        if dense.ndim != 5 or tuple(dense.shape[1:]) not in expected_shapes:
            raise ValueError(f'Z4D dense tensor must match {expected_shapes}, got {tuple(dense.shape)}')
        if z4d.motion.shape[2] != self.structured_motion_slots:
            raise ValueError(
                f"motion slots {z4d.motion.shape[2]} != decoder slots {self.structured_motion_slots}"
            )
        if source_rgb is not None and source_pyramid is not None:
            raise ValueError("pass source_rgb or source_pyramid, not both")
        if self.pre_attention_rgb_query:
            if source_pyramid is None:
                if source_rgb is None:
                    raise ValueError("pre-attention RGB query requires source appearance")
                source_pyramid = self.encode_source_rgb(source_rgb)
                source_rgb = None
            if 32 not in source_pyramid:
                raise ValueError("source RGB pyramid lacks the 32px feature")
        # A native512 run mixes 64x64 Kubric/DR clips with 32x32 PO clips, so the
        # grid follows this clip's dense tensor; every other run keeps the
        # configured query grid (which may be smaller than the dense grid and is
        # then interpolated by the source-local projection).
        query_grid_shape = (tuple(int(v) for v in dense.shape[-2:])
                            if self.native_512 else self.query_grid_shape)
        query_coordinates = self.query_coordinates
        if query_grid_shape != self.query_grid_shape:
            axis = torch.arange(query_grid_shape[0], device=dense.device,
                                dtype=self.query_coordinates.dtype)
            v, u = torch.meshgrid(axis, axis, indexing='ij')
            query_coordinates = torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)
        source, target = self.pair_indices(source, target)
        content = self.query_content(source, target)
        if content.shape[0] not in (1, dense.shape[0]):
            raise ValueError("query batch does not match Z4D batch")
        content = content.expand(dense.shape[0], -1, -1)
        num_query = query_coordinates.shape[0]
        query = content[:, :, None, :].expand(-1, -1, num_query, -1)
        if self.structured_local_queries:
            query = query + self._structured_source_query(z4d, source, content.shape[1], query_grid_shape)
        if self.query_rgb_projection is not None:
            rgb_query = self.query_rgb_projection(source_pyramid[32])
            if rgb_query.shape != (dense.shape[0], self.query_dim, *query_grid_shape):
                raise RuntimeError(
                    f"RGB query projection has unexpected shape {tuple(rgb_query.shape)}"
                )
            rgb_query = rgb_query.flatten(2).transpose(1, 2)[:, None]
            query = query + rgb_query.expand(-1, content.shape[1], -1, -1)
        if self.camera_enabled:
            query = torch.cat((query, content[:, :, None, :] + self.camera_pose.token), dim=2)
            # The camera token carries no pixel location: it uses the grid centre
            # for RoPE, exactly like the non-spatial motion memory tokens.
            query_coordinates = torch.cat(
                (query_coordinates, query_coordinates.mean(dim=0, keepdim=True)), dim=0)
        memory, memory_coordinates = flatten_structured_z4d(z4d)
        for block in self.blocks:
            query = block(query, memory, query_coordinates, memory_coordinates)
        camera = None
        if self.camera_enabled:
            pixel_tokens = query[:, :, :num_query]
            translation, rotation = self.camera_pose(query[:, :, num_query])
            camera = self._camera_readout(
                pixel_tokens, translation, rotation, source, target, query_grid_shape,
            )
        else:
            pixel_tokens = query
        batch, pairs, _, _ = pixel_tokens.shape
        query_height, query_width = query_grid_shape
        feature = pixel_tokens.reshape(
            batch * pairs, query_height, query_width, self.query_dim,
        ).permute(0, 3, 1, 2)
        xyz = self.upsampler(
            feature, source_rgb=source_rgb, source_pyramid=source_pyramid,
            batch=batch, pairs=pairs,
        )
        xyz = xyz.reshape(batch, pairs, 3, *xyz.shape[-2:])
        feature = feature.reshape(batch, pairs, self.query_dim, query_height, query_width)
        return DenseQueryOutput(xyz, feature, camera)

    def _camera_readout(self, pixel_tokens: torch.Tensor, translation: torch.Tensor,
                        rotation: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                        query_grid_shape: tuple[int, int]) -> CameraOutput:
        """Force diagonal identity and read the ray field off the diagonal pair."""
        batch, pairs = pixel_tokens.shape[:2]
        if source.shape != (batch, pairs) or target.shape != (batch, pairs):
            raise ValueError('camera readout requires the per-pair source/target of the decoder')
        diagonal = source == target
        if not bool(diagonal.any(dim=1).all()):
            raise ValueError('camera ray supervision requires one diagonal pair per batch item')
        eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        rotation = torch.where(diagonal[..., None, None], eye, rotation)
        translation = torch.where(diagonal[..., None], torch.zeros_like(translation), translation)
        row = torch.arange(batch, device=pixel_tokens.device)
        diagonal_index = diagonal.long().argmax(dim=1)
        frame = source[row, diagonal_index]
        coordinates = normalised_grid_coordinates(
            query_grid_shape[0], pixel_tokens.device, pixel_tokens.dtype,
        )
        rays = self.camera_rays(pixel_tokens[row, diagonal_index], coordinates)
        return CameraOutput(rotation, translation, rays, frame)
