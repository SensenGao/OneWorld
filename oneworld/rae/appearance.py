"""SVG-faithful low-dimensional appearance branch for the Pi3 boundary RAE.

The branch is deliberately parallel to the frozen Pi3 prefix.  It never writes into a
Pi3 residual stream.  Its output shares the Pi3 patch grid and is concatenated with the
stock block-17 feature only at the RAE/DiT boundary.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _global_scalar_moments(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce SVG's two-stage batch/token then channel moment reduction.

    SVG first computes one mean/std per channel over ``(batch, tokens)`` and then
    averages those channel statistics into one scalar.  In this multi-view adapter a
    view is an image sample, so ``B*V`` is SVG's batch dimension.  The released SVG
    code detaches all four scalar moments before applying the affine transform.
    """
    if value.ndim != 5:
        raise ValueError(f"expected (B,V,C,H,W), got {tuple(value.shape)}")
    dims = (0, 1, 3, 4)
    if not dist.is_available() or not dist.is_initialized():
        channel_mean = value.mean(dim=dims, keepdim=True)
        channel_std = value.std(dim=dims, keepdim=True)
        return channel_mean.mean().detach(), channel_std.mean().detach()

    # A training rank holds one multi-view scene.  Synchronize the detached moments
    # over the logical global image batch so a scene-wise masked branch is normalized
    # together with clean scenes, matching SVG's multi-sample batch behavior instead
    # of degenerating to zero within-channel variance on that rank.
    detached = value.detach()
    channel_sum = detached.sum(dim=dims, keepdim=True)
    channel_square_sum = detached.square().sum(dim=dims, keepdim=True)
    count = torch.tensor(
        detached.shape[0] * detached.shape[1] * detached.shape[3] * detached.shape[4],
        device=detached.device, dtype=detached.dtype)
    dist.all_reduce(channel_sum)
    dist.all_reduce(channel_square_sum)
    dist.all_reduce(count)
    channel_mean = channel_sum / count
    channel_variance = (
        (channel_square_sum - channel_sum.square() / count)
        / (count - 1).clamp_min(1)).clamp_min(0)
    channel_std = channel_variance.sqrt()
    return channel_mean.mean().detach(), channel_std.mean().detach()


def match_pi3_distribution(
    rgb: torch.Tensor,
    pi3: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match appearance-feature moments to the frozen Pi3 feature distribution."""
    if rgb.ndim != 5 or pi3.ndim != 5 or rgb.shape[:2] != pi3.shape[:2]:
        raise ValueError(
            f"expected matching (B,V,C,H,W), got rgb={rgb.shape}, pi3={pi3.shape}")
    if rgb.shape[-2:] != pi3.shape[-2:]:
        raise ValueError("RGB residual and Pi3 features must share one patch grid")
    rgb32 = rgb.float()
    pi332 = pi3.float()
    rgb_mean, rgb_std = _global_scalar_moments(rgb32)
    pi3_mean, pi3_std = _global_scalar_moments(pi332)
    aligned = (rgb32 - rgb_mean) / (rgb_std + eps) * pi3_std + pi3_mean
    aligned_mean, aligned_std = _global_scalar_moments(aligned)
    statistics = {
        "rgb_pre_mean": rgb_mean,
        "rgb_pre_std": rgb_std,
        "pi3_mean": pi3_mean,
        "pi3_std": pi3_std,
        "rgb_aligned_mean": aligned_mean,
        "rgb_aligned_std": aligned_std,
    }
    return aligned.to(rgb.dtype), statistics


class AppearanceEncoder(nn.Module):
    """A shallow per-view ViT producing 32 channels on Pi3's patch grid."""

    def __init__(
        self,
        patch_size: int = 14,
        output_dim: int = 32,
        hidden_dim: int = 384,
        depth: int = 6,
        heads: int = 8,
        branch_drop_prob: float = 0.1,
        transformer_dropout: float = 0.1,
        grad_checkpoint: bool = True,
        base_grid_size: int = 16,
    ):
        super().__init__()
        if output_dim <= 0 or hidden_dim % heads:
            raise ValueError("invalid SVG appearance dimensions")
        if not 0.0 <= branch_drop_prob < 1.0:
            raise ValueError("branch_drop_prob must be in [0,1)")
        if not 0.0 <= transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0,1)")
        self.patch_size = int(patch_size)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.branch_drop_prob = float(branch_drop_prob)
        self.transformer_dropout = float(transformer_dropout)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.base_grid_size = int(base_grid_size)
        self.patch_embed = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        # SVG uses torchvision's ViT-S: a class token and learned positional
        # embeddings.  Interpolation is the only required adaptation for our four
        # rectangular/multi-resolution Pi3 patch grids.
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(
            1, 1 + self.base_grid_size * self.base_grid_size, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(self.transformer_dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=4 * hidden_dim,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-6,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        fan_in = 3 * patch_size * patch_size
        nn.init.trunc_normal_(self.patch_embed.weight, std=fan_in ** -0.5)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _position_embedding(self, ph: int, pw: int, dtype) -> torch.Tensor:
        cls_position = self.pos_embedding[:, :1]
        patch_position = self.pos_embedding[:, 1:].reshape(
            1, self.base_grid_size, self.base_grid_size, self.hidden_dim)
        patch_position = patch_position.permute(0, 3, 1, 2)
        if (ph, pw) != (self.base_grid_size, self.base_grid_size):
            patch_position = F.interpolate(
                patch_position.float(), size=(ph, pw), mode="bicubic",
                align_corners=False).to(self.pos_embedding.dtype)
        patch_position = patch_position.flatten(2).transpose(1, 2)
        return torch.cat((cls_position, patch_position), dim=1).to(dtype)

    def _run_block(self, block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and tokens.requires_grad:
            return checkpoint(block, tokens, use_reentrant=False)
        return block(tokens)

    def forward(
        self,
        images_01: torch.Tensor,
        pi3_z18: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images_01.ndim != 5:
            raise ValueError(f"expected images (B,V,3,H,W), got {images_01.shape}")
        B, V, channels, height, width = images_01.shape
        if channels != 3 or height % self.patch_size or width % self.patch_size:
            raise ValueError("images are incompatible with the appearance patch size")
        ph, pw = height // self.patch_size, width // self.patch_size
        if pi3_z18.shape[:2] != (B, V) or pi3_z18.shape[-2:] != (ph, pw):
            raise ValueError("Pi3 z18 and RGB images disagree on view/grid shape")

        mean = images_01.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images_01.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        x = images_01.reshape(B * V, 3, height, width)
        x = (x - mean) / std
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B * V, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_dropout(x + self._position_embedding(ph, pw, x.dtype))
        for block in self.blocks:
            x = self._run_block(block, x)
        x = self.project(self.norm(x)[:, 1:])
        raw = x.transpose(1, 2).reshape(B, V, self.output_dim, ph, pw)

        dropped = torch.zeros((B,), device=raw.device, dtype=torch.bool)
        if self.training and self.branch_drop_prob > 0:
            dropped = torch.rand((B,), device=raw.device) < self.branch_drop_prob
            mask = self.mask_token.transpose(1, 2).reshape(
                1, 1, self.output_dim, 1, 1).to(raw.dtype)
            raw = torch.where(dropped[:, None, None, None, None], mask, raw)

        aligned, statistics = match_pi3_distribution(raw, pi3_z18)

        return {
            # Keep the pre-alignment value available for the v9 hybrid contract.
            # V8 continues to consume ``rgb_residual`` below, so this is strictly
            # backward compatible.  V9 normalizes geometry and appearance in two
            # independent spaces and therefore must not apply SVG's scalar affine
            # matching before its own LayerNorm.
            "rgb_residual_raw": raw,
            "rgb_residual": aligned,
            "rgb_residual_decode": aligned,
            "rgb_branch_dropped": dropped,
            **statistics,
        }
