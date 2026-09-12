"""Small layers shared by the OneWorld RAE encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


def channel_layer_norm(value: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Apply LayerNorm to channels of a ``(B,V,C,H,W)`` tensor."""
    if value.ndim != 5 or value.shape[2] != norm.normalized_shape[0]:
        raise ValueError(
            f"feature shape {tuple(value.shape)} is incompatible with "
            f"LayerNorm{norm.normalized_shape}"
        )
    return norm(value.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)


class RGBToPi3Residual(nn.Module):
    """Project the appearance latent into the Pi3 decoder width."""

    def __init__(
        self,
        appearance_dim: int = 32,
        hidden_dim: int = 256,
        pi3_dim: int = 1024,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pi3_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        if appearance.ndim != 5:
            raise ValueError(
                f"expected appearance (B,V,C,H,W), got {appearance.shape}"
            )
        value = appearance.permute(0, 1, 3, 4, 2)
        return self.mlp(value).permute(0, 1, 4, 2, 3)
