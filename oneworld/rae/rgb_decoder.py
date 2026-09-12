"""Full-resolution RGB detail decoders for the OneWorld RAE.

The DPT branch is deliberately geometry-only.  The RGB latent is not concatenated to
the low-resolution DPT taps; instead, a learned patch detailer expands each RGB
token directly to a full-resolution feature patch.  The resulting feature is merged
with the coarse DPT feature immediately before the final RGB projection, matching
the role of AnySplat's full-resolution ``input_merger`` without leaking source RGB
pixels into the generative decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oneworld.rae.dpt import RGBPatchResidualHead, _head


class RGBPatchDetailer(nn.Module):
    """Learned exact patch-size expansion from RGB latent tokens to HxW features."""

    def __init__(
        self,
        appearance_dim: int = 32,
        feature_dim: int = 32,
        patch_size: int = 14,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 0:
            raise ValueError("hidden_dim must be non-negative")
        self.pre = (
            nn.Sequential(
                nn.Conv2d(appearance_dim, self.hidden_dim, kernel_size=1),
                nn.GELU(),
            )
            if self.hidden_dim else nn.Identity())
        self.expand = nn.Conv2d(
            self.hidden_dim or appearance_dim,
            self.feature_dim * self.patch_size * self.patch_size,
            kernel_size=1,
        )
        # PixelShuffle is an exact learned 14x patch expansion.  A spatial 3x3
        # refinement lets neighbouring patches remove boundary seams.
        self.refine = nn.Conv2d(
            self.feature_dim, self.feature_dim, kernel_size=3, padding=1)
        nn.init.normal_(self.expand.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.expand.bias)
        nn.init.dirac_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, appearance: torch.Tensor, height: int, width: int):
        if appearance.ndim != 5:
            raise ValueError(
                f"appearance must be (B,V,C,ph,pw), got {appearance.shape}")
        batch, views, channels, patch_h, patch_w = appearance.shape
        expected = (patch_h * self.patch_size, patch_w * self.patch_size)
        if expected != (height, width):
            raise ValueError(
                f"RGB detail grid {patch_h}x{patch_w} with patch size "
                f"{self.patch_size} decodes to {expected}, not {(height, width)}")
        value = appearance.reshape(batch * views, channels, patch_h, patch_w)
        value = self.pre(value)
        value = F.pixel_shuffle(self.expand(value), self.patch_size)
        value = self.refine(value)
        return value


class RGBDetailMerger(nn.Module):
    """Fuse coarse DPT context and full-resolution RGB details before RGB output."""

    def __init__(
        self,
        dpt_channels: int,
        appearance_dim: int,
        patch_size: int,
        feature_dim: int = 32,
        detail_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.coarse = nn.Conv2d(
            dpt_channels, feature_dim, kernel_size=3, padding=1)
        self.detailer = RGBPatchDetailer(
            appearance_dim=appearance_dim,
            feature_dim=feature_dim,
            patch_size=patch_size,
            hidden_dim=detail_hidden_dim,
        )
        self.output = nn.Conv2d(feature_dim, 3, kernel_size=1)
        self.register_buffer(
            "_coarse_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer(
            "_detail_scale", torch.tensor(1.0), persistent=False)

    def forward(
        self,
        dpt_features: torch.Tensor,
        appearance: torch.Tensor,
        height: int,
        width: int,
    ):
        detail = self.detailer(appearance, height, width)
        coarse = self.coarse(dpt_features)
        if coarse.shape != detail.shape:
            raise RuntimeError(
                f"coarse/detail feature mismatch: {coarse.shape} versus {detail.shape}")
        merged = F.relu(
            self._coarse_scale * coarse + self._detail_scale * detail,
            inplace=False)
        return self.output(merged), detail


class RGBPixelResidualMerger(nn.Module):
    """Keep coarse RGB and exact-patch appearance residual independently usable."""

    def __init__(
        self,
        dpt_channels: int,
        appearance_dim: int,
        patch_size: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.coarse = _head(dpt_channels, 3)
        self.detailer = RGBPatchResidualHead(
            appearance_dim=appearance_dim,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
        )
        self.register_buffer(
            "_coarse_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer(
            "_detail_scale", torch.tensor(1.0), persistent=False)

    @property
    def output(self):
        # Expose the last RGB projection through the common decoder interface.
        return self.coarse[-1]

    def forward(
        self,
        dpt_features: torch.Tensor,
        appearance: torch.Tensor,
        height: int,
        width: int,
    ):
        batch, views = appearance.shape[:2]
        coarse = self.coarse(dpt_features)
        detail = self.detailer(appearance).reshape(
            batch * views, 3, height, width)
        return (
            self._coarse_scale * coarse + self._detail_scale * detail,
            detail,
        )

