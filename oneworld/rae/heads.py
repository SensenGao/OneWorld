"""RGB and Gaussian heads used by the released OneWorld RAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oneworld.rae.dpt import GatedDualTapDPTTrunk
from oneworld.rae.rgb_decoder import (
    RGBDetailMerger,
    RGBPatchDetailer,
    RGBPixelResidualMerger,
)


def gaussian_attribute_channels(sh_degree: int) -> int:
    if sh_degree < 0:
        raise ValueError("sh_degree must be non-negative")
    return 1 + 3 + 4 + 3 * (sh_degree + 1) ** 2


class GaussianInputMerger128(nn.Module):
    """Add an exact-patch RGB feature to the 128D fullres GS DPT feature."""

    def __init__(self, dpt_channels: int, appearance_dim: int, patch_size: int,
                 detail_hidden_dim: int = 0) -> None:
        super().__init__()
        self.feature_dim = 128
        self.coarse = (
            nn.Identity() if dpt_channels == self.feature_dim else
            nn.Conv2d(dpt_channels, self.feature_dim, kernel_size=1))
        self.detailer = RGBPatchDetailer(
            appearance_dim=appearance_dim, feature_dim=self.feature_dim,
            patch_size=patch_size, hidden_dim=detail_hidden_dim)

    def forward(self, dpt_features, appearance, height, width):
        coarse = self.coarse(dpt_features)
        detail = self.detailer(appearance, height, width)
        if coarse.shape != detail.shape:
            raise RuntimeError(
                f"coarse/detail feature mismatch: {coarse.shape} versus {detail.shape}")
        return F.relu(coarse + detail, inplace=False)


class GaussianAttributeHead128(nn.Sequential):
    """Exact AnySplat-style 128 -> 128 -> GS-attribute projection."""

    def __init__(self, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )


class RAEHeads(nn.Module):
    """Decode RGB and 3D Gaussian attributes from RAE features."""

    def __init__(
        self, geometry_dim=1024, appearance_dim=32, features=256,
        patch_size=14, level_drop_prob=0.0, sh_degree=4,
        detail_dim=32, detail_hidden_dim=0, detail_merge="feature",
    ) -> None:
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.appearance_dim = int(appearance_dim)
        self.sh_degree = int(sh_degree)
        self.detail_merge = detail_merge
        dpt_kwargs = dict(
            dim_in=self.geometry_dim, features=features, down_ratio=1,
            patch_size=patch_size, level_drop_prob=level_drop_prob)
        self.rgb_dpt = GatedDualTapDPTTrunk(**dpt_kwargs)
        self.gs_dpt = GatedDualTapDPTTrunk(**dpt_kwargs)
        if detail_merge == "feature":
            self.rgb_head = RGBDetailMerger(
                self.rgb_dpt.out_dim, self.appearance_dim, patch_size,
                feature_dim=detail_dim, detail_hidden_dim=detail_hidden_dim)
        elif detail_merge == "pixel_residual":
            self.rgb_head = RGBPixelResidualMerger(
                self.rgb_dpt.out_dim, self.appearance_dim, patch_size,
                hidden_dim=detail_hidden_dim or 256)
        else:
            raise ValueError(f"unsupported RGB detail merge {detail_merge!r}")
        self.gs_input_merger = GaussianInputMerger128(
            self.gs_dpt.out_dim, self.appearance_dim, patch_size,
            detail_hidden_dim=detail_hidden_dim)
        self.gs_channels = gaussian_attribute_channels(self.sh_degree)
        self.gs_head = GaussianAttributeHead128(self.gs_channels)
        self.rgb_patch_head = None
        self._init_gaussians()

    def _init_gaussians(self):
        last = self.gs_head[-1]
        nn.init.zeros_(last.weight)
        bias = torch.zeros(self.gs_channels)
        bias[0], bias[4] = -2.2, 1.0
        with torch.no_grad():
            last.bias.copy_(bias)

    def forward(self, z18, z36, appearance, height, width,
                want=("rgb", "depth", "gs"), gs_appearance=None):
        if z18.shape != z36.shape or z18.ndim != 5:
            raise ValueError("z18/z36 must match (B,V,C,H,W)")
        expected = (*z18.shape[:2], self.appearance_dim, *z18.shape[-2:])
        if appearance.shape != expected:
            raise ValueError(
                f"RGB latent has shape {tuple(appearance.shape)}, expected {expected}")
        gs_appearance = appearance if gs_appearance is None else gs_appearance
        if gs_appearance.shape != expected:
            raise ValueError("Gaussian appearance shape mismatch")
        batch, views = z18.shape[:2]
        out = {"gate": {}}
        if "rgb" in want:
            features, gate = self.rgb_dpt(z18, z36, height, width)
            rgb, detail = self.rgb_head(features, appearance, height, width)
            out["rgb"] = rgb.reshape(batch, views, 3, height, width)
            out["rgb_detail_features"] = detail.reshape(
                batch, views, detail.shape[1], height, width)
            out["gate"]["rgb"] = gate
        if "gs" in want:
            features, gate = self.gs_dpt(z18, z36, height, width)
            merged = self.gs_input_merger(
                features, gs_appearance, height, width)
            out["gs_raw"] = self.gs_head(merged).reshape(
                batch, views, self.gs_channels, height, width)
            out["gs_detail_features"] = merged.reshape(
                batch, views, 128, height, width)
            out["gate"]["gs"] = gate
        return out
