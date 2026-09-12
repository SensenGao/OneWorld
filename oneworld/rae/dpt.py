"""Independent full-resolution RGB and Gaussian DPT components.

This follows 3DGen-Pi's decoder topology: RGB reconstruction and Gaussian attributes are
two real DPT heads, not shallow output convolutions sharing one DPT trunk.  Gaussian
centres do not come from this module: Pi3 supplies full-resolution depth, while source
pixels and K deterministically supply the camera ray for each Gaussian.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The decoder uses VGGT's DPT implementation.
from vggt.heads.dpt_head import _make_fusion_block, _make_scratch, custom_interpolate
from vggt.heads.utils import create_uv_grid, position_grid_to_embed

class GatedDualTapDPTTrunk(nn.Module):
    """Each DPT scale learns a softmax gate over z18 and deterministic z36.

    ``level_drop_prob`` is the total probability of retaining only one level.  Half of
    that probability drops z18 and half drops z36.  A mask is sampled per scene and shared
    by every view, so dropout cannot introduce artificial cross-view inconsistency.
    """

    def __init__(self, dim_in=1024, features=128, out_channels=(96, 192, 384, 768),
                 patch_size=14, down_ratio=1, pos_embed=True, level_drop_prob=0.2):
        super().__init__()
        if not 0.0 <= level_drop_prob < 1.0:
            raise ValueError("level_drop_prob must be in [0,1)")
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.pos_embed = pos_embed
        self.level_drop_prob = float(level_drop_prob)
        self.out_dim = features // 2

        self.norms = nn.ModuleList((nn.LayerNorm(dim_in), nn.LayerNorm(dim_in)))
        self.projects = nn.ModuleList([
            nn.ModuleList((nn.Conv2d(dim_in, oc, 1), nn.Conv2d(dim_in, oc, 1)))
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
        ])

        gate_hidden = max(128, dim_in // 4)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(2 * dim_in), nn.Linear(2 * dim_in, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, 8),
        )
        # High-resolution slots initially favour z18; low-resolution slots favour z36.
        self.base_gate_logits = nn.Parameter(torch.tensor(
            [[1.0, -1.0], [0.5, -0.5], [-0.5, 0.5], [-1.0, 1.0]]))
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)

    def _add_pos(self, x, width, height, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(
            pw, ph, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        return x + pe.permute(2, 0, 1)[None]

    def _level_mask(self, batch: int, device) -> torch.Tensor:
        keep = torch.ones((batch, 2), device=device, dtype=torch.bool)
        if not self.training or self.level_drop_prob == 0:
            return keep
        draw = torch.rand(batch, device=device)
        drop_z18 = draw < self.level_drop_prob / 2
        drop_z36 = (draw >= self.level_drop_prob / 2) & (draw < self.level_drop_prob)
        keep[drop_z18, 0] = False
        keep[drop_z36, 1] = False
        return keep

    def forward(self, z18: torch.Tensor, z36: torch.Tensor, height: int, width: int):
        if z18.shape != z36.shape or z18.ndim != 5:
            raise ValueError(
                f"z18/z36 must match (B,V,C,ph,pw), got {z18.shape}, {z36.shape}")
        B, V, C, ph, pw = z18.shape
        normalized = []
        for norm, tap in zip(self.norms, (z18, z36)):
            x = tap.reshape(B * V, C, ph, pw)
            normalized.append(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        pooled = torch.cat([x.mean(dim=(2, 3)) for x in normalized], dim=1)
        logits = self.gate_mlp(pooled).reshape(B * V, 4, 2)
        logits = logits + self.base_gate_logits[None].to(logits.dtype)
        keep_scene = self._level_mask(B, z18.device)
        keep = keep_scene.repeat_interleave(V, dim=0)
        logits = logits.masked_fill(~keep[:, None], torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1)

        feats = []
        for slot in range(4):
            candidates = [self.projects[slot][tap](normalized[tap]) for tap in range(2)]
            x = (weights[:, slot, 0, None, None, None] * candidates[0]
                 + weights[:, slot, 1, None, None, None] * candidates[1])
            if self.pos_embed:
                x = self._add_pos(x, width, height)
            feats.append(self.resize_layers[slot](x))

        l1, l2, l3, l4 = feats
        l1, l2 = self.scratch.layer1_rn(l1), self.scratch.layer2_rn(l2)
        l3, l4 = self.scratch.layer3_rn(l3), self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1)
        out = self.scratch.output_conv1(out)
        out = custom_interpolate(
            out, (ph * self.patch_size // self.down_ratio,
                  pw * self.patch_size // self.down_ratio),
            mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._add_pos(out, width, height)
        gate_info = {
            "weights": weights.reshape(B, V, 4, 2),
            "level_keep": keep_scene,
        }
        return out, gate_info


def _head(in_channels, out_channels, mid=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(mid, out_channels, 1))


class RGBPatchResidualHead(nn.Module):
    """Decode each RGB32 token into a distinct full 14x14 RGB patch."""

    def __init__(self, appearance_dim, patch_size, hidden_dim=256):
        super().__init__()
        self.patch_size = int(patch_size)
        self.net = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.patch_size * self.patch_size),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, appearance):
        B, V, _C, ph, pw = appearance.shape
        value = self.net(appearance.permute(0, 1, 3, 4, 2)).reshape(
            B, V, ph, pw, 3, self.patch_size, self.patch_size)
        return value.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            B, V, 3, ph * self.patch_size, pw * self.patch_size)

