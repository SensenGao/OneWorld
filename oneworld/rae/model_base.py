"""Base implementation for the normalized geometry and appearance RAE latent.

The DiT latent contains patch tokens only.  Five shared learned register parameters
are inserted at the block-18 boundary and never become generative targets.  RGB32
conditions blocks 18--35 and is also exposed directly to two independent DPT heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oneworld.rae.appearance import AppearanceEncoder
from oneworld.rae.encoder import RAEEncoder
from oneworld.rae.gaussian_adapter import build_gaussians_from_depth
from oneworld.rae.layers import RGBToPi3Residual, channel_layer_norm


class BaseRAE(nn.Module):
    """Shared encoder and decoder continuation used by the released RAE."""

    def __init__(
        self,
        pi3x,
        appearance_dim: int = 32,
        appearance_hidden_dim: int = 384,
        appearance_depth: int = 6,
        appearance_heads: int = 8,
        appearance_transformer_dropout: float = 0.0,
        rgb_skip_drop: float = 0.1,
        trunk_condition_hidden_dim: int = 256,
        features: int = 256,
        sh_degree: int = 4,
        grad_checkpoint: bool = False,
        level_drop_prob: float = 0.2,
        allow_variable_appearance_dim: bool = False,
        gaussian_ray_mode: str = "z_depth",
    ):
        super().__init__()
        if appearance_dim != 32 and not allow_variable_appearance_dim:
            raise ValueError("the released appearance latent has 32 channels")
        if not 0.0 <= rgb_skip_drop < 1.0:
            raise ValueError("rgb_skip_drop must be in [0,1)")
        self.encoder = RAEEncoder(
            pi3x, grad_checkpoint=grad_checkpoint,
            rgb_injection=False, max_views=8)
        # Branch masking is disabled; only the direct decoder shortcut may be dropped.
        self.appearance_encoder = AppearanceEncoder(
            patch_size=pi3x.patch_size,
            output_dim=appearance_dim,
            hidden_dim=appearance_hidden_dim,
            depth=appearance_depth,
            heads=appearance_heads,
            branch_drop_prob=0.0,
            transformer_dropout=appearance_transformer_dropout,
            grad_checkpoint=grad_checkpoint,
        )
        self.geometry_dim = int(pi3x.dec_embed_dim)
        self.appearance_dim = int(appearance_dim)
        self.latent_dim = self.geometry_dim + self.appearance_dim
        self.rgb_skip_drop = float(rgb_skip_drop)
        self.sh_degree = int(sh_degree)
        if gaussian_ray_mode not in ("z_depth", "unit_ray"):
            raise ValueError(
                f"unsupported Gaussian ray mode {gaussian_ray_mode!r}")
        self.gaussian_ray_mode = gaussian_ray_mode

        # These normalized patch tensors are the actual Stage-2 targets.
        self.geometry_norm = nn.LayerNorm(self.geometry_dim, eps=1e-6)
        self.appearance_norm = nn.LayerNorm(self.appearance_dim, eps=1e-6)
        self.rgb_to_trunk = RGBToPi3Residual(
            appearance_dim=self.appearance_dim,
            hidden_dim=trunk_condition_hidden_dim,
            pi3_dim=self.geometry_dim)
        self.fusion_norm = nn.LayerNorm(self.geometry_dim, eps=1e-6)
        self.z36_norm = nn.LayerNorm(self.geometry_dim, eps=1e-6)

        register_count = int(pi3x.patch_start_idx)
        initial_registers = pi3x.register_token.detach().reshape(
            1, register_count, self.geometry_dim).clone()
        self.boundary_registers = nn.Parameter(initial_registers)

        # Diagonal-Gaussian posterior for the 32-channel RGB latent.
        self.posterior_mean = nn.Conv2d(
            self.appearance_dim, self.appearance_dim, kernel_size=1)
        self.posterior_logvar = nn.Conv2d(
            self.appearance_dim, self.appearance_dim, kernel_size=1)

    @property
    def rgb_trunk_gate(self):
        # The decoder uses unit-scale appearance conditioning.
        return self.boundary_registers.new_ones(())

    def _register_tokens(self, batch: int, views: int, dtype) -> torch.Tensor:
        return self.boundary_registers.to(dtype).expand(
            batch * views, -1, -1)

    def _skip_appearance(self, appearance: torch.Tensor):
        batch = appearance.shape[0]
        dropped = torch.zeros(
            batch, device=appearance.device, dtype=torch.bool)
        if self.training and self.rgb_skip_drop > 0:
            dropped = torch.rand(batch, device=appearance.device) < self.rgb_skip_drop
            keep = (~dropped).to(appearance.dtype).reshape(batch, 1, 1, 1, 1)
            appearance = appearance * keep / (1.0 - self.rgb_skip_drop)
        return appearance, dropped

    def _decode_continuation(
        self, z_sem, z_rgb, c2w_norm, height, width, pose_mask,
    ):
        rgb_delta = self.rgb_to_trunk(z_rgb)
        h18 = channel_layer_norm(z_sem + rgb_delta, self.fusion_norm)
        batch, views = z_sem.shape[:2]
        z35_raw, z36_raw, point_hidden, final_pos = self.encoder.propagate_z18(
            h18, c2w_norm, height, width,
            pose_mask=pose_mask, return_hidden=True, return_taps=True,
            register_tokens=self._register_tokens(batch, views, h18.dtype))
        z36 = channel_layer_norm(z36_raw, self.z36_norm)
        return {
            "decoder_input": h18,
            "rgb_trunk_delta": rgb_delta,
            "z35_raw": z35_raw,
            "z36_raw": z36_raw,
            "z36": z36,
            "pi3_point_hidden": point_hidden,
            "pi3_final_pos": final_pos,
        }

    def _gaussians_and_renders(
        self, out, K, c2w_norm, height, width, renders, random_background,
    ):
        out["gs_ok"] = False
        if "gs_raw" not in out or "depth" not in out or renders is False:
            return out
        gaussians = build_gaussians_from_depth(
            out["gs_raw"], out["depth"], K, c2w_norm,
            (height, width), sh_degree=self.sh_degree,
            ray_mode=self.gaussian_ray_mode)
        if gaussians is None:
            return out
        out["gaussians"] = gaussians
        out["gs_ok"] = True
        from oneworld.geometry.render import render_diagonal_views, render_views
        results = []
        for spec in renders or [{}]:
            render_gaussians = gaussians
            if (spec.get("source_rgb_dc") is not None
                    or spec.get("opacity_override") is not None
                    or float(spec.get("analytic_footprint", 0.0)) > 0):
                render_gaussians = dict(gaussians)
            source_rgb_dc = spec.get("source_rgb_dc")
            if source_rgb_dc is not None:
                if source_rgb_dc.shape != (
                        out["depth"].shape[0], out["depth"].shape[1],
                        3, height, width):
                    raise ValueError("source RGB DC override has the wrong shape")
                dc = source_rgb_dc.float().permute(
                    0, 1, 3, 4, 2).reshape(source_rgb_dc.shape[0], -1, 3)
                sh = torch.zeros_like(render_gaussians["sh"])
                sh[..., 0] = (dc - 0.5) / 0.28209479177387814
                render_gaussians["sh"] = sh
            opacity_override = spec.get("opacity_override")
            if opacity_override is not None:
                if not 0.0 <= float(opacity_override) <= 1.0:
                    raise ValueError("opacity override must be in [0,1]")
                render_gaussians["opacities"] = torch.full_like(
                    render_gaussians["opacities"], float(opacity_override))
            footprint = float(spec.get("analytic_footprint", 0.0))
            if footprint > 0:
                from vggt.utils.rotation import mat_to_quat
                depth = out["depth"].float().clamp_min(1e-4)
                sx = footprint * depth / K[..., 0, 0, None, None]
                sy = footprint * depth / K[..., 1, 1, None, None]
                sz = 0.1 * torch.minimum(sx, sy)
                render_gaussians["scales"] = torch.stack(
                    (sx, sy, sz), dim=-1).reshape(depth.shape[0], -1, 3)
                camera_xyzw = mat_to_quat(c2w_norm[..., :3, :3].float())
                camera_wxyz = camera_xyzw[..., [3, 0, 1, 2]]
                quats = camera_wxyz[:, :, None, None].expand(
                    -1, -1, height, width, -1)
                render_gaussians["quats"] = quats.reshape(
                    depth.shape[0], -1, 4)
            if spec.get("diagonal", False):
                render_fn = render_diagonal_views
            else:
                render_fn = render_views
            render_kwargs = {}
            if render_fn is render_views:
                render_kwargs["view_mask"] = spec.get("view_mask")
            rgb, depth, alpha = render_fn(
                render_gaussians,
                spec.get("c2w", c2w_norm), spec.get("K", K),
                width, height, sh_degree=self.sh_degree,
                random_background=spec.get(
                    "random_background", random_background),
                **render_kwargs)
            results.append({"rgb": rgb, "depth": depth, "alpha": alpha})
        out["renders"] = results
        return out

    def _appearance_posterior(self, appearance: torch.Tensor):
        batch, views, channels, height, width = appearance.shape
        flat = appearance.reshape(batch * views, channels, height, width)
        mean = self.posterior_mean(flat).reshape(
            batch, views, self.appearance_dim, height, width)
        logvar = self.posterior_logvar(flat).clamp(-30.0, 20.0).reshape(
            batch, views, self.appearance_dim, height, width)
        variance = logvar.exp()
        if self.training:
            latent = mean + torch.randn_like(mean) * variance.sqrt()
        else:
            latent = mean
        return latent, mean, logvar, variance

    def forward(
        self,
        images_01,
        c2w_norm,
        K,
        want=("rgb", "depth", "gs"),
        renders=None,
        random_background=True,
        pose_mask=None,
        boundary_noise: float = 0.0,
        boundary_noise_prob: float = 1.0,
        boundary_noise_mask=None,
    ):
        del boundary_noise_prob, boundary_noise_mask
        if boundary_noise != 0:
            raise ValueError("RAE reconstruction training is clean-only")
        batch, _views, _channels, height, width = images_01.shape
        prefix = self.encoder.encode_z18(
            images_01, c2w_norm, K, pose_mask=pose_mask)
        rgb = self.appearance_encoder(images_01, prefix["z18"])
        z_sem = channel_layer_norm(prefix["z18"], self.geometry_norm)
        appearance_features = channel_layer_norm(
            rgb["rgb_residual_raw"], self.appearance_norm)
        z_rgb, posterior_mean, posterior_logvar, posterior_variance = (
            self._appearance_posterior(appearance_features))
        latent = torch.cat((z_sem, z_rgb), dim=2)
        continuation = self._decode_continuation(
            z_sem, z_rgb, c2w_norm, height, width, pose_mask)
        rgb_skip, skip_dropped = self._skip_appearance(z_rgb)
        head_kwargs = {}
        if hasattr(self.heads, "detail_merge"):
            # Use one scene-shared mask for both full-resolution input mergers.
            head_kwargs["gs_appearance"] = (
                rgb_skip if self.CONTRACT_VERSION >= 17 else z_rgb)
        out = self.heads(
            z_sem, continuation["z36"], rgb_skip,
            height, width, want=want, **head_kwargs)
        if "gs" in want or "depth" in want:
            out.update(self.encoder.decode_pi3_centres(
                continuation["pi3_point_hidden"],
                continuation["pi3_final_pos"],
                c2w_norm, height, width))

        z_sem_tokens = z_sem.permute(0, 1, 3, 4, 2).float()
        z_rgb_tokens = z_rgb.permute(0, 1, 3, 4, 2).float()
        out.update(
            z18=prefix["z18"],
            z18_raw=prefix["z18"],
            z18_norm=z_sem,
            z_sem=z_sem,
            z_rgb=z_rgb,
            z36=continuation["z36"],
            z36_raw=continuation["z36_raw"],
            decoder_input=continuation["decoder_input"],
            rgb_trunk_delta=continuation["rgb_trunk_delta"],
            semantic=prefix["semantic"],
            evolved_registers=prefix["evolved_registers"].detach(),
            rgb_residual=z_rgb,
            rgb_residual_raw=rgb["rgb_residual_raw"],
            latent=latent,
            latent_decode=latent,
            rgb_branch_dropped=skip_dropped,
            rgb_skip_dropped=skip_dropped,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            posterior_variance=posterior_variance,
            kl_loss=0.5 * (
                posterior_mean.float().square()
                + posterior_variance.float()
                - posterior_logvar.float()
                - 1.0
            ).mean(),
            rgb_trunk_gate=self.rgb_trunk_gate.detach(),
            latent_sample_std=latent.detach().float().std(
                dim=(1, 2, 3, 4), keepdim=True).to(latent.dtype),
            z18_raw_std=prefix["z18"].detach().float().std(
                dim=(1, 2, 3, 4), keepdim=True).to(latent.dtype),
            boundary_sigma=torch.zeros(
                (batch, 1, 1, 1, 1), device=latent.device,
                dtype=latent.dtype),
            appearance_mean_error=z_rgb_tokens.mean(dim=-1).abs().mean(),
            appearance_std_error=(
                z_rgb_tokens.std(dim=-1, correction=0) - 1).abs().mean(),
            geometry_mean_error=z_sem_tokens.mean(dim=-1).abs().mean(),
            geometry_std_error=(
                z_sem_tokens.std(dim=-1, correction=0) - 1).abs().mean(),
        )
        return self._gaussians_and_renders(
            out, K, c2w_norm, height, width, renders, random_background)

    def encode_latent(self, images_01, c2w_norm, K, pose_mask=None):
        prefix = self.encoder.encode_z18(
            images_01, c2w_norm, K, pose_mask=pose_mask)
        rgb = self.appearance_encoder(images_01, prefix["z18"])
        z_sem = channel_layer_norm(prefix["z18"], self.geometry_norm)
        appearance_features = channel_layer_norm(
            rgb["rgb_residual_raw"], self.appearance_norm)
        _sample, z_rgb, posterior_logvar, posterior_variance = (
            self._appearance_posterior(appearance_features))
        return {
            "latent": torch.cat((z_sem, z_rgb), dim=2),
            "z18": prefix["z18"],
            "z18_norm": z_sem,
            "z_sem": z_sem,
            "z_rgb": z_rgb,
            "rgb_residual": z_rgb,
            "rgb_residual_raw": rgb["rgb_residual_raw"],
            "posterior_mean": z_rgb,
            "posterior_logvar": posterior_logvar,
            "posterior_variance": posterior_variance,
        }

    def decode_generated_latent(
        self, latent, c2w_norm, K, height, width, normalized=True,
        want=("rgb", "depth", "gs"), renders=None,
        random_background=False, pose_mask=None,
    ):
        if not normalized:
            raise ValueError("DiT output must be converted to the normalized RAE latent")
        if latent.shape[2] != self.latent_dim:
            raise ValueError(
                f"generated latent has C={latent.shape[2]}, expected {self.latent_dim}")
        z_sem = latent[:, :, :self.geometry_dim]
        z_rgb = latent[:, :, self.geometry_dim:]
        continuation = self._decode_continuation(
            z_sem, z_rgb, c2w_norm, height, width, pose_mask)
        out = self.heads(
            z_sem, continuation["z36"], z_rgb,
            height, width, want=want)
        if "gs" in want or "depth" in want:
            out.update(self.encoder.decode_pi3_centres(
                continuation["pi3_point_hidden"],
                continuation["pi3_final_pos"],
                c2w_norm, height, width))
        out.update(
            latent=latent, z18_norm=z_sem, z_sem=z_sem, z_rgb=z_rgb,
            z36=continuation["z36"],
            decoder_input=continuation["decoder_input"],
            rgb_trunk_delta=continuation["rgb_trunk_delta"],
            rgb_trunk_gate=self.rgb_trunk_gate.detach())
        return self._gaussians_and_renders(
            out, K, c2w_norm, height, width, renders, random_background)
