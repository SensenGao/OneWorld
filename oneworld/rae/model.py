"""OneWorld representation autoencoder and latent encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from oneworld.rae.appearance import AppearanceEncoder
from oneworld.rae.encoder import RAEEncoder
from oneworld.rae.layers import channel_layer_norm
from oneworld.rae.model_decoder import ReconstructionRAE


class RAELatentEncoder(nn.Module):
    """Frozen encoder for the 1024-channel geometry and 32-channel appearance latent."""

    CONTRACT_VERSION = 19

    def __init__(
        self,
        pi3x,
        appearance_dim: int = 32,
        appearance_hidden_dim: int = 384,
        appearance_depth: int = 6,
        appearance_heads: int = 8,
        appearance_transformer_dropout: float = 0.0,
        max_views: int = 8,
    ):
        super().__init__()
        if appearance_dim != 32:
            raise ValueError("appearance_dim must be 32")
        self.encoder = RAEEncoder(
            pi3x, grad_checkpoint=False, rgb_injection=False,
            max_views=max_views)
        self.appearance_encoder = AppearanceEncoder(
            patch_size=pi3x.patch_size,
            output_dim=appearance_dim,
            hidden_dim=appearance_hidden_dim,
            depth=appearance_depth,
            heads=appearance_heads,
            branch_drop_prob=0.0,
            transformer_dropout=appearance_transformer_dropout,
            grad_checkpoint=False,
        )
        self.geometry_dim = int(pi3x.dec_embed_dim)
        self.appearance_dim = int(appearance_dim)
        self.latent_dim = self.geometry_dim + self.appearance_dim
        self.geometry_norm = nn.LayerNorm(
            self.geometry_dim, eps=1e-6, elementwise_affine=False)
        self.appearance_norm = nn.LayerNorm(
            self.appearance_dim, eps=1e-6, elementwise_affine=False)
        self.posterior_mean = nn.Conv2d(
            self.appearance_dim, self.appearance_dim, kernel_size=1)
        self.posterior_logvar = nn.Conv2d(
            self.appearance_dim, self.appearance_dim, kernel_size=1)

    def encode_latent(self, images_01, c2w_norm, K, pose_mask=None):
        prefix = self.encoder.encode_z18(
            images_01, c2w_norm, K, pose_mask=pose_mask)
        appearance = self.appearance_encoder(images_01, prefix["z18"])
        z_sem = channel_layer_norm(prefix["z18"], self.geometry_norm)
        appearance_features = channel_layer_norm(
            appearance["rgb_residual_raw"], self.appearance_norm)
        batch, views, channels, height, width = appearance_features.shape
        flat = appearance_features.reshape(batch * views, channels, height, width)
        z_rgb = self.posterior_mean(flat).reshape(
            batch, views, self.appearance_dim, height, width)
        posterior_logvar = self.posterior_logvar(flat).clamp(-30.0, 20.0).reshape(
            batch, views, self.appearance_dim, height, width)
        return {
            "latent": torch.cat((z_sem, z_rgb), dim=2),
            "z_sem": z_sem,
            "z_rgb": z_rgb,
            "z18": prefix["z18"],
            "rgb_residual_raw": appearance["rgb_residual_raw"],
            "posterior_mean": z_rgb,
            "posterior_logvar": posterior_logvar,
            "posterior_variance": posterior_logvar.exp(),
        }

    def forward(self, images_01, c2w_norm, K, pose_mask=None):
        return self.encode_latent(
            images_01, c2w_norm, K, pose_mask=pose_mask)


class RAE(ReconstructionRAE):
    """Final RAE with fixed latent-to-decoder denormalization."""

    CONTRACT_VERSION = 19
    DENORM_FORMAT = "pi3-v19-ln-to-raw-channel-affine-v1"

    def __init__(self, pi3x, *args, **kwargs):
        super().__init__(pi3x, *args, **kwargs)

        self.geometry_norm = nn.LayerNorm(
            self.geometry_dim, eps=1e-6, elementwise_affine=False)
        self.appearance_norm = nn.LayerNorm(
            self.appearance_dim, eps=1e-6, elementwise_affine=False)
        self.fusion_norm = nn.LayerNorm(
            self.geometry_dim, eps=1e-6, elementwise_affine=False)

        shape = (1, 1, self.geometry_dim, 1, 1)
        self.decoder_denorm_scale = nn.Parameter(
            torch.ones(shape), requires_grad=False)
        self.decoder_denorm_bias = nn.Parameter(
            torch.zeros(shape), requires_grad=False)

    def load_decoder_denorm_stats(self, payload: dict) -> None:
        if payload.get("format") != self.DENORM_FORMAT:
            raise RuntimeError(
                f"invalid decoder denorm format {payload.get('format')!r}")
        if int(payload.get("channels", -1)) != self.geometry_dim:
            raise RuntimeError("decoder denorm channel count does not match Pi3")
        scale = torch.as_tensor(payload["scale"], dtype=torch.float32)
        bias = torch.as_tensor(payload["bias"], dtype=torch.float32)
        if scale.shape != (self.geometry_dim,) or bias.shape != (self.geometry_dim,):
            raise RuntimeError("decoder denorm scale/bias must be one value per channel")
        if not torch.isfinite(scale).all() or not torch.isfinite(bias).all():
            raise RuntimeError("decoder denorm statistics contain non-finite values")
        if bool((scale <= 0).any()):
            raise RuntimeError("decoder denorm scale must be strictly positive")
        with torch.no_grad():
            self.decoder_denorm_scale.copy_(scale.reshape_as(
                self.decoder_denorm_scale))
            self.decoder_denorm_bias.copy_(bias.reshape_as(
                self.decoder_denorm_bias))

    def _decode_continuation(
        self, z_sem, z_rgb, c2w_norm, height, width, pose_mask,
    ):
        rgb_delta = self.rgb_to_trunk(z_rgb)
        h18_normalized = channel_layer_norm(
            z_sem + rgb_delta, self.fusion_norm)
        h18 = (
            h18_normalized * self.decoder_denorm_scale.to(h18_normalized.dtype)
            + self.decoder_denorm_bias.to(h18_normalized.dtype))
        batch, views = z_sem.shape[:2]
        z35_raw, z36_raw, point_hidden, final_pos = self.encoder.propagate_z18(
            h18, c2w_norm, height, width,
            pose_mask=pose_mask, return_hidden=True, return_taps=True,
            register_tokens=self._register_tokens(
                batch, views, h18.dtype))
        z36 = channel_layer_norm(z36_raw, self.z36_norm)
        return {
            "decoder_input": h18,
            "decoder_input_normalized": h18_normalized,
            "rgb_trunk_delta": rgb_delta,
            "z35_raw": z35_raw,
            "z36_raw": z36_raw,
            "z36": z36,
            "pi3_point_hidden": point_hidden,
            "pi3_final_pos": final_pos,
        }
