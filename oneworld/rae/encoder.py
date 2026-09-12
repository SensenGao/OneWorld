"""Single-boundary Pi3 encoder used by the OneWorld RAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

BOUNDARY_BLOCK = 17
FINAL_BLOCK = 35
POSE_INJECT_BLOCKS = (1, 9, 17, 25, 33)
POSE_BLOCK_TO_MODULE = {block: i for i, block in enumerate(POSE_INJECT_BLOCKS)}


class RGBResidualInjector(nn.Module):
    """Raw RGB -> Pi3 patch-grid residual, injected before decoder block zero."""

    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        if images_01.ndim != 5:
            raise ValueError(f"expected images (B,V,3,H,W), got {tuple(images_01.shape)}")
        B, V, C, H, W = images_01.shape
        if C != 3 or H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"image shape {tuple(images_01.shape)} incompatible with patch "
                f"size {self.patch_size}")
        x = images_01.mul(2.0).sub(1.0).reshape(B * V, C, H, W)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class RAEEncoder(nn.Module):
    """Full Pi3 path with one global generative boundary and deterministic continuation."""

    def __init__(
        self,
        pi3x: nn.Module,
        boundary_block: int = BOUNDARY_BLOCK,
        grad_checkpoint: bool = False,
        rgb_injection: bool = True,
        max_views: int = 8,
    ):
        super().__init__()
        if boundary_block != BOUNDARY_BLOCK:
            raise ValueError(
                f"the latent contract fixes boundary_block={BOUNDARY_BLOCK} (the global "
                f"18th block), got {boundary_block}")
        if len(pi3x.decoder) != FINAL_BLOCK + 1:
            raise ValueError(f"expected 36 Pi3 decoder blocks, got {len(pi3x.decoder)}")
        self.pi3x = pi3x
        self.boundary_block = boundary_block
        self.final_block = FINAL_BLOCK
        self.dim = pi3x.dec_embed_dim
        self.patch_size = pi3x.patch_size
        self.grad_checkpoint = grad_checkpoint
        self.rgb_injector = (
            RGBResidualInjector(self.dim, self.patch_size) if rgb_injection else None)
        self.freeze_pre_boundary = False
        # View zero is distinguished by the camera gauge, not by an absolute tensor-slot
        # embedding.  Keeping Pi3 permutation-equivariant over views 1..V-1 lets the
        # later DiT augment/reorder target views without changing the RAE latent contract.
        self.max_views = max_views

        self.register_buffer("z18_mean", torch.zeros(1, 1, self.dim, 1, 1))
        self.register_buffer("z18_std", torch.ones(1, 1, self.dim, 1, 1))
        self.register_buffer("z18_stats_fitted", torch.zeros((), dtype=torch.bool))

    def train(self, mode: bool = True):
        super().train(mode)
        # DINO defines the semantic anchor and must remain deterministic even while the
        # Pi3 decoder, RGB adapter and camera/view adapter are being fine-tuned.
        dino_encoder = getattr(self.pi3x, "encoder", None)
        if dino_encoder is not None:
            dino_encoder.eval()
        depth_encoder = getattr(self.pi3x, "depth_encoder", None)
        if depth_encoder is not None:
            depth_encoder.eval()
        if self.freeze_pre_boundary:
            for block in self.pi3x.decoder[:self.boundary_block + 1]:
                block.eval()
            for block_index, module_index in POSE_BLOCK_TO_MODULE.items():
                if block_index <= self.boundary_block:
                    self.pi3x.pose_inject_blk[module_index].eval()
            if self.rgb_injector is not None:
                self.rgb_injector.eval()
        return self

    def freeze_encoder_half(self) -> None:
        """Keep the learned RGB+Pi3 block-17 encoder fixed during RAE fine-tuning."""
        self.freeze_pre_boundary = True
        for block in self.pi3x.decoder[:self.boundary_block + 1]:
            block.requires_grad_(False)
        for block_index, module_index in POSE_BLOCK_TO_MODULE.items():
            if block_index <= self.boundary_block:
                self.pi3x.pose_inject_blk[module_index].requires_grad_(False)
        if self.rgb_injector is not None:
            self.rgb_injector.requires_grad_(False)
        self.train(self.training)

    # ------------------------------------------------------------------ latent stats

    def normalize_z18(self, raw: torch.Tensor) -> torch.Tensor:
        if not bool(self.z18_stats_fitted):
            raise RuntimeError("z18 statistics have not been fitted")
        return (raw - self.z18_mean) / self.z18_std

    def denormalize_z18(self, normalized: torch.Tensor) -> torch.Tensor:
        if not bool(self.z18_stats_fitted):
            raise RuntimeError("z18 statistics have not been fitted")
        return normalized * self.z18_std + self.z18_mean

    # ------------------------------------------------------------------ token helpers

    def _base_tokens(self, images_01: torch.Tensor, K: torch.Tensor):
        """Frozen-DINO/ray tokens with Pi3X's random pose normalization bypassed."""
        p = self.pi3x
        B, V = images_01.shape[:2]
        dev = images_01.device
        images_norm = (images_01 - p.image_mean) / p.image_std
        ones = torch.ones((B, V), device=dev, dtype=torch.bool)
        zeros = torch.zeros((B, V), device=dev, dtype=torch.bool)
        encoded = p.encode(
            images_norm, with_prior=True, intrinsics=K, poses=None,
            mask_add_ray=ones, mask_add_pose=zeros, mask_add_depth=zeros)
        if len(encoded) < 2:
            raise RuntimeError("Pi3X.encode must return patch and DINO semantic tokens")
        return encoded[0], encoded[1]

    def _positions(self, B: int, V: int, ph: int, pw: int, device):
        p = self.pi3x
        pos = p.position_getter(B * V, ph, pw, device)
        if p.patch_start_idx:
            special = torch.zeros(
                B * V, p.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat((special, pos + 1), dim=1)
        return pos

    def _zero_registers(self, B: int, V: int, dtype, device):
        p = self.pi3x
        return torch.zeros(
            B * V, p.patch_start_idx, self.dim, dtype=dtype, device=device)

    def _initial_registers(self, B: int, V: int):
        p = self.pi3x
        return p.register_token.repeat(B, V, 1, 1).reshape(
            B * V, p.patch_start_idx, self.dim)

    def _run_block(self, block: nn.Module, hidden: torch.Tensor, pos: torch.Tensor):
        if self.grad_checkpoint and self.training:
            return checkpoint(
                lambda h, p: block(h, xpos=p), hidden, pos, use_reentrant=False)
        return block(hidden, xpos=pos)

    def _inject_pose(
        self,
        hidden: torch.Tensor,
        block_index: int,
        c2w_norm: torch.Tensor,
        pose_mask: torch.Tensor,
        B: int,
        V: int,
        hw: int,
        H: int,
        W: int,
        ph: int,
        pw: int,
    ) -> torch.Tensor:
        p = self.pi3x
        if (not getattr(p, "use_multimodal", False)
                or block_index not in POSE_BLOCK_TO_MODULE):
            return hidden
        hidden = hidden.reshape(B, V, hw, self.dim)
        patch = hidden[..., p.patch_start_idx:, :].reshape(B, V * ph * pw, self.dim)
        pose_feature = p.pose_inject_blk[POSE_BLOCK_TO_MODULE[block_index]](
            patch, c2w_norm, H, W, ph, pw, attn_mask=None,
        ).reshape(B, V, ph * pw, self.dim)
        hidden = hidden.clone()
        hidden[..., p.patch_start_idx:, :] += (
            pose_feature * pose_mask.view(B, V, 1, 1).to(pose_feature.dtype))
        return hidden.reshape(B, V * hw, self.dim)

    @staticmethod
    def _map(hidden: torch.Tensor, B: int, V: int, C: int, ph: int, pw: int,
             patch_start_idx: int) -> torch.Tensor:
        tokens = hidden.reshape(B * V, -1, C)[:, patch_start_idx:]
        return tokens.transpose(1, 2).reshape(B, V, C, ph, pw)

    @staticmethod
    def _tokens(feature_map: torch.Tensor) -> torch.Tensor:
        B, V, C, ph, pw = feature_map.shape
        return feature_map.reshape(B * V, C, ph * pw).transpose(1, 2)

    # ------------------------------------------------------------------ propagation

    def propagate_z18(
        self,
        z18: torch.Tensor,
        c2w_norm: torch.Tensor,
        height: int,
        width: int,
        pose_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_taps: bool = False,
        register_tokens: torch.Tensor | None = None,
    ):
        """Re-enter block 18 and run the original Pi3 blocks 18..35."""
        p = self.pi3x
        B, V, C, ph, pw = z18.shape
        if C != self.dim or (ph, pw) != (height // self.patch_size, width // self.patch_size):
            raise ValueError(
                f"z18 shape {tuple(z18.shape)} incompatible with {(height, width)}")
        if pose_mask is None:
            pose_mask = torch.ones((B, V), device=z18.device, dtype=torch.bool)
        if pose_mask.shape != (B, V):
            raise ValueError(
                f"pose_mask must have shape {(B, V)}, got {tuple(pose_mask.shape)}")

        patches = self._tokens(z18)
        if register_tokens is None:
            register_tokens = self._zero_registers(
                B, V, patches.dtype, patches.device)
        expected_registers = (B * V, p.patch_start_idx, self.dim)
        if register_tokens.shape != expected_registers:
            raise ValueError(
                f"register tokens have shape {tuple(register_tokens.shape)}, "
                f"expected {expected_registers}")
        hidden = torch.cat((register_tokens.to(patches.dtype), patches), dim=1)
        hw = hidden.shape[1]
        pos = self._positions(B, V, ph, pw, z18.device)

        point_penultimate = None
        for i in range(self.boundary_block + 1, self.final_block + 1):
            if i % 2 == 0:
                hidden = hidden.reshape(B * V, hw, self.dim)
                pos_i = pos.reshape(B * V, hw, -1)
            else:
                hidden = hidden.reshape(B, V * hw, self.dim)
                pos_i = pos.reshape(B, V * hw, -1)
            hidden = self._run_block(p.decoder[i], hidden, pos_i)
            hidden = self._inject_pose(
                hidden, i, c2w_norm, pose_mask, B, V, hw,
                height, width, ph, pw)
            if i == self.final_block - 1:
                # The released Pi3X point decoder was trained with the concatenation of
                # decoder blocks 34 and 35 (2048 channels). The vendored Pi3X.decode has
                # this concatenation commented out even though point_decoder.projects
                # still has in_features=2048; restoring it is required by the checkpoint.
                point_penultimate = hidden.reshape(B * V, hw, self.dim)

        z36 = self._map(
            hidden, B, V, self.dim, ph, pw, p.patch_start_idx)
        if not return_hidden:
            if return_taps:
                raise ValueError("return_taps=True requires return_hidden=True")
            return z36
        final_hidden = hidden.reshape(B * V, hw, self.dim)
        if point_penultimate is None:
            raise RuntimeError("Pi3 block34 feature was not captured for point decoding")
        point_hidden = torch.cat((point_penultimate, final_hidden), dim=-1)
        final_pos = pos.reshape(B * V, hw, -1)
        if return_taps:
            z35 = self._map(
                point_penultimate, B, V, self.dim, ph, pw,
                p.patch_start_idx)
            return z35, z36, point_hidden, final_pos
        return z36, point_hidden, final_pos

    def decode_pi3_centres(
        self,
        point_hidden: torch.Tensor,
        final_pos: torch.Tensor,
        c2w_norm: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Run Pi3's original point decoder and expose its camera-space depth.

        Pi3 parameterises a full-resolution camera-space point as ``[xy*z, z]``.  We
        retain that auxiliary point map, but Gaussian construction consumes only its
        Z depth. Gaussian XY is reconstructed separately from source pixels and K, so
        point-head XY cannot move centres off their source rays.
        """
        p = self.pi3x
        B, V = c2w_norm.shape[:2]
        ph, pw = height // self.patch_size, width // self.patch_size
        expected_hw = p.patch_start_idx + ph * pw
        expected_dim = getattr(
            getattr(p.point_decoder, "projects", None), "in_features",
            point_hidden.shape[-1])
        if point_hidden.shape != (B * V, expected_hw, expected_dim):
            raise ValueError(
                f"Pi3 point tokens have shape {tuple(point_hidden.shape)}, expected "
                f"{(B * V, expected_hw, expected_dim)}")

        # Pi3's pretrained point decoder/head are the geometry anchor. RAE fine-tuning
        # updates them together with the complete Pi3 decoder continuum.
        point_tokens = p.point_decoder(point_hidden, xpos=final_pos)
        with torch.amp.autocast(device_type=point_hidden.device.type, enabled=False):
            xy, z = p.point_head(
                point_tokens[:, p.patch_start_idx:].float(),
                patch_h=ph, patch_w=pw)
            xy = xy.permute(0, 2, 3, 1).reshape(B, V, height, width, 2)
            z = z.permute(0, 2, 3, 1).reshape(B, V, height, width, 1)
            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat((xy * z, z), dim=-1)
            rotation = c2w_norm[..., :3, :3].float()
            translation = c2w_norm[..., :3, 3].float()
            world_points = torch.einsum(
                "bvij,bvhwj->bvhwi", rotation, local_points)
            world_points = world_points + translation[:, :, None, None]
        return {
            "local_points": local_points,
            "world_points": world_points,
            "depth": local_points[..., 2],
        }

    # ------------------------------------------------------------------ boundary encode

    def encode_z18(
        self,
        images_01: torch.Tensor,
        c2w_norm: torch.Tensor,
        K: torch.Tensor,
        pose_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run only the frozen RGB+Pi3 prefix and return raw global block-17 z18.

        Stage-2 statistics, training and inference must use this exact entry point so
        they see the same RGB injection, pose injection and register-token evolution as
        the RAE, without unnecessarily executing decoder blocks 18..35.
        """
        p = self.pi3x
        if images_01.ndim != 5:
            raise ValueError(
                f"expected images (B,V,3,H,W), got {tuple(images_01.shape)}")
        B, V, _, H, W = images_01.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"resolution {(H, W)} must be divisible by {self.patch_size}")
        ph, pw = H // self.patch_size, W // self.patch_size
        if c2w_norm.shape[:2] != (B, V) or K.shape[:2] != (B, V):
            raise ValueError("images, cameras and intrinsics disagree on B,V")
        if V > self.max_views:
            raise ValueError(f"V={V} exceeds max_views={self.max_views}")
        if pose_mask is None:
            pose_mask = torch.ones((B, V), device=images_01.device, dtype=torch.bool)
        if pose_mask.shape != (B, V):
            raise ValueError(
                f"pose_mask must have shape {(B, V)}, got {tuple(pose_mask.shape)}")

        hidden, semantic = self._base_tokens(images_01, K)
        hidden = hidden.reshape(B * V, ph * pw, self.dim)
        if self.rgb_injector is not None:
            hidden = hidden + self.rgb_injector(images_01).to(hidden.dtype)

        hidden = torch.cat((self._initial_registers(B, V).to(hidden.dtype), hidden), dim=1)
        hw = hidden.shape[1]
        pos = self._positions(B, V, ph, pw, images_01.device)

        for i in range(self.boundary_block + 1):
            if i % 2 == 0:
                hidden = hidden.reshape(B * V, hw, self.dim)
                pos_i = pos.reshape(B * V, hw, -1)
            else:
                hidden = hidden.reshape(B, V * hw, self.dim)
                pos_i = pos.reshape(B, V * hw, -1)
            hidden = self._run_block(p.decoder[i], hidden, pos_i)
            hidden = self._inject_pose(
                hidden, i, c2w_norm, pose_mask, B, V, hw, H, W, ph, pw)

        z18_raw = self._map(
            hidden, B, V, self.dim, ph, pw, p.patch_start_idx)
        evolved_registers = hidden.reshape(B * V, hw, self.dim)[
            :, :p.patch_start_idx]
        semantic = semantic.transpose(1, 2).reshape(B, V, self.dim, ph, pw)
        return {
            "z18": z18_raw,
            "semantic": semantic.detach(),
            # Register tokens are intentionally not part of the generative latent.
            # Keeping the frozen-prefix value available lets the frozen stock
            # continuation provide exact Pi3 geometry targets during RAE training.
            "evolved_registers": evolved_registers,
        }

    # ------------------------------------------------------------------ full encode

    def forward(
        self,
        images_01: torch.Tensor,
        c2w_norm: torch.Tensor,
        K: torch.Tensor,
        pose_mask: torch.Tensor | None = None,
        boundary_noise: float = 0.0,
        boundary_noise_prob: float = 1.0,
    ):
        if not 0.0 <= boundary_noise_prob <= 1.0:
            raise ValueError("boundary_noise_prob must be in [0,1]")
        encoded = self.encode_z18(images_01, c2w_norm, K, pose_mask=pose_mask)
        z18_raw = encoded["z18"]
        semantic = encoded["semantic"]
        B, V, _, H, W = images_01.shape
        # Do not insert a new token LayerNorm into a pretrained Pi3 residual stream:
        # blocks 18..35 and the point head were trained on this raw scale.  The encoder
        # half is frozen below, so the latent scale is fixed; dataset mean/std for DiT
        # is fitted separately without changing the RAE decoder's clean input.
        z18_clean = z18_raw
        z18_for_decode = z18_clean
        sample_std = z18_clean.detach().float().std(
            dim=(1, 2, 3, 4), keepdim=True).to(z18_clean.dtype)
        raw_std = z18_raw.detach().float().std(
            dim=(1, 2, 3, 4), keepdim=True).to(z18_clean.dtype)
        sigma = torch.zeros(
            (B, 1, 1, 1, 1), device=z18_clean.device, dtype=z18_clean.dtype)
        if self.training and boundary_noise > 0:
            # Official RAE samples one Uniform(0,tau) scalar per training sample and
            # multiplies it by randn_like(z).  A Pi3 training sample is a complete
            # multi-view scene, hence all views share sigma while every latent element
            # receives an independent standard-normal epsilon.
            sigma = boundary_noise * torch.rand(
                (B, 1, 1, 1, 1), device=z18_clean.device, dtype=z18_clean.dtype)
            use_noise = (
                torch.rand((B, 1, 1, 1, 1), device=z18_clean.device)
                < boundary_noise_prob)
            sigma = sigma * use_noise.to(sigma.dtype)
            z18_for_decode = z18_clean + sigma * torch.randn_like(z18_clean)

        z36, point_hidden, final_pos = self.propagate_z18(
            z18_for_decode, c2w_norm, H, W, pose_mask=pose_mask,
            return_hidden=True)
        return {
            "z18": z18_clean,
            "z18_decode": z18_for_decode,
            "z36": z36,
            "pi3_point_hidden": point_hidden,
            "pi3_final_pos": final_pos,
            "semantic": semantic.detach(),
            "z18_sample_std": sample_std.detach(),
            "z18_raw_std": raw_std.detach(),
            "boundary_sigma": sigma.detach(),
        }
