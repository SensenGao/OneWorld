"""Wan with a decoupled head for conditional Pi3 latent generation.

The main branch keeps all pretrained Wan-1.3B transformer blocks at width 1536.
The prediction branch is genuinely decoupled: it embeds the noisy/condition input
again at width 2048, then applies a configurable number of independently-parameterised transformer
blocks conditioned token-wise on the Wan main features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from oneworld.models.wan.modules.model import (
    WanModel,
    WanRMSNorm,
    WanSelfAttention,
    sinusoidal_embedding_1d,
)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward layer used by the prediction head."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2.0 / 3.0 * dim * mlp_ratio)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class DecoupledHeadBlock(nn.Module):
    """Prediction-head block using Wan's fused 3D-RoPE attention kernel.

    ``condition`` is the projected Wan-main feature at every token.  It produces
    per-token shift, scale, and gates for main-to-head conditioning.
    """

    def __init__(self, dim: int = 2048, num_heads: int = 16,
                 mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = WanRMSNorm(dim, eps=eps)
        self.norm2 = WanRMSNorm(dim, eps=eps)
        self.attn = WanSelfAttention(
            dim, num_heads, window_size=(-1, -1), qk_norm=False, eps=eps)
        self.mlp = SwiGLUFFN(dim, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, condition, seq_lens, grid_sizes, freqs):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1))
        y = self.norm1(x) * (1.0 + scale_attn) + shift_attn
        x = x + gate_attn * self.attn(y, seq_lens, grid_sizes, freqs)
        y = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(y)


class DecoupledFinalLayer(nn.Module):
    """Per-token output layer conditioned on the Wan main representation."""

    def __init__(self, dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = WanRMSNorm(dim, eps=eps)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)
        # Zero initialization preserves the pretrained main branch at startup.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, condition):
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1.0 + scale) + shift)


class WanPi3Flow(nn.Module):
    default_latent_dim = 1024
    head_dim = 2048
    default_head_depth = 6
    head_num_heads = 16

    def __init__(self, pretrained: WanModel, max_views: int = 8,
                 head_depth: int | None = None,
                 latent_dim: int = default_latent_dim,
                 input_alignment: str = ""):
        super().__init__()
        if pretrained.model_type != "t2v":
            raise ValueError("Stage-2 expects the Wan T2V backbone")
        if pretrained.dim != 1536:
            raise ValueError(f"Wan-1.3B main width must be 1536, got {pretrained.dim}")
        if pretrained.dim // pretrained.num_heads != self.head_dim // self.head_num_heads:
            raise ValueError("Wan main and DH must share the same 3D-RoPE head dimension")
        self.dim = pretrained.dim
        self.freq_dim = pretrained.freq_dim
        self.num_heads = pretrained.num_heads
        self.max_views = max_views
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        self.head_depth = (
            self.default_head_depth if head_depth is None else int(head_depth))
        if self.head_depth < 1:
            raise ValueError(f"head_depth must be positive, got {self.head_depth}")

        # Full Wan fine-tuning: retain every pretrained time and transformer block.
        self.time_embedding = pretrained.time_embedding.float()
        self.time_projection = pretrained.time_projection.float()
        self.blocks = pretrained.blocks
        # Complex frequencies intentionally stay outside buffers, matching upstream Wan.
        self.freqs = pretrained.freqs

        self.aligned_input = bool(input_alignment)
        if not self.aligned_input:
            input_dim = 2 * self.latent_dim
            # Condition and noisy values are concatenated per token.
            self.main_ref_input = nn.Linear(input_dim, self.dim)
            self.main_target_input = nn.Linear(input_dim, self.dim)
        else:
            # V19 is already a patch-token latent on Pi3's H/14 x W/14 grid.
            # Stage 1 aligns only this per-token projection to Wan's input space;
            # there is intentionally no second spatial patchification.
            self.main_latent_input = nn.Linear(self.latent_dim, self.dim)
            self.main_condition_input = nn.Linear(self.latent_dim, self.dim)
        input_dim = 2 * self.latent_dim
        self.head_ref_input = nn.Linear(input_dim, self.head_dim)
        self.head_target_input = nn.Linear(input_dim, self.head_dim)

        # Reference-relative Pluecker rays and the condition mask enter both paths.
        self.main_camera = nn.Linear(7, self.dim)
        self.head_camera = nn.Linear(7, self.head_dim)
        self.main_view_embedding = nn.Parameter(
            torch.zeros(1, max_views, 1, self.dim))
        self.head_view_embedding = nn.Parameter(
            torch.zeros(1, max_views, 1, self.head_dim))
        self.null_context = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.main_to_head = nn.Linear(self.dim, self.head_dim)
        self.head_blocks = nn.ModuleList([
            DecoupledHeadBlock(
                self.head_dim, self.head_num_heads, mlp_ratio=4.0,
                eps=pretrained.eps)
            for _ in range(self.head_depth)
        ])
        self.output_head = DecoupledFinalLayer(
            self.head_dim, self.latent_dim, eps=pretrained.eps)
        self.reset_new_parameters()
        if input_alignment:
            self.load_input_alignment(input_alignment)

    @classmethod
    def from_pretrained(cls, path: str, max_views: int = 8,
                        head_depth: int | None = None,
                        latent_dim: int = default_latent_dim,
                        input_alignment: str = ""):
        model, _ = cls.from_pretrained_with_text_projector(
            path, max_views=max_views, head_depth=head_depth,
            latent_dim=latent_dim, input_alignment=input_alignment)
        return model

    @classmethod
    def from_pretrained_with_text_projector(
            cls, path: str, max_views: int = 8,
            head_depth: int | None = None,
            latent_dim: int = default_latent_dim,
            input_alignment: str = ""):
        """Load the Pi3 flow model and Wan's native 4096->main-dim text MLP.

        The text projector is returned as a separate module on purpose.  Keeping it
        outside the FSDP root preserves the exact FlatParameter layout of the
        text-free 30k checkpoint, while allowing the projector to be trained and
        checkpointed independently during mandatory text fine-tuning.
        """
        pretrained = WanModel.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.bfloat16)
        text_projector = pretrained.text_embedding.float()
        model = cls(
            pretrained, max_views=max_views, head_depth=head_depth,
            latent_dim=latent_dim, input_alignment=input_alignment)
        return model, text_projector

    def reset_new_parameters(self):
        common = (self.main_camera, self.head_camera, self.main_to_head)
        if not self.aligned_input:
            input_layers = (
                self.main_ref_input, self.main_target_input,
                self.head_ref_input, self.head_target_input)
        else:
            input_layers = (
                self.main_latent_input,
                self.head_ref_input, self.head_target_input)
        for layer in (*input_layers, *common):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.aligned_input:
            # The aligned noisy-token path is useful from step zero.  Start the
            # reference condition contribution at zero and let Stage 2 learn it.
            nn.init.zeros_(self.main_condition_input.weight)
            nn.init.zeros_(self.main_condition_input.bias)
        nn.init.normal_(self.main_view_embedding, std=0.02)
        nn.init.normal_(self.head_view_embedding, std=0.02)
        nn.init.normal_(self.null_context, std=0.02)
        # Keep DecoupledFinalLayer's deliberate zero initialisation.

    def load_input_alignment(self, path: str):
        if not self.aligned_input:
            raise ValueError("input alignment requires the aligned-input architecture")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "pi3-v19-dc-videogen-input-align-v1":
            raise ValueError("incompatible input-alignment checkpoint")
        contract = payload.get("contract", {})
        expected = {
            "input_channels": self.latent_dim,
            "output_channels": self.dim,
        }
        mismatch = {
            key: (contract.get(key), value)
            for key, value in expected.items() if contract.get(key) != value
        }
        if mismatch:
            raise ValueError(f"input-alignment contract mismatch: {mismatch}")
        self.main_latent_input.load_state_dict(
            payload["input_projection"], strict=True)
        self.input_alignment_step = int(payload.get("step", -1))
        self.input_alignment_path = path

    def adapter_parameters(self):
        if not self.aligned_input:
            inputs = (
                self.main_ref_input, self.main_target_input,
                self.head_ref_input, self.head_target_input)
        else:
            inputs = (
                self.main_latent_input, self.main_condition_input,
                self.head_ref_input, self.head_target_input)
        modules = (*inputs, self.main_camera, self.head_camera,
                   self.main_to_head, self.head_blocks, self.output_head)
        parameters = [p for module in modules for p in module.parameters()]
        parameters.extend((
            self.main_view_embedding, self.head_view_embedding,
            self.null_context))
        return parameters

    @staticmethod
    def _split_embed(tokens, ref_layer, target_layer):
        batch, views, spatial, channels = tokens.shape
        ref = ref_layer(tokens[:, :1].reshape(batch, spatial, channels))
        target = target_layer(tokens[:, 1:].reshape(
            batch, (views - 1) * spatial, channels))
        return torch.cat((ref, target), dim=1)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        condition_latent: torch.Tensor,
        plucker: torch.Tensor,
        condition_mask: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor | None = None,
        text_context_lens: torch.Tensor | None = None,
        gradient_checkpointing: bool | str = True,
    ) -> torch.Tensor:
        """Predict clean x0 in normalized Pi3 latent space."""
        if noisy_latent.shape != condition_latent.shape:
            raise ValueError("noisy and condition latent shapes must match")
        batch, views, channels, height, width = noisy_latent.shape
        if channels != self.latent_dim or views > self.max_views:
            raise ValueError(f"unexpected latent shape {tuple(noisy_latent.shape)}")
        if plucker.shape != (batch, views, 6, height, width):
            raise ValueError(f"unexpected Pluecker shape {tuple(plucker.shape)}")
        if condition_mask.shape != (batch, views, 1, height, width):
            raise ValueError(f"unexpected condition mask {tuple(condition_mask.shape)}")

        # Pi3 has already patchified the image at stride 14.  Every H/14 x W/14
        # latent token remains one Wan sequence token and one prediction target.
        spatial = height * width

        def view_tokens(value):
            return value.permute(0, 1, 3, 4, 2).reshape(
                batch, views, spatial, value.shape[2])

        condition_tokens = view_tokens(condition_latent)
        noisy_tokens = view_tokens(noisy_latent)
        latent_input = torch.cat((condition_tokens, noisy_tokens), dim=-1)
        camera_tokens = view_tokens(torch.cat((plucker, condition_mask), dim=2))
        if not self.aligned_input:
            main_x = self._split_embed(
                latent_input, self.main_ref_input, self.main_target_input)
        else:
            main_x = (
                self.main_latent_input(noisy_tokens)
                + self.main_condition_input(condition_tokens)
            ).reshape(batch, views * spatial, self.dim)
        head_x = self._split_embed(
            latent_input, self.head_ref_input, self.head_target_input)
        main_x = (
            main_x
            + self.main_camera(camera_tokens).reshape(
                batch, views * spatial, self.dim)
            + self.main_view_embedding[:, :views].expand(
                batch, -1, spatial, -1).reshape(
                batch, views * spatial, self.dim)
        )

        # This module is run under FSDP mixed precision: the gathered compute
        # parameters are BF16 even though their sharded masters remain FP32.  Keep
        # this path inside the caller's autocast context so the FP32 sinusoidal
        # embedding is cast consistently with those compute parameters.
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).float())
        # WanAttentionBlock deliberately performs AdaLN modulation in FP32 and
        # asserts this contract at its boundary.
        time_modulation = self.time_projection(time).float().unflatten(
            1, (6, self.dim))

        sequence_length = views * spatial
        sequence_lengths = torch.full(
            (batch,), sequence_length, dtype=torch.long, device=main_x.device)
        grid_sizes = torch.tensor(
            [[views, height, width]] * batch,
            dtype=torch.long, device=main_x.device)
        if text_context is None:
            if text_context_lens is not None:
                raise ValueError("text_context_lens requires text_context")
            context = self.null_context.expand(batch, -1, -1).to(
                dtype=main_x.dtype)
            context_lens = None
        else:
            if (text_context.ndim != 3 or text_context.shape[0] != batch
                    or text_context.shape[2] != self.dim):
                raise ValueError(
                    f"unexpected text context {tuple(text_context.shape)}")
            if text_context_lens is None or text_context_lens.shape != (batch,):
                raise ValueError(
                    "text_context_lens must have shape [batch] for text conditioning")
            if (bool((text_context_lens < 1).any())
                    or bool((text_context_lens > text_context.shape[1]).any())):
                raise ValueError("text context lengths are outside the padded context")
            context = text_context.to(dtype=main_x.dtype)
            context_lens = text_context_lens
        if self.freqs.device != main_x.device:
            self.freqs = self.freqs.to(main_x.device)

        if isinstance(gradient_checkpointing, bool):
            checkpoint_mode = "full" if gradient_checkpointing else "none"
        else:
            checkpoint_mode = str(gradient_checkpointing)
        if checkpoint_mode not in {"none", "half", "full"}:
            raise ValueError(
                f"invalid gradient checkpointing mode {checkpoint_mode!r}")

        def checkpoint_block(block_index: int) -> bool:
            # Alternating blocks gives an exact 15/30 main + 1/2 DH split while
            # distributing retained activations uniformly through the network.
            return (
                self.training
                and (checkpoint_mode == "full"
                     or (checkpoint_mode == "half" and block_index % 2 == 1)))

        main_kwargs = dict(
            e=time_modulation, seq_lens=sequence_lengths,
            grid_sizes=grid_sizes, freqs=self.freqs,
            context=context, context_lens=context_lens)
        for block_index, block in enumerate(self.blocks):
            if checkpoint_block(block_index):
                main_x = torch.utils.checkpoint.checkpoint(
                    block, main_x, **main_kwargs, use_reentrant=False)
            else:
                main_x = block(main_x, **main_kwargs)

        # Combine the main tokens with time before projecting to head width.
        main_condition = self.main_to_head(
            F.silu(main_x + time[:, None].to(dtype=main_x.dtype)))

        head_x = (
            head_x
            + self.head_camera(camera_tokens).reshape(
                batch, sequence_length, self.head_dim)
            + self.head_view_embedding[:, :views].expand(
                batch, -1, spatial, -1).reshape(
                batch, sequence_length, self.head_dim)
        )
        for block_index, block in enumerate(self.head_blocks):
            if checkpoint_block(block_index):
                head_x = torch.utils.checkpoint.checkpoint(
                    block, head_x, main_condition,
                    sequence_lengths, grid_sizes, self.freqs,
                    use_reentrant=False)
            else:
                head_x = block(
                    head_x, main_condition, sequence_lengths,
                    grid_sizes, self.freqs)

        output = self.output_head(head_x, main_condition)
        return output.reshape(
            batch, views, height, width, self.latent_dim
        ).permute(0, 1, 4, 2, 3)


def wan_flow_sample(target: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor):
    """Rectified-flow interpolation where sigma is the noise coefficient."""
    while sigma.ndim < target.ndim:
        sigma = sigma.unsqueeze(-1)
    noisy = (1.0 - sigma) * target + sigma * noise
    velocity = noise - target
    return noisy, velocity


def shifted_sigma(uniform: torch.Tensor, shift: float = 5.0):
    """Apply the rational timestep shift used by the diffusion transformer."""
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


def dynamic_timestep_shift(latent_height: int, latent_width: int, views: int,
                           latent_dim: int = 1024,
                           shift_base: int = 4096) -> float:
    """Return sqrt((H*W*C*V) / shift_base) for the latent grid."""
    shift_dim = latent_height * latent_width * latent_dim * views
    return math.sqrt(shift_dim / shift_base)


def sample_x0_training_timestep(batch: int, device, shift: float,
                                max_weight: float = 100.0,
                                steps: int = 1000,
                                low_noise_mixture_prob: float = 0.0,
                                low_noise_uniform_max_sigma: float = 0.3,
                                return_low_noise_mask: bool = False):
    """Sample shifted/low-noise-mixture levels and capped x0 weights."""
    sigma = shifted_sigma(torch.rand(batch, device=device), shift=shift)
    low_noise_mask = torch.zeros(batch, device=device, dtype=torch.bool)
    if low_noise_mixture_prob > 0.0:
        low_noise_mask = (
            torch.rand(batch, device=device) < low_noise_mixture_prob)
        low_noise_sigma = (
            torch.rand(batch, device=device) * low_noise_uniform_max_sigma)
        sigma = torch.where(low_noise_mask, low_noise_sigma, sigma)
    timestep = sigma * steps
    weight = sigma.float().square().clamp_min(1e-8).reciprocal()
    result = (sigma, timestep, weight.clamp(max=max_weight))
    if return_low_noise_mask:
        return (*result, low_noise_mask)
    return result


def velocity_to_x0(noisy: torch.Tensor, velocity: torch.Tensor,
                   sigma: torch.Tensor):
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return noisy - sigma * velocity
