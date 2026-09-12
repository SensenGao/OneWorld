#!/usr/bin/env python3
"""Distill the OneWorld DiT with four-step 3DGS-rendering feedback."""

from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for package_root in (Path(__file__).resolve().parent, REPO_ROOT):
    sys.path.insert(0, str(package_root))

from train_dit import (  # noqa: E402
    _copy_parameter_shards,
    _copy_tensor_shards,
    _module_fingerprint,
    _tensor_fingerprint,
    latent_plucker,
    resolution_stats,
)

LATENT_DIM = 1056
GEOMETRY_DIM = 1024
VIEWS = 8
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 448
HEAD_DEPTH = 2
GENERATOR_PERIOD = 5
DEFAULT_STAGE_INDICES = (0, 39, 46, 49)
SOURCE_STEP = 120000
CAMERA_CFG = 2.0
TEXT_CFG = 3.0
TEXT_DROPOUT = 0.5
NOVEL_FEEDBACK_PROB = 0.25
SHIFT_BASE = 4096
BASE_SAMPLING_STEPS = 50
SCORE_MINIMUM_SIGMA = 0.02
SCORE_LOCAL_INTERVAL_PROB = 0.1
FAKE_MAX_X0_WEIGHT = 100.0
DMD_WEIGHT_MAX = 100.0
STAGE_EMBEDDING_INIT = 0.001
DMD2_FORMAT = "oneworld-3dgs-feedback-distill-v1"
DISTILL_CHECKPOINT_FORMAT = "oneworld-distill-rank-local-v1"


def optimizer_kind(update_index: int) -> str:
    if update_index < 0:
        raise ValueError("update index must be non-negative")
    return "generator" if update_index % GENERATOR_PERIOD == 0 else "fake_score"


def completed_update_counts(completed_updates: int) -> tuple[int, int]:
    if completed_updates < 0:
        raise ValueError("completed update count must be non-negative")
    generator_updates = (completed_updates + GENERATOR_PERIOD - 1) // GENERATOR_PERIOD
    return generator_updates, completed_updates - generator_updates


def build_student_sigmas(
    shift: float,
    stage_indices: tuple[int, ...] = DEFAULT_STAGE_INDICES,
    base_steps: int = 50,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    if not math.isfinite(shift) or shift <= 1.0:
        raise ValueError("shift must be finite and greater than one")
    if base_steps < 1 or stage_indices[-1] >= base_steps:
        raise ValueError("stage indices must fall inside the base sampling schedule")
    base = torch.linspace(1.0, 0.0, base_steps + 1, device=device)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    result = shifted[list(stage_indices)]
    if not bool(torch.all(result[:-1] > result[1:])):
        raise RuntimeError(f"student sigmas are not strictly decreasing: {result}")
    return result


def sample_score_sigmas(
    batch: int,
    stage: int,
    student_sigmas: torch.Tensor,
    minimum_sigma: float,
    local_interval_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch < 1:
        raise ValueError("batch must be positive")
    if stage < 0 or stage >= student_sigmas.numel():
        raise ValueError("stage is outside the student schedule")
    if not 0.0 < minimum_sigma < float(student_sigmas[-1]):
        raise ValueError("minimum score sigma must be below the last student sigma")
    if not 0.0 <= local_interval_probability <= 1.0:
        raise ValueError("local interval probability must be in [0,1]")
    device = student_sigmas.device
    current_upper = float(student_sigmas[stage])
    current_lower = (
        float(student_sigmas[stage + 1])
        if stage + 1 < student_sigmas.numel()
        else minimum_sigma
    )
    local_mask = torch.rand(batch, device=device, generator=generator) < local_interval_probability
    broad_uniform = torch.rand(batch, device=device, generator=generator)
    local_uniform = torch.rand(batch, device=device, generator=generator)
    broad = current_lower + broad_uniform * (1.0 - current_lower)
    local = current_lower + local_uniform * (current_upper - current_lower)
    return torch.where(local_mask, local, broad), local_mask


def learning_rate(
    completed_updates: int,
    total_updates: int,
    base_lr: float,
    final_lr: float,
    warmup_updates: int,
) -> float:
    if completed_updates < warmup_updates:
        return base_lr * (completed_updates + 1) / max(1, warmup_updates)
    progress = (completed_updates - warmup_updates) / max(
        1, total_updates - warmup_updates - 1)
    progress = min(max(progress, 0.0), 1.0)
    return final_lr + 0.5 * (base_lr - final_lr) * (
        1.0 + math.cos(math.pi * progress))


def add_stage_token(
    text_context: torch.Tensor,
    text_context_lens: torch.Tensor,
    stage_embeddings: torch.Tensor,
    stage: int,
    detach: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = stage_embeddings[stage].reshape(1, 1, -1).expand(
        text_context.shape[0], -1, -1)
    if detach:
        token = token.detach()
    return torch.cat((token.to(text_context.dtype), text_context), dim=1), text_context_lens + 1


def direct_x0_dmd_target(
    generated_x0: torch.Tensor,
    real_score_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
) -> torch.Tensor:
    return generated_x0.detach() + real_score_x0.detach() - fake_score_x0.detach()


def combine_independent_guidance(
    full: torch.Tensor,
    no_camera: torch.Tensor | None,
    no_text: torch.Tensor | None,
    camera_cfg: float,
    text_cfg: float,
    text_present: torch.Tensor,
) -> torch.Tensor:
    """Apply camera CFG to every sample and text CFG only when text is present."""
    if text_present.shape != (full.shape[0],):
        raise ValueError(
            f"text_present must have shape {(full.shape[0],)}, got "
            f"{tuple(text_present.shape)}"
        )
    guided = full
    if camera_cfg != 1.0:
        if no_camera is None:
            raise ValueError("camera CFG requires a no-camera prediction")
        guided = guided + (camera_cfg - 1.0) * (full - no_camera)
    if text_cfg != 1.0:
        if no_text is None:
            raise ValueError("text CFG requires a no-text prediction")
        text_scale = text_present.to(full.dtype).reshape(
            full.shape[0], *([1] * (full.ndim - 1))
        )
        guided = guided + (text_cfg - 1.0) * text_scale * (full - no_text)
    return guided


def sample_text_present(
    batch: int,
    text_dropout: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch < 1:
        raise ValueError("batch must be positive")
    if not 0.0 <= text_dropout <= 1.0:
        raise ValueError("text dropout must be in [0,1]")
    return torch.rand(batch, device=device, generator=generator) >= text_dropout


def prepare_dmd_camera_batch(
    images: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    source_slots: torch.Tensor,
    canonicalize_cameras,
    return_valid: bool = False,
):
    """Reorder the reference while retaining canonicalized held-out NVS cameras."""
    if images.ndim != 5 or images.shape[1] != VIEWS:
        raise ValueError(f"expected eight context images, got {tuple(images.shape)}")
    if c2w.shape[1:] != (2 * VIEWS, 4, 4):
        raise ValueError(f"expected sixteen cameras, got {tuple(c2w.shape)}")
    if intrinsics.shape[1:] != (2 * VIEWS, 3, 3):
        raise ValueError(f"expected sixteen intrinsics, got {tuple(intrinsics.shape)}")
    if source_slots.shape != (images.shape[0],):
        raise ValueError(f"invalid source slots: {tuple(source_slots.shape)}")
    if bool(((source_slots < 0) | (source_slots >= VIEWS)).any()):
        raise ValueError(f"source slot outside [0,{VIEWS - 1}]")

    device = images.device
    context_base = torch.arange(VIEWS, device=device)
    context_order = torch.stack(
        [
            torch.cat((slot[None], context_base[context_base != slot]))
            for slot in source_slots
        ]
    )
    heldout_order = torch.arange(VIEWS, 2 * VIEWS, device=device)[None].expand(
        images.shape[0], -1
    )
    all_order = torch.cat((context_order, heldout_order), dim=1)
    images = images.gather(
        1, context_order[:, :, None, None, None].expand_as(images)
    )
    intrinsics = intrinsics.gather(
        1, all_order[:, :, None, None].expand_as(intrinsics)
    )
    c2w = c2w.gather(1, all_order[:, :, None, None].expand_as(c2w))
    canonical = canonicalize_cameras(
        c2w, reference_index=0, scale_view_count=VIEWS, strict=not return_valid
    )
    result = (
        images,
        canonical.c2w[:, :VIEWS],
        intrinsics[:, :VIEWS],
        canonical.c2w[:, VIEWS:],
        intrinsics[:, VIEWS:],
        context_order,
    )
    if return_valid:
        result += (canonical.valid, canonical.scale)
    return result


def select_feedback_cameras(
    context_c2w: torch.Tensor,
    context_intrinsics: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsics: torch.Tensor,
    novel_feedback: torch.Tensor,
    target_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose context renders or reference-plus-seven held-out novel renders."""
    batch = context_c2w.shape[0]
    if novel_feedback.shape != (batch,) or target_offsets.shape != (batch,):
        raise ValueError("feedback masks and offsets must have one value per scene")
    slots = (
        torch.arange(VIEWS - 1, device=context_c2w.device)[None]
        + target_offsets[:, None]
    ) % VIEWS
    novel_c2w = torch.cat(
        (
            context_c2w[:, :1],
            target_c2w.gather(
                1, slots[:, :, None, None].expand(-1, -1, 4, 4)
            ),
        ),
        dim=1,
    )
    novel_intrinsics = torch.cat(
        (
            context_intrinsics[:, :1],
            target_intrinsics.gather(
                1, slots[:, :, None, None].expand(-1, -1, 3, 3)
            ),
        ),
        dim=1,
    )
    camera_mask = novel_feedback[:, None, None, None]
    selected_c2w = torch.where(camera_mask, novel_c2w, context_c2w)
    selected_intrinsics = torch.where(
        camera_mask, novel_intrinsics, context_intrinsics
    )
    return selected_c2w, selected_intrinsics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill the OneWorld generator to four 3DGS-feedback steps.")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", required=True, help="NVS-Refined root")
    parser.add_argument(
        "--re10k-root", required=True,
        help="RealEstate10K .torch shard root")
    parser.add_argument("--pi3", required=True, help="Pi3X checkpoint directory")
    parser.add_argument("--wan", required=True, help="Wan2.1-T2V-1.3B directory")
    parser.add_argument("--rae", required=True, help="OneWorld RAE checkpoint")
    parser.add_argument("--stats", required=True, help="RAE latent statistics")
    parser.add_argument("--input-align", required=True)
    parser.add_argument("--text-store", required=True)
    parser.add_argument(
        "--source-checkpoint", required=True,
        help="rank-local DiT checkpoint directory used as the distillation teacher")
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "outputs" / "distill"))
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--generator-lr", type=float, default=1e-6)
    parser.add_argument("--generator-final-lr", type=float, default=2e-7)
    parser.add_argument("--fake-score-lr", type=float, default=5e-7)
    parser.add_argument("--fake-score-final-lr", type=float, default=1e-7)
    parser.add_argument("--generator-warmup-updates", type=int, default=100)
    parser.add_argument("--fake-score-warmup-updates", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing", choices=("none", "half", "full"), default="full"
    )
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--checkpoint-slots", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def validate_args(args):
    if args.updates <= 0 or args.updates % GENERATOR_PERIOD:
        raise ValueError("updates must be positive and divisible by five")
    generator_updates, fake_updates = completed_update_counts(args.updates)
    if generator_updates * 4 != fake_updates:
        raise RuntimeError("DMD2 update plan is not exactly one generator plus four fake-score")
    if args.batch_size not in (1, 2):
        raise ValueError("DMD2 supports per-GPU batch size one or two")
    if args.checkpoint_every < 1 or args.checkpoint_slots < 1:
        raise ValueError("checkpoint cadence and slot count must be positive")
    for path in (
        args.data_root,
        args.re10k_root,
        args.pi3,
        args.wan,
        args.rae,
        args.stats,
        args.input_align,
        args.text_store,
        args.source_checkpoint,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(path)


def synchronized_stage(device: torch.device, number_of_stages: int) -> int:
    value = torch.zeros((), device=device, dtype=torch.int64)
    if dist.get_rank() == 0:
        value.random_(0, number_of_stages)
    dist.broadcast(value, src=0)
    return int(value)


def collective_finite(value: torch.Tensor, label: str):
    valid = torch.tensor(int(bool(torch.isfinite(value))), device=value.device, dtype=torch.int32)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid):
        raise FloatingPointError(f"non-finite distributed {label}")


def has_any_gradient(module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


def assert_gradient_route(
    phase: str,
    student: FSDP,
    fake_score: FSDP,
    real_score: FSDP,
    stage_embeddings: torch.Tensor,
    device: torch.device,
):
    if phase == "generator":
        local_valid = (
            has_any_gradient(student)
            and not has_any_gradient(fake_score)
            and not has_any_gradient(real_score)
            and stage_embeddings.grad is None
        )
    else:
        local_valid = (
            not has_any_gradient(student)
            and has_any_gradient(fake_score)
            and not has_any_gradient(real_score)
            and stage_embeddings.grad is not None
        )
    valid = torch.tensor(int(local_valid), device=device, dtype=torch.int32)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid):
        raise RuntimeError(f"gradient routing failed during {phase} update")


def model_x0(
    model: FSDP,
    noisy: torch.Tensor,
    condition: torch.Tensor,
    plucker: torch.Tensor,
    condition_mask: torch.Tensor,
    sigma: torch.Tensor | float,
    text_context: torch.Tensor,
    text_context_lens: torch.Tensor,
    gradient_checkpointing: str,
) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        timestep = sigma.float().reshape(-1) * 1000.0
    else:
        timestep = torch.full(
            (noisy.shape[0],), float(sigma) * 1000.0, device=noisy.device
        )
    with torch.autocast(
        device_type=noisy.device.type,
        dtype=torch.bfloat16,
        enabled=noisy.is_cuda,
    ):
        prediction = model(
            noisy,
            condition,
            plucker,
            condition_mask,
            timestep,
            text_context=text_context,
            text_context_lens=text_context_lens,
            gradient_checkpointing=gradient_checkpointing,
        )
    return prediction.float()


def guided_real_score_x0(
    real_score: FSDP,
    noisy: torch.Tensor,
    condition: torch.Tensor,
    plucker: torch.Tensor,
    condition_mask: torch.Tensor,
    sigma: torch.Tensor,
    text_context: torch.Tensor,
    text_context_lens: torch.Tensor,
    null_context: torch.Tensor,
    null_context_lens: torch.Tensor,
    camera_cfg: float,
    text_cfg: float,
    text_present: torch.Tensor,
) -> torch.Tensor:
    full = model_x0(
        real_score,
        noisy,
        condition,
        plucker,
        condition_mask,
        sigma,
        text_context,
        text_context_lens,
        "none",
    )
    no_camera = None
    if camera_cfg != 1.0:
        no_camera = model_x0(
            real_score,
            noisy,
            condition,
            torch.zeros_like(plucker),
            condition_mask,
            sigma,
            text_context,
            text_context_lens,
            "none",
        )
    no_text = None
    if text_cfg != 1.0:
        no_text = model_x0(
            real_score,
            noisy,
            condition,
            plucker,
            condition_mask,
            sigma,
            null_context,
            null_context_lens,
            "none",
        )
    return combine_independent_guidance(
        full,
        no_camera,
        no_text,
        camera_cfg,
        text_cfg,
        text_present,
    )


def render_encode_3dgs_feedback(
    rae,
    normalized_latent: torch.Tensor,
    context_c2w: torch.Tensor,
    context_intrinsics: torch.Tensor,
    render_c2w: torch.Tensor,
    render_intrinsics: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    image_height: int,
    image_width: int,
    random_background: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Decode 3DGS, render selected cameras, and re-encode the rendered RGB."""
    latent = normalized_latent.float() * target_std + target_mean
    with torch.autocast(
        device_type=normalized_latent.device.type,
        dtype=torch.bfloat16,
        enabled=normalized_latent.is_cuda,
    ):
        decoded = rae.decode_generated_latent(
            latent,
            context_c2w,
            context_intrinsics,
            image_height,
            image_width,
            normalized=True,
            want=("depth", "gs"),
            renders=[
                {
                    "c2w": render_c2w,
                    "K": render_intrinsics,
                    "random_background": random_background,
                }
            ],
            random_background=random_background,
        )
        local_valid = bool(decoded["gs_ok"])
        globally_valid = torch.tensor(
            int(local_valid), device=normalized_latent.device, dtype=torch.int32
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(globally_valid, op=dist.ReduceOp.MIN)
        if not bool(globally_valid):
            if not local_valid:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"rank={rank} generated latent produced invalid 3D Gaussians",
                    file=sys.stderr,
                    flush=True,
                )
            return None, None
        rendered_rgb = decoded["renders"][0]["rgb"].clamp(0.0, 1.0)
        feedback_raw = rae.encode_latent(
            rendered_rgb, render_c2w, render_intrinsics
        )["latent"]
    feedback = (feedback_raw.float() - target_mean) / target_std
    if feedback.shape != normalized_latent.shape:
        raise RuntimeError(
            f"3D feedback shape {tuple(feedback.shape)} does not match "
            f"student output {tuple(normalized_latent.shape)}"
        )
    return feedback, rendered_rgb.float()


def rollout_student(
    student: FSDP,
    target_stage: int,
    student_sigmas: torch.Tensor,
    condition: torch.Tensor,
    plucker: torch.Tensor,
    condition_mask: torch.Tensor,
    text_context: torch.Tensor,
    text_context_lens: torch.Tensor,
    feedback_function,
    require_gradient: bool,
    gradient_checkpointing: str,
) -> torch.Tensor:
    noisy = torch.randn_like(condition)
    for stage in range(target_stage):
        with torch.no_grad():
            generated_x0 = model_x0(
                student,
                noisy,
                condition,
                plucker,
                condition_mask,
                float(student_sigmas[stage]),
                text_context,
                text_context_lens,
                "none",
            )
            generated_x0, _ = feedback_function(generated_x0)
            if generated_x0 is None:
                return None
            next_sigma = float(student_sigmas[stage + 1])
            noisy = (1.0 - next_sigma) * generated_x0 + next_sigma * torch.randn_like(
                generated_x0
            )
    if require_gradient:
        return model_x0(
            student,
            noisy,
            condition,
            plucker,
            condition_mask,
            float(student_sigmas[target_stage]),
            text_context,
            text_context_lens,
            gradient_checkpointing,
        )
    with torch.no_grad():
        return model_x0(
            student,
            noisy,
            condition,
            plucker,
            condition_mask,
            float(student_sigmas[target_stage]),
            text_context,
            text_context_lens,
            "none",
        )


def load_source_contract(path: str, source_step: int, world_size: int) -> dict:
    contract_path = os.path.join(path, "contract.json")
    with open(contract_path, encoding="utf-8") as handle:
        contract = json.load(handle)
    expected = {
        "world_size": world_size,
        "step": source_step,
        "latent_target": "rae_1056",
        "prediction": "direct_x0",
        "prediction_channels_per_token": LATENT_DIM,
        "resolutions_hw": [[224, 448]],
        "patch_grids_hw": [[16, 32]],
        "views": VIEWS,
        "condition_views": 1,
        "decoupled_head": "dim2048 depth2 heads16",
    }
    mismatch = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"source checkpoint contract mismatch: {mismatch}")
    return contract


def load_source_weights(
    source_checkpoint: str,
    student: FSDP,
    real_score: FSDP,
    fake_score: FSDP,
    rank: int,
    world_size: int,
    decoder_parameters,
):
    rank_file = os.path.join(source_checkpoint, f"rank{rank:05d}.pt")
    payload = torch.load(rank_file, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") != "rank-local-fsdp-weights-ema-v2":
        raise RuntimeError(f"unsupported source checkpoint format: {payload.get('format')}")
    if payload.get("rank") != rank or payload.get("world_size") != world_size:
        raise RuntimeError("source checkpoint rank/world mismatch")
    for label, model in (
        ("student", student),
        ("real_score", real_score),
        ("fake_score", fake_score),
    ):
        _copy_parameter_shards(model, payload["ema"], label)
        actual = _tensor_fingerprint(model.parameters(), next(model.parameters()).device)
        if not torch.equal(actual, payload["fingerprints"]["ema"]):
            raise RuntimeError(f"{label} source EMA fingerprint mismatch")
    if "decoder" not in payload:
        raise RuntimeError("source checkpoint does not contain MDF decoder weights")
    _copy_tensor_shards(decoder_parameters, payload["decoder"], "MDF decoder")
    del payload
    gc.collect()


def load_text_projector(source_checkpoint: str, text_projector):
    path = os.path.join(source_checkpoint, "text_projector.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "wan-native-text-projector-v1":
        raise RuntimeError("unsupported text projector checkpoint")
    text_projector.load_state_dict(payload["ema"], strict=True)
    if not torch.equal(_module_fingerprint(text_projector), payload["ema_fingerprint"]):
        raise RuntimeError("text projector EMA fingerprint mismatch")
    text_projector.eval().requires_grad_(False)


def find_resume(out_dir: str, requested: str, rank: int) -> str:
    if requested == "":
        path = ""
    elif requested != "auto":
        path = requested
    elif rank == 0:
        pointer = os.path.join(out_dir, "latest.txt")
        if os.path.isfile(pointer):
            slot, _ = open(pointer, encoding="utf-8").read().strip().split()
            path = os.path.join(out_dir, "checkpoints", f"slot{slot}")
        else:
            path = ""
    else:
        path = ""
    values = [path]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def load_dmd_resume(
    path: str,
    student: FSDP,
    fake_score: FSDP,
    stage_embeddings: torch.Tensor,
    rank: int,
    world_size: int,
    decoder_parameters,
) -> tuple[int, list[int], list[int], list[int], int]:
    payload = torch.load(
        os.path.join(path, f"rank{rank:05d}.pt"),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("format") != DISTILL_CHECKPOINT_FORMAT:
        raise RuntimeError("unsupported DMD2 resume checkpoint format")
    if payload.get("dmd2_format") != DMD2_FORMAT:
        raise RuntimeError("checkpoint is not a OneWorld distillation checkpoint")
    if payload.get("rank") != rank or payload.get("world_size") != world_size:
        raise RuntimeError("DMD2 resume rank/world mismatch")
    _copy_parameter_shards(student, payload["model"], "student resume")
    _copy_parameter_shards(fake_score, payload["fake_score"], "fake-score resume")
    _copy_tensor_shards(
        decoder_parameters, payload["decoder_ema"], "decoder resume")
    stage_embeddings.data.copy_(payload["fake_stage_embeddings"].to(stage_embeddings.device))
    student_fingerprint = _tensor_fingerprint(
        student.parameters(), next(student.parameters()).device
    )
    fake_fingerprint = _tensor_fingerprint(
        fake_score.parameters(), next(fake_score.parameters()).device
    )
    if not torch.equal(student_fingerprint, payload["fingerprints"]["model"]):
        raise RuntimeError("student resume fingerprint mismatch")
    if not torch.equal(fake_fingerprint, payload["fingerprints"]["fake_score"]):
        raise RuntimeError("fake-score resume fingerprint mismatch")
    update_step = int(payload["step"])
    histogram = [int(value) for value in payload["stage_histogram"]]
    if len(histogram) != 4 or sum(histogram) != update_step:
        raise RuntimeError("invalid saved stage histogram")
    conditioning_histogram = [
        int(value) for value in payload["conditioning_histogram"]
    ]
    feedback_histogram = [int(value) for value in payload["feedback_histogram"]]
    expected_samples = update_step * world_size * int(payload["batch_size"])
    if (
        len(conditioning_histogram) != 2
        or sum(conditioning_histogram) != expected_samples
        or len(feedback_histogram) != 2
        or sum(feedback_histogram) != expected_samples
    ):
        raise RuntimeError("invalid saved mixed-conditioning histograms")
    skipped_invalid_gaussian_batches = int(
        payload.get("skipped_invalid_gaussian_batches", 0)
    )
    del payload
    gc.collect()
    return (
        update_step,
        histogram,
        conditioning_histogram,
        feedback_histogram,
        skipped_invalid_gaussian_batches,
    )


def save_checkpoint(
    args,
    slot,
    update_step: int,
    student: FSDP,
    fake_score: FSDP,
    stage_embeddings: torch.Tensor,
    stage_histogram: list[int],
    conditioning_histogram: list[int],
    feedback_histogram: list[int],
    skipped_invalid_gaussian_batches: int,
    source_contract: dict,
    rank: int,
    decoder_parameters,
):
    world_size = dist.get_world_size()
    dist.barrier()
    student_fingerprint = _tensor_fingerprint(
        student.parameters(), next(student.parameters()).device
    )
    fake_fingerprint = _tensor_fingerprint(
        fake_score.parameters(), next(fake_score.parameters()).device
    )
    checkpoint_root = os.path.join(args.out, "checkpoints")
    final_path = os.path.join(checkpoint_root, f"slot{slot}")
    temporary_path = os.path.join(
        checkpoint_root,
        f".slot{slot}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}",
    )
    if rank == 0:
        os.makedirs(checkpoint_root, exist_ok=True)
        if os.path.exists(temporary_path):
            shutil.rmtree(temporary_path)
    dist.barrier()
    fingerprints = {
        "model": student_fingerprint,
        "ema": student_fingerprint,
        "fake_score": fake_fingerprint,
    }
    student_cpu = [parameter.detach().cpu() for parameter in student.parameters()]
    payload = {
        "format": DISTILL_CHECKPOINT_FORMAT,
        "dmd2_format": DMD2_FORMAT,
        "rank": rank,
        "world_size": world_size,
        "step": update_step,
        "model": student_cpu,
        "ema": [tensor.clone() for tensor in student_cpu],
        "fake_score": [parameter.detach().cpu() for parameter in fake_score.parameters()],
        "fake_stage_embeddings": stage_embeddings.detach().cpu(),
        "stage_histogram": list(stage_histogram),
        "conditioning_histogram": list(conditioning_histogram),
        "feedback_histogram": list(feedback_histogram),
        "skipped_invalid_gaussian_batches": skipped_invalid_gaussian_batches,
        "batch_size": args.batch_size,
        "fingerprints": fingerprints,
        "decoder_ema": [
            parameter.detach().cpu() for parameter in decoder_parameters],
    }
    os.makedirs(temporary_path, exist_ok=True)
    rank_file = os.path.join(temporary_path, f"rank{rank:05d}.pt")
    torch.save(payload, rank_file + ".tmp")
    os.replace(rank_file + ".tmp", rank_file)
    del payload, student_cpu
    gc.collect()
    dist.barrier()
    if rank == 0:
        shutil.copy2(
            os.path.join(args.source_checkpoint, "text_projector.pt"),
            os.path.join(temporary_path, "text_projector.pt"),
        )
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.replace(temporary_path, final_path)
        generator_updates, fake_updates = completed_update_counts(update_step)
        student_sigmas = build_student_sigmas(
            math.sqrt((16 * 32 * LATENT_DIM * VIEWS) / SHIFT_BASE),
            DEFAULT_STAGE_INDICES,
            BASE_SAMPLING_STEPS,
        )
        contract = dict(source_contract)
        contract.update(
            {
                "step": update_step,
                "training_target_updates": args.updates,
                "source_step": SOURCE_STEP,
                "source_checkpoint": os.path.abspath(args.source_checkpoint),
                "slot": slot,
                "checkpoint_format": DISTILL_CHECKPOINT_FORMAT,
                "optimizer_resume": "reset",
                "checkpoint_every_steps": args.checkpoint_every,
                "checkpoint_slots": args.checkpoint_slots,
                "world_size": world_size,
                "ema": "student_current_weights_aliased_for_sampler_compatibility",
                "dmd2": {
                    "format": DMD2_FORMAT,
                    "total_optimizer_updates": args.updates,
                    "completed_optimizer_updates": update_step,
                    "generator_updates": generator_updates,
                    "fake_score_updates": fake_updates,
                    "update_cycle": "GFFFF",
                    "generator_to_fake_score_ratio": [1, 4],
                    "random_student_rollout_depths": [1, 2, 3, 4],
                    "stage_indices_in_50_step_shifted_schedule": list(
                        DEFAULT_STAGE_INDICES
                    ),
                    "student_sigmas": [float(value) for value in student_sigmas],
                    "stage_histogram": list(stage_histogram),
                    "generator_loss_channels": (
                        "3dgs_render_reencoded_views1_to7_all1056"
                    ),
                    "fake_score_loss_channels": (
                        "3dgs_render_reencoded_views1_to7_all1056"
                    ),
                    "conditioning_modes": {
                        "image_camera_no_text_probability": TEXT_DROPOUT,
                        "image_camera_text_probability": 1.0 - TEXT_DROPOUT,
                        "histogram": list(conditioning_histogram),
                    },
                    "feedback_tasks": {
                        "context_probability": 1.0 - NOVEL_FEEDBACK_PROB,
                        "heldout_nvs_probability": NOVEL_FEEDBACK_PROB,
                        "histogram": list(feedback_histogram),
                        "context_to_novel_ratio": [3, 1],
                    },
                    "rollout_feedback": (
                        "decode_3dgs_render_context_rae_encode_before_next_stage"
                    ),
                    "real_score": (
                        "frozen_100k_ema_camera_cfg_all_text_cfg_text_samples_only"
                    ),
                    "fake_score": "trainable_100k_ema_initialization_with_stage_token",
                    "camera_cfg": CAMERA_CFG,
                    "text_cfg": TEXT_CFG,
                    "score_sigma_sampling": {
                        "broad_probability": 1.0 - SCORE_LOCAL_INTERVAL_PROB,
                        "local_stage_interval_probability": SCORE_LOCAL_INTERVAL_PROB,
                        "minimum_sigma": SCORE_MINIMUM_SIGMA,
                    },
                    "generator_lr": args.generator_lr,
                    "fake_score_lr": args.fake_score_lr,
                    "decoder_frozen_but_differentiable_to_student": True,
                    "text_projector_frozen": True,
                    "student_and_fake_score_cfg": False,
                    "post_distillation_inference_cfg": False,
                    "skipped_invalid_gaussian_batches": (
                        skipped_invalid_gaussian_batches
                    ),
                },
            }
        )
        contract["rae_checkpoint"] = os.path.abspath(args.rae)
        contract["rae_frozen_during_distillation"] = True
        with open(os.path.join(final_path, "contract.json"), "w", encoding="utf-8") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
        pointer_tmp = os.path.join(args.out, "latest.txt.tmp")
        with open(pointer_tmp, "w", encoding="utf-8") as handle:
            handle.write(f"{slot} {update_step}\n")
        os.replace(pointer_tmp, os.path.join(args.out, "latest.txt"))
        with open(os.path.join(args.out, "progress.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{update_step}\n")
        with open(os.path.join(args.out, "progress.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "optimizer_updates": update_step,
                    "generator_updates": generator_updates,
                    "fake_score_updates": fake_updates,
                    "stage_histogram": list(stage_histogram),
                    "conditioning_histogram": list(conditioning_histogram),
                    "feedback_histogram": list(feedback_histogram),
                    "skipped_invalid_gaussian_batches": (
                        skipped_invalid_gaussian_batches
                    ),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    dist.barrier()


def build_fsdp_model(
    WanPi3Flow,
    fsdp_kwargs: dict,
    args,
    frozen: bool,
    with_text_projector: bool,
):
    torch.manual_seed(20260818)
    if with_text_projector:
        unwrapped, text_projector = WanPi3Flow.from_pretrained_with_text_projector(
            args.wan,
            max_views=VIEWS,
            head_depth=HEAD_DEPTH,
            latent_dim=LATENT_DIM,
            input_alignment=args.input_align,
        )
    else:
        unwrapped = WanPi3Flow.from_pretrained(
            args.wan,
            max_views=VIEWS,
            head_depth=HEAD_DEPTH,
            latent_dim=LATENT_DIM,
            input_alignment=args.input_align,
        )
        text_projector = None
    unwrapped = unwrapped.float()
    if frozen:
        unwrapped.eval().requires_grad_(False)
    wrapped = FSDP(unwrapped, **fsdp_kwargs)
    del unwrapped
    gc.collect()
    return wrapped, text_projector


def main():
    args = parse_args()
    validate_args(args)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    main_process = rank == 0
    if world_size != 16:
        raise RuntimeError(f"DMD2 training requires the source world size 16, got {world_size}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if main_process:
        os.makedirs(args.out, exist_ok=True)

    from oneworld.models.wan.modules.attention import (
        FLASH_ATTN_2_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
    )
    from oneworld.models.wan.modules.model import WanAttentionBlock
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import opencv_camera_to_plucker
    from oneworld.rae.camera import canonicalize_pi3_cameras
    from oneworld.rae.checkpoint import (
        load_rae_model,
        set_rae_inference,
        wrap_rae_mdf_decoder_fsdp,
    )
    from oneworld.rae.nvs_refined import MixedMultiResolutionRAESampler, NVSRefined
    from oneworld.rae.text_condition import WanTextEmbeddingStore
    from oneworld.rae.wan_flow import (
        DecoupledHeadBlock,
        WanPi3Flow,
        dynamic_timestep_shift,
    )
    from cut3r_data.datasets.realestate10k_torch import RE10K_Torch_Multi

    attention_backend = (
        "external_flash_attention_3"
        if FLASH_ATTN_3_AVAILABLE
        else "external_flash_attention_2"
        if FLASH_ATTN_2_AVAILABLE
        else "torch_scaled_dot_product_attention"
    )
    source_contract = load_source_contract(
        args.source_checkpoint, SOURCE_STEP, world_size
    )
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={WanAttentionBlock, DecoupledHeadBlock},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": mixed_precision,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": device,
        "sync_module_states": True,
        "limit_all_gathers": True,
        "use_orig_params": False,
    }
    initialization_start = time.time()
    student, text_projector = build_fsdp_model(
        WanPi3Flow, fsdp_kwargs, args, frozen=False, with_text_projector=True
    )
    real_score, _ = build_fsdp_model(
        WanPi3Flow, fsdp_kwargs, args, frozen=True, with_text_projector=False
    )
    fake_score, _ = build_fsdp_model(
        WanPi3Flow, fsdp_kwargs, args, frozen=False, with_text_projector=False
    )
    text_projector = text_projector.float().to(device)
    load_text_projector(args.source_checkpoint, text_projector)

    pi3x = Pi3X.from_pretrained(args.pi3).to(device).eval()
    rae, rae_checkpoint = load_rae_model(pi3x, args.rae, device)
    _decoder_modules, decoder_parameters = wrap_rae_mdf_decoder_fsdp(
        rae,
        {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "mixed_precision": mixed_precision,
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "device_id": device,
            "sync_module_states": True,
            "limit_all_gathers": True,
            "use_orig_params": False,
        },
    )
    load_source_weights(
        args.source_checkpoint,
        student,
        real_score,
        fake_score,
        rank,
        world_size,
        decoder_parameters,
    )
    set_rae_inference(rae)

    torch.manual_seed(args.seed)
    stage_embeddings = torch.nn.Parameter(
        torch.randn(4, 1536, device=device) * STAGE_EMBEDDING_INIT
    )
    dist.broadcast(stage_embeddings.data, src=0)
    student_optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.generator_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    fake_optimizer = torch.optim.AdamW(
        list(fake_score.parameters()) + [stage_embeddings],
        lr=args.fake_score_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    resume_path = find_resume(args.out, args.resume, rank)
    if resume_path:
        (
            update_step,
            stage_histogram,
            conditioning_histogram,
            feedback_histogram,
            skipped_invalid_gaussian_batches,
        ) = load_dmd_resume(
            resume_path,
            student,
            fake_score,
            stage_embeddings,
            rank,
            world_size,
            decoder_parameters,
        )
        initialization = f"resume:{resume_path}"
    else:
        update_step = 0
        stage_histogram = [0, 0, 0, 0]
        conditioning_histogram = [0, 0]
        feedback_histogram = [0, 0]
        skipped_invalid_gaussian_batches = 0
        initialization = f"source_ema:{args.source_checkpoint}"
    statistics = torch.load(args.stats, map_location="cpu", weights_only=False)
    latent_statistics = resolution_stats(
        statistics, device, IMAGE_HEIGHT, IMAGE_WIDTH
    )
    if int(statistics.get("rae_step", -1)) != int(rae_checkpoint["step"]):
        raise RuntimeError("RAE and latent statistics step mismatch")
    text_store = WanTextEmbeddingStore(args.text_store)

    remaining_updates = max(0, args.updates - update_step)
    reserve_updates = max(128, math.ceil(remaining_updates * 0.01))
    re10k_dataset = RE10K_Torch_Multi(
        ROOT=args.re10k_root,
        split="train",
        num_views=16,
        resolution=[(IMAGE_WIDTH, IMAGE_HEIGHT)],
        ordered_views=True,
        aug_crop=False,
        seed=20260906 + rank * 1000003,
        min_interval=1,
        max_interval=128,
    )
    refined_dataset = NVSRefined(
        args.data_root,
        subsets=("ACID", "DL3DV", "Re10K"),
    )
    samples = MixedMultiResolutionRAESampler(
        re10k_dataset,
        refined_dataset,
        ((IMAGE_HEIGHT, IMAGE_WIDTH),),
        context_views=8,
        target_views=8,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank,
        length=(remaining_updates + reserve_updates) * args.batch_size,
        start_step=SOURCE_STEP + update_step,
        seed=18000017,
        min_span=48,
        max_span=160,
        return_metadata=True,
    )
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2 if args.workers else None,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    flow_shift = dynamic_timestep_shift(
        IMAGE_HEIGHT // 14,
        IMAGE_WIDTH // 14,
        VIEWS,
        latent_dim=LATENT_DIM,
        shift_base=SHIFT_BASE,
    )
    student_sigmas = build_student_sigmas(
        flow_shift, DEFAULT_STAGE_INDICES, BASE_SAMPLING_STEPS, device
    )
    generator_target_updates, fake_target_updates = completed_update_counts(args.updates)
    if main_process:
        print(
            "DMD2 contract: total_updates="
            f"{args.updates} generator={generator_target_updates} fake_score={fake_target_updates} "
            "cycle=GFFFF random_rollout_depth=1..4 GAN=false",
            flush=True,
        )
        print(
            f"student_sigmas={[round(float(value), 6) for value in student_sigmas]} "
            f"shift={flow_shift:.6f} source=100K_EMA real_score=frozen "
            "generator_loss=3DGS_render_reencoded_views1..7_all1056 "
            "fake_loss=3DGS_render_reencoded_views1..7_all1056",
            flush=True,
        )
        print(
            f"world={world_size} batch/gpu={args.batch_size} camera_cfg={CAMERA_CFG:g} "
            f"text_cfg={TEXT_CFG:g} attention={attention_backend} "
            f"conditioning=image_camera:{TEXT_DROPOUT:.2f},"
            f"image_camera_text:{1.0-TEXT_DROPOUT:.2f} "
            f"feedback=context:{1.0-NOVEL_FEEDBACK_PROB:.2f},"
            f"heldout_nvs:{NOVEL_FEEDBACK_PROB:.2f} "
            f"gradient_checkpointing={args.gradient_checkpointing} "
            f"initialization={initialization} init_sec={time.time()-initialization_start:.1f}",
            flush=True,
        )

    torch.manual_seed(args.seed + rank * 1000003)
    np.random.seed(args.seed + rank * 1000003)
    student.train()
    fake_score.train()
    real_score.eval()
    last_log = time.time()
    metrics_sum = {
        "generator_loss": 0.0,
        "fake_loss": 0.0,
        "fake_geometry_mse": 0.0,
        "fake_rgb32_mse": 0.0,
        "score_sigma": 0.0,
        "local_interval": 0.0,
        "gradient_norm": 0.0,
    }
    metric_count = 0
    skipped_invalid_camera_batches = 0
    torch.cuda.reset_peak_memory_stats(device)

    for loader_batch in loader:
        if update_step >= args.updates:
            break
        (
            all_images,
            all_c2w,
            all_intrinsics,
            _frame_ids,
            _refined_source,
            _resolution_index,
            scene_keys,
            _captions,
        ) = loader_batch
        images = all_images[:, :8].to(device, non_blocking=True)
        all_c2w = all_c2w.to(device, non_blocking=True).float()
        all_intrinsics = all_intrinsics.to(device, non_blocking=True).float()
        source_slots = torch.zeros(images.shape[0], device=device, dtype=torch.long)
        (
            images,
            context_c2w,
            context_intrinsics,
            target_c2w,
            target_intrinsics,
            _,
            camera_valid,
            camera_scale,
        ) = prepare_dmd_camera_batch(
            images,
            all_c2w,
            all_intrinsics,
            source_slots,
            canonicalize_pi3_cameras,
            return_valid=True,
        )
        invalid_camera_batch = torch.tensor(
            int(bool((~camera_valid).any())), device=device, dtype=torch.int32
        )
        dist.all_reduce(invalid_camera_batch, op=dist.ReduceOp.MAX)
        if bool(invalid_camera_batch):
            skipped_invalid_camera_batches += 1
            if bool((~camera_valid).any()):
                print(
                    f"rank={rank} skipped invalid camera before update={update_step} "
                    f"scale={camera_scale[~camera_valid].tolist()}",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reference_raw = rae.encode_latent(
                images[:, :1], context_c2w[:, :1], context_intrinsics[:, :1]
            )["latent"]
        reference = (
            reference_raw.float() - latent_statistics["ref_mean"]
        ) / latent_statistics["ref_std"]
        batch, _, channels, latent_height, latent_width = reference.shape
        if channels != LATENT_DIM:
            raise RuntimeError(f"unexpected latent channels: {channels}")
        condition = torch.zeros(
            batch,
            VIEWS,
            channels,
            latent_height,
            latent_width,
            device=device,
            dtype=reference.dtype,
        )
        condition[:, :1] = reference
        condition_mask = torch.zeros(
            batch,
            VIEWS,
            1,
            latent_height,
            latent_width,
            device=device,
            dtype=reference.dtype,
        )
        condition_mask[:, :1] = 1
        plucker = latent_plucker(
            context_c2w,
            context_intrinsics,
            latent_height,
            latent_width,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            opencv_camera_to_plucker,
        ).to(reference.dtype)

        global_sample_base = update_step * world_size * batch + rank * batch
        long_choice = (
            torch.arange(batch, device=device) + global_sample_base
        ) % 2 == 0
        variants = [
            "long_caption" if choice else "short_caption"
            for choice in long_choice.tolist()
        ]
        text_present = sample_text_present(
            batch, TEXT_DROPOUT, device
        )
        raw_text_cpu, text_lens_cpu = text_store.load_batch(
            scene_keys, variants, text_present.tolist()
        )
        raw_null_cpu, null_lens_cpu = text_store.load_batch(
            scene_keys, variants, [False] * batch
        )
        raw_text = raw_text_cpu.pin_memory().to(device, non_blocking=True)
        raw_null = raw_null_cpu.pin_memory().to(device, non_blocking=True)
        text_lens = text_lens_cpu.pin_memory().to(device, non_blocking=True)
        null_lens = null_lens_cpu.pin_memory().to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            text_context = text_projector(raw_text).float()
            null_context = text_projector(raw_null).float()

        novel_feedback = (
            torch.rand(batch, device=device) < NOVEL_FEEDBACK_PROB
        )
        target_offsets = (
            torch.arange(batch, device=device) + global_sample_base
        ).remainder(VIEWS).long()
        feedback_c2w, feedback_intrinsics = select_feedback_cameras(
            context_c2w,
            context_intrinsics,
            target_c2w,
            target_intrinsics,
            novel_feedback,
            target_offsets,
        )
        feedback_plucker = latent_plucker(
            feedback_c2w,
            feedback_intrinsics,
            latent_height,
            latent_width,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            opencv_camera_to_plucker,
        ).to(reference.dtype)

        def context_feedback_function(value):
            return render_encode_3dgs_feedback(
                rae,
                value,
                context_c2w,
                context_intrinsics,
                context_c2w,
                context_intrinsics,
                latent_statistics["target_mean"],
                latent_statistics["target_std"],
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                False,
            )

        target_stage = synchronized_stage(device, student_sigmas.numel())
        mode_counts = torch.stack(
            (
                (~text_present).sum(),
                text_present.sum(),
                (~novel_feedback).sum(),
                novel_feedback.sum(),
            )
        ).to(torch.int64)
        dist.all_reduce(mode_counts)
        phase = optimizer_kind(update_step)
        score_sigma, local_interval = sample_score_sigmas(
            batch,
            target_stage,
            student_sigmas,
            SCORE_MINIMUM_SIGMA,
            SCORE_LOCAL_INTERVAL_PROB,
        )
        score_sigma_view = score_sigma.reshape(batch, 1, 1, 1, 1)

        student_optimizer.zero_grad(set_to_none=True)
        fake_optimizer.zero_grad(set_to_none=True)
        generator_loss = torch.zeros((), device=device)
        fake_loss = torch.zeros((), device=device)
        fake_geometry_mse = torch.zeros((), device=device)
        fake_rgb32_mse = torch.zeros((), device=device)

        if phase == "generator":
            generated_x0 = rollout_student(
                student,
                target_stage,
                student_sigmas,
                condition,
                plucker,
                condition_mask,
                text_context,
                text_lens,
                context_feedback_function,
                True,
                args.gradient_checkpointing,
            )
            if generated_x0 is None:
                student_optimizer.zero_grad(set_to_none=True)
                fake_optimizer.zero_grad(set_to_none=True)
                skipped_invalid_gaussian_batches += 1
                if main_process:
                    print(
                        f"skipped invalid 3DGS batch before update={update_step} "
                        f"phase={phase} stage={target_stage}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            generated_x0, _ = render_encode_3dgs_feedback(
                rae,
                generated_x0,
                context_c2w,
                context_intrinsics,
                feedback_c2w,
                feedback_intrinsics,
                latent_statistics["target_mean"],
                latent_statistics["target_std"],
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                True,
            )
            if generated_x0 is None:
                student_optimizer.zero_grad(set_to_none=True)
                fake_optimizer.zero_grad(set_to_none=True)
                skipped_invalid_gaussian_batches += 1
                if main_process:
                    print(
                        f"skipped invalid 3DGS batch before update={update_step} "
                        f"phase={phase} stage={target_stage}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            noisy_score = (
                (1.0 - score_sigma_view) * generated_x0.detach()
                + score_sigma_view * torch.randn_like(generated_x0)
            )
            fake_text, fake_lens = add_stage_token(
                text_context,
                text_lens,
                stage_embeddings,
                target_stage,
                detach=True,
            )
            with torch.no_grad():
                real_x0 = guided_real_score_x0(
                    real_score,
                    noisy_score,
                    condition,
                    feedback_plucker,
                    condition_mask,
                    score_sigma,
                    text_context,
                    text_lens,
                    null_context,
                    null_lens,
                    CAMERA_CFG,
                    TEXT_CFG,
                    text_present,
                )
                fake_x0 = model_x0(
                    fake_score,
                    noisy_score,
                    condition,
                    feedback_plucker,
                    condition_mask,
                    score_sigma,
                    fake_text,
                    fake_lens,
                    "none",
                )
                generator_target = direct_x0_dmd_target(
                    generated_x0, real_x0, fake_x0
                )
                normalization = (
                    generated_x0[:, 1:].detach() - real_x0[:, 1:]
                ).abs().mean(dim=(1, 2, 3, 4)).clamp_min(1e-4).reciprocal()
                normalization = normalization.clamp(max=DMD_WEIGHT_MAX)
            feedback_error = (
                generated_x0[:, 1:]
                - generator_target[:, 1:]
            ).square().mean(dim=(1, 2, 3, 4))
            generator_loss = (feedback_error * normalization).mean()
            collective_finite(generator_loss, "generator DMD loss")
            generator_loss.backward()
            assert_gradient_route(
                phase, student, fake_score, real_score, stage_embeddings, device
            )
            gradient_norm = student.clip_grad_norm_(args.grad_clip).float()
            generator_updates, _ = completed_update_counts(update_step)
            current_lr = learning_rate(
                generator_updates,
                generator_target_updates,
                args.generator_lr,
                args.generator_final_lr,
                args.generator_warmup_updates,
            )
            for group in student_optimizer.param_groups:
                group["lr"] = current_lr
            collective_finite(gradient_norm, "generator gradient")
            student_optimizer.step()
            student_optimizer.zero_grad(set_to_none=True)
        else:
            generated_x0 = rollout_student(
                student,
                target_stage,
                student_sigmas,
                condition,
                plucker,
                condition_mask,
                text_context,
                text_lens,
                context_feedback_function,
                False,
                "none",
            )
            if generated_x0 is None:
                student_optimizer.zero_grad(set_to_none=True)
                fake_optimizer.zero_grad(set_to_none=True)
                skipped_invalid_gaussian_batches += 1
                if main_process:
                    print(
                        f"skipped invalid 3DGS batch before update={update_step} "
                        f"phase={phase} stage={target_stage}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            with torch.no_grad():
                generated_x0, _ = render_encode_3dgs_feedback(
                    rae,
                    generated_x0,
                    context_c2w,
                    context_intrinsics,
                    feedback_c2w,
                    feedback_intrinsics,
                    latent_statistics["target_mean"],
                    latent_statistics["target_std"],
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    True,
                )
            if generated_x0 is None:
                student_optimizer.zero_grad(set_to_none=True)
                fake_optimizer.zero_grad(set_to_none=True)
                skipped_invalid_gaussian_batches += 1
                if main_process:
                    print(
                        f"skipped invalid 3DGS batch before update={update_step} "
                        f"phase={phase} stage={target_stage}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            generated_x0 = generated_x0.detach()
            noisy_score = (
                (1.0 - score_sigma_view) * generated_x0
                + score_sigma_view * torch.randn_like(generated_x0)
            )
            fake_text, fake_lens = add_stage_token(
                text_context,
                text_lens,
                stage_embeddings,
                target_stage,
                detach=False,
            )
            fake_x0 = model_x0(
                fake_score,
                noisy_score,
                condition,
                feedback_plucker,
                condition_mask,
                score_sigma,
                fake_text,
                fake_lens,
                args.gradient_checkpointing,
            )
            geometry_per_scene = (
                fake_x0[:, 1:, :GEOMETRY_DIM] - generated_x0[:, 1:, :GEOMETRY_DIM]
            ).square().mean(dim=(1, 2, 3, 4))
            rgb32_per_scene = (
                fake_x0[:, 1:, GEOMETRY_DIM:] - generated_x0[:, 1:, GEOMETRY_DIM:]
            ).square().mean(dim=(1, 2, 3, 4))
            full_per_scene = (
                fake_x0[:, 1:] - generated_x0[:, 1:]
            ).square().mean(dim=(1, 2, 3, 4))
            x0_weight = score_sigma.square().clamp_min(1e-8).reciprocal().clamp(
                max=FAKE_MAX_X0_WEIGHT
            )
            fake_loss = (full_per_scene * x0_weight).mean()
            fake_geometry_mse = geometry_per_scene.mean()
            fake_rgb32_mse = rgb32_per_scene.mean()
            collective_finite(fake_loss, "fake-score loss")
            fake_loss.backward()
            if stage_embeddings.grad is not None:
                dist.all_reduce(stage_embeddings.grad)
                stage_embeddings.grad.div_(world_size)
            assert_gradient_route(
                phase, student, fake_score, real_score, stage_embeddings, device
            )
            model_gradient_norm = fake_score.clip_grad_norm_(args.grad_clip).float()
            stage_gradient_norm = torch.nn.utils.clip_grad_norm_(
                [stage_embeddings], args.grad_clip
            ).float()
            gradient_norm = torch.maximum(model_gradient_norm, stage_gradient_norm)
            _, fake_updates = completed_update_counts(update_step)
            current_lr = learning_rate(
                fake_updates,
                fake_target_updates,
                args.fake_score_lr,
                args.fake_score_final_lr,
                args.fake_score_warmup_updates,
            )
            for group in fake_optimizer.param_groups:
                group["lr"] = current_lr
            collective_finite(gradient_norm, "fake-score gradient")
            fake_optimizer.step()
            fake_optimizer.zero_grad(set_to_none=True)

        stage_histogram[target_stage] += 1
        conditioning_histogram[0] += int(mode_counts[0])
        conditioning_histogram[1] += int(mode_counts[1])
        feedback_histogram[0] += int(mode_counts[2])
        feedback_histogram[1] += int(mode_counts[3])
        update_step += 1
        reduced = torch.stack(
            (
                generator_loss.detach(),
                fake_loss.detach(),
                fake_geometry_mse.detach(),
                fake_rgb32_mse.detach(),
                score_sigma.mean(),
                local_interval.float().mean(),
                gradient_norm.detach(),
            )
        ).double()
        dist.all_reduce(reduced)
        reduced.div_(world_size)
        for name, value in zip(metrics_sum, reduced.tolist()):
            metrics_sum[name] += value
        metric_count += 1

        if main_process and (
            update_step % args.log_every == 0 or update_step == 1
        ):
            elapsed = time.time() - last_log
            generator_updates, fake_updates = completed_update_counts(update_step)
            averages = {
                name: value / metric_count for name, value in metrics_sum.items()
            }
            print(
                f"update={update_step:06d}/{args.updates} "
                f"gen={generator_updates:05d} fake={fake_updates:05d} "
                f"phase={phase} depth={target_stage + 1} "
                f"stage_hist={stage_histogram} sigma={averages['score_sigma']:.4f} "
                f"conditioning_hist={conditioning_histogram} "
                f"feedback_hist={feedback_histogram} "
                f"local_frac={averages['local_interval']:.3f} "
                f"loss_g={averages['generator_loss']:.6f} "
                f"loss_fake={averages['fake_loss']:.6f} "
                f"fake_geo={averages['fake_geometry_mse']:.6f} "
                f"fake_rgb32={averages['fake_rgb32_mse']:.6f} "
                f"grad={averages['gradient_norm']:.4f} lr={current_lr:.2e} "
                f"sec/update={elapsed/metric_count:.3f} "
                f"peak_GiB={torch.cuda.max_memory_allocated(device)/2**30:.2f}",
                flush=True,
            )
            metrics_sum = {name: 0.0 for name in metrics_sum}
            metric_count = 0
            last_log = time.time()

        save_now = update_step % args.checkpoint_every == 0
        if save_now or update_step >= args.updates:
            slot = (
                (update_step // args.checkpoint_every) % args.checkpoint_slots
                if save_now
                else "resume"
            )
            save_checkpoint(
                args,
                slot,
                update_step,
                student,
                fake_score,
                stage_embeddings,
                stage_histogram,
                conditioning_histogram,
                feedback_histogram,
                skipped_invalid_gaussian_batches,
                source_contract,
                rank,
                decoder_parameters,
            )
            if main_process:
                print(f"checkpoint update={update_step} slot={slot}", flush=True)

    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device), device=device, dtype=torch.float64
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    generator_updates, fake_updates = completed_update_counts(update_step)
    if main_process:
        print(
            f"training complete updates={update_step}/{args.updates} "
            f"generator={generator_updates} fake_score={fake_updates} "
            f"stage_hist={stage_histogram} "
            f"conditioning_hist={conditioning_histogram} "
            f"feedback_hist={feedback_histogram} "
            f"skipped_invalid_camera_batches={skipped_invalid_camera_batches} "
            f"skipped_invalid_gaussian_batches={skipped_invalid_gaussian_batches} "
            f"max_peak_GiB={float(peak)/2**30:.2f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
