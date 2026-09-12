#!/usr/bin/env python3
"""Train the OneWorld diffusion transformer for 100K steps."""

from __future__ import annotations

import argparse
import copy
import functools
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
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRAIN_RESOLUTIONS = ((224, 448),)
HEAD_DEPTH = 2
LATENT_DIM = 1056
CVC_THRESHOLD = 0.9
CVC_WEIGHT = 0.2
CVC_TEMPERATURE = 0.07


def cross_view_correspondence_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    threshold: float = CVC_THRESHOLD,
    temperature: float = CVC_TEMPERATURE,
):
    """Token correspondence objective from target views to the source view."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if condition.shape[:2] != (target.shape[0], 1):
        raise ValueError("condition must contain exactly one source view")
    predicted_tokens = prediction[:, 1:].permute(0, 1, 3, 4, 2).flatten(0, 1)
    target_tokens = target[:, 1:].permute(0, 1, 3, 4, 2).flatten(0, 1)
    source_tokens = condition.permute(0, 1, 3, 4, 2)[:, 0]
    source_tokens = source_tokens[:, None].expand(
        -1, target.shape[1] - 1, -1, -1, -1).flatten(0, 1)
    predicted_tokens = F.normalize(predicted_tokens.flatten(1, 2), dim=-1)
    target_tokens = F.normalize(target_tokens.flatten(1, 2), dim=-1)
    source_tokens = F.normalize(source_tokens.flatten(1, 2), dim=-1)
    with torch.no_grad():
        target_similarity = torch.bmm(
            target_tokens.float(), source_tokens.float().transpose(1, 2))
        confidence, labels = target_similarity.max(dim=-1)
        selected = confidence >= threshold
    logits = torch.bmm(
        predicted_tokens.float(), source_tokens.float().transpose(1, 2)
    ) / temperature
    token_loss = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    selected_count = selected.sum()
    loss = (token_loss * selected).sum() / selected_count.clamp_min(1)
    return loss, selected.float().mean()


def prepare_camera_batch(images, c2w, intrinsics, source_slots,
                         camera_canonicalizer, return_valid=False):
    """Move the source to slot zero and canonicalize the eight training views."""
    if images.ndim != 5 or images.shape[1] != 8:
        raise ValueError(f"expected E8 images, got {tuple(images.shape)}")
    if c2w.shape[1:] != (16, 4, 4) or intrinsics.shape[1:] != (8, 3, 3):
        raise ValueError(
            f"invalid camera batch: c2w={tuple(c2w.shape)} K={tuple(intrinsics.shape)}")
    if source_slots.shape != (images.shape[0],):
        raise ValueError(f"invalid source slots: {tuple(source_slots.shape)}")
    if bool(((source_slots < 0) | (source_slots >= 8)).any()):
        raise ValueError(f"source slot outside [0,7]: {source_slots.tolist()}")
    device = images.device
    context_base = torch.arange(8, device=device)
    context_order = torch.stack([
        torch.cat((slot[None], context_base[context_base != slot]))
        for slot in source_slots
    ])
    heldout_order = torch.arange(8, 16, device=device)[None].expand(
        images.shape[0], -1)
    all_order = torch.cat((context_order, heldout_order), dim=1)
    images = images.gather(
        1, context_order[:, :, None, None, None].expand_as(images))
    intrinsics = intrinsics.gather(
        1, context_order[:, :, None, None].expand_as(intrinsics))
    c2w = c2w.gather(1, all_order[:, :, None, None].expand_as(c2w))
    canonical = camera_canonicalizer(
        c2w, reference_index=0, scale_view_count=images.shape[1],
        strict=not return_valid)
    c2w = canonical.c2w[:, :8]
    result = (images, c2w, intrinsics, context_order)
    if return_valid:
        result += (canonical.valid, canonical.scale)
    return result


def rotation_span_deg(c2w, reference_index):
    relative = torch.linalg.inv(c2w[reference_index]) @ c2w
    trace = relative[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine)).max()


def latent_plucker(c2w, intrinsics, latent_height, latent_width,
                   image_height, image_width, geometry_function):
    batch, views = c2w.shape[:2]
    latent_intrinsics = intrinsics.clone()
    latent_intrinsics[..., 0, :] *= latent_width / image_width
    latent_intrinsics[..., 1, :] *= latent_height / image_height
    rays = geometry_function(
        c2w.reshape(batch * views, 4, 4),
        latent_intrinsics.reshape(batch * views, 3, 3),
        latent_height, latent_width)
    return rays.reshape(
        batch, views, latent_height, latent_width, 6).permute(0, 1, 4, 2, 3)


def learning_rate(step: int, args) -> float:
    """Keep the base LR through 10k, then cosine-decay to the final step."""
    if step < args.decay_start:
        return args.lr
    progress = (step - args.decay_start) / max(
        1, args.steps - 1 - args.decay_start)
    progress = min(max(progress, 0.0), 1.0)
    return args.final_lr + 0.5 * (args.lr - args.final_lr) * (
        1.0 + math.cos(math.pi * progress))


def resolution_stats(all_stats, device, height, width):
    expected = {
        "name": "v19_ln_z18_concat_ln_rgb32",
        "rae_contract_version": 19,
        "channels": LATENT_DIM,
        "geometry_channels": 1024,
        "appearance_channels": 32,
        "boundary_block_zero_based": 17,
        "resolutions_hw": [list(value) for value in TRAIN_RESOLUTIONS],
        "target_views": 8,
        "reference_views": 1,
        "camera": (
            "pi3_native_ref0_context8_mean_baseline_scale;"
            "heldout_NVS_poses_excluded"),
    }
    contract = all_stats.get("latent_contract", {})
    mismatch = {
        key: (contract.get(key), value)
        for key, value in expected.items() if contract.get(key) != value
    }
    if (all_stats.get("format")
            != "pi3-v19-separate-ln-channel-stats-v1" or mismatch):
        raise ValueError(f"incompatible latent statistics: {mismatch}")
    key, channels = f"{height}x{width}", LATENT_DIM
    values = all_stats["by_resolution"][key]
    shaped = {}
    for name in ("target_mean", "target_std", "ref_mean", "ref_std"):
        if values[name].shape != (channels,) or not bool(torch.isfinite(values[name]).all()):
            raise ValueError(f"invalid {name} in latent statistics")
        shaped[name] = values[name].to(device).reshape(1, 1, -1, 1, 1)
    shaped["target_std"] = shaped["target_std"].clamp_min(1e-6)
    shaped["ref_std"] = shaped["ref_std"].clamp_min(1e-6)
    return shaped


@torch.no_grad()
def update_fsdp_ema(ema_model, model, decay: float):
    """Update matching FP32 local FSDP shards without materialising full weights."""
    ema_parameters = list(ema_model.parameters())
    model_parameters = list(model.parameters())
    if len(ema_parameters) != len(model_parameters):
        raise RuntimeError("model/EMA FSDP layouts differ")
    for ema_parameter, parameter in zip(ema_parameters, model_parameters):
        if ema_parameter.shape != parameter.shape:
            raise RuntimeError(
                f"model/EMA shard mismatch: {ema_parameter.shape} != {parameter.shape}")
        ema_parameter.mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)


@torch.no_grad()
def update_module_ema(ema_module, module, decay: float):
    source = module.module if isinstance(module, DDP) else module
    for ema_parameter, parameter in zip(
            ema_module.parameters(), source.parameters(), strict=True):
        ema_parameter.mul_(decay).add_(
            parameter.detach().float(), alpha=1.0 - decay)


@torch.no_grad()
def _module_fingerprint(module):
    source = module.module if isinstance(module, DDP) else module
    result = torch.zeros(3, dtype=torch.float64)
    for parameter in source.parameters():
        value = parameter.detach().cpu().double()
        result[0] += value.numel()
        result[1] += value.sum()
        result[2] += value.square().sum()
    return result


@torch.no_grad()
def _tensor_fingerprint(tensors, device):
    """Global [number of values, sum, squared sum] over rank-local shards."""
    result = torch.zeros(3, device=device, dtype=torch.float64)
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach().double()
        result[0] += value.numel()
        result[1] += value.sum()
        result[2] += value.square().sum()
    dist.all_reduce(result)
    return result.cpu()


def _checkpoint_fingerprints(model, ema_model):
    device = next(model.parameters()).device
    return {
        "model": _tensor_fingerprint(model.parameters(), device),
        "ema": _tensor_fingerprint(ema_model.parameters(), device),
    }


def _copy_tensor_shards(parameters, saved, label):
    parameters = list(parameters)
    if len(parameters) != len(saved):
        raise RuntimeError(
            f"{label} shard count mismatch: {len(parameters)} != {len(saved)}")
    with torch.no_grad():
        for index, (parameter, source) in enumerate(zip(parameters, saved)):
            if parameter.shape != source.shape:
                raise RuntimeError(
                    f"{label} shard {index} shape mismatch: "
                    f"{tuple(parameter.shape)} != {tuple(source.shape)}")
            parameter.copy_(source.to(parameter.device, dtype=parameter.dtype))


def _copy_parameter_shards(module, saved, label):
    _copy_tensor_shards(module.parameters(), saved, label)


def _verify_fingerprints(expected, actual):
    for name in expected:
        if name not in actual:
            raise RuntimeError(f"checkpoint fingerprint {name} is missing")
        if not torch.equal(expected[name], actual[name]):
            raise RuntimeError(
                f"checkpoint {name} fingerprint mismatch: "
                f"expected={expected[name].tolist()} actual={actual[name].tolist()}")


def save_checkpoint(out_dir, slot, step, model, ema_model, args, rank,
                    text_projector, text_projector_ema, text_contract,
                    decoder_parameters=None):
    """Atomically write same-world-size rank-local FSDP shards."""
    world_size = dist.get_world_size()
    dist.barrier()
    fingerprints = _checkpoint_fingerprints(model, ema_model)
    checkpoint_root = os.path.join(out_dir, "checkpoints")
    final_path = os.path.join(checkpoint_root, f"slot{slot}")
    temporary_path = os.path.join(
        checkpoint_root, f".slot{slot}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}")
    if rank == 0:
        os.makedirs(checkpoint_root, exist_ok=True)
        if os.path.exists(temporary_path):
            shutil.rmtree(temporary_path)
    dist.barrier()
    checkpoint_format = "rank-local-fsdp-weights-ema-v2"
    payload = {
        "format": checkpoint_format,
        "rank": rank,
        "world_size": world_size,
        "step": step,
        "model": [parameter.detach().cpu() for parameter in model.parameters()],
        "ema": [parameter.detach().cpu() for parameter in ema_model.parameters()],
        "fingerprints": fingerprints,
    }
    if decoder_parameters is not None:
        payload["decoder"] = [
            parameter.detach().cpu() for parameter in decoder_parameters]
    rank_file = os.path.join(temporary_path, f"rank{rank:05d}.pt")
    os.makedirs(temporary_path, exist_ok=True)
    torch.save(payload, rank_file + ".tmp")
    os.replace(rank_file + ".tmp", rank_file)
    dist.barrier()
    if rank == 0:
        source = (text_projector.module if isinstance(text_projector, DDP)
                  else text_projector)
        torch.save({
            "format": "wan-native-text-projector-v1",
            "model": {name: value.detach().cpu()
                      for name, value in source.state_dict().items()},
            "ema": {name: value.detach().cpu()
                    for name, value in text_projector_ema.state_dict().items()},
            "model_fingerprint": _module_fingerprint(source),
            "ema_fingerprint": _module_fingerprint(text_projector_ema),
        }, os.path.join(temporary_path, "text_projector.pt"))
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.replace(temporary_path, final_path)
        contract = {
            "step": step,
            "training_target_steps": args.steps,
            "slot": slot,
            "checkpoint_format": checkpoint_format,
            "optimizer_resume": "reset",
            "checkpoint_every_steps": args.ckpt_every,
            "checkpoint_slots": args.checkpoint_slots,
            "world_size": world_size,
            "prediction": "direct_x0",
            "loss": "x0_mse*min(1/sigma^2,100)+0.2*cvc",
            "cvc": {
                "weight": CVC_WEIGHT,
                "threshold": CVC_THRESHOLD,
                "temperature": CVC_TEMPERATURE,
            },
            "timestep_sampling": {
                "distribution": "shifted_uniform_v1",
                "shift_base": args.shift_base,
            },
            "latent_target": "rae_1056",
            "resolutions_hw": [list(value) for value in TRAIN_RESOLUTIONS],
            "patch_grids_hw": [
                [height // 14, width // 14]
                for height, width in TRAIN_RESOLUTIONS],
            "latent_stats": os.path.abspath(args.stats),
            "rae_checkpoint": os.path.abspath(args.rae),
            "dataset": "re10k_nvs_refined",
            "data_mix": "equal_nvs_refined_and_re10k",
            "views": 8,
            "condition_views": 1,
            "batch_per_gpu": args.batch_size,
            "lr": args.lr,
            "decay_start": args.decay_start,
            "final_lr": args.final_lr,
            "ema": args.ema,
            "fsdp": "FULL_SHARD fp32-master bf16-compute",
            "gradient_checkpointing": args.gradient_checkpointing_mode != "none",
            "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
            "attention_backend": args.attention_backend,
            "main": "Wan-1.3B dim1536 depth30",
            "decoupled_head": f"dim2048 depth{HEAD_DEPTH} heads16",
            "latent_token_grid": "Pi3_H/14_x_W/14_preserved_no_unpatchify",
            "prediction_channels_per_token": LATENT_DIM,
            "input_alignment": os.path.abspath(args.input_align),
            "mdf": {
                "enabled": args.mdf,
                "decoder_loss_weight": args.decoder_loss_weight,
                "lpips_weight": args.decoder_lpips_weight,
            },
            "view_interval": [1, 128],
            "condition_source": "first_context_view",
            "camera_normalization": (
                "selected_source_ref0_mean_baseline_over_8_DiT_views;"
                "heldout_NVS_poses_excluded_from_scale"),
            "camera_dropout": args.camera_dropout,
            "cfg_unconditional": "keep_rgb; independently empty_text and zero_plucker",
        }
        contract["text_condition"] = text_contract
        with open(os.path.join(final_path, "contract.json"), "w") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True)
        pointer_tmp = os.path.join(out_dir, "latest.txt.tmp")
        with open(pointer_tmp, "w") as handle:
            handle.write(f"{slot} {step}\n")
        os.replace(pointer_tmp, os.path.join(out_dir, "latest.txt"))
    dist.barrier()


def find_resume(out_dir, requested, rank):
    if requested == "":
        path = ""
    elif requested != "auto":
        path = requested
    elif rank == 0:
        pointer = os.path.join(out_dir, "latest.txt")
        if os.path.exists(pointer):
            slot, _ = open(pointer).read().strip().split()
            path = os.path.join(out_dir, "checkpoints", f"slot{slot}")
        else:
            path = ""
    else:
        path = ""
    values = [path]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def load_checkpoint(path, model, ema_model, text_projector,
                    text_projector_ema, decoder_parameters=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_file = os.path.join(path, f"rank{rank:05d}.pt")
    if not os.path.isfile(rank_file):
        raise FileNotFoundError(rank_file)
    payload = torch.load(rank_file, map_location="cpu", weights_only=False)
    if payload.get("format") != "rank-local-fsdp-weights-ema-v2":
        raise RuntimeError(f"unsupported checkpoint format in {rank_file}")
    if payload.get("rank") != rank or payload.get("world_size") != world_size:
        raise RuntimeError(
            f"rank/world mismatch in {rank_file}: "
            f"saved rank={payload.get('rank')} world={payload.get('world_size')}, "
            f"current rank={rank} world={world_size}")

    _copy_parameter_shards(model, payload["model"], "model")
    _copy_parameter_shards(ema_model, payload["ema"], "ema")
    if decoder_parameters is not None and "decoder" in payload:
        _copy_tensor_shards(
            decoder_parameters, payload["decoder"], "decoder")
    actual = _checkpoint_fingerprints(model, ema_model)
    _verify_fingerprints(payload["fingerprints"], actual)
    text_path = os.path.join(path, "text_projector.pt")
    if not os.path.isfile(text_path):
        raise FileNotFoundError(text_path)
    text_payload = torch.load(text_path, map_location="cpu", weights_only=False)
    if text_payload.get("format") != "wan-native-text-projector-v1":
        raise RuntimeError("unsupported text projector checkpoint")
    source = (text_projector.module if isinstance(text_projector, DDP)
              else text_projector)
    source.load_state_dict(text_payload["model"], strict=True)
    text_projector_ema.load_state_dict(text_payload["ema"], strict=True)
    if not torch.equal(
            _module_fingerprint(source), text_payload["model_fingerprint"]):
        raise RuntimeError("text projector checkpoint fingerprint mismatch")
    if not torch.equal(
            _module_fingerprint(text_projector_ema),
            text_payload["ema_fingerprint"]):
        raise RuntimeError("text projector EMA checkpoint fingerprint mismatch")
    if rank == 0:
        print(
            f"resume verification passed step={payload['step']} "
            f"format={payload['format']} world={world_size} optimizer_reset=true",
            flush=True)
    return int(payload["step"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the OneWorld latent flow-matching generator.")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--data-root", required=True,
        help="NVS-Refined root containing the released parquet shards")
    parser.add_argument(
        "--re10k-root", required=True,
        help="RealEstate10K .torch shard root containing train/ and index_train.json")
    parser.add_argument("--pi3", required=True, help="Pi3X checkpoint directory")
    parser.add_argument("--wan", required=True, help="Wan2.1-T2V-1.3B directory")
    parser.add_argument(
        "--input-align", required=True,
        help="RAE-to-Wan input projection checkpoint")
    parser.add_argument(
        "--rae", required=True, help="OneWorld RAE checkpoint")
    parser.add_argument(
        "--stats", required=True, help="RAE latent normalization statistics")
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "outputs" / "dit"))
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay-start", type=int, default=10000)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--shift-base", type=int, default=4096)
    parser.add_argument("--max-x0-weight", type=float, default=100.0)
    parser.add_argument("--camera-dropout", type=float, default=0.1)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--text-store", required=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema", type=float, default=0.9995)
    parser.add_argument("--mdf", action="store_true")
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoder-lpips-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--checkpoint-slots", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing-half", action="store_true",
        help="checkpoint alternating blocks (15/30 main and 1/2 decoupled-head blocks)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gradient_checkpointing and args.gradient_checkpointing_half:
        raise ValueError(
            "choose either full or half gradient checkpointing, not both")
    args.gradient_checkpointing_mode = (
        "full" if args.gradient_checkpointing else
        "half" if args.gradient_checkpointing_half else "none")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size not in (1, 2, 4, 8):
        raise ValueError("supported per-GPU batch sizes are 1, 2, 4 and 8")
    if args.ckpt_every < 1 or args.checkpoint_slots < 1:
        raise ValueError("checkpoint cadence and slot count must be positive")
    if not 0.0 <= args.camera_dropout < 1.0:
        raise ValueError("camera-dropout must be in [0, 1)")
    if not 0.0 <= args.text_dropout < 1.0:
        raise ValueError("text-dropout must be in [0, 1)")
    if args.decay_start >= args.steps:
        raise ValueError("decay-start must be before the final step")
    if args.decoder_loss_weight < 0 or args.decoder_lpips_weight < 0:
        raise ValueError("decoder loss weights must be non-negative")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    main_process = rank == 0
    if (world_size * args.batch_size) % 2:
        raise ValueError("exact 1:1 long/short mixing requires an even global batch")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for path in (args.pi3, args.wan, args.rae, args.stats):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if not os.path.isfile(args.input_align):
        raise FileNotFoundError(args.input_align)
    if not os.path.isfile(args.text_store):
        raise FileNotFoundError(args.text_store)
    if main_process:
        os.makedirs(args.out, exist_ok=True)

    from oneworld.models.wan.modules.model import WanAttentionBlock
    from oneworld.models.wan.modules.attention import (
        FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE)
    args.attention_backend = (
        "external_flash_attention_3" if FLASH_ATTN_3_AVAILABLE else
        "external_flash_attention_2" if FLASH_ATTN_2_AVAILABLE else
        "torch_scaled_dot_product_attention")
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import opencv_camera_to_plucker
    from oneworld.rae.camera import canonicalize_pi3_cameras
    from oneworld.rae.checkpoint import (
        load_latent_encoder,
        load_rae_model,
        wrap_rae_mdf_decoder_fsdp,
    )
    from oneworld.rae.model import RAELatentEncoder
    from oneworld.rae.text_condition import WanTextEmbeddingStore
    from oneworld.rae.wan_flow import (
        DecoupledHeadBlock,
        WanPi3Flow,
        dynamic_timestep_shift,
        sample_x0_training_timestep,
        wan_flow_sample,
    )
    from cut3r_data.datasets.realestate10k_torch import RE10K_Torch_Multi

    pi3x = Pi3X.from_pretrained(args.pi3).to(device).eval()
    if args.mdf:
        rae, encoder_checkpoint = load_rae_model(
            pi3x, args.rae, device)
        encoder = rae
    else:
        rae = None
        encoder = RAELatentEncoder(pi3x, max_views=8).to(device).eval()
        encoder_checkpoint = load_latent_encoder(encoder, args.rae)
    encoder.requires_grad_(False)

    torch.manual_seed(20260818)
    model_unwrapped, text_projector_unwrapped = (
        WanPi3Flow.from_pretrained_with_text_projector(
            args.wan, max_views=8, head_depth=HEAD_DEPTH,
            latent_dim=LATENT_DIM, input_alignment=args.input_align))
    model_unwrapped = model_unwrapped.float()
    text_projector_unwrapped = text_projector_unwrapped.float().to(device)
    text_projector_ema = copy.deepcopy(
        text_projector_unwrapped).eval().requires_grad_(False)
    text_projector = DDP(
        text_projector_unwrapped, device_ids=[local_rank],
        broadcast_buffers=False, gradient_as_bucket_view=True)
    text_parameters = sum(
        parameter.numel() for parameter in text_projector.parameters())
    total_parameters = sum(p.numel() for p in model_unwrapped.parameters())
    adapter_parameters = sum(
        p.numel() for p in model_unwrapped.adapter_parameters())
    ema_unwrapped = copy.deepcopy(model_unwrapped).eval()
    ema_unwrapped.requires_grad_(False)

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={WanAttentionBlock, DecoupledHeadBlock})
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        # Do not blanket-cast nested FSDP inputs: WanAttentionBlock requires its
        # AdaLN timestep tensor to remain FP32. Tensor-heavy paths are still BF16
        # through the surrounding autocast and BF16 gathered compute parameters.
        cast_forward_inputs=False)
    fsdp_kwargs = dict(
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=False)
    model = FSDP(model_unwrapped, **fsdp_kwargs)
    ema_model = FSDP(ema_unwrapped, **fsdp_kwargs)
    ema_model.eval()

    decoder_modules, decoder_parameters = [], None
    decoder_lpips = None
    if args.mdf:
        decoder_modules, decoder_parameters = wrap_rae_mdf_decoder_fsdp(
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
        for module in decoder_modules:
            module.train()
        from oneworld.losses import LPIPS
        decoder_lpips = LPIPS().to(device).eval().requires_grad_(False)

    optimizer_parameters = list(model.parameters())
    optimizer_parameters.extend(text_projector.parameters())
    if decoder_parameters is not None:
        optimizer_parameters.extend(decoder_parameters)
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay)
    resume_path = find_resume(args.out, args.resume, rank)
    if resume_path:
        start_step = load_checkpoint(
            resume_path, model, ema_model, text_projector, text_projector_ema,
            decoder_parameters=decoder_parameters)
        initialization = f"resume:{resume_path}"
    else:
        if args.mdf:
            raise RuntimeError("MDF joint training requires a Stage-1 checkpoint")
        start_step = 0
        initialization = "wan_pretrained"

    text_store = WanTextEmbeddingStore(args.text_store)
    manifest = text_store.manifest
    text_contract = {
        "required": True,
        "format": manifest["format"],
        "manifest": os.path.abspath(args.text_store),
        "caption_sha256": manifest["caption_sha256"],
        "caption_fields": manifest["caption_fields"],
        "training_mix": manifest["training_mix"],
        "t5_checkpoint": manifest["t5_checkpoint"],
        "t5_sha256": manifest["t5_sha256"],
        "embedding_dim": manifest["embedding_dim"],
        "max_length": manifest["max_length"],
        "text_dropout": args.text_dropout,
        "projection": "trainable native Wan 4096->1536 MLP",
    }

    statistics = torch.load(args.stats, map_location="cpu", weights_only=False)
    stats_by_resolution = {
        resolution: resolution_stats(statistics, device, *resolution)
        for resolution in TRAIN_RESOLUTIONS}
    stats_rae_step = int(statistics.get("rae_step", -1))
    if stats_rae_step != int(encoder_checkpoint["step"]):
        raise ValueError(
            f"latent stats/checkpoint step mismatch: {stats_rae_step} != "
            f"{encoder_checkpoint['step']}")
    remaining_steps = max(0, args.steps - start_step)
    # Invalid camera records are rare but must be skipped synchronously by every
    # rank.  Give the finite map-style loader enough deterministic reserve samples
    # for those skips; the optimizer loop still stops at exactly ``args.steps``.
    retry_reserve_steps = (
        max(128, math.ceil(remaining_steps * 0.01))
        if remaining_steps else 0)
    loader_steps = remaining_steps + retry_reserve_steps
    from oneworld.rae.nvs_refined import (
        MixedMultiResolutionRAESampler, NVSRefined)
    re10k_dataset = RE10K_Torch_Multi(
        ROOT=args.re10k_root, split="train", num_views=16,
        resolution=[(width, height) for height, width in TRAIN_RESOLUTIONS],
        ordered_views=True, aug_crop=False,
        seed=20260827 + rank * 1000003,
        min_interval=1, max_interval=128)
    refined_dataset = NVSRefined(
        args.data_root, subsets=("ACID", "DL3DV", "Re10K"))
    samples = MixedMultiResolutionRAESampler(
        re10k_dataset, refined_dataset, TRAIN_RESOLUTIONS,
        context_views=8, target_views=8,
        batch_size=args.batch_size, world_size=world_size, rank=rank,
        length=loader_steps * args.batch_size,
        start_step=start_step, seed=18000017, min_span=48, max_span=160,
        return_metadata=True)
    loader = DataLoader(
        samples, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, prefetch_factor=2,
        persistent_workers=args.workers > 0, drop_last=True)

    if main_process:
        print(
            f"contract: resolutions={TRAIN_RESOLUTIONS} | "
            "latent=RAE-1056 | E1->E8(1->7) | V=8 gap=[1,128] | "
            "prediction=x0 | target/ref normalization | "
            "weight=min(1/sigma^2,100) | resolution-aware shift",
            flush=True)
        print(
            f"FSDP=FULL_SHARD world={world_size} batch/gpu={args.batch_size} "
            f"global_batch={world_size * args.batch_size} "
            f"parameters={total_parameters/1e9:.3f}B "
            f"new_DH_and_adapters={adapter_parameters/1e9:.3f}B "
            f"trainable_text_projection={text_parameters/1e6:.2f}M "
            f"DH_depth={HEAD_DEPTH} "
            f"gradient_checkpointing={args.gradient_checkpointing_mode} "
            f"attention_backend={args.attention_backend}",
            flush=True)
        print(
            f"lr={args.lr:.2e} constant_until={args.decay_start} "
            f"cosine_final={args.final_lr:.2e}@{args.steps} "
            f"ema={args.ema} fp32_sharded=true local_step={start_step} "
            f"initialization={initialization} camera_dropout={args.camera_dropout} "
            f"text_dropout={args.text_dropout} long_short=1:1",
            flush=True)

    torch.manual_seed(20260818 + rank)
    np.random.seed(20260818 + rank)
    model.train()
    text_projector.train()
    step = start_step
    last_log = time.time()
    metrics_sum = {}
    metric_count = 0
    skipped_invalid_camera_batches = 0
    torch.cuda.reset_peak_memory_stats(device)

    for loader_batch in loader:
        if step >= args.steps:
            break
        (all_images, c2w, all_intrinsics, _frame_ids, refined_source,
         _resolution_index, scene_keys, _captions) = loader_batch
        images = all_images[:, :8]
        intrinsics = all_intrinsics[:, :8]
        source_slots = torch.zeros(images.shape[0], dtype=torch.long)
        pose_span_deg = torch.stack([
            rotation_span_deg(cameras, 0) for cameras in c2w])
        images = images.to(device, non_blocking=True)
        c2w = c2w.to(device, non_blocking=True).float()
        intrinsics = intrinsics.to(device, non_blocking=True).float()
        source_slots = source_slots.to(device, non_blocking=True).long()
        pose_span_deg = pose_span_deg.to(device, non_blocking=True).float()
        refined_source = refined_source.to(device, non_blocking=True).float()
        image_height, image_width = images.shape[-2:]
        if ((image_height, image_width) not in TRAIN_RESOLUTIONS):
            raise RuntimeError(
                f"resolution contract violated: {tuple(images.shape)}")
        local_hw = torch.tensor(
            [image_height, image_width], device=device, dtype=torch.int64)
        minimum_hw, maximum_hw = local_hw.clone(), local_hw.clone()
        dist.all_reduce(minimum_hw, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum_hw, op=dist.ReduceOp.MAX)
        if not bool((minimum_hw == maximum_hw).all()):
            raise RuntimeError(
                f"FSDP ranks disagree on resolution: min={minimum_hw.tolist()} "
                f"max={maximum_hw.tolist()}")

        if c2w.shape[1] != 16 or images.shape[1] != 8:
            raise RuntimeError("training requires E8 RGB plus all 16 RAE cameras")
        (images, c2w, intrinsics, _, camera_valid,
         camera_scale) = prepare_camera_batch(
            images, c2w, intrinsics, source_slots,
            canonicalize_pi3_cameras, return_valid=True)
        local_invalid_camera = ~camera_valid
        invalid_camera_batch = torch.tensor(
            int(bool(local_invalid_camera.any())),
            device=device, dtype=torch.int32)
        dist.all_reduce(invalid_camera_batch, op=dist.ReduceOp.MAX)
        if bool(invalid_camera_batch):
            skipped_invalid_camera_batches += 1
            if bool(local_invalid_camera.any()):
                bad_indices = torch.where(local_invalid_camera)[0].tolist()
                bad_scenes = [str(scene_keys[index]) for index in bad_indices]
                bad_scales = camera_scale[local_invalid_camera].tolist()
                print(
                    f"rank={rank} skipped invalid camera batch before "
                    f"step={step}: batch_indices={bad_indices} "
                    f"scenes={bad_scenes} scales={bad_scales}",
                    file=sys.stderr, flush=True)
            continue

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            target_raw = encoder.encode_latent(
                images, c2w, intrinsics)["latent"]
            ref_raw = encoder.encode_latent(
                images[:, :1], c2w[:, :1], intrinsics[:, :1])["latent"]
        current_stats = stats_by_resolution[(image_height, image_width)]
        target = (
            target_raw.float() - current_stats["target_mean"]
        ) / current_stats["target_std"]
        ref = (
            ref_raw.float() - current_stats["ref_mean"]
        ) / current_stats["ref_std"]
        if not bool(torch.isfinite(target).all() and torch.isfinite(ref).all()):
            raise FloatingPointError("non-finite normalized latent")

        batch, views, channels, latent_height, latent_width = target.shape
        condition = torch.zeros_like(target)
        condition[:, :1] = ref
        condition_mask = torch.zeros(
            batch, views, 1, latent_height, latent_width,
            device=device, dtype=target.dtype)
        condition_mask[:, :1] = 1
        plucker = latent_plucker(
            c2w, intrinsics, latent_height, latent_width,
            image_height, image_width,
            opencv_camera_to_plucker).to(target.dtype)
        # Reference appearance remains present while camera pose is absent on
        # unconditional examples; the condition mask is kept unchanged.
        camera_keep = (
            torch.rand(batch, device=device) >= args.camera_dropout
        ).view(batch, 1, 1, 1, 1)
        plucker.mul_(camera_keep)
        # Retaining this small input gradient provides an end-to-end camera-control
        # assertion that remains valid even when FSDP exposes only FlatParameters.
        plucker.requires_grad_(True)

        global_sample_base = step * world_size * batch + rank * batch
        long_choice = (
            (torch.arange(batch, device=device) + global_sample_base) % 2 == 0)
        text_keep = torch.rand(batch, device=device) >= args.text_dropout
        variants = [
            "long_caption" if retained else "short_caption"
            for retained in long_choice.tolist()
        ]
        raw_text_cpu, text_context_lens_cpu = text_store.load_batch(
            scene_keys, variants, text_keep.tolist())
        raw_text = raw_text_cpu.pin_memory().to(
            device, non_blocking=True).requires_grad_(True)
        text_context_lens = text_context_lens_cpu.pin_memory().to(
            device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            text_context = text_projector(raw_text)

        flow_shift = dynamic_timestep_shift(
            latent_height, latent_width, views,
            latent_dim=channels, shift_base=args.shift_base)
        sigma, timestep, x0_weight = sample_x0_training_timestep(
            batch, device, shift=flow_shift,
            max_weight=args.max_x0_weight)
        noise = torch.randn_like(target)
        noisy, _ = wan_flow_sample(target, noise, sigma)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x0_pred = model(
                noisy, condition, plucker, condition_mask, timestep,
                text_context=text_context,
                text_context_lens=text_context_lens,
                gradient_checkpointing=args.gradient_checkpointing_mode)
            squared_error = (x0_pred.float() - target).square()
            if channels != LATENT_DIM:
                raise RuntimeError(
                    f"expected {LATENT_DIM} latent channels, got {channels}")
            geometry_per_scene = squared_error[:, :, :1024].mean(
                dim=(1, 2, 3, 4))
            appearance_per_scene = squared_error[:, :, 1024:].mean(
                dim=(1, 2, 3, 4))
            x0_per_scene = squared_error.mean(dim=(1, 2, 3, 4))
            diffusion_loss = (x0_per_scene * x0_weight).mean()
            predicted_raw = (
                x0_pred.float() * current_stats["target_std"]
                + current_stats["target_mean"]
            )
            cvc_loss, cvc_fraction = cross_view_correspondence_loss(
                predicted_raw, target_raw.float(), ref_raw.float())
            loss = diffusion_loss + CVC_WEIGHT * cvc_loss
            decoder_pixel_loss = loss.new_zeros(())
            decoder_lpips_loss = loss.new_zeros(())
            mdf_alpha = loss.new_zeros(())
            if args.mdf:
                alpha = torch.rand(
                    batch, 1, 1, 1, 1, device=device, dtype=x0_pred.dtype)
                mixed_raw = alpha * predicted_raw + (1.0 - alpha) * target_raw.float()
                decoded = rae.decode_generated_latent(
                    mixed_raw,
                    c2w,
                    intrinsics,
                    image_height,
                    image_width,
                    normalized=True,
                    want=("rgb", "depth", "gs"),
                    renders=[{
                        "c2w": c2w,
                        "K": intrinsics,
                        "random_background": False,
                    }],
                    random_background=False,
                )
                if not decoded["gs_ok"]:
                    raise RuntimeError("MDF decoder produced invalid Gaussians")
                direct_rgb = decoded["rgb"].float()
                rendered_rgb = decoded["renders"][0]["rgb"].float()
                decoder_pixel_loss = 0.5 * (
                    F.mse_loss(direct_rgb, images.float())
                    + F.mse_loss(rendered_rgb, images.float())
                )
                decoder_lpips_loss = 0.5 * (
                    decoder_lpips(
                        direct_rgb.flatten(0, 1) * 2.0 - 1.0,
                        images.float().flatten(0, 1) * 2.0 - 1.0,
                    )
                    + decoder_lpips(
                        rendered_rgb.flatten(0, 1) * 2.0 - 1.0,
                        images.float().flatten(0, 1) * 2.0 - 1.0,
                    )
                )
                decoder_loss = (
                    decoder_pixel_loss
                    + args.decoder_lpips_weight * decoder_lpips_loss
                )
                loss = loss + args.decoder_loss_weight * decoder_loss
                mdf_alpha = alpha.float().mean()

        finite_loss = torch.tensor(
            int(bool(torch.isfinite(loss))), device=device, dtype=torch.int32)
        dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
        if not bool(finite_loss):
            raise FloatingPointError("non-finite distributed x0 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = model.clip_grad_norm_(args.grad_clip)
        decoder_gradient = loss.new_zeros(())
        if decoder_parameters is not None:
            decoder_gradient = torch.nn.utils.clip_grad_norm_(
                decoder_parameters, args.grad_clip).float()
        text_projector_gradient = torch.nn.utils.clip_grad_norm_(
            text_projector.parameters(), args.grad_clip).float()
        finite_gradient = torch.tensor(
            int(bool(torch.isfinite(grad_norm)
                     and torch.isfinite(text_projector_gradient)
                     and torch.isfinite(decoder_gradient))),
            device=device, dtype=torch.int32)
        dist.all_reduce(finite_gradient, op=dist.ReduceOp.MIN)
        if not bool(finite_gradient):
            raise FloatingPointError("non-finite generator/text gradient")
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        update_fsdp_ema(ema_model, model, args.ema)
        update_module_ema(text_projector_ema, text_projector, args.ema)
        step += 1

        with torch.no_grad():
            camera_gradient = (
                plucker.grad.float().square().sum()
                if plucker.grad is not None
                else torch.zeros((), device=device))
            text_input_gradient = (
                raw_text.grad.float().square().sum()
                if raw_text.grad is not None
                else torch.zeros((), device=device))
            values = {
                "loss": loss.detach(),
                "diffusion_loss": diffusion_loss.detach(),
                "cvc_loss": cvc_loss.detach(),
                "cvc_fraction": cvc_fraction.detach(),
                "decoder_pixel_loss": decoder_pixel_loss.detach(),
                "decoder_lpips_loss": decoder_lpips_loss.detach(),
                "decoder_gnorm": decoder_gradient.detach(),
                "mdf_alpha": mdf_alpha.detach(),
                "x0_mse": squared_error.mean().detach(),
                "geometry_mse": geometry_per_scene.mean().detach(),
                "appearance_mse": appearance_per_scene.mean().detach(),
                "ref_mse": squared_error[:, :1].mean().detach(),
                "tgt_mse": squared_error[:, 1:].mean().detach(),
                "x0_weight": x0_weight.mean().detach(),
                "gnorm": grad_norm.detach().float(),
                "camera_gnorm": camera_gradient.sqrt(),
                "camera_keep": camera_keep.float().mean(),
                "text_input_gnorm": text_input_gradient.sqrt(),
                "text_projector_gnorm": text_projector_gradient,
                "text_keep": text_keep.float().mean(),
                "long_fraction": long_choice.float().mean(),
                "pose_span_deg": pose_span_deg.mean(),
                "refined_fraction": refined_source.mean(),
                "resolution_height": loss.new_tensor(float(image_height)),
                "resolution_width": loss.new_tensor(float(image_width)),
            }
            pose_masks = {
                "pose_local": pose_span_deg < 20.0,
                "pose_medium": (pose_span_deg >= 20.0) & (pose_span_deg < 45.0),
                "pose_large": pose_span_deg >= 45.0,
            }
            for pose_name, pose_mask in pose_masks.items():
                values[f"{pose_name}_count"] = pose_mask.float().sum()
                values[f"{pose_name}_mse_sum"] = (
                    x0_per_scene.detach() * pose_mask.float()).sum()
            reduced = torch.stack([value.float() for value in values.values()])
            dist.all_reduce(reduced)
            reduced /= world_size
            values = dict(zip(values, reduced))
            for name, value in values.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            metric_count += 1

        if main_process and step % args.log_every == 0:
            elapsed = time.time() - last_log
            summary_names = [
                "loss", "diffusion_loss", "cvc_loss", "cvc_fraction",
                "decoder_pixel_loss", "decoder_lpips_loss", "decoder_gnorm",
                "mdf_alpha",
                "x0_mse", "geometry_mse", "appearance_mse",
                "ref_mse", "tgt_mse", "x0_weight", "gnorm",
                "camera_gnorm", "camera_keep", "text_input_gnorm",
                "text_projector_gnorm", "text_keep", "long_fraction",
                "pose_span_deg", "refined_fraction",
                "resolution_height", "resolution_width",
            ]
            summary = " ".join(
                f"{name}={metrics_sum[name]/metric_count:.5f}" for name in
                summary_names)
            for pose_name in ("pose_local", "pose_medium", "pose_large"):
                count = metrics_sum[f"{pose_name}_count"]
                fraction = count / (metric_count * args.batch_size)
                mse = (metrics_sum[f"{pose_name}_mse_sum"] / count
                       if count > 0 else float("nan"))
                summary += f" {pose_name}_frac={fraction:.3f} {pose_name}_mse={mse:.5f}"
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"step={step:06d} shift={flow_shift:.4f} "
                f"sigma={sigma.mean():.4f} {summary} lr={lr:.2e} "
                f"sec/it={elapsed/metric_count:.3f} "
                f"peak_GiB={peak:.2f}", flush=True)
            metrics_sum, metric_count, last_log = {}, 0, time.time()

        save_now = step % args.ckpt_every == 0
        if save_now or step >= args.steps:
            slot = (step // args.ckpt_every) % args.checkpoint_slots
            save_checkpoint(
                args.out, slot, step, model, ema_model, args, rank,
                text_projector, text_projector_ema, text_contract,
                decoder_parameters=decoder_parameters)
            if main_process:
                print(
                    f"checkpoint step={step} slot={slot}", flush=True)
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device), device=device, dtype=torch.float64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if main_process:
        print(
            f"training complete step={step}/{args.steps} "
            f"skipped_invalid_camera_batches={skipped_invalid_camera_batches} "
            f"max_peak_GiB={float(peak)/2**30:.2f}", flush=True)
        with open(os.path.join(args.out, "progress.txt"), "w") as handle:
            handle.write(f"{step}\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
