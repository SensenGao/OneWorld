#!/usr/bin/env python3
"""Train the OneWorld representation autoencoder with 3DGS rendering losses."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESOLUTIONS = ((224, 448), (448, 448), (252, 448), (336, 448))
APPEARANCE_DIM = 32
APPEARANCE_HIDDEN_DIM = 384
APPEARANCE_DEPTH = 6
APPEARANCE_HEADS = 8
FEATURES = 256
SH_DEGREE = 1
LPIPS_WEIGHT = 0.05
LPIPS_START = 500
KL_WEIGHT = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Train the OneWorld RAE.")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--re10k-root", required=True)
    parser.add_argument("--pi3", required=True)
    parser.add_argument("--decoder-denorm-stats", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "rae"))
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=18000017)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def learning_rate(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.steps - args.warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    return args.final_lr + 0.5 * (args.lr - args.final_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def configure_trainable(model):
    model.requires_grad_(False)
    pi3x = model.encoder.pi3x
    for block in pi3x.decoder[model.encoder.boundary_block + 1 :]:
        block.requires_grad_(True)
    for index in (3, 4):
        pi3x.pose_inject_blk[index].requires_grad_(True)
    pi3x.point_decoder.requires_grad_(True)
    pi3x.point_head.requires_grad_(True)
    model.appearance_encoder.requires_grad_(True)
    model.posterior_mean.requires_grad_(True)
    model.posterior_logvar.requires_grad_(True)
    if model.appearance_encoder.branch_drop_prob == 0:
        model.appearance_encoder.mask_token.requires_grad_(False)
    if not len(model.appearance_encoder.blocks):
        model.appearance_encoder.class_token.requires_grad_(False)
    model.fusion_norm.requires_grad_(True)
    model.z36_norm.requires_grad_(True)
    model.boundary_registers.requires_grad_(True)
    model.rgb_to_trunk.requires_grad_(True)
    model.heads.requires_grad_(True)
    model.encoder.freeze_encoder_half()

    trunk_prefixes = (
        "encoder.pi3x.decoder",
        "encoder.pi3x.pose_inject_blk",
        "encoder.pi3x.point_decoder",
        "encoder.pi3x.point_head",
    )
    trunk, adapters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (trunk if name.startswith(trunk_prefixes) else adapters).append(parameter)
    return trunk, adapters


def checkpoint_state(model):
    def keep(name: str) -> bool:
        if name.startswith("encoder.pi3x.decoder."):
            return int(name.split(".")[3]) >= model.encoder.boundary_block + 1
        if name.startswith("encoder.pi3x.pose_inject_blk."):
            return int(name.split(".")[3]) >= 3
        return name.startswith(
            (
                "encoder.pi3x.point_decoder.",
                "encoder.pi3x.point_head.",
                "appearance_encoder.",
                "rgb_to_trunk.",
                "geometry_norm.",
                "appearance_norm.",
                "fusion_norm.",
                "z36_norm.",
                "boundary_registers",
                "decoder_denorm_scale",
                "decoder_denorm_bias",
                "posterior_mean.",
                "posterior_logvar.",
                "heads.",
            )
        )

    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if keep(name)
    }


def contract():
    return {
        "version": 19,
        "boundary_block_zero_based": 17,
        "boundary_attention": "global",
        "context_views": 8,
        "heldout_target_views": 8,
        "latent_concat_dim": 1056,
        "resolutions_hw": [list(value) for value in DEFAULT_RESOLUTIONS],
        "patch_grids_hw": [
            [height // 14, width // 14]
            for height, width in DEFAULT_RESOLUTIONS
        ],
        "gaussians": {
            "sh_degree": SH_DEGREE,
            "ray_mode": "z_depth",
        },
    }


def save_checkpoint(path: Path, step: int, model, optimizer, args, world_size: int):
    payload = {
        "step": step,
        "rae": checkpoint_state(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "world_size": world_size,
        "contract": contract(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    parameters = dict(model.named_parameters())
    extra = [name for name in payload["rae"] if name not in parameters]
    if extra:
        raise RuntimeError(f"checkpoint contains unknown RAE parameters: {extra[:8]}")
    with torch.no_grad():
        for name, value in payload["rae"].items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])


def main():
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    for path in (
        args.data_root,
        args.re10k_root,
        args.pi3,
        args.decoder_denorm_stats,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    main_process = rank == 0
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from cut3r_data.datasets.realestate10k_torch import RE10K_Torch_Multi
    from pi3.models.pi3x import Pi3X

    from oneworld.losses import LPIPS
    from oneworld.rae.camera import canonicalize_pi3_cameras
    from oneworld.rae.model import RAE
    from oneworld.rae.nvs_refined import MixedMultiResolutionRAESampler, NVSRefined

    output_dir = Path(args.out)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    pi3x = Pi3X.from_pretrained(args.pi3).to(device)
    model = RAE(
        pi3x,
        appearance_dim=APPEARANCE_DIM,
        appearance_hidden_dim=APPEARANCE_HIDDEN_DIM,
        appearance_depth=APPEARANCE_DEPTH,
        appearance_heads=APPEARANCE_HEADS,
        appearance_transformer_dropout=0.0,
        rgb_skip_drop=0.0,
        features=FEATURES,
        sh_degree=SH_DEGREE,
        grad_checkpoint=args.gradient_checkpointing,
        level_drop_prob=0.0,
        gaussian_ray_mode="z_depth",
        detail_hidden_dim=0,
        detail_merge="feature",
    ).to(device)
    denorm = torch.load(
        args.decoder_denorm_stats,
        map_location="cpu",
        weights_only=False,
    )
    model.load_decoder_denorm_stats(denorm)
    model.heads.rgb_head._coarse_scale.fill_(1.0)
    model.heads.rgb_head._detail_scale.fill_(1.0)
    trunk_parameters, adapter_parameters = configure_trainable(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_parameters, "lr_scale": 1.0},
            {"params": adapter_parameters, "lr_scale": 1.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
    start_step = 0
    if args.resume and resume_path.is_file():
        start_step = load_checkpoint(resume_path, model, optimizer)

    wrapped_model = (
        DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else model
    )
    lpips = LPIPS().to(device).eval().requires_grad_(False)

    remaining_steps = max(0, args.steps - start_step)
    reserve_steps = max(128, math.ceil(remaining_steps * 0.01)) if remaining_steps else 0
    re10k_dataset = RE10K_Torch_Multi(
        ROOT=args.re10k_root,
        split="train",
        num_views=16,
        resolution=[(width, height) for height, width in DEFAULT_RESOLUTIONS],
        ordered_views=True,
        aug_crop=False,
        seed=20260827 + rank * 1000003,
        min_interval=1,
        max_interval=128,
    )
    refined_dataset = NVSRefined(args.data_root, subsets=("ACID", "DL3DV", "Re10K"))
    samples = MixedMultiResolutionRAESampler(
        re10k_dataset,
        refined_dataset,
        DEFAULT_RESOLUTIONS,
        context_views=8,
        target_views=8,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank,
        length=(remaining_steps + reserve_steps) * args.batch_size,
        start_step=start_step,
        seed=args.seed,
        min_span=48,
        max_span=160,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "drop_last": True,
    }
    if args.workers:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(samples, **loader_options)

    step = start_step
    accumulated = {
        "loss": 0.0,
        "pixel": 0.0,
        "novel": 0.0,
        "context": 0.0,
        "lpips": 0.0,
        "kl": 0.0,
    }
    window = 0
    last_log = time.time()
    for batch in loader:
        if step >= args.steps:
            break
        images, cameras, intrinsics = batch[:3]
        images = images.to(device, non_blocking=True)
        cameras = cameras.to(device, non_blocking=True).float()
        intrinsics = intrinsics.to(device, non_blocking=True).float()
        canonical = canonicalize_pi3_cameras(
            cameras,
            reference_index=0,
            scale_view_count=16,
            strict=False,
        )
        valid = canonical.valid.to(device=device, dtype=torch.int32).min()
        if distributed:
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(valid):
            continue

        context_images, target_images = images[:, :8], images[:, 8:]
        context_cameras, target_cameras = canonical.c2w[:, :8], canonical.c2w[:, 8:]
        context_intrinsics, target_intrinsics = intrinsics[:, :8], intrinsics[:, 8:]
        height, width = context_images.shape[-2:]
        renders = [
            {"c2w": target_cameras, "K": target_intrinsics, "random_background": False},
            {"c2w": context_cameras, "K": context_intrinsics, "random_background": False},
        ]

        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped_model(
                context_images,
                context_cameras,
                context_intrinsics,
                want=("rgb", "depth", "gs"),
                renders=renders,
                random_background=False,
            )
            gaussian_valid = torch.tensor(
                int(output["gs_ok"]), device=device, dtype=torch.int32
            )
            if distributed:
                dist.all_reduce(gaussian_valid, op=dist.ReduceOp.MIN)
            if not bool(gaussian_valid):
                continue
            reconstructed = output["rgb"].float()
            novel_render = output["renders"][0]["rgb"].float()
            context_render = output["renders"][1]["rgb"].float()
            pixel_loss = torch.nn.functional.mse_loss(reconstructed, context_images)
            novel_loss = torch.nn.functional.mse_loss(novel_render, target_images)
            context_loss = torch.nn.functional.mse_loss(context_render, context_images)
            reconstruction_loss = (pixel_loss + novel_loss + context_loss) / 3.0
            perceptual_loss = reconstruction_loss.new_zeros(())
            if step >= LPIPS_START:
                direct_lpips = lpips(
                    context_images.reshape(-1, 3, height, width) * 2 - 1,
                    reconstructed.reshape(-1, 3, height, width) * 2 - 1,
                )
                novel_lpips = lpips(
                    target_images.reshape(-1, 3, height, width) * 2 - 1,
                    novel_render.reshape(-1, 3, height, width) * 2 - 1,
                )
                perceptual_loss = 0.5 * (direct_lpips + novel_lpips)
            kl_loss = output["kl_loss"]
            loss = (
                reconstruction_loss
                + LPIPS_WEIGHT * perceptual_loss
                + KL_WEIGHT * kl_loss
            )

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite RAE loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        step += 1

        values = {
            "loss": loss,
            "pixel": pixel_loss,
            "novel": novel_loss,
            "context": context_loss,
            "lpips": perceptual_loss,
            "kl": kl_loss,
        }
        reduced = torch.stack([value.detach() for value in values.values()]).double()
        if distributed:
            dist.all_reduce(reduced)
            reduced.div_(world_size)
        for name, value in zip(accumulated, reduced.tolist()):
            accumulated[name] += value
        window += 1

        if main_process and (step == 1 or step % args.log_every == 0):
            metrics = " ".join(
                f"{name}={value / window:.5f}" for name, value in accumulated.items()
            )
            print(
                f"step={step:06d}/{args.steps} {metrics} "
                f"grad={float(gradient_norm):.4f} lr={lr:.2e} "
                f"sec/step={(time.time() - last_log) / window:.3f}",
                flush=True,
            )
            accumulated = {name: 0.0 for name in accumulated}
            window = 0
            last_log = time.time()

        if step % args.checkpoint_every == 0:
            if distributed:
                dist.barrier()
            if main_process:
                save_checkpoint(output_dir / "last.pt", step, model, optimizer, args, world_size)

    if distributed:
        dist.barrier()
    if main_process:
        save_checkpoint(output_dir / "final.pt", step, model, optimizer, args, world_size)
        print(f"finished RAE training at step={step}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
