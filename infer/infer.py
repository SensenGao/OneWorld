#!/usr/bin/env python3
"""OneWorld image-conditioned four-step 3D scene inference."""

from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from oneworld.rae.release import (
    check_flat_layout, fingerprints_close, load_named_parameters, module_fingerprint,
)

HEIGHT = 224
WIDTH = 448
VIEWS = 8
LATENT_DIM = 1056


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an eight-view 3D Gaussian scene from one image."
    )
    parser.add_argument("--model", required=True, help="downloaded Sensen02/OneWorld")
    parser.add_argument("--wan", required=True, help="Wan2.1-T2V-1.3B directory")
    parser.add_argument("--pi3", default="yyfz233/Pi3X")
    parser.add_argument("--image", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--video-frames", type=int, default=48)
    parser.add_argument("--video-fps", type=int, default=24)
    return parser.parse_args()


def initialize_distributed():
    if "RANK" not in os.environ:
        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29500",
        )
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def resolve_model_files(model_root: Path):
    config_path = model_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") not in (None, "oneworld-4step-v1", "oneworld-4step-v2", "oneworld-4step-v3"):
        raise RuntimeError("unsupported OneWorld package format")
    for key, expected in (("height", HEIGHT), ("width", WIDTH), ("views", VIEWS)):
        if key in config and config[key] != expected:
            raise RuntimeError(f"unsupported model configuration: {key}={config[key]}")
    required = {
        "checkpoint": model_root / config["checkpoint"],
        "rae": model_root / config["rae"],
        "stats": model_root / config["stats"],
    }
    # The final generator shards contain the trained input projection.  An
    # alignment initializer is optional when restoring those complete shards.
    if config.get("input_alignment"):
        required["input_alignment"] = model_root / config["input_alignment"]
    if config.get("format") == "oneworld-4step-v2":
        if config.get("aligned_input") is not True or config.get("decoder_layout") != "rae_decoder_v1":
            raise RuntimeError("invalid v2 input architecture or decoder layout")
        required["manifest"] = model_root / config["manifest"]
        required["layout"] = required["checkpoint"] / "layout.json"
    if config.get("format") == "oneworld-4step-v3":
        if config.get("aligned_input") is not True or config.get("weight_layout") != "named_parameters_v1":
            raise RuntimeError("invalid merged input architecture or weight layout")
        required["manifest"] = model_root / config["manifest"]
        required["weights"] = model_root / config["weights"]
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"model package is incomplete: {missing}")
    return config, required


@torch.no_grad()
def tensor_fingerprint(tensors, device):
    result = torch.zeros(3, device=device, dtype=torch.float64)
    for tensor in tensors:
        value = tensor.detach().double()
        result[0] += value.numel()
        result[1] += value.sum()
        result[2] += value.square().sum()
    dist.all_reduce(result)
    return result.cpu()


@torch.no_grad()
def load_rank_local_tensors(path, parameters, field, fingerprint_name):
    rank, runtime_world = dist.get_rank(), dist.get_world_size()
    metadata = torch.load(
        path / "rank00000.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    accepted_formats = {
        "rank-local-fsdp-weights-ema-v2",
        "rank-local-fsdp-weights-ema-joint-decoder-v3",
        "oneworld-distill-rank-local-v1",
        "oneworld-rank-local-inference-v1",
    }
    if metadata.get("format") not in accepted_formats:
        raise RuntimeError(f"unsupported checkpoint format: {metadata.get('format')}")
    saved_world = int(metadata.get("world_size", -1))
    if saved_world <= 0:
        raise RuntimeError("invalid checkpoint world size")
    if saved_world % runtime_world:
        raise RuntimeError(
            f"saved world_size={saved_world} cannot use runtime world_size={runtime_world}"
        )
    if field not in metadata:
        raise RuntimeError(f"checkpoint is missing {field}")
    parameters = list(parameters)
    layout_path = path / "layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text())
        if layout.get("format") != "oneworld-parameter-layout-v1":
            raise RuntimeError("unsupported checkpoint parameter layout")
        check_flat_layout(parameters, layout[field])
    source_sizes = [int(tensor.numel()) for tensor in metadata[field]]
    if len(parameters) != len(source_sizes):
        raise RuntimeError(
            f"{field} parameter groups differ: {len(parameters)} != {len(source_sizes)}"
        )

    full_numels = []
    required_ranks = set()
    for parameter, source_size in zip(parameters, source_sizes):
        unpadded = getattr(parameter, "_unpadded_unsharded_size", None)
        if unpadded is None:
            raise RuntimeError("FSDP parameter metadata is unavailable")
        full_numel = int(unpadded.numel())
        if source_size != (full_numel + saved_world - 1) // saved_world:
            raise RuntimeError(f"{field} saved shard size differs from model parameter size")
        if parameter.numel() != (full_numel + runtime_world - 1) // runtime_world:
            raise RuntimeError(f"{field} runtime shard size differs from model parameter size")
        full_numels.append(full_numel)
        begin = rank * parameter.numel()
        end = min(begin + parameter.numel(), full_numel)
        if begin < end:
            required_ranks.update(
                range(begin // source_size, (end - 1) // source_size + 1)
            )

    payloads = {0: metadata} if 0 in required_ranks else {}
    for source_rank in sorted(required_ranks):
        if source_rank not in payloads:
            payloads[source_rank] = torch.load(
                path / f"rank{source_rank:05d}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )

    for source_rank, payload in {0: metadata, **payloads}.items():
        if any(payload.get(key) != expected for key, expected in (
                ("rank", source_rank), ("step", metadata["step"]),
                ("world_size", saved_world), ("format", metadata["format"]))):
            raise RuntimeError(f"inconsistent checkpoint shard metadata: rank {source_rank}")
        tensors = payload.get(field, ())
        if len(tensors) != len(source_sizes) or any(
                t.ndim != 1 or t.numel() != size or t.dtype != reference.dtype
                for t, size, reference in zip(tensors, source_sizes, metadata[field])):
            raise RuntimeError(f"inconsistent checkpoint shard tensors: rank {source_rank}")
        if not torch.equal(payload["fingerprints"][fingerprint_name],
                           metadata["fingerprints"][fingerprint_name]):
            raise RuntimeError(f"inconsistent checkpoint fingerprint metadata: rank {source_rank}")

    step = int(metadata["step"])
    for index, (parameter, source_size, full_numel) in enumerate(
        zip(parameters, source_sizes, full_numels)
    ):
        begin = rank * parameter.numel()
        end = min(begin + parameter.numel(), full_numel)
        restored = torch.zeros(parameter.numel(), dtype=metadata[field][index].dtype)
        source_offset_global = begin
        destination_offset = 0
        while source_offset_global < end:
            source_rank = source_offset_global // source_size
            source_offset = source_offset_global % source_size
            count = min(end - source_offset_global, source_size - source_offset)
            restored[destination_offset : destination_offset + count].copy_(
                payloads[source_rank][field][index][
                    source_offset : source_offset + count
                ]
            )
            source_offset_global += count
            destination_offset += count
        parameter.copy_(restored.to(parameter.device, dtype=parameter.dtype))

    expected = metadata["fingerprints"][fingerprint_name]
    actual = tensor_fingerprint(parameters, parameters[0].device)
    expected_source_count = sum(source_sizes) * saved_world
    expected_runtime_count = sum(p.numel() for p in parameters) * runtime_world
    valid = (expected[0].item() == expected_source_count
             and actual[0].item() == expected_runtime_count
             and fingerprints_close(expected, actual, compare_count=False))
    if not valid:
        raise RuntimeError(f"{field} fingerprint mismatch")
    del payloads, metadata
    gc.collect()
    return step, saved_world


def load_text_projector(checkpoint: Path, projector):
    payload = torch.load(
        checkpoint / "text_projector.pt", map_location="cpu", weights_only=False
    )
    if payload.get("format") != "wan-native-text-projector-v1":
        raise RuntimeError("unsupported text projector checkpoint")
    projector.load_state_dict(payload["ema"], strict=True)
    actual = module_fingerprint(projector)
    expected = payload["ema_fingerprint"]
    if not fingerprints_close(expected, actual):
        raise RuntimeError("text projector fingerprint mismatch")


def load_statistics(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload["by_resolution"][f"{HEIGHT}x{WIDTH}"]
    result = {}
    for name in ("target_mean", "target_std", "ref_mean", "ref_std"):
        value = values[name]
        if tuple(value.shape) != (LATENT_DIM,):
            raise RuntimeError(f"invalid latent statistics: {name}={tuple(value.shape)}")
        if not torch.isfinite(value).all() or (name.endswith("std") and bool((value < 0).any())):
            raise RuntimeError(f"invalid latent statistics values: {name}")
        result[name] = value.to(device).reshape(1, 1, LATENT_DIM, 1, 1)
    result["target_std"] = result["target_std"].clamp_min(1e-6)
    result["ref_std"] = result["ref_std"].clamp_min(1e-6)
    return result


def load_contract(checkpoint: Path):
    contract = json.loads((checkpoint / "contract.json").read_text(encoding="utf-8"))
    if contract.get("decoder_layout", "rae_decoder_v1") != "rae_decoder_v1":
        raise RuntimeError("unsupported decoder parameter layout")
    if contract.get("format") == "oneworld-4step-v2" and not (checkpoint / "layout.json").is_file():
        raise RuntimeError("v2 checkpoint is missing its parameter layout")
    if contract.get("format") == "oneworld-4step-v3" and contract.get("weight_layout") != "named_parameters_v1":
        raise RuntimeError("unsupported merged checkpoint layout")
    expected = {
        "prediction": "direct_x0",
        "prediction_channels_per_token": LATENT_DIM,
        "views": VIEWS,
        "condition_views": 1,
    }
    mismatch = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"checkpoint contract mismatch: {mismatch}")
    sigmas = list(contract.get("dmd2", {}).get("student_sigmas", ()))
    if len(sigmas) != 4:
        raise RuntimeError("checkpoint does not contain the four-step schedule")
    if sigmas[-1] != 0.0:
        sigmas.append(0.0)
    return contract, sigmas


def encode_prompt(prompt, wan_path: Path, max_length: int, device, rank):
    length = torch.zeros(1, dtype=torch.long, device=device)
    raw = None
    if rank == 0:
        from oneworld.models.wan.modules.t5 import T5EncoderModel

        encoder = T5EncoderModel(
            text_len=max_length,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=wan_path / "models_t5_umt5-xxl-enc-bf16.pth",
            tokenizer_path=wan_path / "google" / "umt5-xxl",
        )
        raw = encoder([prompt], device)[0].float().contiguous()
        length[0] = raw.shape[0]
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    dist.broadcast(length, src=0)
    if rank != 0:
        raw = torch.empty(int(length.item()), 4096, device=device)
    dist.broadcast(raw, src=0)
    return raw[None], length


def load_image(path: Path, device):
    image = Image.open(path).convert("RGB")
    scale = max(WIDTH / image.width, HEIGHT / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - WIDTH) // 2
    top = (resized.height - HEIGHT) // 2
    image = resized.crop((left, top, left + WIDTH, top + HEIGHT))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return image, tensor[None, None]


def load_cameras(path: Path, device):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if [payload.get("height"), payload.get("width")] != [HEIGHT, WIDTH]:
        raise ValueError(f"camera resolution must be {HEIGHT}x{WIDTH}")
    c2w = torch.tensor(payload["c2w"], dtype=torch.float32, device=device)
    intrinsics = torch.tensor(
        payload["intrinsics"], dtype=torch.float32, device=device
    )
    if c2w.shape != (VIEWS, 4, 4) or intrinsics.shape != (VIEWS, 3, 3):
        raise ValueError("camera file must contain eight c2w and intrinsics matrices")
    from oneworld.rae.camera import canonicalize_pi3_cameras

    canonical = canonicalize_pi3_cameras(
        c2w[None], reference_index=0, scale_view_count=VIEWS, strict=True
    )
    return canonical.c2w, intrinsics[None]


def latent_plucker(c2w, intrinsics, geometry_function):
    latent_height, latent_width = HEIGHT // 14, WIDTH // 14
    latent_intrinsics = intrinsics.clone()
    latent_intrinsics[..., 0, :] *= latent_width / WIDTH
    latent_intrinsics[..., 1, :] *= latent_height / HEIGHT
    rays = geometry_function(
        c2w.reshape(VIEWS, 4, 4),
        latent_intrinsics.reshape(VIEWS, 3, 3),
        latent_height,
        latent_width,
    )
    return rays.reshape(1, VIEWS, latent_height, latent_width, 6).permute(
        0, 1, 4, 2, 3
    )


@torch.inference_mode()
def predict_x0(model, noisy, condition, plucker, condition_mask, text, text_lens, sigma):
    timestep = torch.full(
        (noisy.shape[0],), float(sigma * 1000.0), device=noisy.device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(
            noisy,
            condition,
            plucker,
            condition_mask,
            timestep,
            text_context=text,
            text_context_lens=text_lens,
            gradient_checkpointing=False,
        ).float()


@torch.inference_mode()
def feedback_latent(rae, normalized, c2w, intrinsics, statistics):
    latent = normalized * statistics["target_std"] + statistics["target_mean"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        decoded = rae.decode_generated_latent(
            latent,
            c2w,
            intrinsics,
            HEIGHT,
            WIDTH,
            normalized=True,
            want=("depth", "gs"),
            renders=[{"c2w": c2w, "K": intrinsics, "random_background": False}],
            random_background=False,
        )
        if not decoded["gs_ok"]:
            raise RuntimeError("invalid 3D Gaussians during student feedback")
        rendered = decoded["renders"][0]["rgb"].clamp(0, 1)
        encoded = rae.encode_latent(rendered, c2w, intrinsics)["latent"]
    return (encoded.float() - statistics["target_mean"]) / statistics["target_std"]


@torch.inference_mode()
def sample(model, rae, condition, plucker, condition_mask, text, text_lens,
           c2w, intrinsics, statistics, sigmas, seed):
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    noisy = torch.randn(
        condition.shape, device=condition.device, generator=generator
    )
    sigma_values = torch.tensor(sigmas, device=condition.device, dtype=torch.float32)
    for sigma, sigma_next in zip(sigma_values[:-1], sigma_values[1:]):
        clean = predict_x0(
            model, noisy, condition, plucker, condition_mask, text, text_lens, sigma
        )
        if float(sigma_next) == 0.0:
            noisy = clean
        else:
            clean = feedback_latent(rae, clean, c2w, intrinsics, statistics)
            noise = torch.randn(
                clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
            )
            noisy = (1.0 - sigma_next) * clean + sigma_next * noise
    return noisy


def interpolate_cameras(c2w, intrinsics, frame_count):
    key_c2w = c2w[0].float().cpu().numpy()
    key_intrinsics = intrinsics[0].float().cpu().numpy()
    key_times = np.linspace(0.0, 1.0, VIEWS)
    frame_times = np.linspace(0.0, 1.0, frame_count)
    rotations = Slerp(key_times, Rotation.from_matrix(key_c2w[:, :3, :3]))(
        frame_times
    ).as_matrix().astype(np.float32)
    translations = np.stack(
        [
            np.interp(frame_times, key_times, key_c2w[:, axis, 3])
            for axis in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    result_c2w = np.broadcast_to(
        np.eye(4, dtype=np.float32), (frame_count, 4, 4)
    ).copy()
    result_c2w[:, :3, :3] = rotations
    result_c2w[:, :3, 3] = translations
    result_intrinsics = np.stack(
        [
            np.interp(frame_times, key_times, key_intrinsics[:, row, column])
            for row in range(3)
            for column in range(3)
        ],
        axis=-1,
    ).reshape(frame_count, 3, 3).astype(np.float32)
    return (
        torch.from_numpy(result_c2w)[None].to(c2w.device),
        torch.from_numpy(result_intrinsics)[None].to(intrinsics.device),
    )


@torch.inference_mode()
def decode_scene(rae, latent, c2w, intrinsics, video_frames):
    renders = [{"c2w": c2w, "K": intrinsics, "random_background": False}]
    if video_frames > 0:
        video_c2w, video_intrinsics = interpolate_cameras(
            c2w, intrinsics, video_frames
        )
        renders.append(
            {
                "c2w": video_c2w,
                "K": video_intrinsics,
                "random_background": False,
            }
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        decoded = rae.decode_generated_latent(
            latent,
            c2w,
            intrinsics,
            HEIGHT,
            WIDTH,
            normalized=True,
            want=("depth", "gs"),
            renders=renders,
            random_background=False,
        )
    if not decoded["gs_ok"]:
        raise RuntimeError("generated latent produced invalid 3D Gaussians")
    views = decoded["renders"][0]["rgb"].float().clamp(0, 1)
    video = decoded["renders"][1]["rgb"].float().clamp(0, 1) if video_frames > 0 else None
    return views, video


def tensor_image(value):
    array = (
        value.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array)


def save_outputs(out: Path, source_image, views, video, prompt, seed, fps):
    out.mkdir(parents=True, exist_ok=True)
    views_dir = out / "views"
    views_dir.mkdir(exist_ok=True)
    source_image.save(out / "source.png")
    view_images = []
    for index, view in enumerate(views[0]):
        image = tensor_image(view)
        image.save(views_dir / f"v{index:02d}.png")
        view_images.append(image)

    cells = [source_image, *view_images]
    cell_width, cell_height = WIDTH // 2, HEIGHT // 2
    canvas = Image.new("RGB", (3 * cell_width, 3 * (cell_height + 24)), "black")
    draw = ImageDraw.Draw(canvas)
    labels = ["source", *[f"3DGS view {index}" for index in range(VIEWS)]]
    for index, (image, label) in enumerate(zip(cells, labels)):
        row, column = divmod(index, 3)
        x, y = column * cell_width, row * (cell_height + 24)
        canvas.paste(image.resize((cell_width, cell_height)), (x, y + 24))
        draw.text((x + 6, y + 5), label, fill="white")
    canvas.save(out / "grid.png")

    if video is not None:
        forward = [np.asarray(tensor_image(frame)) for frame in video[0]]
        frames = forward + forward[-2:0:-1]
        imageio.mimsave(
            out / "sweep.mp4",
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "conditioning": "image+camera+text" if prompt else "image+camera",
                "seed": seed,
                "views": VIEWS,
                "video_frames_forward": 0 if video is None else int(video.shape[1]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    model_config, paths = resolve_model_files(Path(args.model))
    _contract, sigmas = load_contract(paths["checkpoint"])
    rank, world, device = initialize_distributed()
    merged = model_config.get("format") == "oneworld-4step-v3"
    if not merged and 16 % world:
        raise RuntimeError("runtime GPU count must divide the released 16-rank checkpoint")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    max_text_length = int(model_config.get("text_length", 256))
    raw_text, text_lens = encode_prompt(
        args.prompt, Path(args.wan), max_text_length, device, rank
    )

    from oneworld.models.wan.modules.model import WanAttentionBlock
    from oneworld.rae.checkpoint import (
        load_rae_model,
        set_rae_inference,
        wrap_rae_decoder_fsdp,
    )
    from oneworld.rae.wan_flow import DecoupledHeadBlock, WanPi3Flow
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import opencv_camera_to_plucker

    statistics = load_statistics(paths["stats"], device)
    torch.manual_seed(20260818)
    unwrapped, text_projector = WanPi3Flow.from_pretrained_with_text_projector(
        args.wan,
        max_views=VIEWS,
        head_depth=2,
        latent_dim=LATENT_DIM,
        input_alignment=str(paths["input_alignment"]) if "input_alignment" in paths else "",
        aligned_input=model_config.get("aligned_input", True),
    )
    if merged:
        # Wan's pretrained blocks start in BF16. Restore full-precision EMA
        # values before FSDP applies inference mixed precision, as in v1/v2.
        unwrapped.float()
        load_named_parameters(paths["weights"], unwrapped, "generator.")
        load_named_parameters(paths["weights"], text_projector, "text_projector.")
        expected = torch.tensor(model_config["text_projector_fingerprint"], dtype=torch.float64)
        if not fingerprints_close(expected, module_fingerprint(text_projector)):
            raise RuntimeError("merged text projector fingerprint mismatch")
    else:
        load_text_projector(paths["checkpoint"], text_projector)
    text_projector = text_projector.float().to(device).eval().requires_grad_(False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text = text_projector(raw_text).float()
    del raw_text

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
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": mixed_precision,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": device,
        "sync_module_states": True,
        "limit_all_gathers": True,
        "use_orig_params": False,
    }
    model = FSDP(
        unwrapped.float().eval().requires_grad_(False),
        auto_wrap_policy=auto_wrap,
        **fsdp_kwargs,
    ).eval()
    if merged:
        checkpoint_step = int(model_config["step"])
        saved_world = int(model_config["source_world_size"])
    else:
        checkpoint_step, saved_world = load_rank_local_tensors(
            paths["checkpoint"], model.parameters(), "ema", "ema"
        )

    pi3x = Pi3X.from_pretrained(args.pi3).to(device)
    rae, _rae_info = load_rae_model(pi3x, str(paths["rae"]), device)
    if model_config.get("format") in ("oneworld-4step-v2", "oneworld-4step-v3") and _rae_info["legacy_migration"]:
        raise RuntimeError("v2 release requires a complete repaired RAE checkpoint")
    if merged:
        # EMA decoder parameters were merged by name into the saved RAE. Keep
        # the same mixed-precision execution, independently of checkpoint layout.
        wrap_rae_decoder_fsdp(rae, fsdp_kwargs)
    else:
        metadata = torch.load(
            paths["checkpoint"] / "rank00000.pt",
            map_location="cpu", weights_only=False, mmap=True,
        )
        if "decoder_ema" in metadata:
            decoder_parameters = wrap_rae_decoder_fsdp(rae, fsdp_kwargs)
            decoder_step, decoder_world = load_rank_local_tensors(
                paths["checkpoint"], decoder_parameters, "decoder_ema", "decoder_ema"
            )
            if (decoder_step, decoder_world) != (checkpoint_step, saved_world):
                raise RuntimeError("generator and decoder checkpoint metadata differ")
        del metadata
    set_rae_inference(rae)

    source_image, image = load_image(Path(args.image), device)
    c2w, intrinsics = load_cameras(Path(args.cameras), device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = rae.encode_latent(image, c2w[:, :1], intrinsics[:, :1])[
            "latent"
        ]
    latent_height, latent_width = HEIGHT // 14, WIDTH // 14
    condition = torch.zeros(
        1, VIEWS, LATENT_DIM, latent_height, latent_width, device=device
    )
    condition[:, :1] = (
        reference.float() - statistics["ref_mean"]
    ) / statistics["ref_std"]
    condition_mask = torch.zeros(
        1, VIEWS, 1, latent_height, latent_width, device=device
    )
    condition_mask[:, :1] = 1
    plucker = latent_plucker(c2w, intrinsics, opencv_camera_to_plucker).float()

    normalized = sample(
        model,
        rae,
        condition,
        plucker,
        condition_mask,
        text,
        text_lens,
        c2w,
        intrinsics,
        statistics,
        sigmas,
        args.seed,
    )
    latent = normalized * statistics["target_std"] + statistics["target_mean"]
    views, video = decode_scene(rae, latent, c2w, intrinsics, args.video_frames)
    if rank == 0:
        save_outputs(
            Path(args.out),
            source_image,
            views.cpu(),
            None if video is None else video.cpu(),
            args.prompt,
            args.seed,
            args.video_fps,
        )
        print(
            f"OneWorld inference complete: step={checkpoint_step} "
            f"saved_world={saved_world} runtime_world={world} out={args.out}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
