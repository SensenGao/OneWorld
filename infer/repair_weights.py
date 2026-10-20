#!/usr/bin/env python3
"""Export a self-contained, strictly loadable release from the original weights.

CPU only. The source package is never modified. Run verify_weights.py afterwards.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oneworld.rae.release import (
    decoder_layout, file_sha256, fingerprints_close, generator_layout,
    module_fingerprint, verify_manifest,
)


RAE_ARGS = (
    "appearance_dim", "appearance_hidden_dim", "appearance_depth", "appearance_heads",
    "appearance_transformer_dropout", "appearance_drop", "features", "level_drop",
    "sh_degree", "gs_ray_mode", "rgb_detail_hidden_dim", "rgb_detail_merge",
    "rgb_coarse_scale", "rgb_detail_scale",
)


def build_meta_models(wan_path, pi3_path, checkpoint):
    from oneworld.models.wan.modules.model import WanModel
    from oneworld.rae.wan_flow import WanPi3Flow
    from oneworld.rae.model import RAE
    from pi3.models.pi3x import Pi3X
    wan_config = {k: v for k, v in json.loads((wan_path / "config.json").read_text()).items()
                  if not k.startswith("_")}
    saved = checkpoint["args"]
    with torch.device("meta"):
        generator = WanPi3Flow(WanModel(**wan_config), max_views=8, head_depth=2,
                               latent_dim=1056, aligned_input=True)
    # Public DINO computes scalar dropout rates during construction.  Build on
    # CPU before discarding its values; no backbone weights or GPU are needed.
    with torch.device("cpu"):
        pi3 = Pi3X(**json.loads((pi3_path / "config.json").read_text()))
    pi3.to("meta")
    with torch.device("meta"):
        rae = RAE(
            pi3, appearance_dim=int(saved.get("appearance_dim", 32)),
            appearance_hidden_dim=int(saved.get("appearance_hidden_dim", 384)),
            appearance_depth=int(saved.get("appearance_depth", 6)),
            appearance_heads=int(saved.get("appearance_heads", 8)),
            appearance_transformer_dropout=float(saved.get("appearance_transformer_dropout", 0)),
            rgb_skip_drop=float(saved.get("appearance_drop", 0)),
            features=int(saved.get("features", 256)), grad_checkpoint=False,
            level_drop_prob=float(saved.get("level_drop", 0)),
            sh_degree=int(saved.get("sh_degree", checkpoint["contract"]["gaussians"]["sh_degree"])),
            gaussian_ray_mode=saved.get("gs_ray_mode", "z_depth"),
            detail_hidden_dim=int(saved.get("rgb_detail_hidden_dim", 0)),
            detail_merge=saved.get("rgb_detail_merge", "feature"))
    return generator, rae


def validate_shards(checkpoint, layout):
    first = torch.load(checkpoint / "rank00000.pt", map_location="cpu", mmap=True, weights_only=False)
    world = int(first["world_size"])
    if world != 16 or first["format"] != "oneworld-rank-local-inference-v1":
        raise RuntimeError("expected the official 16-shard inference release")
    actual = {field: torch.zeros(3, dtype=torch.float64) for field in ("ema", "decoder_ema")}
    sizes = {field: [t.numel() for t in first[field]] for field in actual}
    for field, groups in layout.items():
        if field == "format":
            continue
        full_sizes = [sum(p["numel"] for p in g["parameters"]) for g in groups]
        if sizes[field] != [(n + world - 1) // world for n in full_sizes]:
            raise RuntimeError(f"source {field} shard layout differs from model")
    for rank in range(world):
        payload = torch.load(checkpoint / f"rank{rank:05d}.pt", map_location="cpu", mmap=True, weights_only=False)
        for key, value in (("rank", rank), ("world_size", world), ("step", first["step"]), ("format", first["format"])):
            if payload.get(key) != value:
                raise RuntimeError(f"source shard metadata differs: {rank}, {key}")
        for field in actual:
            if len(payload[field]) != len(sizes[field]):
                raise RuntimeError(f"source parameter groups differ: {rank}, {field}")
            if not torch.equal(payload["fingerprints"][field], first["fingerprints"][field]):
                raise RuntimeError(f"source fingerprint metadata differs: {rank}, {field}")
            for index, tensor in enumerate(payload[field]):
                if tensor.ndim != 1 or tensor.numel() != sizes[field][index] or tensor.dtype != first[field][index].dtype:
                    raise RuntimeError(f"source shard shape/dtype differs: {rank}, {field}, {index}")
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("nonfinite source checkpoint tensor")
                value = tensor.double()
                actual[field] += torch.tensor([value.numel(), value.sum(), value.square().sum()], dtype=torch.float64)
        print(f"Validated source shard {rank + 1}/{world}", flush=True)
    for field in actual:
        if not fingerprints_close(first["fingerprints"][field], actual[field]):
            raise RuntimeError(f"source fingerprint mismatch: {field}")
    return first


def extract_input_projection(checkpoint, root_group, source_world):
    names = {"main_latent_input.weight", "main_latent_input.bias"}
    offsets = {}; offset = 0
    for parameter in root_group["parameters"]:
        if parameter["name"] in names:
            offsets[parameter["name"]] = (offset, parameter)
        offset += parameter["numel"]
    if set(offsets) != names:
        raise RuntimeError("aligned input projection is absent from generator layout")
    shards = [torch.load(checkpoint / f"rank{rank:05d}.pt", map_location="cpu", mmap=True,
                         weights_only=False)["ema"][0] for rank in range(source_world)]
    root = torch.cat(shards)[:offset]
    return {name.removeprefix("main_latent_input."): root[start:start + p["numel"]].reshape(p["shape"]).clone()
            for name, (start, p) in offsets.items()}


def export(source, destination, wan, pi3):
    if destination.exists():
        raise FileExistsError(f"export destination must not exist: {destination}")
    source_config = json.loads((source / "config.json").read_text())
    source_checkpoint = source / source_config["checkpoint"]
    original = torch.load(source / source_config["rae"], map_location="cpu", mmap=True, weights_only=False)
    generator, rae = build_meta_models(wan, pi3, original)
    layout = {"format": "oneworld-parameter-layout-v1",
              "ema": generator_layout(generator), "decoder_ema": decoder_layout(rae)}
    metadata = validate_shards(source_checkpoint, layout)
    projection = extract_input_projection(source_checkpoint, layout["ema"][0], metadata["world_size"])
    state = dict(original["rae"])
    removed = []
    for name in ("wan_alignment_projection.weight", "wan_alignment_projection.bias"):
        if name in state:
            removed.append(name); del state[name]
    posterior = {f"posterior_{branch}.{kind}" for branch in ("mean", "logvar") for kind in ("weight", "bias")}
    if any(name in state for name in posterior):
        raise RuntimeError("source already contains a posterior; this exporter expects the original release")
    if original["contract"].get("dit_latent") != (
            "concat(layernorm_stock_pi3_z18,layernorm_rgb32);"
            "reversible_per_resolution_channel_standardization_deferred_to_dit"):
        raise RuntimeError("source does not use the original deterministic v19 latent contract")
    state["posterior_mean.weight"] = torch.eye(rae.appearance_dim).reshape(rae.appearance_dim, rae.appearance_dim, 1, 1)
    state["posterior_mean.bias"] = torch.zeros(rae.appearance_dim)
    state["posterior_logvar.weight"] = torch.zeros_like(state["posterior_mean.weight"])
    state["posterior_logvar.bias"] = torch.zeros(rae.appearance_dim)
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"nonfinite RAE state: {name}")
    contract = copy.deepcopy(original["contract"])
    contract["appearance_posterior"] = "identity"
    contract["decoder_denorm"]["statistics"] = "embedded_in_rae.pt"
    repaired = {"format": "oneworld-rae-inference-v2", "step": original["step"], "contract": contract,
                "args": {k: original["args"][k] for k in RAE_ARGS if k in original["args"]}, "rae": state}
    from oneworld.rae.checkpoint import _checkpoint_state, validate_rae_parameters
    validate_rae_parameters(rae, _checkpoint_state(rae, repaired))
    destination.mkdir(parents=True)
    output_checkpoint = destination / "checkpoint"
    output_checkpoint.mkdir()
    for rank in range(metadata["world_size"]):
        filename = f"rank{rank:05d}.pt"
        shutil.copy2(source_checkpoint / filename, output_checkpoint / filename)
    torch.save(repaired, destination / "rae.pt")
    reloaded = torch.load(destination / "rae.pt", map_location="cpu", mmap=True, weights_only=False)
    for name, value in original["rae"].items():
        if name not in removed and not torch.equal(value, reloaded["rae"][name]):
            raise RuntimeError(f"export changed a trained RAE parameter: {name}")
    text = torch.load(source_checkpoint / "text_projector.pt", map_location="cpu", weights_only=False)
    projector = nn.Sequential(nn.Linear(4096, 1536), nn.GELU(approximate="tanh"), nn.Linear(1536, 1536))
    projector.load_state_dict(text["ema"], strict=True)
    fingerprint = module_fingerprint(projector)
    if not fingerprints_close(text["ema_fingerprint"], fingerprint):
        raise RuntimeError("source text projector fingerprint mismatch")
    text["ema_fingerprint"] = fingerprint
    text["fingerprint_comparison"] = {"count": "exact", "rtol": 1e-10, "atol": 1e-5}
    torch.save(text, output_checkpoint / "text_projector.pt")
    torch.save({"format": "pi3-v19-dc-videogen-input-align-v1", "step": metadata["step"],
                "contract": {"input_channels": 1056, "output_channels": 1536},
                "input_projection": projection, "source": "final_generator_ema"}, destination / "input_alignment.pt")
    shutil.copy2(source / source_config["stats"], destination / "latent_stats.pt")
    config = {**source_config, "format": "oneworld-4step-v2", "checkpoint": "checkpoint",
              "rae": "rae.pt", "stats": "latent_stats.pt", "input_alignment": "input_alignment.pt",
              "aligned_input": True, "decoder_layout": "rae_decoder_v1", "manifest": "manifest.json"}
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    student_contract = json.loads((source_checkpoint / "contract.json").read_text())
    student_contract.update(format="oneworld-4step-v2", decoder_layout="rae_decoder_v1",
                            appearance_posterior="identity", input_architecture="aligned", parameter_layout="layout.json")
    (output_checkpoint / "contract.json").write_text(json.dumps(student_contract, indent=2) + "\n")
    (output_checkpoint / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# OneWorld repaired inference package\n\n"
        "Use with the matching repaired OneWorld inference code. Trained generator, decoder, "
        "appearance and text parameters are preserved. The v19 appearance posterior is saved "
        "explicitly as identity/zeros, and the actual final EMA input projection is included.\n\n"
        "Run `python infer/verify_weights.py --model /path/to/this/package` before inference.\n")
    files = {str(path.relative_to(destination)): {"size": path.stat().st_size, "sha256": file_sha256(path)}
             for path in sorted(destination.rglob("*")) if path.is_file()}
    source_files = {str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*"))
                    if p.is_file() and ".cache" not in p.relative_to(source).parts}
    for rank in range(metadata["world_size"]):
        name = f"checkpoint/rank{rank:05d}.pt"
        if files[name]["sha256"] != source_files[f"{source_config['checkpoint']}/rank{rank:05d}.pt"]:
            raise RuntimeError(f"export changed source shard bytes: {rank}")
    if files["latent_stats.pt"]["sha256"] != source_files[source_config["stats"]]:
        raise RuntimeError("export changed latent statistics")
    manifest = {"format": "oneworld-release-manifest-v1", "files": files,
                "source_files_sha256": source_files, "trained_parameters_changed": False,
                "removed_training_only_parameters": removed, "added_deterministic_parameters": sorted(posterior),
                "decoder_layout": "rae_decoder_v1", "source_world_size": metadata["world_size"],
                "source_step": metadata["step"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_manifest(destination)
    print(f"Repaired release exported and verified: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wan", type=Path, required=True)
    parser.add_argument("--pi3", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(f"export destination must not exist: {destination}")
    staging = destination.with_name(destination.name + ".exporting")
    export(args.source.resolve(), staging, args.wan.resolve(), args.pi3.resolve())
    staging.rename(destination)
    print(f"Complete release installed atomically at {destination}")


if __name__ == "__main__":
    main()
