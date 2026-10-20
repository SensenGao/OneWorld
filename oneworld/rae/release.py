"""CPU-readable parameter layouts and integrity checks for inference releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints_close(expected, actual, *, compare_count=True):
    if expected.shape != (3,) or actual.shape != (3,):
        return False
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        return False
    return (not compare_count or torch.equal(expected[:1], actual[:1])) and bool(
        torch.allclose(expected[1:], actual[1:], rtol=1e-10, atol=1e-5))


def module_fingerprint(module):
    result = torch.zeros(3, dtype=torch.float64)
    for parameter in module.parameters():
        value = parameter.detach().double().cpu()
        result[0] += value.numel()
        result[1] += value.sum()
        result[2] += value.square().sum()
    return result


def _group(name, parameters):
    return {
        "module": name,
        "parameters": [{"name": key, "shape": list(value.shape),
                        "numel": value.numel()} for key, value in parameters],
    }


def generator_layout(model):
    wrapped = [f"blocks.{i}" for i in range(len(model.blocks))]
    wrapped += [f"head_blocks.{i}" for i in range(len(model.head_blocks))]
    root = [(name, parameter) for name, parameter in model.named_parameters()
            if not any(name.startswith(prefix + ".") for prefix in wrapped)]
    return [_group("", root), *[
        _group(name, model.get_submodule(name).named_parameters()) for name in wrapped]]


def decoder_layout(model):
    names = [f"encoder.pi3x.decoder.{i}" for i in range(
        model.encoder.boundary_block + 1, len(model.encoder.pi3x.decoder))]
    names += ["encoder.pi3x.pose_inject_blk.3", "encoder.pi3x.pose_inject_blk.4",
              "rgb_to_trunk", "z36_norm", "heads.rgb_dpt", "heads.rgb_head"]
    return [_group(name, model.get_submodule(name).named_parameters()) for name in names]


def check_flat_layout(parameters, groups):
    if len(parameters) != len(groups):
        raise RuntimeError("checkpoint parameter layout group count differs")
    for index, (parameter, group) in enumerate(zip(parameters, groups)):
        expected_names = [p["name"] for p in group["parameters"]]
        expected_shapes = [p["shape"] for p in group["parameters"]]
        if list(getattr(parameter, "_fqns", ())) != expected_names:
            raise RuntimeError(f"checkpoint parameter names differ in group {index}")
        if [list(s) for s in getattr(parameter, "_shapes", ())] != expected_shapes:
            raise RuntimeError(f"checkpoint parameter shapes differ in group {index}")


def verify_manifest(model_root):
    root = Path(model_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "oneworld-release-manifest-v1":
        raise RuntimeError("unsupported release manifest")
    files = manifest.get("files", {})
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*")
                    if p.is_file() and ".cache" not in p.relative_to(root).parts
                    and str(p.relative_to(root)) not in ("manifest.json", ".gitattributes")}
    if set(files) != actual_files:
        raise RuntimeError("release file list differs from manifest")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"invalid release path: {name}")
        if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"release checksum mismatch: {name}")
    return manifest


def read_named_tensors(path, prefix):
    """Read one component from a complete, FSDP-independent checkpoint."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}).get("format") != "oneworld-inference-named-v1":
            raise RuntimeError("unsupported named checkpoint format")
        keys = [key for key in handle.keys() if key.startswith(prefix)]
        if not keys:
            raise RuntimeError(f"named checkpoint is missing component {prefix}")
        return {key[len(prefix):]: handle.get_tensor(key) for key in keys}


@torch.no_grad()
def load_named_parameters(path, module, prefix):
    state = read_named_tensors(path, prefix)
    parameters = dict(module.named_parameters())
    if set(state) != set(parameters):
        raise RuntimeError(
            f"named checkpoint parameters differ for {prefix}: "
            f"missing={sorted(set(parameters) - set(state))[:8]} "
            f"extra={sorted(set(state) - set(parameters))[:8]}")
    for name, parameter in parameters.items():
        value = state[name]
        if value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise RuntimeError(f"named checkpoint shape/dtype differs: {prefix}{name}")
    for name, parameter in parameters.items():
        parameter.copy_(state[name].to(parameter.device))
    return len(parameters)
