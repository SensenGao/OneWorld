"""Checkpoint loading for the released OneWorld RAE."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from oneworld.rae.model import RAE
from oneworld.rae.release import read_named_tensors


def read_rae_checkpoint(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix != ".json":
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = json.loads(path.read_text())
    source = checkpoint.get("tensor_source", {})
    if source.get("format") != "safetensors" or source.get("prefix") != "rae.":
        raise RuntimeError("unsupported RAE tensor source")
    weights = (path.parent / source["file"]).resolve()
    if not weights.is_relative_to(path.parent.resolve()):
        raise RuntimeError("RAE tensor source escapes model directory")
    checkpoint["rae"] = read_named_tensors(weights, "rae.")
    return checkpoint


def _compatible_v19_state(model, state, contract):
    """Restore the deterministic appearance path in the original v19 release."""
    state = dict(state)
    # This auxiliary Wan alignment head is absent from the inference graph.
    for name in ("wan_alignment_projection.weight", "wan_alignment_projection.bias"):
        state.pop(name, None)
    posterior = {
        f"posterior_{branch}.{kind}"
        for branch in ("mean", "logvar") for kind in ("weight", "bias")
    }
    saved = posterior.intersection(state)
    if saved and saved != posterior:
        raise RuntimeError(f"incomplete RAE posterior state: {sorted(posterior - saved)}")
    if not saved:
        if contract.get("dit_latent") != (
                "concat(layernorm_stock_pi3_z18,layernorm_rgb32);"
                "reversible_per_resolution_channel_standardization_deferred_to_dit"):
            raise RuntimeError("missing RAE posterior without the legacy v19 latent contract")
        # Original v19 encoded normalized appearance directly.  Only inference
        # uses this mean; zero logvar is a neutral compatibility value.
        state["posterior_mean.weight"] = torch.eye(
            model.appearance_dim, dtype=model.posterior_mean.weight.dtype
        ).reshape_as(model.posterior_mean.weight)
        for name in sorted(posterior - {"posterior_mean.weight"}):
            state[name] = torch.zeros_like(dict(model.named_parameters())[name])
    return state


def _checkpoint_state(model, checkpoint):
    state = checkpoint.get("rae", {})
    legacy_identity = not any(name.startswith(("posterior_mean.", "posterior_logvar."))
                              for name in state)
    repaired = checkpoint.get("format") == "oneworld-rae-inference-v2"
    if not repaired:
        state = _compatible_v19_state(model, state, checkpoint.get("contract", {}))
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name in parameters and tuple(value.shape) != tuple(parameters[name].shape):
            raise RuntimeError(f"RAE parameter shape differs: {name}")
    if repaired:
        posterior = {f"posterior_{branch}.{kind}" for branch in ("mean", "logvar")
                     for kind in ("weight", "bias")}
        if not posterior.issubset(state):
            raise RuntimeError("repaired RAE is missing its saved posterior")
        if any(name.startswith("wan_alignment_projection.") for name in state):
            raise RuntimeError("repaired RAE contains an obsolete alignment module")
        if checkpoint.get("contract", {}).get("appearance_posterior") != "identity":
            raise RuntimeError("repaired RAE has an unknown appearance posterior")
        mean = state["posterior_mean.weight"]
        identity = torch.eye(model.appearance_dim, dtype=mean.dtype, device=mean.device).reshape_as(mean)
        if not torch.equal(mean, identity) or any(
                bool(state[name].count_nonzero()) for name in posterior - {"posterior_mean.weight"}):
            raise RuntimeError("repaired RAE identity posterior differs from its contract")
    # The original v19 graph has no posterior convolution. Even an identity
    # convolution rounds FP32 normalized appearance under BF16 autocast.
    # Select the original operation only after its checkpoint contract validates.
    model.appearance_identity = repaired or legacy_identity
    return state


def validate_rae_parameters(model, state):
    parameters = dict(model.named_parameters())
    extra = [name for name in state if name not in parameters]
    if extra:
        raise RuntimeError(f"RAE state has unknown parameters: {extra[:8]}")
    trained_pi3 = tuple(
        [f"encoder.pi3x.decoder.{i}." for i in range(18, 36)]
        + ["encoder.pi3x.pose_inject_blk.3.", "encoder.pi3x.pose_inject_blk.4.",
           "encoder.pi3x.point_decoder.", "encoder.pi3x.point_head."])
    missing = [name for name in parameters if name not in state and (
        not name.startswith("encoder.pi3x.") or name.startswith(trained_pi3))]
    if missing:
        raise RuntimeError(f"RAE checkpoint is missing parameters: {missing[:8]}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape:
            raise RuntimeError(f"RAE parameter shape differs: {name}")
    return parameters


def load_latent_encoder(encoder, checkpoint_path: str) -> dict:
    checkpoint = read_rae_checkpoint(checkpoint_path)
    contract = checkpoint.get("contract", {})
    expected = {
        "version": 19,
        "boundary_block_zero_based": 17,
        "boundary_attention": "global",
        "context_views": 8,
        "latent_concat_dim": 1056,
        "resolutions_hw": [
            [224, 448],
            [448, 448],
            [252, 448],
            [336, 448],
        ],
    }
    mismatched = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatched:
        raise ValueError(f"unexpected RAE latent contract: {mismatched}")

    parameters = dict(encoder.named_parameters())
    state = _checkpoint_state(encoder, checkpoint)
    prefixes = ("appearance_encoder.", "posterior_mean.", "posterior_logvar.")
    required = {name for name in parameters if name.startswith(prefixes)}
    saved = {name for name in state if name.startswith(prefixes)}
    if required != saved:
        raise RuntimeError(
            "incomplete appearance encoder state: "
            f"missing={sorted(required - saved)[:8]} "
            f"extra={sorted(saved - required)[:8]}"
        )
    with torch.no_grad():
        for name in sorted(required):
            parameters[name].copy_(
                state[name].to(
                    parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )
    return {
        "step": int(checkpoint.get("step", -1)),
        "contract": contract,
        "loaded_parameters": len(required),
    }


def load_rae_model(pi3x, checkpoint_path: str, device):
    checkpoint = read_rae_checkpoint(checkpoint_path)
    contract = checkpoint.get("contract", {})
    expected = {
        "version": 19,
        "boundary_block_zero_based": 17,
        "boundary_attention": "global",
        "context_views": 8,
        "heldout_target_views": 8,
        "latent_concat_dim": 1056,
    }
    mismatched = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"unexpected RAE contract: {mismatched}")

    saved = checkpoint.get("args", {})
    model = RAE(
        pi3x,
        appearance_dim=int(saved.get("appearance_dim", 32)),
        appearance_hidden_dim=int(saved.get("appearance_hidden_dim", 384)),
        appearance_depth=int(saved.get("appearance_depth", 6)),
        appearance_heads=int(saved.get("appearance_heads", 8)),
        appearance_transformer_dropout=float(
            saved.get("appearance_transformer_dropout", 0.0)
        ),
        rgb_skip_drop=float(saved.get("appearance_drop", 0.0)),
        features=int(saved.get("features", 256)),
        grad_checkpoint=False,
        level_drop_prob=float(saved.get("level_drop", 0.0)),
        sh_degree=int(
            saved.get("sh_degree", contract.get("gaussians", {}).get("sh_degree", 4))
        ),
        gaussian_ray_mode=saved.get("gs_ray_mode", "z_depth"),
        detail_hidden_dim=int(saved.get("rgb_detail_hidden_dim", 0)),
        detail_merge=saved.get("rgb_detail_merge", "feature"),
    ).to(device)
    model.heads.rgb_head._coarse_scale.fill_(
        float(saved.get("rgb_coarse_scale", 1.0))
    )
    model.heads.rgb_head._detail_scale.fill_(
        float(saved.get("rgb_detail_scale", 1.0))
    )

    state = _checkpoint_state(model, checkpoint)
    parameters = validate_rae_parameters(model, state)
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(
                value.to(parameters[name].device, dtype=parameters[name].dtype)
            )
    result = {
        "step": int(checkpoint.get("step", -1)),
        "contract": contract,
        "args": saved,
        "loaded_parameters": len(state),
        "state_format": checkpoint.get("format", "legacy-v19"),
        "legacy_migration": checkpoint.get("format") != "oneworld-rae-inference-v2",
    }
    del checkpoint
    gc.collect()
    return model, result


def _decoder_modules(model: RAE):
    pi3x = model.encoder.pi3x
    return [
        *pi3x.decoder[model.encoder.boundary_block + 1 :],
        pi3x.pose_inject_blk[3],
        pi3x.pose_inject_blk[4],
        model.rgb_to_trunk,
        model.z36_norm,
        model.heads.rgb_dpt,
        model.heads.rgb_head,
    ]


def wrap_rae_decoder_fsdp(model: RAE, fsdp_kwargs: dict):
    pi3x = model.encoder.pi3x
    wrapped = []
    for index in range(model.encoder.boundary_block + 1, len(pi3x.decoder)):
        pi3x.decoder[index] = FSDP(pi3x.decoder[index], **fsdp_kwargs)
        wrapped.append(pi3x.decoder[index])
    for index in (3, 4):
        pi3x.pose_inject_blk[index] = FSDP(
            pi3x.pose_inject_blk[index], **fsdp_kwargs
        )
        wrapped.append(pi3x.pose_inject_blk[index])
    for parent, name in (
        (model, "rgb_to_trunk"),
        (model, "z36_norm"),
        (model.heads, "rgb_dpt"),
        (model.heads, "rgb_head"),
    ):
        setattr(parent, name, FSDP(getattr(parent, name), **fsdp_kwargs))
        wrapped.append(getattr(parent, name))
    return [parameter for module in wrapped for parameter in module.parameters()]


def wrap_rae_mdf_decoder_fsdp(model: RAE, fsdp_kwargs: dict):
    """Shard all trainable latent-to-RGB/3DGS decoder modules."""
    pi3x = model.encoder.pi3x
    wrapped = []
    for index in range(model.encoder.boundary_block + 1, len(pi3x.decoder)):
        pi3x.decoder[index] = FSDP(pi3x.decoder[index], **fsdp_kwargs)
        wrapped.append(pi3x.decoder[index])
    for index in (3, 4):
        pi3x.pose_inject_blk[index] = FSDP(
            pi3x.pose_inject_blk[index], **fsdp_kwargs)
        wrapped.append(pi3x.pose_inject_blk[index])
    for parent, name in (
        (pi3x, "point_decoder"),
        (pi3x, "point_head"),
        (model, "rgb_to_trunk"),
        (model, "z36_norm"),
        (model.heads, "rgb_dpt"),
        (model.heads, "rgb_head"),
        (model.heads, "gs_dpt"),
        (model.heads, "gs_input_merger"),
        (model.heads, "gs_head"),
    ):
        setattr(parent, name, FSDP(getattr(parent, name), **fsdp_kwargs))
        wrapped.append(getattr(parent, name))
    parameters = [
        parameter for module in wrapped for parameter in module.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return wrapped, parameters


def set_rae_inference(model: RAE) -> None:
    model.requires_grad_(False)
    for module in _decoder_modules(model):
        module.eval()
    model.eval()
