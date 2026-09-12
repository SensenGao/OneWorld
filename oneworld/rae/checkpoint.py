"""Checkpoint loading for the released OneWorld RAE."""

from __future__ import annotations

import gc

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from oneworld.rae.model import RAE


def load_latent_encoder(encoder, checkpoint_path: str) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
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
    state = checkpoint.get("rae", {})
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
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
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

    parameters = dict(model.named_parameters())
    state = checkpoint.get("rae", {})
    extra = [name for name in state if name not in parameters]
    if extra:
        raise RuntimeError(f"RAE state has unknown parameters: {extra[:8]}")
    for name in ("decoder_denorm_scale", "decoder_denorm_bias"):
        if name not in state:
            raise RuntimeError(f"RAE checkpoint is missing {name}")
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
