"""Single-source camera normalization for the Pi3 boundary RAE.

The same functions in this file are part of the latent contract and must be used by
RAE training, latent-statistics extraction, DiT training and sampling.  In particular,
Pi3X's internal random pose rescaling must not be used for a generative latent target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CanonicalCameraBatch:
    """Canonical OpenCV camera-to-world matrices and their removed scene scale."""

    c2w: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor


def canonicalize_cameras(
    c2w: torch.Tensor,
    reference_index: int = 0,
    scale_view_count: int | None = None,
    max_translation_norm: float | None = None,
    min_baseline: float = 1e-6,
    strict: bool = True,
) -> CanonicalCameraBatch:
    """Apply reference-relative pose and max-component scale normalization.

    Args:
        c2w: ``(B,V,4,4)`` OpenCV camera-to-world matrices.
        reference_index: reference view.  The 1->7 contract uses view 0.
        scale_view_count: optional number of leading cameras used for scene scale. The
            current RAE passes all 16 sampled cameras so context and NVS share one scale.
        max_translation_norm: optional pre-normalisation outlier threshold.
        min_baseline: scenes below this translation baseline are marked invalid.
        strict: raise if a scene is invalid; otherwise return the per-scene mask.

    Returns:
        ``CanonicalCameraBatch`` where max absolute relative translation component is 1.
        ``scale`` is the removed translation scale, useful for mapping predictions back.
    """
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected c2w (B,V,4,4), got {tuple(c2w.shape)}")
    if not -c2w.shape[1] <= reference_index < c2w.shape[1]:
        raise IndexError(
            f"reference_index={reference_index} invalid for V={c2w.shape[1]}")
    if scale_view_count is None:
        scale_view_count = c2w.shape[1]
    if not 1 < scale_view_count <= c2w.shape[1]:
        raise ValueError(
            f"scale_view_count must be in [2,{c2w.shape[1]}], got {scale_view_count}")

    # Pose algebra is cheap and significantly less stable in bf16.
    with torch.autocast(device_type=c2w.device.type, enabled=False):
        cameras = c2w.float()
        finite = torch.isfinite(cameras).all(dim=(-1, -2, -3))
        ref_inv = torch.linalg.inv(cameras[:, reference_index])
        relative = ref_inv[:, None] @ cameras
        translation = relative[..., :3, 3]

        context_translation = translation[:, :scale_view_count]
        scale = context_translation.abs().amax(dim=(1, 2))
        valid = finite & (scale > min_baseline)
        if max_translation_norm is not None:
            max_norm = torch.linalg.vector_norm(
                context_translation, dim=-1).amax(dim=1)
            valid = valid & (max_norm <= float(max_translation_norm))

        if strict and not bool(valid.all()):
            bad = torch.where(~valid)[0][:8].tolist()
            raise ValueError(
                "invalid camera batch before canonicalization: "
                f"batch_indices={bad}, scales={scale[~valid][:8].tolist()}")

        relative = relative.clone()
        relative[..., :3, 3] /= scale.clamp_min(min_baseline)[:, None, None]

    return CanonicalCameraBatch(c2w=relative, scale=scale, valid=valid)


def canonicalize_pi3_cameras(
    c2w: torch.Tensor,
    reference_index: int = 0,
    scale_view_count: int | None = None,
    max_translation_norm: float | None = None,
    min_baseline: float = 1e-6,
    strict: bool = True,
) -> CanonicalCameraBatch:
    """Deterministic form of Pi3X's native no-depth pose normalization.

    Pi3X first expresses every OpenCV camera-to-world pose relative to view zero and,
    when no depth prior is supplied, divides translation by the mean camera distance
    from that reference.  The released training path additionally multiplies this scale
    by a random ``Uniform(0.8, 1.2)`` augmentation.  A generative latent target must be
    deterministic, so this function retains the pretrained pose convention while fixing
    that augmentation to one.

    All cameras that will participate in reconstruction or rendering must be normalized
    together *before* the context/target split.  ``scale`` records the removed scene
    scale for optional metric-space restoration.
    """
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected c2w (B,V,4,4), got {tuple(c2w.shape)}")
    views = c2w.shape[1]
    if not -views <= reference_index < views:
        raise IndexError(f"reference_index={reference_index} invalid for V={views}")
    if scale_view_count is None:
        scale_view_count = views
    if not 1 < scale_view_count <= views:
        raise ValueError(
            f"scale_view_count must be in [2,{views}], got {scale_view_count}")

    with torch.autocast(device_type=c2w.device.type, enabled=False):
        cameras = c2w.float()
        finite = torch.isfinite(cameras).all(dim=(-1, -2, -3))
        ref_inv = torch.linalg.inv(cameras[:, reference_index])
        relative = ref_inv[:, None] @ cameras
        translation = relative[:, :scale_view_count, :3, 3]

        # Pi3X's no-depth branch averages ||t_i|| over views 1..N-1.  Generalize the
        # exclusion to ``reference_index`` while keeping the exact statistic.
        keep = torch.ones(scale_view_count, device=c2w.device, dtype=torch.bool)
        if 0 <= reference_index < scale_view_count:
            keep[reference_index] = False
        distances = torch.linalg.vector_norm(translation[:, keep], dim=-1)
        scale = distances.mean(dim=1)
        valid = finite & torch.isfinite(scale) & (scale > min_baseline)
        if max_translation_norm is not None:
            raw_max = torch.linalg.vector_norm(translation, dim=-1).amax(dim=1)
            valid = valid & (raw_max <= float(max_translation_norm))

        if strict and not bool(valid.all()):
            bad = torch.where(~valid)[0][:8].tolist()
            raise ValueError(
                "invalid camera batch before Pi3 normalization: "
                f"batch_indices={bad}, scales={scale[~valid][:8].tolist()}")

        relative = relative.clone()
        relative[..., :3, 3] /= scale.clamp_min(min_baseline)[:, None, None]

    return CanonicalCameraBatch(c2w=relative, scale=scale, valid=valid)
