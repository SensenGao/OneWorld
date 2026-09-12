"""Convert DPT attributes plus Pi3 depth into pixel-ray-anchored Gaussians.

Pi3's point head predicts ``(xy, log_depth)``.  Only its camera-space depth is part of
the Gaussian-centre contract.  Lateral coordinates are deterministic camera rays from
the source pixel and its pixel-unit intrinsic matrix, matching 3DGen-Pi3.  In
particular, fine-tuning the Pi3 point head can never move a Gaussian off its source ray.
"""

import torch
from vggt.utils.rotation import mat_to_quat

SCALE_MIN = 0.5
SCALE_MAX = 15.0


def scale_multiplier(K_pixel, multiplier=0.1):
    return multiplier * (
        1.0 / K_pixel[..., 0, 0] + 1.0 / K_pixel[..., 1, 1])


def build_gaussians_from_depth(
    gs_raw, depth, K_full, c2w, image_size, sh_degree=1,
    ray_mode="z_depth",
):
    """Use Pi3 depth on exact source-pixel rays as the Gaussian means.

    ``depth`` is Pi3 point-head depth, shaped ``(B,V,H,W)``.  There is no learned XY
    or centre-offset channel. ``z_depth`` preserves the historical Pi3 camera-Z
    convention; ``unit_ray`` reproduces 3DGen-Pi3/AnySplat's normalized-ray
    Gaussian adapter.  Keeping the choice explicit makes the convention testable.
    """
    B, V, _, h, w = gs_raw.shape
    H, W = image_size
    device, dtype = gs_raw.device, gs_raw.dtype
    if (h, w) != (H, W):
        raise ValueError(f"Gaussian DPT must be full resolution, got {(h, w)} != {(H, W)}")
    if depth.shape != (B, V, H, W):
        raise ValueError(
            f"Pi3 depth has shape {tuple(depth.shape)}, expected {(B, V, H, W)}")
    if K_full.shape != (B, V, 3, 3):
        raise ValueError(
            f"pixel intrinsics have shape {tuple(K_full.shape)}, expected {(B, V, 3, 3)}")
    if c2w.shape != (B, V, 4, 4):
        raise ValueError(
            f"c2w has shape {tuple(c2w.shape)}, expected {(B, V, 4, 4)}")
    if ray_mode not in ("z_depth", "unit_ray"):
        raise ValueError(f"unsupported Gaussian ray mode {ray_mode!r}")
    d_sh = (sh_degree + 1) ** 2
    expected_channels = 1 + 3 + 4 + 3 * d_sh
    if gs_raw.shape[2] != expected_channels:
        raise ValueError(
            f"degree-{sh_degree} Gaussian attributes require "
            f"{expected_channels} channels, got {gs_raw.shape[2]}")
    x = gs_raw.permute(0, 1, 3, 4, 2)
    opacity_raw, scale_raw = x[..., 0], x[..., 1:4]
    quaternion_raw, sh_raw = x[..., 4:8], x[..., 8:]

    # K is in pixels at the render resolution.  Evaluate both the historical
    # camera-Z convention and 3DGen-Pi3's normalized-ray convention explicitly.
    # Keep centre geometry in fp32 even when the learned attributes use bf16.  At a
    # 448-pixel render, bf16 ray quantisation is already visible as sub-pixel jitter.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij")
    pixels = torch.stack(
        (xs + 0.5, ys + 0.5, torch.ones_like(xs)), dim=-1)
    rays = torch.einsum(
        "bvij,hwj->bvhwi", torch.linalg.inv(K_full.float()), pixels)
    if ray_mode == "unit_ray":
        rays = torch.nn.functional.normalize(rays, dim=-1, eps=1e-8)
    camera_depth = depth.float().clamp_min(1e-4)
    camera_points = rays * camera_depth.unsqueeze(-1)
    world_points = torch.einsum(
        "bvij,bvhwj->bvhwi", c2w[..., :3, :3].float(), camera_points)
    world_points = world_points + c2w[..., :3, 3].reshape(B, V, 1, 1, 3).float()

    multiplier = scale_multiplier(K_full).reshape(B, V, 1, 1, 1)
    lo, hi = float(multiplier.min()), float(multiplier.max())
    if lo < 1e-5 or hi > 1e-1:
        raise ValueError(
            f"Gaussian scale multiplier {lo:.3e}..{hi:.3e}: K must use pixels")
    if lo <= 1e-4 or hi >= 1e-2:
        return None
    scales = (SCALE_MIN + (SCALE_MAX - SCALE_MIN) * torch.sigmoid(scale_raw))
    scales = scales * camera_depth.unsqueeze(-1) * multiplier
    local_quaternions = (
        quaternion_raw / quaternion_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    # RAE-Pi3 rotates the camera-space covariance by the context c2w rotation. gsplat
    # consumes scale+quaternion rather than a precomputed covariance, so perform the
    # equivalent quaternion composition q_world = q_c2w * q_local.  Pi3/our GS logits
    # use scalar-first WXYZ; VGGT's matrix utility returns scalar-last XYZW.
    camera_xyzw = mat_to_quat(c2w[..., :3, :3].float())
    camera_quaternions = camera_xyzw[..., [3, 0, 1, 2]].to(dtype)
    camera_quaternions = camera_quaternions[:, :, None, None].expand_as(
        local_quaternions)
    aw, ax, ay, az = camera_quaternions.unbind(dim=-1)
    bw, bx, by, bz = local_quaternions.unbind(dim=-1)
    quaternions = torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)
    quaternions = quaternions / quaternions.norm(
        dim=-1, keepdim=True).clamp_min(1e-8)
    opacities = torch.sigmoid(opacity_raw)

    sh = sh_raw.reshape(B, V, h, w, 3, d_sh)
    # Match RAE-Pi3's initialization prior: retain DC and attenuate higher-order SH.
    sh_mask = torch.ones(d_sh, device=device, dtype=dtype)
    for degree in range(1, sh_degree + 1):
        sh_mask[degree ** 2:(degree + 1) ** 2] = 0.1 * 0.25 ** degree
    sh = sh * sh_mask
    count = V * h * w
    return {
        "means": world_points.reshape(B, count, 3),
        "scales": scales.reshape(B, count, 3),
        "quats": quaternions.reshape(B, count, 4),
        "opacities": opacities.reshape(B, count),
        "sh": sh.reshape(B, count, 3, d_sh),
        "per_view": (V, h, w),
    }


def build_gaussians_from_points(
    gs_raw, local_points, K_full, c2w, image_size, sh_degree=1,
    ray_mode="z_depth",
):
    """Backward-compatible wrapper; Pi3-predicted XY is intentionally ignored."""
    if local_points.ndim != 5 or local_points.shape[-1] != 3:
        raise ValueError(
            f"Pi3 local points must end in XYZ, got {tuple(local_points.shape)}")
    return build_gaussians_from_depth(
        gs_raw, local_points[..., 2], K_full, c2w, image_size,
        sh_degree=sh_degree, ray_mode=ray_mode)
