"""Batched gsplat rendering for the OneWorld 3DGS decoder."""

import torch
from gsplat import rasterization


def _voxel_fuse(means, scales, quats, opacities, sh, voxel_size):
    """Aggregate multi-view splats inside world-space voxels."""
    if voxel_size <= 0:
        return means, scales, quats, opacities, sh
    if means.shape[0] == 0:
        raise ValueError("cannot voxel-fuse an empty Gaussian cloud")
    voxel = torch.round(means.detach().float() / float(voxel_size)).to(torch.int64)
    _unique, inverse = torch.unique(voxel, dim=0, return_inverse=True)
    count = int(inverse.max().item()) + 1

    # Opacity is the only learned per-splat reliability signal available here.  Keep
    # the weighting differentiable but detach its denominator role so the model cannot
    # lower the reconstruction loss merely by changing fusion assignments/weights.
    weight = opacities.detach().float().clamp_min(1e-4)
    denom = torch.zeros(count, device=means.device, dtype=torch.float32)
    denom.index_add_(0, inverse, weight)
    denom = denom.clamp_min(1e-8)

    def average(value):
        flat = value.float().reshape(value.shape[0], -1)
        fused = torch.zeros(
            count, flat.shape[1], device=value.device, dtype=torch.float32)
        fused.index_add_(0, inverse, flat * weight[:, None])
        return (fused / denom[:, None]).reshape(count, *value.shape[1:])

    means = average(means)
    scales = average(scales).clamp_min(1e-8)
    # q and -q encode the same rotation.  Canonicalizing the sign prevents their
    # weighted average from cancelling before normalization.
    canonical_quats = quats * torch.where(
        quats[:, :1] < 0, -torch.ones_like(quats[:, :1]),
        torch.ones_like(quats[:, :1]))
    quats = average(canonical_quats)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sh = average(sh)
    opacities = average(opacities[:, None]).squeeze(-1).clamp(0, 1)
    return means, scales, quats, opacities, sh


def render_views(
    gaussians,
    c2w,                  # (B, V_out, 4, 4) cameras to render from
    K,                    # (B, V_out, 3, 3) pixel units at the render resolution
    width,
    height,
    sh_degree=1,
    random_background=True,
    near_plane=0.01,
    far_plane=1000.0,
    view_mask=None,       # (B, V_src) bool; False drops that view's Gaussians
    voxel_size=0.0,       # world-space fusion grid; 0 preserves raw concatenation
):
    """Rasterize per batch element. Returns rgb (B,V,3,H,W), depth (B,V,H,W), alpha."""
    B = gaussians["means"].shape[0]
    V_out = c2w.shape[1]
    dev = gaussians["means"].device

    rgbs, depths, alphas = [], [], []
    for b in range(B):
        means = gaussians["means"][b]
        scales = gaussians["scales"][b]
        quats = gaussians["quats"][b]
        opac = gaussians["opacities"][b]
        sh = gaussians["sh"][b].permute(0, 2, 1)          # (N, d_sh, 3) as gsplat wants

        if view_mask is not None:
            V_src, h, w = gaussians["per_view"]
            m = view_mask[b].reshape(V_src, 1, 1).expand(V_src, h, w).reshape(-1)
            means, scales, quats, opac, sh = (
                value[m] for value in (means, scales, quats, opac, sh))

        means, scales, quats, opac, sh = _voxel_fuse(
            means, scales, quats, opac, sh, float(voxel_size))

        w2c = torch.linalg.inv(c2w[b].float())            # (V_out, 4, 4)

        if random_background:
            bg = torch.rand(V_out, 3, device=dev, dtype=torch.float32)
        else:
            bg = torch.zeros(V_out, 3, device=dev, dtype=torch.float32)

        out, alpha, _ = rasterization(
            means.float(), quats.float(), scales.float(), opac.float(), sh.float(),
            w2c, K[b].float(), width, height,
            sh_degree=sh_degree,
            render_mode="RGB+ED",
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=0.0,
            backgrounds=bg,
            rasterize_mode="classic",
        )
        rgb, dep = out[..., :3], out[..., 3:4]
        rgbs.append(rgb.permute(0, 3, 1, 2))
        depths.append(dep.squeeze(-1))
        alphas.append(alpha.squeeze(-1))

    return (torch.stack(rgbs), torch.stack(depths), torch.stack(alphas))


def render_diagonal_views(
    gaussians,
    c2w,
    K,
    width,
    height,
    sh_degree=1,
    random_background=False,
):
    """Render source cloud ``v`` only into its own source camera ``v``.

    The ordinary renderer merges all source clouds.  That objective admits a bad
    solution in which colour/opacity errors from one cloud are cancelled by another.
    This batched diagonal pass gives every source cloud an independently identifiable
    reconstruction target while leaving the merged context/NVS objectives unchanged.
    """
    batch = gaussians["means"].shape[0]
    source_views, source_height, source_width = gaussians["per_view"]
    if c2w.shape[:2] != (batch, source_views):
        raise ValueError(
            f"diagonal cameras {tuple(c2w.shape[:2])} do not match "
            f"Gaussian source views {(batch, source_views)}")
    if K.shape[:2] != (batch, source_views):
        raise ValueError("diagonal intrinsics do not match Gaussian source views")
    per_view_count = source_height * source_width

    diagonal = {"per_view": (1, source_height, source_width)}
    for key in ("means", "scales", "quats", "opacities", "sh"):
        value = gaussians[key]
        if value.shape[1] != source_views * per_view_count:
            raise ValueError(f"Gaussian field {key} violates per_view layout")
        diagonal[key] = value.reshape(
            batch * source_views, per_view_count, *value.shape[2:])
    rgb, depth, alpha = render_views(
        diagonal,
        c2w.reshape(batch * source_views, 1, 4, 4),
        K.reshape(batch * source_views, 1, 3, 3),
        width,
        height,
        sh_degree=sh_degree,
        random_background=random_background,
    )
    return (
        rgb.reshape(batch, source_views, 3, height, width),
        depth.reshape(batch, source_views, height, width),
        alpha.reshape(batch, source_views, height, width),
    )
