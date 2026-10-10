"""Per-texel gather projection: each valid texel samples one SR screen pixel."""

from __future__ import annotations

import numpy as np
import torch


__all__ = ["project_per_texel_gather", "compute_quality_threshold"]

def _to_tensor(
    x: torch.Tensor | np.ndarray,
    device: torch.device | str | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _squeeze_lut(t: torch.Tensor, expect_last_dim: int | None = None) -> torch.Tensor:
    if t.dim() == 4:
        if t.shape[0] != 1:
            raise ValueError(f"expected leading dim 1, got shape {tuple(t.shape)}")
        t = t.squeeze(0)
    if expect_last_dim is None:
        if t.dim() == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        return t
    if t.dim() == 2:
        if expect_last_dim != 1:
            raise ValueError(
                f"got 2D tensor but expected trailing dim {expect_last_dim}"
            )
        return t.unsqueeze(-1)
    return t


def compute_quality_threshold(
    Q_global: torch.Tensor,
    *,
    Q_view: torch.Tensor | None = None,
    abs_eps: float = 1e-8,
    rel_eps: float = 1e-3,
) -> float:
    if rel_eps <= 0:
        return abs_eps

    q_global_pos = Q_global[Q_global > 0]
    scale_global = float(q_global_pos.median()) if q_global_pos.numel() > 0 else 0.0

    scale_view = 0.0
    if Q_view is not None:
        q_view_pos = Q_view[Q_view > 0]
        if q_view_pos.numel() > 0:
            scale_view = float(q_view_pos.median())

    scale = max(scale_global, scale_view)
    return max(abs_eps, rel_eps * scale)


def project_per_texel_gather(
    sr_image: torch.Tensor,
    *,
    screen_tri_id: torch.Tensor,
    view_matrix: torch.Tensor | np.ndarray,
    proj_matrix: torch.Tensor | np.ndarray,
    texture: torch.Tensor,
    pos_lut: torch.Tensor,
    tex_tri_id: torch.Tensor,
    valid_mask: torch.Tensor,
    Q_global: torch.Tensor,
    Q_view: torch.Tensor,
    normal_lut: torch.Tensor | None = None,
    update_mask: torch.Tensor | None = None,
    alpha_mask: torch.Tensor | None = None,
    edge_erode_px: int = 1,
    depth_map: torch.Tensor | None = None,
    depth_match_eps: float = 1e-2,
    min_cos_view_normal: float = 0.3,
    quality_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sr_image.device
    H, W = sr_image.shape[:2]
    TexH, TexW = texture.shape[:2]

    # ---- Normalize LUT shapes (the rendering.lut module returns (1, H, W, C)) ----
    pos_lut_2d = _squeeze_lut(pos_lut)
    normal_lut_2d: torch.Tensor | None = None
    if normal_lut is not None:
        normal_lut_2d = _squeeze_lut(
            normal_lut.to(device), expect_last_dim=3
        )
    tex_tri_id_2d = _squeeze_lut(tex_tri_id).long()
    valid_mask_2d = _squeeze_lut(valid_mask).bool()

    # ---- Build VP matrix ----
    view_t = _to_tensor(view_matrix, device)
    proj_t = _to_tensor(proj_matrix, device)
    vp = proj_t @ view_t

    # ---- Find indices of valid texels ----
    ti, tj = torch.where(valid_mask_2d)
    n_valid = ti.numel()
    empty_mask = torch.zeros((TexH, TexW), device=device, dtype=torch.bool)
    if n_valid == 0:
        return texture.clone(), Q_global.clone(), empty_mask

    # ---- Project: world pos (N, 3) -> clip (N, 4) ----
    world_pos = pos_lut_2d[ti, tj]
    ones = torch.ones((n_valid, 1), device=device, dtype=world_pos.dtype)
    world_pos_h = torch.cat([world_pos, ones], dim=-1)
    clip = world_pos_h @ vp.T

    w = clip[:, 3]
    w_safe = w.abs().clamp_min(1e-12)
    ndc = clip[:, :3] / w_safe.unsqueeze(-1)
    in_front = w > 1e-6

    in_xy = (
        (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    )
    in_z = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    in_frustum = in_front & in_xy & in_z

    # ---- Map NDC to integer screen pixel ----
    sx_cont = (ndc[:, 0] + 1.0) * 0.5 * W
    sy_cont = (ndc[:, 1] + 1.0) * 0.5 * H
    sx = sx_cont.long().clamp_(0, W - 1)
    sy = sy_cont.long().clamp_(0, H - 1)

    # ---- Gates ----
    gate = in_frustum.clone()

    if update_mask is not None:
        um = update_mask.to(device).bool()
        gate &= um[sy, sx]

    if alpha_mask is not None and edge_erode_px > 0:
        am = alpha_mask.to(device).float().view(1, 1, H, W)
        k = edge_erode_px * 2 + 1
        am_eroded = -torch.nn.functional.max_pool2d(
            -am, kernel_size=k, stride=1, padding=edge_erode_px,
        )
        am_eroded_2d = (am_eroded.view(H, W) > 0.5)
        gate &= am_eroded_2d[sy, sx]
    elif alpha_mask is not None:
        gate &= alpha_mask.to(device).bool()[sy, sx]

    if normal_lut_2d is not None:
        cam_pos = -(view_t[:3, :3].T @ view_t[:3, 3])
        view_dir = torch.nn.functional.normalize(
            cam_pos - world_pos,
            dim=-1, eps=1e-8,
        )
        n = torch.nn.functional.normalize(
            normal_lut_2d[ti, tj],
            dim=-1, eps=1e-8,
        )
        cos_angle = (view_dir * n).sum(dim=-1).abs().clamp(0.0, 1.0)
        gate &= cos_angle > min_cos_view_normal

    actual_tri = screen_tri_id.to(device).long()[sy, sx]
    expected_tri = tex_tri_id_2d[ti, tj]
    tri_match = actual_tri == expected_tri

    if depth_map is not None and depth_match_eps > 0:
        dm = depth_map.to(device)
        if dm.dim() == 3 and dm.shape[-1] == 1:
            dm = dm.squeeze(-1)
        elif dm.dim() == 3:
            dm = dm[..., 0]
        texel_ndc_z = ndc[:, 2]
        screen_ndc_z = dm[sy, sx].to(texel_ndc_z.dtype)
        foreground = actual_tri > 0
        depth_match = (screen_ndc_z - texel_ndc_z).abs() < depth_match_eps
        gate &= tri_match | (foreground & depth_match)
    else:
        gate &= tri_match

    if quality_threshold is None:
        quality_threshold = compute_quality_threshold(Q_global, Q_view=Q_view)
    q_pixel = Q_view.to(device).float()[sy, sx]
    q_existing = Q_global[ti, tj]
    gate &= q_pixel > q_existing + float(quality_threshold)

    # ---- Apply ----
    out_tex = texture.clone()
    Q_new = Q_global.clone()
    write_mask = empty_mask.clone()

    if gate.any():
        win_ti = ti[gate]
        win_tj = tj[gate]
        win_sy = sy[gate]
        win_sx = sx[gate]

        out_tex[win_ti, win_tj] = sr_image[win_sy, win_sx]
        Q_new[win_ti, win_tj] = q_pixel[gate]
        write_mask[win_ti, win_tj] = True

    return out_tex, Q_new, write_mask
