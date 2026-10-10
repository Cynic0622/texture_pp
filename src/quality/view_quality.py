"""Per-view quality and global-quality sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_view_quality(
    world_pos_map: torch.Tensor,
    normal_map: torch.Tensor,
    alpha_mask: torch.Tensor,
    view_mat: np.ndarray,
) -> torch.Tensor:
    cam_pos = -view_mat[:3, :3].T @ view_mat[:3, 3]
    cam_pos_t = torch.from_numpy(cam_pos).to(world_pos_map.device)

    view_vec = cam_pos_t - world_pos_map
    dist_sq = torch.sum(view_vec ** 2, dim=-1)
    view_dirs = F.normalize(view_vec, dim=-1, eps=1e-8)
    n = F.normalize(normal_map, dim=-1, eps=1e-8)

    cos_angles = torch.sum(view_dirs * n, dim=-1).clamp(0.0, 1.0)
    w_angle = cos_angles ** 2
    w_dist = 1.0 / (dist_sq + 1e-4)
    facing = (cos_angles > 0.0).float()
    return w_angle * w_dist * alpha_mask.float() * facing


def sample_global_quality(Q_global: torch.Tensor, uv_map: torch.Tensor) -> torch.Tensor:
    uv_coords = (uv_map * 2.0 - 1.0).unsqueeze(0)
    sampled = (
        F.grid_sample(
            Q_global.unsqueeze(0).unsqueeze(0),
            uv_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .squeeze(-1)
    )
    return sampled
