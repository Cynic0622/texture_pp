"""Texture-space LUTs (position, normal, triangle ID) and morphological dilation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dilate_lut(lut: torch.Tensor, mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    lut_filled = lut.clone()
    mask_curr = mask.clone()
    kernel = torch.ones(1, 1, 3, 3, device=lut.device)

    for _ in range(iterations):
        sum_val = F.conv2d(
            (lut_filled * mask_curr).permute(0, 3, 1, 2),
            kernel.expand(lut.shape[-1], 1, 3, 3),
            padding=1,
            groups=lut.shape[-1],
        )
        sum_mask = F.conv2d(mask_curr.permute(0, 3, 1, 2), kernel, padding=1)
        avg_val = sum_val / (sum_mask + 1e-8)
        avg_val = avg_val.permute(0, 2, 3, 1)
        mask_dilated = (sum_mask > 0).permute(0, 2, 3, 1).float()
        boundary_mask = (mask_dilated - mask_curr) > 0.5
        if boundary_mask.sum() == 0:
            break
        lut_filled[boundary_mask.expand_as(lut_filled)] = avg_val[boundary_mask.expand_as(lut_filled)]
        mask_curr = mask_dilated
    return lut_filled


def dilate_id_lut(lut: torch.Tensor, mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    lut_filled = lut.float().clone()
    mask_curr = mask.clone()
    for _ in range(iterations):
        lut_dilated = F.max_pool2d(
            lut_filled.permute(0, 3, 1, 2),
            kernel_size=3, stride=1, padding=1,
        ).permute(0, 2, 3, 1)
        mask_dilated = F.max_pool2d(
            mask_curr.permute(0, 3, 1, 2),
            kernel_size=3, stride=1, padding=1,
        ).permute(0, 2, 3, 1)
        boundary_mask = (mask_dilated > 0.5) & (mask_curr < 0.5)
        if boundary_mask.sum() == 0:
            break
        lut_filled[boundary_mask] = lut_dilated[boundary_mask]
        mask_curr = (mask_dilated > 0.5).float()
    return lut_filled.int()


def precompute_texture_luts(
    glctx,
    *,
    vtx_pos: torch.Tensor,
    pos_idx: torch.Tensor,
    vtx_uv: torch.Tensor,
    vtx_normals: torch.Tensor,
    tex_hw: tuple,
    ss_factor: int = 4,
) -> Dict[str, torch.Tensor]:
    import nvdiffrast.torch as dr

    t_w, t_h = tex_hw
    uv_flip = vtx_uv.clone()
    uv_flip[..., 1] = 1.0 - uv_flip[..., 1]
    uv_clip = torch.cat(
        [
            uv_flip * 2.0 - 1.0,
            torch.zeros_like(vtx_uv[..., :1]),
            torch.ones_like(vtx_uv[..., :1]),
        ],
        dim=-1,
    )
    ss_w, ss_h = t_w * ss_factor, t_h * ss_factor

    rast_uv, _ = dr.rasterize(glctx, uv_clip.unsqueeze(0), pos_idx, resolution=[ss_w, ss_h])
    pos_lut, _ = dr.interpolate(vtx_pos.unsqueeze(0), rast_uv, pos_idx)
    normal_lut, _ = dr.interpolate(vtx_normals.unsqueeze(0), rast_uv, pos_idx)

    pos_lut = F.interpolate(
        pos_lut.permute(0, 3, 1, 2), size=(t_w, t_h), mode="nearest",
    ).permute(0, 2, 3, 1)
    normal_lut = F.interpolate(
        normal_lut.permute(0, 3, 1, 2), size=(t_w, t_h), mode="nearest",
    ).permute(0, 2, 3, 1)

    valid_mask = (pos_lut.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
    valid_mask_raw = valid_mask.clone()
    pos_lut = dilate_lut(pos_lut, valid_mask, iterations=4)
    normal_lut = dilate_lut(normal_lut, valid_mask, iterations=4)

    tex_tri_id = rast_uv[..., -1].unsqueeze(-1).float()
    tex_tri_id = (
        F.interpolate(
            tex_tri_id.permute(0, 3, 1, 2), size=(t_w, t_h), mode="nearest",
        )
        .permute(0, 2, 3, 1)
        .int()
    )
    tex_tri_id = dilate_id_lut(tex_tri_id, valid_mask, iterations=4)

    return {
        "pos_lut": pos_lut,
        "normal_lut": normal_lut,
        "tex_tri_id": tex_tri_id,
        "valid_mask": valid_mask,
        "valid_mask_raw": valid_mask_raw,
    }
