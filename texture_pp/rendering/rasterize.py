"""Forward rasterisation via nvdiffrast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def _render(glctx, model_mtx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level, normals=None, background_color=None, texture_filter_mode=None):
    import nvdiffrast.torch as dr

    pos_clip = _transform_pos(mtx, pos)
    if pos.shape[-1] == 3:
        pos = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    alpha_mask = rast_out[..., -1] > 0
    uv_flipped = uv.clone()
    uv_flipped[..., 1] = 1.0 - uv_flipped[..., 1]
    screen_tri_id = rast_out[..., -1].int()

    if texture_filter_mode == 'linear':
        texc, _ = dr.interpolate(uv_flipped[None, ...], rast_out, uv_idx)
        image = dr.texture(tex[None, ...] if tex.dim() == 3 else tex,
                           texc,
                           filter_mode='linear')
    elif enable_mip and texture_filter_mode is None:
        texc, texd = dr.interpolate(uv_flipped[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        image = dr.texture(tex[None, ...] if tex.dim() == 3 else tex,
                           texc, texd,
                           filter_mode='linear-mipmap-linear',
                           max_mip_level=max_mip_level)
    else:
        texc, texd = dr.interpolate(uv_flipped[None, ...], rast_out, uv_idx)
        image = dr.texture(tex[None, ...] if tex.dim() == 3 else tex,
                           texc,
                           filter_mode=texture_filter_mode or 'nearest')

    mask = torch.clamp(rast_out[..., -1:], 0.0, 1.0)
    if background_color is not None:
        bg_color = torch.tensor(background_color, device=image.device, dtype=image.dtype)
        bg_color = bg_color.view(1, 1, 1, -1)
        image[..., :3] = image[..., :3] * mask + bg_color * (1.0 - mask)

    uv_map, _ = dr.interpolate(uv_flipped, rast_out, uv_idx)
    depth_ndc = pos_clip[..., 2:3] / (pos_clip[..., 3:4] + 1e-8)
    depth_map, _ = dr.interpolate(depth_ndc.contiguous(), rast_out, pos_idx)

    model_mtx = torch.from_numpy(model_mtx).to(normals.device)
    normal_map = None
    if normals is not None:
        normal_matrix = torch.transpose(torch.inverse(model_mtx), -1, -2)[:3, :3]
        normals_world = torch.matmul(normals.unsqueeze(0), normal_matrix.unsqueeze(0).transpose(-1, -2)).squeeze(0)
        normal_map, _ = dr.interpolate(normals_world, rast_out, pos_idx)
        if normal_map.dim() == 4:
            normal_map = normal_map[0]
        normal_map = F.normalize(normal_map, dim=-1)

    world_pos = pos @ model_mtx.T
    world_pos_map, _ = dr.interpolate(world_pos, rast_out, pos_idx)
    if world_pos_map.dim() == 4:
        world_pos_map = world_pos_map[0]
    world_pos_map = world_pos_map[..., :3]
    return image[0][..., :3], uv_map[0], normal_map, world_pos_map, alpha_mask[0], depth_map[0], screen_tri_id[0]


@dataclass
class RenderResult:
    image: torch.Tensor
    uv_map: torch.Tensor
    normal_map: Optional[torch.Tensor]
    world_pos_map: torch.Tensor
    alpha_mask: torch.Tensor
    depth_map: torch.Tensor
    screen_tri_id: torch.Tensor


def render_view(
    glctx,
    view: dict,
    *,
    vtx_pos: torch.Tensor,
    pos_idx: torch.Tensor,
    vtx_uv: torch.Tensor,
    uv_idx: torch.Tensor,
    tex: torch.Tensor,
    vtx_normals: torch.Tensor,
    ref_res: int,
    enable_mip: bool = True,
    max_mip_level: int = 0,
    background: Sequence[float] = (1.0, 1.0, 1.0),
    texture_filter_mode: str | None = None,
) -> RenderResult:
    from texture_pp.rendering.proj_matrix import make_proj_matrix

    view_mat = view["view"]
    proj_matrix = make_proj_matrix(view["persp"], ref_res)
    vp = np.matmul(proj_matrix, view_mat).astype(np.float32)

    image, uv_map, normal_map, world_pos_map, alpha_mask, depth_map, screen_tri_id = _render(
        glctx,
        np.eye(4, dtype=np.float32),
        vp,
        vtx_pos,
        pos_idx,
        vtx_uv,
        uv_idx,
        tex,
        ref_res,
        enable_mip,
        max_mip_level,
        vtx_normals,
        background_color=list(background),
        texture_filter_mode=texture_filter_mode,
    )
    return RenderResult(
        image=image,
        uv_map=uv_map,
        normal_map=normal_map,
        world_pos_map=world_pos_map,
        alpha_mask=alpha_mask,
        depth_map=depth_map,
        screen_tri_id=screen_tri_id,
    )
