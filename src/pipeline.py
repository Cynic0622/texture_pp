"""texture_pp inference pipeline."""

from __future__ import annotations

import os
import re
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import nvdiffrast.torch as dr

from src.config import Config
from src.io_utils import OBJ, load_texture, save_image
from src.geometry import select_views
from src.rendering import dilate_lut, precompute_texture_luts, render_view
from src.quality import compute_view_quality, get_new_mask, sample_global_quality
from src.projection import project_per_texel_gather
from src.rendering.proj_matrix import make_proj_matrix
from src.sr.local import LocalSRModel

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")


def resolve_model_path(model_path: str) -> str:
    if not os.path.isdir(model_path):
        return model_path
    dirname = os.path.basename(os.path.normpath(model_path))
    candidate = os.path.join(model_path, dirname + ".obj")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"No {dirname}.obj found in {model_path}")


def _prepare_tensors(mesh, tex_np: np.ndarray) -> Tuple[torch.Tensor, ...]:
    pos = np.array(mesh.unwelded_vertices)
    pos_idx = np.array(mesh.unwelded_indices)
    if pos.shape[1] == 4:
        pos = pos[:, 0:3]
    uv = np.array(mesh.unwelded_uvs)
    uv_idx = np.array(mesh.unwelded_indices)
    normals = np.array(mesh.unwelded_normals)

    return (
        torch.from_numpy(pos.astype(np.float32)).cuda(),
        torch.from_numpy(pos_idx.astype(np.int32)).cuda(),
        torch.from_numpy(uv.astype(np.float32)).cuda(),
        torch.from_numpy(uv_idx.astype(np.int32)).cuda(),
        torch.from_numpy(normals.astype(np.float32)).cuda(),
        torch.from_numpy(tex_np.astype(np.float32)).cuda().contiguous(),
    )


def _init_optimizable_texture(tex_np: np.ndarray, scale_factor: int = 4) -> torch.Tensor:
    t = torch.from_numpy(tex_np).permute(2, 0, 1).unsqueeze(0).cuda()
    up = F.interpolate(t, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().detach().clone().requires_grad_(False)


def super_resolve_texture(cfg: Config) -> torch.Tensor:
    # ---- resolve model path (support directory input) --------------------
    model_path = resolve_model_path(cfg.model_path)
    tex_path = cfg.tex_path
    if tex_path is None:
        tex_path = re.sub(r'\.obj$', '.png', model_path, flags=re.IGNORECASE)

    # ---- load mesh and texture ------------------------------------------
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    mesh = OBJ(model_path)
    tex_np = load_texture(tex_path)

    # ---- tensors + LUTs ---------------------------------------------------
    opt_tex = _init_optimizable_texture(tex_np, cfg.upscale)
    vtx_pos, pos_idx, vtx_uv, uv_idx, vtx_normals, tex = _prepare_tensors(mesh, tex_np)
    max_mip_level = max(int(np.log2(max(tex.shape[:2]) / 512)) + 1, 0)

    glctx = dr.RasterizeGLContext()
    t_w, t_h = opt_tex.shape[0], opt_tex.shape[1]
    luts = precompute_texture_luts(
        glctx,
        vtx_pos=vtx_pos, pos_idx=pos_idx,
        vtx_uv=vtx_uv, vtx_normals=vtx_normals,
        tex_hw=(t_w, t_h), ss_factor=cfg.upscale,
    )
    pos_lut = luts["pos_lut"]
    normal_lut = luts["normal_lut"]
    valid_mask_raw = luts["valid_mask_raw"]
    tex_tri_id = luts["tex_tri_id"]

    # ---- view selection ----------------------------------------------------
    background_color = [1.0, 1.0, 1.0]
    Q_global = torch.zeros_like(opt_tex[..., 0], dtype=torch.float32)
    views = select_views(
        mesh,
        fov_y_deg=45.0,
        aspect=1.0,
        split_curved_angle_deg=cfg.split_curved_angle_deg,
        min_view_extent_ratio=cfg.min_view_extent_ratio,
        merge_min_face_count=cfg.merge_min_face_count,
        merge_min_area_ratio=cfg.merge_min_area_ratio,
        seam_min_strip_vertices=cfg.seam_min_strip_vertices,
        seam_split_normal_spread_deg=cfg.seam_split_normal_spread_deg,
        seam_split_max_coherence=cfg.seam_split_max_coherence,
        weld_vertex_eps=cfg.weld_vertex_eps,
        weld_uv_eps=cfg.weld_uv_eps,
        use_xatlas_charts=cfg.use_xatlas_charts,
        log=False,
    )

    # ---- local SR model ----------------------------------------------------
    local_sr = LocalSRModel(cfg)

    pbar = tqdm(range(len(views)), dynamic_ncols=True)
    for i in pbar:
        view = views[i]
        ref = render_view(
            glctx, view,
            vtx_pos=vtx_pos, pos_idx=pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx,
            tex=tex, vtx_normals=vtx_normals,
            ref_res=cfg.ref_res, enable_mip=True,
            max_mip_level=max_mip_level, background=background_color,
        )
        cur = render_view(
            glctx, view,
            vtx_pos=vtx_pos, pos_idx=pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx,
            tex=opt_tex, vtx_normals=vtx_normals,
            ref_res=cfg.ref_res, enable_mip=True,
            max_mip_level=max_mip_level, background=background_color,
        )

        Q_view = compute_view_quality(
            cur.world_pos_map, cur.normal_map, cur.alpha_mask, view["view"],
        )
        Q_old = sample_global_quality(Q_global, cur.uv_map)
        update_mask = torch.gt(Q_view - 1e-4, Q_old)

        m_np = get_new_mask(update_mask.cpu().numpy(), min_block_size=9)
        update_mask_tree = torch.from_numpy(m_np).bool()

        with torch.no_grad():
            input_image_for_sr = cur.image.clone()
            mask_3 = update_mask_tree.unsqueeze(-1).expand_as(cur.image)
            input_image_for_sr[mask_3] = ref.image[mask_3]
            sr_image = local_sr(
                input_image_for_sr,
                mask=update_mask_tree,
                prompt="",
            )

        proj_matrix_np = make_proj_matrix(view["persp"], cfg.ref_res)
        opt_tex, Q_global, _ = project_per_texel_gather(
            sr_image=sr_image,
            screen_tri_id=cur.screen_tri_id,
            view_matrix=view["view"],
            proj_matrix=proj_matrix_np,
            texture=opt_tex,
            pos_lut=pos_lut,
            normal_lut=normal_lut,
            tex_tri_id=tex_tri_id,
            valid_mask=valid_mask_raw,
            Q_global=Q_global,
            Q_view=Q_view,
            update_mask=update_mask,
            alpha_mask=cur.alpha_mask,
            depth_map=cur.depth_map,
        )

    # ---- final dilation + save ----------------------------------------------
    opt_tex_4d = opt_tex.unsqueeze(0).contiguous()
    opt_tex_4d = dilate_lut(opt_tex_4d, valid_mask_raw, iterations=4)
    opt_tex = opt_tex_4d.squeeze(0)

    out_path = os.path.join(os.path.dirname(model_path), f"{model_name}_hr.png")
    save_image(out_path, opt_tex.detach().cpu().numpy())
    print(f"[texture_pp] final texture saved to {out_path}")

    return opt_tex
