"""
Run as: python texture_pp/main.py -m <model.obj> [-t <tex.png>]

The final super-resolved texture is saved next to the mesh as
<model_name>_hr.png.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from texture_pp.config import Config
from texture_pp.pipeline import super_resolve_texture


def _build_argparser() -> argparse.ArgumentParser:
    project_root = pathlib.Path(__file__).absolute().parents[1]
    default_models_path = project_root / "models"
    p = argparse.ArgumentParser(
        description="texture_pp: multi-view texture super-resolution (inference)."
    )

    # --- mesh / texture ---
    p.add_argument("--model_path", "-m", type=str, required=True,
                   help="Path to .obj mesh file, or a directory containing "
                        "<dirname>.obj and <dirname>.png")
    p.add_argument("--tex_path", "-t", type=str, default=None,
                   help="Path to the low-resolution texture PNG.")

    # --- SR model ---
    p.add_argument("--pretrained_model_path", type=str,
                   default=str(default_models_path / "stable-diffusion-2-1-base"),
                   help="SD 2.1 base model directory.")
    p.add_argument("--pretrained_path", type=str,
                   default=str(default_models_path / "texture_sr.pkl"),
                   help="Fine-tuned TextureSRModel checkpoint.")
    p.add_argument("--upscale", type=int, default=4)

    # --- view-selection tuning knobs ---
    p.add_argument("--split_curved_angle_deg", type=float, default=90.0,
                   help="Recursive split threshold (deg) for normal spread within "
                        "an island.")
    p.add_argument("--min_view_extent_ratio", type=float, default=0.30,
                   help="Lower bound on (island_extent / camera_distance).")
    p.add_argument("--merge_min_face_count", type=int, default=200,
                   help="An island with fewer faces is candidate for merging "
                        "into its largest non-small 3D-adjacent neighbour.")
    p.add_argument("--merge_min_area_ratio", type=float, default=0.02,
                   help="Combined with --merge_min_face_count: an island whose "
                        "area is below this fraction of total mesh area AND whose "
                        "face count is below the threshold is treated as 'small'.")
    p.add_argument("--seam_min_strip_vertices", type=int, default=8,
                   help="Discard seam strips with fewer vertices than this")
    p.add_argument("--seam_split_normal_spread_deg", type=float, default=0.0,
                   help="If > 0, a seam strip whose surrounding faces span "
                        "this many degrees of normal direction is split into 2 "
                        "cameras.")
    p.add_argument("--seam_split_max_coherence", type=float, default=0.5,
                   help="If > 0, a seam strip whose area-weighted normal "
                        "coherence |Sum(a_i*n_i)|/Sum(a_i) falls below this is "
                        "split into 2 cameras.")
    p.add_argument("--weld_vertex_eps", type=float, default=1e-6,
                   help="3D-position welding tolerance (L_inf).")
    p.add_argument("--weld_uv_eps", type=float, default=1e-6,
                   help="2D UV-position welding tolerance, analogous to "
                        "--weld_vertex_eps but for UV coords.")
    p.add_argument("--use_xatlas_charts", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Run xatlas after loading the mesh to compute an "
                        "alternative UV partitioning used ONLY for view-selection "
                        "island detection.")
    return p


def _args_to_config(args: argparse.Namespace) -> Config:
    return Config(
        model_path=args.model_path,
        tex_path=args.tex_path,
        pretrained_model_path=args.pretrained_model_path,
        pretrained_path=args.pretrained_path,
        upscale=args.upscale,
        split_curved_angle_deg=args.split_curved_angle_deg,
        min_view_extent_ratio=args.min_view_extent_ratio,
        merge_min_face_count=args.merge_min_face_count,
        merge_min_area_ratio=args.merge_min_area_ratio,
        seam_min_strip_vertices=args.seam_min_strip_vertices,
        seam_split_normal_spread_deg=args.seam_split_normal_spread_deg,
        seam_split_max_coherence=args.seam_split_max_coherence,
        weld_vertex_eps=args.weld_vertex_eps,
        weld_uv_eps=args.weld_uv_eps,
        use_xatlas_charts=args.use_xatlas_charts,
    )


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = _args_to_config(args)
    super_resolve_texture(cfg)


if __name__ == "__main__":
    main()
