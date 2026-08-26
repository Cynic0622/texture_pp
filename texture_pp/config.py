"""Pipeline configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:

    # ---- mesh / texture ----
    model_path: str = ""
    tex_path: Optional[str] = None

    # ---- SR model ----
    pretrained_model_path: str = ""
    pretrained_path: str = ""
    upscale: int = 4

    # ---- rendering ----
    ref_res: int = 1024

    # ---- view selection knobs ----
    split_curved_angle_deg: float = 90.0
    min_view_extent_ratio: float = 0.30
    merge_min_face_count: int = 200
    merge_min_area_ratio: float = 0.02
    seam_min_strip_vertices: int = 8
    seam_split_normal_spread_deg: float = 0.0
    seam_split_max_coherence: float = 0.5
    weld_vertex_eps: float = 1e-6
    weld_uv_eps: float = 1e-6
    use_xatlas_charts: bool = True
