"""Rendering primitives: projection, rasterisation, LUTs."""

from texture_pp.rendering.proj_matrix import get_proj_mtx, make_proj_matrix
from texture_pp.rendering.rasterize import render_view, RenderResult
from texture_pp.rendering.lut import dilate_lut, dilate_id_lut, precompute_texture_luts

__all__ = [
    "get_proj_mtx",
    "make_proj_matrix",
    "render_view",
    "RenderResult",
    "dilate_lut",
    "dilate_id_lut",
    "precompute_texture_luts",
]
