"""Projection matrix construction."""

from __future__ import annotations

from typing import Union

import numpy as np


def get_proj_mtx(near: float, far: float, fov_y: float = 45.0, aspect: float = 1.0) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y) / 2.0)
    proj_mtx = np.zeros((4, 4), dtype=np.float32)
    proj_mtx[0, 0] = f / aspect
    proj_mtx[1, 1] = f
    proj_mtx[2, 2] = (far + near) / (near - far)
    proj_mtx[2, 3] = (2 * far * near) / (near - far)
    proj_mtx[3, 2] = -1.0
    return proj_mtx


def make_proj_matrix(persp: Union[dict, np.ndarray], ref_res: int) -> np.ndarray:
    if isinstance(persp, dict):
        fov_y = persp.get("fov_y_deg", 45.0)
        aspect = persp.get("aspect", 1.0)
        near = persp["near"]
        far = persp["far"]
        return get_proj_mtx(near=near, far=far, fov_y=fov_y, aspect=aspect)
    return persp
