"""Image / texture I/O helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

import imageio


def save_image(fn: str, x: np.ndarray) -> None:
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)


def load_texture(tex_path: str) -> np.ndarray:
    img = np.array(Image.open(tex_path))
    return img[:, :, :3].astype(np.float32) / 255.0
