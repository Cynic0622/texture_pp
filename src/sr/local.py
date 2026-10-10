"""Thin wrapper around TextureSRModel.

Constructor takes the pipeline config. Call signature matches the call site
in pipeline.py: returns the SR image as (H, W, 3) float [0, 1].
"""

from __future__ import annotations

from typing import Optional

import torch

from src.sr.model.texture_sr import TextureSRModel
from src.sr._util import silent_run


class LocalSRModel:
    """Wrapper that constructs and runs the local mask-conditioned SR model."""

    def __init__(self, cfg):
        self._model = TextureSRModel(cfg.sr_model_path)
        self._model.set_eval()

    def __call__(
        self,
        img: torch.Tensor,                   # (H, W, 3) float [0, 1] CUDA
        mask: Optional[torch.Tensor] = None,  # (H, W) bool CUDA
        prompt: str = "",
    ) -> torch.Tensor:
        """Run a single SR pass. Returns (H, W, 3) float [0, 1]."""
        input_tensor = img.permute(2, 0, 1).unsqueeze(0).cuda() * 2.0 - 1.0
        input_mask = None if mask is None else mask.unsqueeze(0).unsqueeze(0).cuda()
        _, sr_image = silent_run(
            self._model,
            input_tensor,
            mask=input_mask,
            prompt=prompt,
        )
        sr_image = ((sr_image[0] + 1.0) / 2.0).permute(1, 2, 0).to(torch.float32)
        return sr_image
