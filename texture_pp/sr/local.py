"""Thin wrapper around the fine-tuned TextureSRModel.

Constructor takes the pipeline config. Call signature matches the call site
in pipeline.py: returns the SR image as (H, W, 3) float [0, 1].
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch

from texture_pp.sr.model.texture_sr import TextureSRModel
from texture_pp.sr._util import silent_run


class LocalSRModel:

    def __init__(self, cfg):
        sd = torch.load(cfg.pretrained_path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(
            pretrained_model_path=cfg.pretrained_model_path,
            lora_rank_unet_mask=sd.get("lora_rank", 16),
            lora_alpha_unet_mask=sd.get("lora_alpha", 32),
            timesteps1=1,
        )
        self._model = TextureSRModel(args, pretrained_path=cfg.pretrained_path)
        self._model.set_eval()

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prompt: str = "",
    ) -> torch.Tensor:
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
