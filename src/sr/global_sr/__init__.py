"""Global super-resolution model (HYPIR SD2 LoRA) and its wrapper."""

from __future__ import annotations

import torch

from src.sr.global_sr.sd2 import SD2Enhancer
from src.sr._util import silent_run


class GlobalSR:
    def __init__(self, cfg):
        self._model = SD2Enhancer(
            base_model_path=cfg.global_base_model_path,
            weight_path=cfg.global_weight_path,
            lora_modules=cfg.global_lora_modules,
            lora_rank=cfg.global_lora_rank,
            model_t=cfg.global_model_t,
            coeff_t=cfg.global_coeff_t,
            device="cuda",
        )
        self._model.init_models()

    def __call__(
        self,
        img: torch.Tensor,
        prompt: str = "",
        *,
        upscale: int,
        patch_size: int,
        stride: int,
    ) -> torch.Tensor:
        lq_tensor = img.permute(2, 0, 1).unsqueeze(0).cuda()
        sr_imgs = silent_run(
            self._model.enhance,
            lq_tensor,
            prompt,
            "factor",
            upscale,
            patch_size=patch_size,
            stride=stride,
            return_type="np",
        )
        sr_image = torch.from_numpy(sr_imgs[0]).cuda()
        if sr_image.dtype == torch.uint8:
            sr_image = sr_image.float() / 255.0
        if sr_image.ndim == 4:
            sr_image = sr_image.squeeze(0)
        return sr_image


__all__ = ["GlobalSR"]
