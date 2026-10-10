import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from src.sr.model.autoencoder_kl import AutoencoderKL
from src.sr.model.unet_2d_condition import UNet2DConditionModel


class TextureSRModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        self.vae_fix = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

        self.unet.to("cuda")
        self.text_encoder.to("cuda")
        self.vae_fix.to("cuda")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()
        self.timesteps1 = torch.tensor([1], device="cuda").long()
        self.empty_prompt_embeds = self.encode_prompt([""]).to(self.unet.dtype)

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            embeds = []
            for caption in prompt_batch:
                tokens = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                ).input_ids.to(self.text_encoder.device)
                embeds.append(self.text_encoder(tokens)[0])
        return torch.cat(embeds, dim=0)

    def set_eval(self):
        self.eval()
        self.unet.requires_grad_(False)
        self.vae_fix.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def _create_soft_mask(self, mask, blur_kernel_size=31, blur_sigma=10.0):
        from torchvision.transforms import GaussianBlur
        original_dtype = mask.dtype
        blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        mask_soft = blur(mask.float())
        mask_soft = torch.clamp(mask_soft, 0.0, 1.0)
        return mask_soft.to(dtype=original_dtype)

    @torch.no_grad()
    def forward(self, c_t, prompt="", mask=None):
        torch.cuda.synchronize()
        start_time = time.time()
        c_t = c_t.to(self.unet.dtype)
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        if prompt:
            prompt_embeds = self.encode_prompt([prompt]).to(self.unet.dtype)
        else:
            prompt_embeds = self.empty_prompt_embeds
        if mask is None:
            raise ValueError("Mask is required for TextureSRModel inference.")
        mask = mask.to(device=c_t.device, dtype=c_t.dtype)
        latent_mask = F.interpolate(mask, size=encoded_control.shape[-2:], mode="bilinear", align_corners=False)
        latent_mask_in = latent_mask * 2.0 - 1.0
        encoded_control = torch.cat([encoded_control, latent_mask_in], dim=1)
        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        x_denoised = encoded_control[:, :4] - model_pred
        output_image = self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample.clamp(-1, 1)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        return (total_time, output_image)
