"""TextureSRModel: mask-conditioned local super-resolution model."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from texture_pp.sr.model.autoencoder_kl import AutoencoderKL
from texture_pp.sr.model.unet_2d_condition import UNet2DConditionModel

def _build_mask_layer_targets(unet):
    layer_patterns = ['to_k', 'to_q', 'to_v', 'to_out.0', 'conv', 'conv1', 'conv2', 'conv_in', 'conv_shortcut', 'conv_out', 'proj_out', 'proj_in', 'ff.net.2', 'ff.net.0.proj']
    (encoder, decoder, others) = ([], [], [])
    for (n, _p) in unet.named_parameters():
        if 'bias' in n or 'norm' in n:
            continue
        for pat in layer_patterns:
            if pat in n:
                if 'down_blocks' in n or 'conv_in' in n:
                    encoder.append(n.replace('.weight', ''))
                elif 'up_blocks' in n or 'conv_out' in n:
                    decoder.append(n.replace('.weight', ''))
                else:
                    others.append(n.replace('.weight', ''))
                break
    return (list(dict.fromkeys(encoder)), list(dict.fromkeys(decoder)), list(dict.fromkeys(others)))

class TextureSRModel(nn.Module):
    def __init__(self, args, pretrained_path=None):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder='text_encoder')
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder='unet')
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder='vae')
        self._expand_conv_in_to_5ch()
        (enc_tgts, dec_tgts, oth_tgts) = _build_mask_layer_targets(self.unet)
        rank = int(args.lora_rank_unet_mask)
        alpha = int(getattr(args, 'lora_alpha_unet_mask', 2 * rank))
        self.lora_rank = rank
        self.lora_alpha = alpha
        mask_pix_specs = [('default_encoder_mask_pix', enc_tgts, 'gaussian'), ('default_decoder_mask_pix', dec_tgts, 'gaussian'), ('default_others_mask_pix', oth_tgts, 'gaussian')]
        mask_sem_specs = [('default_encoder_mask_sem', enc_tgts, True), ('default_decoder_mask_sem', dec_tgts, True), ('default_others_mask_sem', oth_tgts, True)]
        for (name, targets, init) in mask_pix_specs + mask_sem_specs:
            self.unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights=init, target_modules=targets), adapter_name=name)
        n = len(mask_pix_specs) + len(mask_sem_specs)
        n_layers = len(enc_tgts) + len(dec_tgts) + len(oth_tgts)
        print(f'[TextureSRModel] registered {n} adapters (encoder={len(enc_tgts)}, decoder={len(dec_tgts)}, others={len(oth_tgts)} layers each, rank={rank})')
        if pretrained_path is not None:
            print(f'[TextureSRModel] Loading weights from {pretrained_path}')
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self._load_resume_state(ckpt)
        self.unet.to('cuda')
        self.text_encoder.to('cuda')
        self.vae_fix.to('cuda')
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()
        self.timesteps1 = torch.tensor([args.timesteps1], device='cuda').long()
        self.empty_prompt_embeds = self.encode_prompt(['']).to(self.unet.dtype)
        self.mask_pix_names = [name for (name, _t, _i) in mask_pix_specs]
        self.mask_sem_names = [name for (name, _t, _i) in mask_sem_specs]
        self.pix_and_sem_names = self.mask_pix_names + self.mask_sem_names
        self.mask_names = self.pix_and_sem_names

    def _expand_conv_in_to_5ch(self):
        old = self.unet.conv_in
        assert isinstance(old, nn.Conv2d)
        assert old.in_channels == 4
        new = nn.Conv2d(in_channels=5, out_channels=old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=hasattr(old, 'bias') and old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :4, :, :].copy_(old.weight)
            if hasattr(new, 'bias') and new.bias is not None:
                new.bias.copy_(old.bias)
        self.unet.conv_in = new
        self.unet.config.in_channels = 5
        print('[TextureSRModel] conv_in expanded to 5 channels (5th zero-init).')

    def encode_prompt(self, prompt_batch):
        with torch.no_grad():
            embeds = []
            for caption in prompt_batch:
                tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(self.text_encoder.device)
                embeds.append(self.text_encoder(tokens)[0])
        return torch.cat(embeds, dim=0)

    def set_eval(self):
        self.eval()
        self.unet.requires_grad_(False)
        self.vae_fix.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        set_weights_and_activate_adapters(self.unet, self.mask_names, [1.0] * len(self.mask_names))
        self.unet.merge_and_unload()

    @torch.no_grad()
    def forward(self, c_t, prompt='', mask=None):
        torch.cuda.synchronize()
        start_time = time.time()
        if c_t.ndim == 3:
            c_t = c_t.unsqueeze(0)
        c_t = c_t.to(self.unet.dtype)
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        if prompt:
            prompt_embeds = self.encode_prompt([prompt]).to(self.unet.dtype)
        else:
            prompt_embeds = self.empty_prompt_embeds
        if mask is not None:
            mask = mask.to(device=c_t.device, dtype=c_t.dtype)
            latent_mask = F.interpolate(mask, size=encoded_control.shape[-2:], mode='bilinear', align_corners=False)
        else:
            raise ValueError('Mask is required for TextureSRModel inference.')
        latent_mask_in = latent_mask * 2.0 - 1.0
        encoded_control = torch.cat([encoded_control, latent_mask_in], dim=1)
        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        x_denoised = encoded_control[:, :4] - model_pred
        output_image = self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample.clamp(-1, 1)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        return (total_time, output_image)

    def _load_resume_state(self, ckpt):
        conv_sd = ckpt['state_dict_unet_conv_in']
        (missing, unexpected) = self.unet.conv_in.load_state_dict(conv_sd, strict=False)
        if missing:
            print(f'[TextureSRModel] conv_in missing keys: {missing}')
        if unexpected:
            print(f'[TextureSRModel] conv_in unexpected keys: {unexpected}')
        loaded = 0
        for (n, p) in self.unet.named_parameters():
            if 'lora' in n and '_pix' in n:
                if n in ckpt.get('state_dict_unet_pix', {}):
                    p.data.copy_(ckpt['state_dict_unet_pix'][n])
                    loaded += 1
        print(f'[TextureSRModel] Loaded {loaded} mask_pix params from ckpt.')
        loaded_sem = 0
        for (n, p) in self.unet.named_parameters():
            if 'lora' in n and '_sem' in n:
                if n in ckpt.get('state_dict_unet_sem', {}):
                    p.data.copy_(ckpt['state_dict_unet_sem'][n])
                    loaded_sem += 1
        print(f'[TextureSRModel] Loaded {loaded_sem} mask_sem params from ckpt.')
