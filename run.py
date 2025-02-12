import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images


@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    #return images


def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    depth_image = latent_utils.load_depth_image(cfg=cfg) if cfg.use_control else None
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    #cfg.cross_attn_64_range = Range(start=50, end=70)
    #cfg.cross_attn_32_range = Range(start=50, end=70)
    #cfg.adain_range = Range(start=60, end=90)
    #cfg.contrast_strength = 1.67
    #cfg.swap_guidance_scale = 3.5
    for range_start_CA in range(0,101,10):
        for range_end_CA in range(0,101,10):
            for range_start_ADAIN in range(0,101,10):
                for range_end_ADAIN in range(0,101,10):
                    if range_start_CA >= range_end_CA or range_start_ADAIN >= range_end_ADAIN:
                        continue
                    cfg.cross_attn_64_range = Range(start=range_start_CA, end=range_end_CA)
                    cfg.adain_range = Range(start=range_start_ADAIN, end=range_end_ADAIN)
                    model.config.adain_range = Range(start=range_start_ADAIN, end=range_end_ADAIN)
                    #start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
                    #end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
                    images = model.pipe(
                        prompt=[cfg.prompt] * 3,
                        negative_prompt=["deformed, bad anatomy, bad lighting, monochrome"] * 3,
                        latents=init_latents,
                        control_image=depth_image,
                        guidance_scale=1.0,
                        num_inference_steps=cfg.num_timesteps,
                        swap_guidance_scale=cfg.swap_guidance_scale,
                        callback=model.get_adain_callback(),
                        eta=1,
                        zs=init_zs,
                        generator=torch.Generator('cuda').manual_seed(cfg.seed),
                        cross_image_attention_range=cfg.cross_attn_64_range,
                        # controlnet_conditioning_scale = 0.3,
                        # control_guidance_start = 0.0,
                        # control_guidance_end = 0.4,
                    ).images
                    # Save images
                    #print("AdaIN:", cfg.adain_range.start, cfg.adain_range.end)
                    images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}__CA64_{range_start_CA}_{range_end_CA}__AdaIN_{range_start_ADAIN}_{range_end_ADAIN}.png")
                    #images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}__CA64_{range_start_CA}_{range_end_CA}__AdaIN_{range_start_ADAIN}_{range_end_ADAIN}.png")
                    #images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}__CA64_{range_start_CA}_{range_end_CA}__AdaIN_{range_start_ADAIN}_{range_end_ADAIN}.png")
                    #joined_images = np.concatenate(images[::-1], axis=1)
                    #Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}__CA64_{range_start_CA}_{range_end_CA}__AdaIN_{range_start_ADAIN}_{range_end_ADAIN}.png")
                    
    #return images


if __name__ == '__main__':
    main()
