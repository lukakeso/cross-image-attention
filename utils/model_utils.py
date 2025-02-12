import torch
from diffusers import DDIMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline, CrossImageAttentionControlStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model(use_control=False):
    print("Loading Stable Diffusion model...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    if use_control:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        pipe = CrossImageAttentionControlStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            controlnet=controlnet, safety_checker=None).to(device)
    else:
        pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                        safety_checker=None).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    print("Done.")
    return pipe
