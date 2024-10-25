from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers import FluxPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput
from diffusers.utils import make_image_grid
from diffusers.pipelines.flux.pipeline_flux import *

import fire
import numpy as np

EXTRASIGMAS = True

    
class FlowMatchEulerDiscreteSchedulerInverse(FlowMatchEulerDiscreteScheduler):
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        sigmas = torch.flip(sigmas, [0])
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([torch.zeros(1, device=sigmas.device), sigmas])

        self._step_index = None
        self._begin_index = None


def latent_to_pil(latents, height, width, pipe):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image


def main(seed=42):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    
    height, width = 512, 512
    numsteps = 20

    prompt = "A cat holding a sign that says hello world"
    z0 = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=numsteps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="latent",
    ).images[0]
    
    with torch.no_grad():
        image = latent_to_pil(z0[None], height=height, width=width, pipe=pipe)[0]
        
    image.save("flux-dev.png")    
    
    # invert
    pipe.scheduler.__class__ = FlowMatchEulerDiscreteSchedulerInverse
    prompt = ""
    zT_inv = pipe(
        prompt,
        latents=z0[None],
        height=height,
        width=width,
        guidance_scale=1,
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="latent",
    ).images[0]
    
    
    # reconstruct
    pipe.scheduler.__class__ = FlowMatchEulerDiscreteScheduler
    prompt = "A cat holding a sign that says hello world"
    z0_rec = pipe(
        prompt,
        latents=zT_inv[None],
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=numsteps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(69),
        output_type="latent",
    ).images[0]
    
    with torch.no_grad():
        image_rec = latent_to_pil(z0_rec[None], height=height, width=width, pipe=pipe)[0]
    
    image_rec.save("flux-dev-inv-recons.png")    
    
    comp = make_image_grid([image, image_rec], 1, 2)
    print("done")
    

if __name__ == "__main__":
    fire.Fire(main)