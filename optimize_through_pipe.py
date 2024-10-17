from copy import deepcopy
from functools import partial
from typing import List, Optional, Union
import PIL
import torch
import numpy as np
from scipy.stats import norm
import fire

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
# from diffusers.pipelines.stable_diffusion.pipeline_flax_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import make_image_grid

from torchvision.transforms.functional import to_pil_image, to_tensor

from torch.utils.checkpoint import checkpoint
import tqdm


def wiggle_latents(latents: torch.Tensor, l=1) -> torch.Tensor:
    """
    Reroll latents

    @param latents: latent tensor with batch dim
    @return: torch.Tensor with batch dim 
    """

    # reverse sampling back to barcode pixels in [0, 2**self.l - 1]
    _latents = latents.detach().cpu().numpy()
    _latents = norm.cdf(_latents) * 2**l
    _latents = _latents.astype(np.int32)
    # fix bug where we sometimes get 2**l
    _latents[_latents == 2**l] = 2**l - 1
    # latents is now integers in [0, 2**self.l - 1]

    # forward sampling with randomnes
    # y we already have
    y = _latents
    # u we draw
    u = np.random.uniform(low=0, high=1, size=y.shape).astype(np.float32)
    # sampling a gaussian
    new_latent = norm.ppf((u + y) / 2**l)
    
    return torch.tensor(new_latent).to(latents.device, latents.dtype)


def load_pipe(modelid=None, device="cuda"):
    modelid = "stabilityai/stable-diffusion-2-1-base" if modelid is None else modelid
    DEVICE = torch.device(device)
    # Load the pre-trained Stable Diffusion 2.1 model from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(modelid,
                                                scheduler=DDIMScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32),  # DPMSolverMultistepScheduler.from_pretrained(...)
                                                safety_checker=None,
                                                torch_dtype=torch.float32).to(DEVICE)

    forward_scheduler = pipe.scheduler
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)  # DPMSolverMultistepInverseScheduler.from_pretrained(...)
    
    return pipe, forward_scheduler, inverse_scheduler


def generate_image(pipe=None, scheduler=None, 
                   prompt="a futuristic city skyline at sunset",
                   latents=None,
                   num_inference_steps=50,
                   guidance_scale=7.5,
                   resolution=512):
    orig_sched = pipe.scheduler
    
    pipe.scheduler = scheduler
    # Generate an image
    image = pipe(prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, latents=latents).images[0]
    
    pipe.scheduler = orig_sched
    return image


class DiffPipe(torch.nn.Module):
    """ A class for backpropagating through a pipeline's generation loop using gradient checkpointing.
    Before use, must check if one step of the pipeline (unet) is deterministic.
    """
    def __init__(self, pipe, scheduler=None, device=None):
        super().__init__()
        self.pipe = pipe
        self.device = device
        self.scheduler = scheduler
        
        self.unet = UnetWrapper(self.pipe.unet)
        
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        
    def forward(self, 
                latents=None,
                prompt="", 
                negative_prompt="",
                prompt_embeds=None,
                guidance_scale=7.5,
                num_inference_steps: int = 50,
                device=None,
                do_classifier_free_guidance=None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
                ):
        oldsched = self.pipe.scheduler
        self.pipe.scheduler = self.scheduler
        
        sigmas = None
        batch_size = 1
        num_images_per_prompt = 1
        timesteps = None
        do_classifier_free_guidance = (guidance_scale > 1.) if do_classifier_free_guidance is None else do_classifier_free_guidance
        
        device = device if device is not None else self.device
        
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                guidance_scale > 0,
                negative_prompt
            )
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds = prompt_embeds
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        # num_channels_latents = self.pipe.unet.config.in_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        latents = latents * self.scheduler.init_noise_sigma    # copied from prepare_latents, latents variable must be tensor on the right device

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipe._num_timesteps = len(timesteps)
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = checkpoint(self.unet, 
                    latent_model_input,
                    t,
                    prompt_embeds,
                    timestep_cond,
                    self.pipe.cross_attention_kwargs,
                    added_cond_kwargs,
                    False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     step_idx = i // getattr(self.scheduler, "order", 1)
                    #     callback(step_idx, t, latents)
        self.pipe.scheduler = oldsched
        return latents
            
    
class UnetWrapper(torch.nn.Module):
    """ Necessary to wrap Unet because gradient checkpoint doesn't allow kwargs so need to make used args a tuple. """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        
    def forward(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    timestep_cond,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict,):
        ret = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,)
        return ret
    
    
def latent_to_pil(latents, pipe):
    with torch.no_grad():
        image = pipe.vae.decode(latents.detach() / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
    # do_denormalize = [True] * image.shape[0]
    # ret = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
    ret = [to_pil_image((img * 0.5 + 0.5).clamp(0, 1)) for img in image]
    return ret
    
    
def pixel_to_latent(image, pipe):
    if not isinstance(image, torch.Tensor):
        image = to_tensor(image).to(pipe.device)[None]
        image = 2. * image - 1.
    
    posterior = pipe.vae.encode(image).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor  # * 0.18215 for SD 15, 21, Mitsua...
    
    return z0
    

def invert_image(pipe=None, scheduler=None,
                 image=None,
                 image_pt=None,
                    num_inference_steps=50,
                    guidance_scale=1,
                    resolution=512
                 ):
    # need to get z0 first
    if image_pt is None:
        image_pt = to_tensor(image).to(pipe.device)[None]
        image_pt = 2. * image_pt - 1.
    else:
        assert image is None
        
    print(image_pt.shape)
    posterior = pipe.vae.encode(image_pt).latent_dist
    z0 = posterior.mean
    z0 = z0.to(pipe.device)
    z0 = z0 * pipe.vae.config.scaling_factor  # * 0.18215 for SD 15, 21, Mitsua...

    # set to inverse scheduler
    orig_sched = pipe.scheduler
    pipe.scheduler = scheduler
    
    # invert z0 to zT
    zT_retrieved = pipe(latents=z0,
                        num_inference_steps=num_inference_steps,
                        prompt=[""] * z0.shape[0],
                        negative_prompt="",
                        guidance_scale=guidance_scale,
                        width=resolution,
                        height=resolution,
                        output_type='latent',
                        return_dict=False,
                        )[0]
    pipe.scheduler = orig_sched
    return zT_retrieved


def main_forward(lr=1e-3,
         updates=100):
    # first test: generate an image, invert it and optimize inverted zT to produce something closer to original image
    pipe, forward_scheduler, inverse_scheduler = load_pipe()
    
    # ############ create example
    # prompt = "a futuristic city skyline at sunset"
    prompt = "a black and white photograph of a cow"
    original_zT = torch.randn(1, 4, 64, 64)
    image = generate_image(pipe, forward_scheduler, prompt=prompt, latents=original_zT)
    image_pt = to_tensor(image)
    
    _inverted = invert_image(pipe=pipe, scheduler=inverse_scheduler, image_pt=image_pt.to(pipe.device)[None])
    
    # check what reconstruction looks like, to later compare to what differentiable pipe produces
    # _recons = generate_image(pipe, forward_scheduler, prompt=prompt, latents=inverted)
    print(image_pt.shape, _inverted.shape)
    # ############ end of create example
    
    diffpipe = DiffPipe(pipe, scheduler=forward_scheduler, device=pipe.device)
    
    # with torch.no_grad():       # verify that with original noise, our adapted pipeline generates the same image
    #     check_image = ztmodel(original_zT.to(pipe.vae.device), prompt)
    #     check_image = latent_to_pil(check_image, pipe)[0]
    
    inverted = torch.nn.Parameter(_inverted.detach().clone())
    optim = torch.optim.Adam([inverted], lr=lr)
    
    # make trainable noise
    # zT = torch.nn.Parameter(inverted)
    # optim = torch.optim.SGD([zT], lr=lr)
    
    # define loss
    latent_target = pixel_to_latent(image, pipe)
    
    recons_history_latents = [None]
    recons_history_pil = [image]
    zT_history = []
    loss_history = []
    
    recons_overview = None
    
    # training loop
    for update_number in tqdm.tqdm(range(updates)):
        optim.zero_grad()
        
        recons = diffpipe(inverted, prompt)
        
        # zT_history.append(inverted.detach().clone())    
        recons_history_latents.append(recons.detach().clone())
        recons_history_pil.append(latent_to_pil(recons_history_latents[-1], pipe)[0])
        
        recons_overview = make_image_grid(recons_history_pil, 1, len(recons_history_pil))
        
        l = torch.nn.functional.mse_loss(recons, latent_target.detach())
        loss_history.append(l.detach().item())
        l.backward()
        
        print(f"Step {update_number}: z_T delta norm: {(inverted - _inverted).norm().item():.5f}, z_T grad norm: {inverted.grad.norm().item():.5f}, loss: {l.item():.5f}")
        
        optim.step()
    
    
    print("Inverted grad norm:", inverted.grad.norm())
    
    
def main_inverse(lr=1e-3,
         updates=100):
    # inverse test: generate an image, invert it and optimize reconstruction to produce a latent closer to original latent
    pipe, forward_scheduler, inverse_scheduler = load_pipe()
    
    # ############ create example
    # prompt = "a futuristic city skyline at sunset"
    prompt = "a black and white photograph of a cow"
    original_zT = torch.randn(1, 4, 64, 64)
    image = generate_image(pipe, forward_scheduler, prompt=prompt, latents=original_zT)
    image_pt = to_tensor(image)
    _z0 = pixel_to_latent(image, pipe)
    
    inverted = invert_image(pipe=pipe, scheduler=inverse_scheduler, image=image)
    
    # check what reconstruction looks like, to later compare to what differentiable pipe produces
    _recons = generate_image(pipe, forward_scheduler, prompt=prompt, latents=inverted)
    print(image_pt.shape, inverted.shape)
    # ############ end of create example
    
    diffpipe = DiffPipe(pipe, scheduler=inverse_scheduler, device=pipe.device)
    
    # check that diffpipe does the same inversion as invert_image()
    with torch.no_grad():
        inverted_test = diffpipe(_z0, "", guidance_scale=1.) 
        assert( torch.allclose(inverted_test, inverted) )
    
    z0 = torch.nn.Parameter(_z0.detach().clone())
    optim = torch.optim.Adam([z0], lr=lr)
    
    # define loss
    target_latent = torch.zeros_like(z0)
    
    inverted_history = []
    loss_history = []
    
    recons_overview = None
    
    # training loop
    for update_number in tqdm.tqdm(range(updates)):
        optim.zero_grad()
        
        # recons = diffpipe(zT, prompt)
        inverted_latent = diffpipe(z0, "", guidance_scale=1.)
        
        # zT_history.append(inverted.detach().clone())    
        inverted_history.append(inverted_latent.detach().clone())
        
        l = torch.nn.functional.mse_loss(inverted_latent, target_latent.detach())
        loss_history.append(l.detach().item())
        l.backward()
        
        print(f"Step {update_number}: z_0 delta norm: {(z0 - _z0).norm().item():.5f}, z_0 grad norm: {z0.grad.norm().item():.5f}, loss: {l.item():.5f}")
        
        optim.step()
        
    print("Done training")
    
    


if __name__ == "__main__":
    fire.Fire(main_forward)