import typing

import os

from typing import List, Optional, Union
import PIL
import PIL.Image
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

import argparse
import wandb

from utils.wm.wm_utils import WmProviders
from utils.wm import tr_provider
from utils.wm import gs_provider
from utils.wm import pc_provider
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.pc_provider import parser as pc_parser

from utils.pipe import pipe_utils



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "OUT/imprint/"

# args
parser = argparse.ArgumentParser(description="attack", parents=[gs_parser, tr_parser, pc_parser])
parser.add_argument("--target_image_name", type=str, default="cat0.jpg")
parser.add_argument("--modelid_attacker", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--modelid_target", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument('--lora_checkpoint_dir_target', default=None, type=str)
parser.add_argument('--unet_id_or_checkpoint_dir_target', default=None, type=str)
parser.add_argument("--prompt", type=str, default="1girl The image features a female")
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--wm_type", type=str, default="GS")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--updates", type=int, default=300)
parser.add_argument("--save_steps", type=int, default=1)
parser.add_argument("--validation_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mse_on_z0_factor", type=float, default=0.0)
parser.add_argument("--mse_on_pixels_factor", type=float, default=0.0)
args = parser.parse_args()



def diff_and_glue_2_PIL_images(_image_original: PIL.Image.Image,
                               _reconstructed_from_z0_original: PIL.Image.Image,
                               image_from_z0: PIL.Image.Image) -> typing.Tuple[PIL.Image.Image,
                                                                            PIL.Image.Image,
                                                                            PIL.Image.Image]:
    """
    Concatenate images side by side and compute difference images

    All have same shape
    """

    width, height = _image_original.size

    _0_0 = _image_original.copy()
    _0_1 = image_from_z0.copy()
    _1_0 = _reconstructed_from_z0_original.copy()
    _1_1 = image_from_z0.copy()
    
    # diff (original - optimized)
    _0_2 = to_pil_image(         (to_tensor(_0_0) - to_tensor(_0_1)) / 2.0 + 0.5)
    _0_3 = to_pil_image(torch.abs(to_tensor(_0_0) - to_tensor(_0_1)) )
    # diff (z0_reconstructed - optimized)
    _1_2 = to_pil_image(         (to_tensor(_1_0) - to_tensor(_1_1)) / 2.0 + 0.5)
    _1_3 = to_pil_image(torch.abs(to_tensor(_1_0) - to_tensor(_1_1)) )
    
    # Concatenate images side by side
    glued = PIL.Image.new('RGB', (width * 4, height * 2))
    glued.paste(_0_0, (0,         0))
    glued.paste(_0_1, (width,     0))
    glued.paste(_0_2, (width * 2, 0))
    glued.paste(_0_3, (width * 3, 0))
    glued.paste(_1_0, (0,         height))
    glued.paste(_1_1, (width,     height))
    glued.paste(_1_2, (width * 2, height))
    glued.paste(_1_3, (width * 3, height))

    return {"glued": glued,
            "_0_2": _0_2,
            "_0_3": _0_3,
            "_1_2": _1_2,
            "_1_3": _1_3}


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


def load_pipe(modelid=None):
    modelid = "stabilityai/stable-diffusion-2-1-base" if modelid is None else modelid
    # Load the pre-trained Stable Diffusion 2.1 model from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(modelid,
                                                scheduler=DDIMScheduler.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32),  # DPMSolverMultistepScheduler.from_pretrained(...)
                                                safety_checker=None,
                                                torch_dtype=torch.float32).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

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
    
    
def main_inverse(target_image_name: str = "stalin0.jpg",
                 target_image_mask_name: str = "stalin0.mask.png",              # ADDED
                 modelid_attacker: str = "stabilityai/stable-diffusion-2-1-base",
                 modelid_target: str = "stabilityai/stable-diffusion-2-1-base",
                 unet_id_or_checkpoint_dir_target: str = None,
                 lora_checkpoint_dir_target: str = None,
                 prompt: str = "1girl The image features a female",
                 guidance_scale: float = 7.5,
                 resolution: int = 512,
                 wm_type: str = "gs",
                 num_inference_steps: int = 50,
                 lr: float = 1e-2,
                 updates: int = 100,
                 save_steps: int = 1,  # TODO
                 validation_steps: int = 10,
                 seed: int = 1,
                 mse_on_z0_factor: float = 0.0,
                 mse_on_pixels_factor: float = 0.0,
                 **kwargs):
    torch.manual_seed(seed)

    #init wandb
    wandb.init(project="imprint", config=args)
    
    target_image_path = os.path.join(OUT_DIR, "target_images", target_image_name)
    out_dir = os.path.join(OUT_DIR,
                           "results",
                           f"target_image_name={target_image_name}",
                           f"modelid_attacker={modelid_attacker}",
                           f"modelid_target={modelid_target}"
                           f"unet_id_or_checkpoint_dir_target={'/'.join(unet_id_or_checkpoint_dir_target.split('/')[-2:]) if unet_id_or_checkpoint_dir_target is not None else 'None'}",
                           f"lora_checkpoint_dir_target={'/'.join(lora_checkpoint_dir_target.split('/')[-2:]) if lora_checkpoint_dir_target is not None else 'None'}",
                           f"prompt={prompt}"
                           f"guidance_scale={guidance_scale}",
                           f"resolution={resolution}",
                           f"lr={lr}",
                           f"seed={seed}",
                           f"mse_on_z0_factor={mse_on_z0_factor}",
                           f"mse_on_pixels_factor={mse_on_pixels_factor}",
                           f"wm_type={wm_type}")
    os.makedirs(out_dir, exist_ok=True)

    # read im age
    _image_original = PIL.Image.open(target_image_path).resize((resolution, resolution))
    
    # ADDED ###########
    mask_replace = PIL.Image.open(target_image_mask_name).convert("L")
    blurred_mask_replace = mask_replace.filter(PIL.ImageFilter.GaussianBlur(radius=5))  # Adjust the radius as needed
    blurred_mask_replace_pt = to_tensor(blurred_mask_replace)
    latent_blurred_mask_replace = torch.nn.functional.interpolate(blurred_mask_replace_pt[None], size=64, mode="bilinear")[0]

    pixel_replace = _image_original.copy()
    # Apply the mask to the image
    pixel_replace.putalpha(blurred_mask_replace)  # This adds the mask to the image's alpha channel
    # END ADDED

    # inverse test: generate an image, invert it and optimize reconstruction to produce a latent closer to original latent
    pipe_attacker, forward_scheduler, inverse_scheduler = load_pipe()

    # pipe_provider for checking watermark
    pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=modelid_target,
                                           unet_id_or_checkpoint_dir=unet_id_or_checkpoint_dir_target,
                                           lora_checkpoint_dir=lora_checkpoint_dir_target,
                                           resolution=_image_original.width,
                                           device=DEVICE,
                                           eager_loading=True,
                                           disable_tqdm=True,)

    _z0_original = pixel_to_latent(_image_original, pipe_attacker)
    _zT_original = invert_image(pipe=pipe_attacker, image=_image_original, scheduler=inverse_scheduler, num_inference_steps=num_inference_steps)
    
    # check what reconstruction looks like, to later compare to what differentiable pipe produces
    _reconstructed_from_z0_original = latent_to_pil(_z0_original, pipe_attacker)[0]
    
    diffpipe = DiffPipe(pipe_attacker, scheduler=inverse_scheduler, device=pipe_attacker.device)
    
    # check that diffpipe does the same inversion as invert_image()
    with torch.no_grad():
        inverted_test = diffpipe(_z0_original,
                                 prompt="",
                                 num_inference_steps=num_inference_steps,
                                 guidance_scale=1.) 
        assert(torch.allclose(inverted_test, _zT_original))
    
    z0 = torch.nn.Parameter(_z0_original.detach().clone())
    optim = torch.optim.Adam([z0], lr=lr)
    
    # define target latent
    #_zT_target = torch.randn_like(z0)
    wm_provider = WmProviders[wm_type].value(latent_shape=z0.shape, **kwargs)
    _zT_target_pristine = wm_provider.get_wm_latents()["zT_torch"]
    _generated = pipe_provider_target.generate(prompts=prompt, latents=_zT_target_pristine, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)["images_torch"]
    with torch.no_grad():
        _zT_target_retrieved = invert_image(pipe=pipe_attacker,
                                            image_pt=_generated,
                                            scheduler=inverse_scheduler,
                                            num_inference_steps=num_inference_steps)
    _zT_target = _zT_target_retrieved

    # log
    wandb.log({"image_original": wandb.Image(_image_original),
               "_reconstructed_from_z0_original": wandb.Image(_reconstructed_from_z0_original),
               #"_z0_original": wandb.Image(_z0_original.detach().cpu()),
               #"_zT_original": wandb.Image(_zT_original.detach().cpu()),
               #"_zT_target_pristine": wandb.Image(_zT_target.detach().cpu())
               }),
    # save reconstructed from z0 as well on disk
    _reconstructed_from_z0_original_path = os.path.join(out_dir, "_reconstructed_from_z0_original")
    os.makedirs(_reconstructed_from_z0_original_path, exist_ok=True)
    _reconstructed_from_z0_original.save(os.path.join(_reconstructed_from_z0_original_path, f"{target_image_name}_reconstructed_from_z0_original.png"))
    
    # training loop
    inverted_history = []
    loss_history = []
    for update_number in tqdm.tqdm(range(updates)):
        optim.zero_grad()
        
        # recons = diffpipe(zT, prompt)
        inverted_latent = diffpipe(z0, "", guidance_scale=1.)
        
        # zT_history.append(inverted.detach().clone())    
        inverted_history.append(inverted_latent.detach().clone())
        
        # get loss in zT space
        l = torch.nn.functional.mse_loss(inverted_latent, _zT_target.detach())

        # get loss on in pixel space
        if mse_on_z0_factor > 0.0:
            l_z0 = torch.nn.functional.mse_loss(z0, _z0_original.detach())
            l = l + mse_on_z0_factor * l_z0

        if mse_on_pixels_factor > 0.0:
            with torch.no_grad():
                image_from_z0_original = latent_to_pil(_z0_original, pipe_attacker)[0]
            image_from_z0 = latent_to_pil(z0, pipe_attacker)[0]
            l_pixels = torch.nn.functional.mse_loss(to_tensor(image_from_z0).to(DEVICE)[None], to_tensor(image_from_z0_original).to(DEVICE)[None])
            l = l + mse_on_pixels_factor * l_pixels

        loss_history.append(l.detach().item())
        l.backward()
        optim.step()
        
        # ADDED
        # paste the original z0 where mask is one
        latent_blurred_mask_replace = latent_blurred_mask_replace.to(z0.device)
        z0.data = latent_blurred_mask_replace * _z0_original.data + (1 - latent_blurred_mask_replace) * z0.data
        # END ADDED
        
        # validate
        if update_number % validation_steps == 0:
            with torch.no_grad():
                image_from_z0 = latent_to_pil(z0, pipe_attacker)[0]
                
                # ADDED
                # paste original pixels in masked areas before running them through watermark checking
                image_from_z0.paste(pixel_replace, (0, 0), pixel_replace)
                # END ADDED
                
                glue_results = diff_and_glue_2_PIL_images(_image_original, _reconstructed_from_z0_original, image_from_z0)
                z_delta_norm = (z0 - _z0_original).norm().item()

                # watermark test
                #inv_wm = pipe_wm.invert_z0(z0, num_inference_steps=num_inference_steps)  # This is a zT_torch with batch dim
                inv_wm = pipe_provider_target.invert_images(image_from_z0, num_inference_steps=num_inference_steps)["zT_torch"]
                accuracy_results = wm_provider.get_accuracies(inv_wm)
                accuracy = accuracy_results["accuracies"][0]
                p_value = accuracy_results["p_values"][0] if "p_values" in accuracy_results else 0.0
            
            # log
            print(f"Step {update_number}: z_0 delta norm: {(z0 - _z0_original).norm().item():.5f}, z_0 grad norm: {z0.grad.norm().item():.5f}, loss: {l.item():.5f}, accuracy: {accuracy:.5f}, p_value: {p_value}")
            wandb.log({#"zT": wandb.Image(inverted_latent.detach().cpu()),
                       #"z0": wandb.Image(z0.detach().cpu()),
                       "image_from_z0": wandb.Image(image_from_z0),
                       "glued": wandb.Image(glue_results["glued"]),
                       "diff_to_original_image_(color)": wandb.Image(glue_results["_0_2"]),
                       "diff_to_original_image_(abs)": wandb.Image(glue_results["_0_3"]),
                       "diff_to_reconstructed_from_z0_(color)": wandb.Image(glue_results["_1_2"]),
                       "diff_to_reconstructed_from_z0_(abs)": wandb.Image(glue_results["_1_3"]),
                       "z0_delta_norm": z_delta_norm,
                       "z0_grad_norm": z0.grad.norm().item(),
                       "loss": l.item(),
                       "accuracy": accuracy,
                       "p_value": p_value,
                       },
                       step=update_number+1)
            # save
            for name, img in zip(["image_from_z0",
                                  "glued",
                                  "diff_to_original_image_(color)",
                                  "diff_to_original_image_(abs)",
                                  "diff_to_reconstructed_from_z0_(color)",
                                  "diff_to_reconstructed_from_z0_(abs)"],
                                 [image_from_z0,
                                  glue_results["glued"],
                                  glue_results["_0_2"],
                                  glue_results["_0_3"],
                                  glue_results["_1_2"],
                                  glue_results["_1_3"],]):
                dir_path = os.path.join(out_dir, name)
                os.makedirs(dir_path, exist_ok=True)
                img.save(os.path.join(dir_path, f"{target_image_name}_{update_number}.png"))


if __name__ == "__main__":
    #fire.Fire(main_inverse)
    #fire.Fire(main_forward)

    main_inverse(**vars(args))
