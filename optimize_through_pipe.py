from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
import torch
import numpy as np
from scipy.stats import norm
import fire

from diffusers import AutoPipelineForText2Image as AutoPipeline, StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler, Transformer2DModel, PixArtSigmaPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
# from diffusers.pipelines.stable_diffusion.pipeline_flax_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import make_image_grid, deprecate

from torchvision.transforms.functional import to_pil_image, to_tensor, gaussian_blur

from torch.utils.checkpoint import checkpoint
import tqdm

import PIL


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


def load_pipe(modelid=None, device="cuda", **kwargs):
    modelid = "stabilityai/stable-diffusion-2-1-base" if modelid is None else modelid
    DEVICE = torch.device(device)
    # Load the pre-trained Stable Diffusion 2.1 model from Hugging Face
    transformer = None
    unet = None
    if modelid == "PixArt-alpha/PixArt-Sigma-XL-2-512-MS":
        unetmodelid = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        transformer = Transformer2DModel.from_pretrained(unetmodelid, subfolder="transformer", torch_dtype=torch.float32).to(DEVICE)
        modelid = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    pipe = AutoPipeline.from_pretrained(modelid, # DPMSolverMultistepScheduler.from_pretrained(...)
                                                safety_checker=None,
                                                torch_dtype=torch.float32).to(DEVICE)
    if modelid in ("PixArt-alpha/PixArt-Sigma-XL-2-512-MS", "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"):
        pipe.scheduler = pipe.scheduler # DPMSolverMultistepScheduler with order = 1
    else:
        pipe.scheduler = DDIMScheduler.from_pretrained(modelid, subfolder="scheduler", torch_dtype=torch.float32)
    
    if unet is not None:
        pipe.unet = unet
    if transformer is not None:
        pipe.transformer = transformer

    scheduler_class_map = {
        DDIMScheduler: DDIMInverseScheduler,
        DPMSolverMultistepScheduler: DPMSolverMultistepInverseScheduler,
    }
    inverse_scheduler_class = None
    for scheduler_class, inverse_scheduler_class in scheduler_class_map.items():
        if isinstance(pipe.scheduler, scheduler_class):
            inverse_scheduler = inverse_scheduler_class.from_pretrained(modelid, subfolder='scheduler', torch_dtype=torch.float32)  # DPMSolverMultistepInverseScheduler.from_pretrained(...)
    
    return pipe, inverse_scheduler


def generate_image(pipe=None, scheduler=None, 
                   prompt="a futuristic city skyline at sunset",
                   latents=None,
                   num_inference_steps=20,
                   guidance_scale=7.5,
                   resolution=512):
    orig_sched = pipe.scheduler
    
    pipe.scheduler = scheduler
    # Generate an image
    image = pipe(prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            latents=latents).images[0]
    
    pipe.scheduler = orig_sched
    return image


class PixArtSigmaDiffPipe(PixArtSigmaPipeline):
    
    @classmethod
    def convert(cls, pipe):
        pipe.__class__ = cls
        # wrap transformer into one that's gradient checkpointable
        pipe.transformer = TransformerWrapper(pipe.transformer)
        return pipe
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 300): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 256:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        device = self.vae.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        with torch.no_grad():
            # 3. Encode input prompt
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas
                )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = checkpoint(self.transformer,
                    latent_model_input,
                    prompt_embeds,
                    prompt_attention_mask,
                    current_timestep,
                    added_cond_kwargs,
                    False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class StableDiffusionDiffPipe(StableDiffusionPipeline):
    
    @classmethod
    def convert(cls, pipe):
        pipe.__class__ = cls
        # wrap transformer into one that's gradient checkpointable
        pipe.unet = UnetWrapper(pipe.unet)
        return pipe
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.vae.device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = checkpoint(self.unet,
                    latent_model_input,
                    t,
                    prompt_embeds,
                    timestep_cond,
                    self.cross_attention_kwargs,
                    added_cond_kwargs,
                    False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return ImagePipelineOutput(images=image)


class DiffPipe(torch.nn.Module):
    """ A class for backpropagating through a pipeline's generation loop using gradient checkpointing.
    Before use, must check if one step of the pipeline (unet) is deterministic.
    """
    def __init__(self, pipe, scheduler=None, device=None):
        super().__init__()
        self.pipe = pipe
        self.device = device
        self.scheduler = scheduler
        
        if isinstance(self.pipe, PixArtSigmaPipeline):
            self.pipe = PixArtSigmaDiffPipe.convert(self.pipe)
        if hasattr(self.pipe, "unet"):
            self.diffusionmodel = UnetWrapper(self.pipe.unet)
        if hasattr(self.pipe, "transformer"):
            self.diffusionmodel = TransformerWrapper(self.pipe.transformer)
        
        for param in self.diffusionmodel.parameters():
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
            
    
class WrapperBase(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    @property
    def config(self): return self.model.config
    
    
class UnetWrapper(WrapperBase):
    """ Necessary to wrap Unet because gradient checkpoint doesn't allow kwargs so need to make used args a tuple. """
        
    def forward(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    timestep_cond,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict,):
        ret = self.model(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,)
        return ret
    
    
class TransformerWrapper(WrapperBase):
    """ Necessary to wrap Unet because gradient checkpoint doesn't allow kwargs so need to make used args a tuple. """
    
    def forward(self, latent_model_input,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    added_cond_kwargs,
                    return_dict):
        ret = self.model(
                    latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=return_dict)
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


# def main_forward(lr=1e-2,
#                  updates=100,
#                  num_inference_steps=20,
#                  guidance_scale=4.5,
#                  modelid="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
#                  ):

def main_forward(lr=1e-2,
                 updates=100,
                 num_inference_steps=20,
                 guidance_scale=7.5,
                 modelid="stabilityai/stable-diffusion-2-1-base",
                 device=1,
                 ):
    
    device = torch.device("cuda", device)
    # first test: generate an image, invert it and optimize inverted zT to produce something closer to original image
    pipe, inverse_scheduler = load_pipe(modelid=modelid, device=device)
    
    # ############ create example
    # prompt = "a futuristic city skyline at sunset"
    prompt = "a black and white photograph of a cow"
    original_zT = torch.randn(1, 4, 64, 64)
    image = generate_image(pipe, pipe.scheduler, prompt=prompt, latents=original_zT, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image_pt = to_tensor(image)
    
    _inverted = invert_image(pipe=pipe, scheduler=inverse_scheduler, image_pt=image_pt.to(pipe.device)[None], 
                             num_inference_steps=num_inference_steps)
    
    # check what reconstruction looks like, to later compare to what differentiable pipe produces
    # _recons = generate_image(pipe, forward_scheduler, prompt=prompt, latents=inverted)
    print(image_pt.shape, _inverted.shape)
    # ############ end of create example
    
    if isinstance(pipe, PixArtSigmaPipeline):
        diffpipe = PixArtSigmaDiffPipe.convert(pipe)
    elif isinstance(pipe, StableDiffusionPipeline):
        diffpipe = StableDiffusionDiffPipe.convert(pipe)
    
    with torch.no_grad():       # verify that with original noise, our adapted pipeline generates the same image
        check_image = diffpipe(latents=original_zT.to(diffpipe.vae.device), prompt=prompt, num_inference_steps=num_inference_steps, 
                               guidance_scale=guidance_scale, output_type="latent")[0]
        check_image = latent_to_pil(check_image, diffpipe)[0]
    
    inverted = torch.nn.Parameter(_inverted.detach().clone())
    optim = torch.optim.Adam([inverted], lr=lr)
    
    # make trainable noise
    # zT = torch.nn.Parameter(inverted)
    # optim = torch.optim.SGD([zT], lr=lr)
    
    # define loss
    latent_target = pixel_to_latent(image, diffpipe)
    
    recons_history_latents = [None]
    recons_history_pil = [image]
    zT_history = []
    loss_history = []
    
    recons_overview = None
    
    # training loop
    for update_number in tqdm.tqdm(range(updates)):
        optim.zero_grad()
        
        recons = diffpipe(latents=inverted, prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, output_type="latent")[0]
        
        # zT_history.append(inverted.detach().clone())    
        recons_history_latents.append(recons.detach().clone())
        recons_history_pil.append(latent_to_pil(recons_history_latents[-1], pipe)[0])
        
        grid = [recons_history_pil[0], recons_history_pil[-1], recons_history_pil[1]]
        recons_overview = make_image_grid(grid, 1, len(grid))
        
        l = torch.nn.functional.mse_loss(recons, latent_target.detach())
        loss_history.append(l.detach().item())
        l.backward()
        
        print(f"Step {update_number}: z_T delta norm: {(inverted - _inverted).norm().item():.5f}, z_T grad norm: {inverted.grad.norm().item():.5f}, loss: {l.item():.5f}")
        
        optim.step()
    
    print("Inverted grad norm:", inverted.grad.norm())
    
    
def main_inverse(lr=1e-3,
         image="trump2.jpg",
         mask="trump2.mask.png",
         prompt="a photograph of stalin and his advisers taking a walk by the river",
         updates=100):
    
    if isinstance(image, str):      # probably a path
        image = PIL.Image.open(image).convert("RGB")
        
        if mask is None:
            mask = PIL.Image.new('L', image.size, 0)  # 'L' for grayscale, 0 for black (transparent)
            # Create a drawing context for the mask
            draw = PIL.ImageDraw.Draw(mask)

            # # STALIN MASKS
            # # Draw a shape on the mask (e.g., an ellipse)
            # # stalin's face
            # draw.ellipse((160, 131, 227, 192), fill=255)  # Fill the ellipse with white (opaque)
            # # removed face
            # draw.ellipse((350, 183, 401, 240), fill=255)  # Fill the ellipse with white (opaque)
            # # face on the left
            # draw.ellipse((10, 150, 77, 200), fill=255)  # Fill the ellipse with white (opaque)
            
            # TRUMP MASKS
            draw.ellipse((520, 130, 673, 300), fill=255)  # Fill the ellipse with white (opaque)
            
        else:
            mask = PIL.Image.open(mask).convert("L")
        blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=5))  # Adjust the radius as needed
        blurred_mask_pt = to_tensor(blurred_mask)
        latent_blurred_mask = torch.nn.functional.interpolate(blurred_mask_pt[None], size=64, mode="bilinear")[0]

        pixel_replace = deepcopy(image)
        # Apply the mask to the image
        pixel_replace.putalpha(blurred_mask)  # This adds the mask to the image's alpha channel
        
    # inverse test: generate an image, invert it and optimize reconstruction to produce a latent closer to original latent
    pipe, inverse_scheduler = load_pipe()
    
    if image is None:
    # ############ create example
        # prompt = "a futuristic city skyline at sunset"
        prompt = "a black and white photograph of a cow"
        original_zT = torch.randn(1, 4, 64, 64)
        image = generate_image(pipe, pipe.scheduler, prompt=prompt, latents=original_zT)
        
        
    image_pt = to_tensor(image)
    _z0 = pixel_to_latent(image, pipe)
    
    inverted = invert_image(pipe=pipe, scheduler=inverse_scheduler, image=image)
    
    # check what reconstruction looks like, to later compare to what differentiable pipe produces
    # _recons = generate_image(pipe, forward_scheduler, prompt=prompt, latents=inverted)
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
    
    
    with torch.no_grad():
        z0_init = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
        z0_init = to_pil_image(z0_init[0].clamp(-1, 1) * 0.5 + 0.5)
        # compose faces into reconstruction
        z0_init.paste(pixel_replace, (0, 0), pixel_replace)
    
    # training loop
    for update_number in tqdm.tqdm(range(updates)):
        optim.zero_grad()
        
        with torch.no_grad():
            z0_image = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
            z0_image = to_pil_image(z0_image[0].clamp(-1, 1) * 0.5 + 0.5)
            z0_image.paste(pixel_replace, (0, 0), pixel_replace)
        
        # recons = diffpipe(zT, prompt)
        inverted_latent = diffpipe(z0, "", guidance_scale=1.)
        
        # zT_history.append(inverted.detach().clone())    
        inverted_history.append(inverted_latent.detach().clone())
        
        l = torch.nn.functional.mse_loss(inverted_latent, target_latent.detach())
        loss_history.append(l.detach().item())
        l.backward()
        
        optim.step()
        
        latent_blurred_mask = latent_blurred_mask.to(z0.device)
        z0.data = latent_blurred_mask * _z0.data + (1 - latent_blurred_mask) * z0.data
        print(f"Step {update_number}: z_0 delta norm: {(z0 - _z0).norm().item():.5f}, z_0 grad norm: {z0.grad.norm().item():.5f}, loss: {l.item():.5f}")
        
        
    print("Done training")
    
    


if __name__ == "__main__":
    # fire.Fire(main_inverse)
    fire.Fire(main_forward)