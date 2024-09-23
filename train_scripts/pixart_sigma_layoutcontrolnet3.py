# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from functools import partial
import html
import inspect
import re, os, random
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora, LoRAParametrization, get_lora_state_dict

import torch

from diffusers import PixArtSigmaPipeline

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN

from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers.utils.import_utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.models.attention_processor import Attention, AttnProcessor2_0


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess_example(example, caption_column="captions", proportion_empty_prompts=0, max_length=300, tokenizer=None, omit_global_caption=False, ):
    globalcaption = example[caption_column]
    if omit_global_caption:
        globalcaption = ""
    captions = [globalcaption] + example["seg_captions"]
    example["seg_input_ids"] = []
    example["seg_attention_mask"] = []
    example["whichlayer"] = []
    for i, seg_caption in enumerate(captions):
        if seg_caption is not None:
            # if omit_global_caption and i == 0:
            #     continue
            seg_input_ids, seg_attention_mask = tokenize_caption(caption=seg_caption, proportion_empty_prompts=proportion_empty_prompts, max_length=max_length, tokenizer=tokenizer)
            seglen = int((seg_input_ids != 0).float().sum().item())
            example["seg_input_ids"].append(seg_input_ids[0, :seglen-1])
            example["whichlayer"].append(torch.ones_like(example["seg_input_ids"][-1]) * i )    # if global caption is not at zero: (i+1))
    example["seg_input_ids"].append(torch.ones_like(seg_input_ids[0, :1]))
    example["whichlayer"].append(torch.zeros_like(seg_input_ids[0, :1]))
            
    example["seg_input_ids"] =  torch.cat(example["seg_input_ids"], 0)
    example["seg_attention_mask"] =  torch.ones_like(example["seg_input_ids"])
    example["whichlayer"] =  torch.cat(example["whichlayer"], 0) 
    return example


def tokenize_caption(example=None, caption=None, is_train=True, proportion_empty_prompts=0., max_length=300, tokenizer=None, caption_column="captions"):
    if random.random() < proportion_empty_prompts:
        caption = ""
    else:
        if caption is None:
            assert example is not None
            caption = example[caption_column]
        else:
            assert example is None
        if isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption = random.choice(caption) if is_train else caption[0]
        if not caption.endswith("."):
            caption += "."
    # if not seg_caption.endswith("."):
    #     seg_caption += "."
    inputs = tokenizer(caption, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids, inputs.attention_mask


def collate_fn(examples):
    images = torch.stack([example["image"] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    cond_images = torch.stack([example["cond_image"] for example in examples])
    cond_images = cond_images.to(memory_format=torch.contiguous_format).float()
    whichexample, input_ids, attention_mask, whichlayer = [], [], [], []
    for i, example in enumerate(examples):
        input_ids.append(example["seg_input_ids"])
        attention_mask.append(example["seg_attention_mask"])
        whichlayer.append(example["whichlayer"])
        # whichexample.append(torch.tensor(i, dtype=torch.long, device=input_ids[-1].device))
    input_ids, attention_mask, whichlayer = [torch.nn.utils.rnn.pad_sequence(x, batch_first=True) for x in [input_ids, attention_mask, whichlayer]]
    return {"image": images, 
            "cond_image": cond_images,
            "input_ids": input_ids, 
            "prompt_attention_mask": attention_mask,
            "whichlayer": whichlayer,
            }
    

def encode_text(batch, encoder):
    # minimize length:
    input_ids, attention_mask, whichlayer = batch["input_ids"], batch["prompt_attention_mask"], batch["whichlayer"]    
    
    # run encoder model:
    encoded = encoder(input_ids.to(encoder.device), attention_mask=attention_mask)[0]    
    return encoded, attention_mask, whichlayer



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def zero_module(m):
    for param in m.parameters():
        torch.fill_(param.data, 0)
        
        
def create_zeroconv2d(inpch, outch, kernel, padding=0):
    conv = torch.nn.Conv2d(inpch, outch, kernel, padding=padding)
    zero_module(conv)
    return conv


def create_zerolin(inpch, outch):
    conv = torch.nn.Linear(inpch, outch)
    zero_module(conv)
    return conv


def create_identlin(inpch, outch):
    conv = torch.nn.Linear(inpch, outch)
    conv.weight.data = torch.diagflat(torch.ones_like(torch.diagonal(conv.weight.data))) * 0.5
    conv.bias.data = torch.zeros_like(conv.bias.data)
    return conv


class FeatureGate(torch.nn.Module):
    init_main_frac = 0.9
    
    def __init__(self, numfeats):
        self.numfeats = numfeats
        self.gate_main = torch.nn.Parameter(torch.ones(self.numfeats) * self.init_main_frac)
        self.gate_branch = torch.nn.Parameter(torch.ones(self.numfeats) * (1 - self.init_main_frac))
        
    def forward(self, main, branch):
        gated_main = main * self.gate_main[None, None]
        gated_branch = branch * self.gate_branch[None, None]
        return gated_main + gated_branch
    
    
class AttnMixer(torch.nn.Module):
    def __init__(self, dim, numheads):
        super().__init__()
        self.dim, self.numheads = dim, numheads
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.dim, self.dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim // 2, self.dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim, self.numheads),
            torch.nn.Sigmoid(),
        )
        
    def forward(self, x):
        batsize, numheads, numpix, dim = x.shape
        ret = x.transpose(1, 2).view(batsize, numpix, numheads * dim)
        ret = self.model(ret)
        return ret 
    
    
class AttnMixerLight(torch.nn.Module):
    def __init__(self, dim, numheads):
        super().__init__()
        self.dim, self.numheads = dim, numheads
        self.headweights = torch.nn.Parameter(torch.zeros(self.numheads))
        
    def forward(self, x):
        ret = torch.sigmoid(self.headweights[None, None])
        return ret
    

class CustomAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    
    mask_attention = True

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
    @classmethod
    def adapt(cls, obj, attn, use_attention_embeddings=False, mask_attention=True, attention_mix=False, attention_mix_light=False):
        obj.__class__ = cls
        NUM_OBJ_LAYERS = 25
        T5_DIM = 1152
        if attention_mix:
            attn.attnmix = AttnMixer(T5_DIM, attn.heads)
        
        obj.mask_attention = True if (mask_attention or attention_mix) else False
        obj.attention_mix = attention_mix
        
        if use_attention_embeddings:
            attn.obj_layer_embed = torch.nn.Embedding(NUM_OBJ_LAYERS, T5_DIM)
            torch.nn.init.zeros_(attn.obj_layer_embed.weight)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        layer_ids: Optional[torch.Tensor] = None,   # BEWARE: layer ids are +1 higher than the actual index of their object mask in object_masks
        object_masks: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        # embed control layer ids and add them to control_encoder_hidden_states
        # obj_layer_embed = self.obj_layer_embed(layer_ids)
        # obj_layer_embed *= attention_mask.float()[:, :, None]
        encoder_hidden_states_keys = encoder_hidden_states
        if hasattr(attn, "obj_layer_embed") and attn.obj_layer_embed is not None:
            encoder_hidden_states_keys = encoder_hidden_states_keys + attn.obj_layer_embed(layer_ids)
        
        key = attn.to_k(encoder_hidden_states_keys)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        
        # 1. compute attention without object masks
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        if self.mask_attention:
            # 2. compute attention with object masks
            object_masks = torch.nn.functional.avg_pool2d(object_masks, 16, 16) > 0
            # globalonly = torch.where(object_masks.sum(1) > 0, torch.zeros_like(object_masks[:, 0]), torch.ones_like(object_masks[:, 0]))
            globalonly = torch.ones_like(object_masks[:, 0])
            object_masks = torch.cat([globalonly[:, None], object_masks], 1)
            object_masks = object_masks.flatten(2)
            
            object_attention_mask = torch.take_along_dim(object_masks, layer_ids[:, :, None], 1)
            object_attention_mask = (object_attention_mask.transpose(1, 2)[:, None].float() - 1.) * 10000.
            
            attention_mask = torch.min(attention_mask, object_attention_mask)

            masked_hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            
            if self.attention_mix:
                # 3. combine attentions with and without object masks
                mixweights = attn.attnmix(query).transpose(1, 2)[:, :, :, None]
                hidden_states = hidden_states * mixweights + masked_hidden_states * (1 - mixweights)
            else:
                hidden_states = masked_hidden_states

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

class PixArtTransformer2DModelWithLayoutControlNet(PixArtTransformer2DModel):
    def initialize_adapter(self, control_blocks, use_identlin=False):
        self.control_blocks = torch.nn.ModuleList(control_blocks)
        
        self.zeroconvs = torch.nn.ModuleList([None for _ in self.control_blocks])
        
        self.use_identlin = use_identlin
        
        if not self.use_identlin:
            self.zeroconvs = torch.nn.ModuleList([
                create_zerolin(block.ff.net[2].out_features, block.ff.net[2].out_features) if block is not None else None \
                    for block in self.control_blocks
            ])
        else:
            # self.zeroconvs = torch.nn.ModuleList([
            #     create_identlin(block.ff.net[2].out_features, block.ff.net[2].out_features) if block is not None else None \
            #         for block in self.control_blocks
            # ])
            self.zeroconvs = torch.nn.ModuleList([
                FeatureGate(block.ff.net[2].out_features) if block is not None else None for block in self.control_blocks
            ])
    
    @classmethod
    def adapt(cls, main, control_encoder=None, control_encoder2=None, num_layers=-1, use_controlnet=False, 
              use_adapters=False, use_controllora=False, lora_rank=64, use_identlin=False, 
              use_attention_embeddings=False, use_masked_attention=False, use_attention_mix=False, use_attention_mix_light=False):
        main.__class__ = cls
        main.control_blocks, main.zeroconvs, main.simple_connectors = None, None, None
        
        main.control_encoder = control_encoder
        main.simple_encoder = control_encoder2
            
        if use_controlnet:
            control_blocks = [None] * len(main.transformer_blocks)
            for i in range(len(main.transformer_blocks)):
                if num_layers == -1 or i < num_layers:
                    control_blocks[i] = deepcopy(main.transformer_blocks[i])
        
            main.initialize_adapter(control_blocks, use_identlin=use_identlin)
                    
            for block in main.control_blocks:
                if block is not None:
                    CustomAttnProcessor2_0.adapt(block.attn2.processor, block.attn2, use_attention_embeddings=use_attention_embeddings, 
                                                 mask_attention=use_masked_attention, attention_mix=use_attention_mix, attention_mix_light=use_attention_mix_light)
                    
        
        if use_adapters:
            connectors = [None] * len(main.transformer_blocks)
            tm_dim = main.transformer_blocks[0].ff.net[2].out_features
            for i in range(len(main.transformer_blocks)):
                connectors[i] = torch.nn.Sequential(
                    torch.nn.Linear(control_encoder2.outchannels, tm_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(tm_dim, tm_dim),
                )
                
            main.simple_connectors = torch.nn.ModuleList(connectors)
            
        if use_controllora:
            lora_config = {  # specify which layers to add lora to, by default only add to linear layers
                torch.nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=lora_rank, lora_alpha=lora_rank),
                },
            }
            add_lora(main.transformer_blocks, lora_config)
        
        return main
    
    def get_trainable_parameters(self):
        ret = []
        if self.control_encoder is not None:
            ret += list(self.control_encoder.parameters())
        if self.control_blocks is not None:
            ret += list(self.control_blocks.parameters())
        if self.zeroconvs is not None:
            ret += list(self.zeroconvs.parameters())
        if self.simple_encoder is not None:
            ret += list(self.simple_encoder.parameters()) 
        if self.simple_connectors is not None:
            ret += list(self.simple_connectors.parameters())
        
        ret += list(get_lora_params(self))
        # ret += list(self.obj_layer_embed.parameters())
        
        return ret
    
    def get_control_state_dict(self):
        ret = {}
        for k, v in self.state_dict().items():
            if k.split(".")[0] in {"control_encoder", "control_blocks", "zeroconvs", "simple_encoder", "simple_connectors"}:
                ret[k] = v
        for k, v in get_lora_state_dict(self).items():
            ret[k] = v
        return ret
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        control_image: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        encoder_hidden_states, control_layer_ids = encoder_hidden_states
        encoder_attention_mask = encoder_attention_mask
        
        control_cross_attention_kwargs = {"layer_ids": control_layer_ids, "object_masks": control_image}
        
        # Encode control image and add to hidden states
        if self.control_encoder is not None:
            control_latents = self.control_encoder(control_image)[0]       # control encoder should already have zeroconv on it
            hidden_states_control = hidden_states + control_latents
            
        if self.simple_encoder is not None:
            simple_latents = self.simple_encoder(control_image)[0]
            simple_latents = simple_latents.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)       # patching happens here
        if self.control_encoder is not None:
            hidden_states_control = self.pos_embed(hidden_states_control)
        prev_hidden_state_control = None

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            control_block = self.control_blocks[i] if self.control_blocks is not None else None
            zeroconv = self.zeroconvs[i] if self.zeroconvs is not None else None
            simple_connector = self.simple_connectors[i] if self.simple_connectors is not None else None
            
            hidden_states_in = hidden_states
            if simple_connector is not None:
                hidden_states_in += simple_connector(simple_latents)
                
            # if prev_hidden_state_control is not None:
            #     hidden_states_in += prev_hidden_state_control
                    
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states_in,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
                if control_block is not None:
                    hidden_states_control = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(control_block),
                        hidden_states_control,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        control_cross_attention_kwargs,
                        None,
                        **ckpt_kwargs,
                    )
                    if self.use_identlin:
                        prev_hidden_state_control = zeroconv(hidden_states, hidden_states_control)
                    else:
                        prev_hidden_state_control = zeroconv(hidden_states_control)
                    hidden_states += prev_hidden_state_control
            else:
                hidden_states = block(
                    hidden_states_in,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )
                if control_block is not None:
                    hidden_states_control = control_block(
                        hidden_states_control,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=control_cross_attention_kwargs,
                        class_labels=None,
                    )
                    if self.use_identlin:
                        prev_hidden_state_control = zeroconv(hidden_states, hidden_states_control)
                    else:
                        prev_hidden_state_control = zeroconv(hidden_states_control)
                    hidden_states += prev_hidden_state_control

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class PixArtSigmaLayoutControlNetPipeline(PixArtSigmaPipeline):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        ret = super(PixArtSigmaLayoutControlNetPipeline, cls).from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        ret.__class__ = cls
        return ret
    
    def check_cond(self, condimage, height, width):
        pass        # TODO
    
    def set_preprocess_example(self, fn):
        self.preprocess_example = fn

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        obj_prompts: List[str] = None,
        negative_prompt: str = "",
        obj_masks:Optional[torch.Tensor] = None,
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
        control_image = obj_masks
        
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
        
        self.check_cond(control_image, height, width)

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        # # MAIN PROMPT ENCODING
        # (
        #     prompt_embeds,
        #     prompt_attention_mask,
        #     negative_prompt_embeds,
        #     negative_prompt_attention_mask,
        # ) = self.encode_prompt(
        #     prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt=negative_prompt,
        #     num_images_per_prompt=num_images_per_prompt,
        #     device=device,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     prompt_attention_mask=prompt_attention_mask,
        #     negative_prompt_attention_mask=negative_prompt_attention_mask,
        #     clean_caption=clean_caption,
        #     max_sequence_length=max_sequence_length,
        # )
        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            
        # CONTROL PROMPT ENCODING:
        # create example and preprocess it
        example = {"captions": [prompt],
                   "seg_captions": obj_prompts,
                   "cond_image": obj_masks,
                   "image": obj_masks}
        example = self.preprocess_example(example, tokenizer=self.tokenizer)
        example = collate_fn([example])
        example = {k: v.to(self.text_encoder.device) for k, v in example.items()}
        prompt_embeds, prompt_attention_mask, layer_ids = encode_text(example, self.text_encoder)
        
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
             negative_layer_ids = torch.zeros_like(layer_ids)
             layer_ids = torch.cat([negative_layer_ids, layer_ids], 0)
            
        prompt_embeds = (prompt_embeds, layer_ids)

        dtype = prompt_embeds[0].dtype

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
            dtype,
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
                control_image_input = torch.cat([control_image] * 2) if do_classifier_free_guidance else control_image
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
                noise_pred = self.transformer(
                    latent_model_input,
                    control_image=control_image_input.to(dtype=dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
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
