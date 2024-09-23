# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
from copy import deepcopy
from functools import partial
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import fire
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtSigmaPipeline, PixArtAlphaPipeline, \
    Transformer2DModel as Transformer2DModel
    # PixArtTransformer2DModel as Transformer2DModel

from pixart_sigma_layoutcontrolnet3 import PixArtSigmaLayoutControlNetPipeline, PixArtTransformer2DModelWithLayoutControlNet, create_zeroconv2d, preprocess_example, collate_fn, encode_text
    
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

import wandb

from cocodata import COCOInstancesDataset, COCOPanopticDataset, masktensor_to_colorimage
from train_control_ae import ControlSignalEncoder, ControlSignalEncoderV2, ControlSignalEncoderV3

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


ROOTNAME = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

def parse_args(inpargs):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    
    parser.add_argument(
        "--startfrom",
        type=str,
        default=None,
        help="Custom checkpoint to start from.",
    )
    
    parser.add_argument(
        "--num_control_layers",
        type=int,
        default=-1,
        help="Number of layers of the DiT to copy for ControlNet adaption. If -1, all layers are Control-adapted.",
    )
    
    parser.add_argument(
        "--control_encoder",
        type=str,
        default=None,
        help="Path to pretrained control encoder",
    )
    parser.add_argument(
        "--control_encoder2",
        type=str,
        default=None,
        help="Path to second pretrained control encoder for simple adapter",
    )
    
    parser.add_argument(
        "--use_controlnet",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    
    parser.add_argument(
        "--use_bbox",
        default=False,
        action="store_true",
        help=(
            "Whether to use bbox masks instead of instance masks."
        ),
    )
    
    parser.add_argument(
        "--no_masked_attention",
        default=False,
        action="store_true",
        help=(
            "Whether to use bbox masks instead of instance masks."
        ),
    )
    
    
    parser.add_argument(
        "--use_attention_mix",
        default=False,
        action="store_true",
        help=(
            "Whether to use attention mix instead of just masking attention."
        ),
    )
    
    parser.add_argument(
        "--use_attention_mix_light",
        default=False,
        action="store_true",
        help=(
            "Whether to use attention mix light instead of full attention mix. Light mix only learns per-head per-layer, independently of input."
        ),
    )
    
    parser.add_argument(
        "--omit_global_caption",
        default=False,
        action="store_true",
        help=(
            "Whether to omit global caption"
        ),
    )
    
    parser.add_argument(
        "--use_attention_embeddings",
        default=False,
        action="store_true",
        help=(
            "Whether to use attention embeddings"
        ),
    )
    
    parser.add_argument(
        "--use_identlin",
        default=False,
        action="store_true",
        help=(
            "Initialize controlnet's zero lin's to identity matrices instead"
        ),
    )
    
    
    parser.add_argument(
        "--use_controllora",
        default=False,
        action="store_true",
        help=(
            "Use control lora on base transformer model"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help=(
            "LoRA rank"
        ),
    )
    
    parser.add_argument(
        "--use_adapters",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    
    parser.add_argument(
        "--pretrained_model_name_or_path", "--model",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir", "--traindir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default="a stylized portrait drawing the style of thisnewartistA", help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=None,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_steps", type=int, default=10000)
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=20, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="Whether or not to use Dora. For more information, see"
             " https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.use_dora"
    )
    parser.add_argument(
        "--use_rslora",
        action="store_true",
        default=False,
        help="Whether or not to use RS Lora. For more information, see"
             " https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.use_rslora"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--micro_conditions",
        default=False,
        action="store_true",
        help="Only set to true for `PixArt-alpha/PixArt-XL-2-1024-MS`"
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=300,
        help="max length for the tokenized text embedding.",
    )

    parser.add_argument("--local-rank", type=int, default=-1)

    args = parser.parse_args(inpargs)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def validate_args(args):
    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")


DATASET_NAME_MAPPING = {"lambdalabs/pokemon-blip-captions": ("image", "text"),
                        "svjack/pokemon-blip-captions-en-zh": ("image", "en_text")}


def default(f, fallback):
    try:
        return f()
    except AttributeError as e:
        return fallback
    

NUMOBJ = 20
def load_data(args, tokenizer, split="train"):
    # See Section 3.1. of the paper.
    max_length = args.max_token_length
    
    # if accelerator.is_main_process:
    dataset = COCOInstancesDataset(maindir=args.train_data_dir, split=split, upscale_to=512, max_masks=NUMOBJ, use_bbox=default(lambda: args.use_bbox, None))
    
    # with accelerator.is_main_process:
    dataset.transforms.append(partial(preprocess_example, proportion_empty_prompts=args.proportion_empty_prompts, max_length=max_length, tokenizer=tokenizer, omit_global_caption=args.omit_global_caption))


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # batch = next(iter(train_dataloader))
    # print(batch)
    
    return dataset, train_dataloader


def load_model(args, mixed_precision, device):
    # For mixed precision training we cast all non-trainable weights
    # (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(ROOTNAME, subfolder="scheduler", torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(ROOTNAME, subfolder="tokenizer", torch_dtype=weight_dtype)
    text_encoder = T5EncoderModel.from_pretrained(ROOTNAME, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    vae = AutoencoderKL.from_pretrained(ROOTNAME, subfolder="vae", torch_dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device)

    transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

    # freeze parameters of models to save more memory
    transformer.requires_grad_(False)
    
    # Freeze the transformer parameters before adding adapters
    for param in transformer.parameters():
        param.requires_grad_(False)
        
        
    return transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype


def load_control_encoders(args, device, weight_dtype):
    # control encoder
    controlencoder = None
    if args.control_encoder is not None:
        controlencoder = ControlSignalEncoderV3(21, 4).to(weight_dtype)
        controlencoder.load_state_dict(torch.load(Path(args.control_encoder)))
        controlencoder.zeroconv = create_zeroconv2d(4, 4, 1, padding=0)
        controlencoder.to(device)
    
    controlencoder2 = None
    if args.control_encoder2 is not None:
        controlencoder2 = ControlSignalEncoderV3(21, 512, patchsize=2).to(weight_dtype)
        controlencoder2.load_state_dict(torch.load(Path(args.control_encoder2)))
        controlencoder2.to(device)
        
    return controlencoder, controlencoder2


def load_pretrained(path, device=torch.device("cuda"), return_pipe=True, load_saved_state_dict=True):
    path = Path(path)
    with open(path / "args.json") as f:
        args = argparse.Namespace(**json.load(f))
        
    if hasattr(args, "startfrom") and args.startfrom is not None:
        transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype = load_pretrained(args.startfrom, device=device, return_pipe=False, load_saved_state_dict=True)
    else:
        transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype = load_model(args, None, device)
    control_encoder, control_encoder2 = load_control_encoders(args, device, weight_dtype)
    transformer = adapt_transformer_controlnet(transformer, args=args, control_encoder=control_encoder, control_encoder2=control_encoder2, num_layers=args.num_control_layers, weight_dtype=weight_dtype)
    
    if load_saved_state_dict:
        savedpaths = list(path.glob("saved-*"))
        savedpaths = sorted(savedpaths, key=lambda x: int(x.stem[6:]), reverse=True)
        print("Loading from: ", savedpaths[0])
            
        sd = torch.load(savedpaths[0] / "controlnet_weights.pt")
        transformer.load_state_dict(sd, strict=False)
    
    print("loaded pipeline components, loading pipeline")
    
    if return_pipe:
        pipeline = PixArtSigmaLayoutControlNetPipeline.from_pretrained(ROOTNAME)
        pipeline.vae = vae
        pipeline.text_encoder = text_encoder
        pipeline.transformer = transformer
        
        pipeline.set_preprocess_example(partial(preprocess_example, proportion_empty_prompts=args.proportion_empty_prompts, max_length=args.max_token_length, omit_global_caption=args.omit_global_caption))
        
        return pipeline, tokenizer, noise_scheduler, args
    else:
        return transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype


def setup_accelerator(args, logging_dir):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    return accelerator


def configure_optimizer(args, trainable_layers, transformer, accelerator):
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = \
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return optimizer


def adapt_transformer_controlnet(transformer, args=None, control_encoder=None, control_encoder2=None, num_layers=-1, _return_trainable=False, weight_dtype=torch.float32, use_attention_embeddings=False):
    # convert vanilla Transformer model to one that also takes the control signal and has a controlnet branch
    transformer = PixArtTransformer2DModelWithLayoutControlNet.adapt(transformer, control_encoder=control_encoder, control_encoder2=control_encoder2, 
                                                                     num_layers=num_layers, use_controlnet=args.use_controlnet, use_adapters=args.use_adapters, 
                                                                     use_controllora=args.use_controllora, lora_rank=args.lora_rank, 
                                                                     use_identlin=args.use_identlin, 
                                                                     use_attention_embeddings=use_attention_embeddings,
                                                                     use_masked_attention=default(lambda: not args.no_masked_attention, True),
                                                                     use_attention_mix=default(lambda: args.use_attention_mix, False),
                                                                     use_attention_mix_light=default(lambda: args.use_attention_mix_light, False))
    transformer.to(weight_dtype)
    
    if _return_trainable:
        trainable = transformer.get_trainable_parameters()
        for param in trainable:
            param.requires_grad = True
        # trainable = list(transformer.control_blocks.parameters()) 
        # list(transformer.control_encoder.parameters()) + list(transformer.zeroconvs.parameters())        
        return transformer, trainable
    else:
        return transformer


def configure_lr_scheduler(args, train_dataloader, optimizer, accelerator):
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_train_steps * accelerator.num_processes,
    )
    return lr_scheduler


def save_checkpoint(args, accelerator, transformer, global_step):
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, "
                f"removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)

    # unwrapped_transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)

    # unwrapped_transformer.save_pretrained(
    #     save_directory=save_path,
    #     safe_serialization=True,
    # )

    logger.info(f"Saved state to {save_path}")
    
    
def save_dit(args, accelerator, transformer, global_step):
    return _save_dit(transformer, accelerator, args, global_step)


def save_dit_path(transformer, path):
    return _save_dit(transformer, savedir=path)
    
    
def _save_dit(transformer, accelerator=None, args=None, global_step=None, savedir=None):
    if savedir is None:
        save_directory = Path(args.output_dir) / f"saved-{global_step}"
        save_directory.mkdir(parents=True, exist_ok=True)
    else:
        save_directory = savedir
        save_directory.mkdir(parents=True, exist_ok=True)
    
    if accelerator is not None:
        transformer = accelerator.unwrap_model(transformer)
        
    # config = deepcopy(transformer.config)
    # config["num_control_layers"] = len([block for block in transformer.control_blocks if block is not None])
    # with open(save_directory / "configc.json", "w") as f:
    #     json.dump(config, f, indent=4)
    transformer.save_config(save_directory)
    retdict = transformer.get_control_state_dict() if hasattr(transformer, "get_control_state_dict") else transformer.state_dict()
    # retdict = {k: v for k, v in statedict.items() \
    #     if k.startswith("control_blocks") or k.startswith("control_encoder") or k.startswith("zeroconvs")}
    torch.save(retdict, save_directory / "controlnet_weights.pt")


def compareModelWeights(model_a, model_b):
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    
    if set(sd_a.keys()) != set(sd_b.keys()):
        return False
    
    for k in sd_a.keys():
        if not torch.all(sd_a[k] == sd_b[k].to(sd_a[k].device)):
            return False
        
    return True


class Validator:
    
    def __init__(self, transformer=None, vae=None, text_encoder=None, accelerator=None, examples=None, weight_dtype=torch.float32, preprocess_example=None):
        self.init_pipeline(transformer=transformer, vae=vae, text_encoder=text_encoder, accelerator=accelerator, preprocess_example=preprocess_example)
        self.weight_dtype = weight_dtype
        self.init_data()
        self.examples = examples
    
    def init_pipeline(self, transformer=None, vae=None, text_encoder=None, accelerator=None, preprocess_example=None):
        self.pipeline = PixArtSigmaLayoutControlNetPipeline.from_pretrained(ROOTNAME)
        self.pipeline.vae = vae
        self.pipeline.text_encoder = text_encoder
        self.pipeline.transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False) if transformer is not None else None
        self.accelerator = accelerator
        self.pipeline.to(accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.set_preprocess_example(preprocess_example)
        torch.cuda.empty_cache()
        
    def init_data(self):        # get required number of examples
        pass
    
    def generate(self, transformer=None, seed=None, device=None, global_step=None):
        # logger.info(
        #     f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        #     f" {args.validation_prompt}."
        # )
        # create pipeline
        # pipeline = DiffusionPipeline.from_pretrained(
        #     ROOTNAME,
        #     transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
        #     text_encoder=text_encoder, vae=vae,
        #     torch_dtype=weight_dtype,
        # )
        # pipeline = pipeline.to(accelerator.device)
        # pipeline.set_progress_bar_config(disable=True)
        
        device = self.accelerator.device if device is None else device
        
        if transformer is not None:     # override transformer
            self.pipeline.transformer = self.accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
            
        self.pipeline.to(device)

        # run inference
        generator = torch.Generator(device=device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        images = []
        cond_images = []
        captions = []
        for example in self.examples:
            images.append(
                self.pipeline(prompt=example["captions"][0], obj_prompts=example["seg_captions"], obj_masks=example["cond_image"][None].to(device, dtype=self.weight_dtype), num_inference_steps=20, generator=generator, height=512, width=512).images[0])
            cond_images.append(masktensor_to_colorimage(example["cond_image"], bw=True))
            captions.append(example["captions"][0] + "// " + ". ".join([str(ex) for ex in example["seg_captions"] if ex is not None]))

        self.pipeline.transformer = None
        
        # make grid and store in trackers
        # images = [transforms.ToPILImage()(img) for img in images]
        cond_images = [transforms.ToPILImage()(img) for img in cond_images]
        grid = make_image_grid(cond_images + images, 2, len(images))
        
        text = "\n".join(captions)
        
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                # np_images = np.stack([np.asarray(img) for img in images])
                np_image = np.asarray(grid)
                tracker.writer.add_images("validation", np_image, global_step, dataformats="HWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(grid, caption=text)
                        ]
                    }
                )
        
        torch.cuda.empty_cache()
        
        return images


def main(args):
    validate_args(args)
    
    argdumppath = Path(args.output_dir)
    argdumppath.mkdir(parents=True, exist_ok=True)
    
    with open(argdumppath / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = setup_accelerator(args, logging_dir)
    if args.startfrom is not None:
        transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype = load_pretrained(args.startfrom, device=accelerator.device, return_pipe=False)
    else:   
        transformer, tokenizer, text_encoder, noise_scheduler, vae, weight_dtype = load_model(args, accelerator.mixed_precision, accelerator.device)
    control_encoder, control_encoder2 = load_control_encoders(args, accelerator.device, weight_dtype)
    transformer, trainable_layers = adapt_transformer_controlnet(transformer, args=args, control_encoder=control_encoder, control_encoder2=control_encoder2, num_layers=args.num_control_layers, _return_trainable=True, weight_dtype=weight_dtype, use_attention_embeddings=args.use_attention_embeddings)
    optimizer = configure_optimizer(args, trainable_layers, transformer, accelerator)
    train_dataset, train_dataloader = load_data(args, tokenizer, split="train")
    lr_scheduler = configure_lr_scheduler(args, train_dataloader, optimizer, accelerator)


    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("pixart-layoutcontrolnet2-coco-v2", config=vars(args))

    print("use identlin: ", args.use_identlin)


    # TRAIN LOOP !
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"\n Resuming from checkpoint {path} \n")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.num_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    train_dl_iter = iter(train_dataloader)
    
    validation_examples = []
    while len(validation_examples) < args.num_validation_images:
        i = random.randint(0, len(train_dataset))
        validation_example = train_dataset[i]
        validation_examples.append(validation_example)
        
    unwrapped_transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
    validator = Validator(transformer=unwrapped_transformer, vae=vae, text_encoder=text_encoder, accelerator=accelerator, examples=validation_examples, weight_dtype=weight_dtype, 
                          preprocess_example=partial(preprocess_example, proportion_empty_prompts=args.proportion_empty_prompts, max_length=args.max_token_length, omit_global_caption=args.omit_global_caption))
    
    last_val_global_step = -1
    
    if True:
        save_dit_path(unwrapped_transformer, Path(args.output_dir))
        # reloaded = load_pretrained(Path(args.output_dir))
    
    # for i in range(global_step, args.num_train_steps):
    while True:
        transformer.train()
        train_loss = 0.0
        
        # get batch
        try:
            batch = next(train_dl_iter)
        except StopIteration as e:
            train_dl_iter = iter(train_dataloader)
            batch = next(train_dl_iter)
        
        # do step
        with accelerator.accumulate(transformer):
            # Convert images to latent space
            latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            control_image = batch["cond_image"].to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            prompt_embeds, attention_mask, layer_ids = encode_text(batch, text_encoder)
            # text_encoder(batch["input_ids"], attention_mask=batch['prompt_attention_mask'])[0]
            # prompt_attention_mask = batch['prompt_attention_mask']
            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Prepare micro-conditions.
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if getattr(transformer, 'module', transformer).config.sample_size == 128 and args.micro_conditions:
                resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
                aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
                resolution = resolution.to(dtype=weight_dtype, device=latents.device)
                aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=latents.device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            # Predict the noise residual and compute loss
            model_pred = transformer(noisy_latents,
                                     control_image=control_image,
                                        encoder_hidden_states=(prompt_embeds, layer_ids),
                                        encoder_attention_mask=attention_mask,
                                        timestep=timesteps,
                                        added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = \
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = trainable_layers
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        save_checkpoint(args, accelerator, transformer, global_step)
                        
                if global_step % args.save_every == 0:
                    if accelerator.is_main_process:
                        save_dit(args, accelerator, transformer, global_step)
                

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.num_train_steps:
                print("global step reached args.num_train_steps")
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and global_step % args.validate_every == 0 and global_step != last_val_global_step:
                
                images = validator.generate(transformer, seed=args.seed, global_step=global_step)
                
                last_val_global_step = global_step

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        save_dit_path(transformer, savedir=Path(args.output_dir) / "dit-final")
        # lora_state_dict = get_peft_model_state_dict(transformer)
        # StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "transformer_lora"), lora_state_dict)

    # Final inference
    del transformer
    torch.cuda.empty_cache()
    # Load previous transformer
    pipe = load_pretrained(Path(args.output_dir), return_pipe=True)
    pipeline = pipeline.to(accelerator.device)


    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=20, generator=generator).images[0])

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, global_step, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()



def mainfire_controllora(
        train_data_dir="/USERSPACE/lukovdg1/coco2017",
        output_dir="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_layoutcontrolnet2b",
        # resume_from_checkpoint="latest",
        pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        use_controllora=True,
        lora_rank=128,
        # use_controlnet=True,
        control_encoder="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_controlnet/encoder.pth",
        # num_control_layers=14,
        # use_adapters=True,
        # control_encoder2="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_simpleadapter/encoder.pth",
        validate_every=250,
        learning_rate=1e-5,
        max_grad_norm=1.,
        num_train_steps=50000,
        checkpointing_steps=250,
        save_every=5000,
        # mixed_precision="fp16",
        train_batch_size=4,
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        num_validation_images=6,
        seed=42,
        **kwargs,
    ):
        fargs = locals().copy()
        
        dels = ["kwargs", "ModuleType", "_python_view_image_mod"]
        for d in dels:
            if d in fargs:
                del fargs[d]
                
        print("kw", kwargs)
        
        actualargs = []
        for k, v in fargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
        
        for k, v in kwargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
                
        args = parse_args(actualargs)
            
        main(args)
        


def mainfire_simpleadapt_controllora(
        train_data_dir="/USERSPACE/lukovdg1/coco2017",
        # use_identlin=True,
        startfrom="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_simpleadapters",
        output_dir="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_simpleadapters+controllora",
        # resume_from_checkpoint="latest",
        pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        use_controllora=True,
        lora_rank=128,
        # use_controlnet=True,
        control_encoder="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_controlnet/encoder.pth",
        # num_control_layers=14,
        # use_adapters=True,
        # control_encoder2="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_simpleadapter/encoder.pth",
        validate_every=250,
        learning_rate=1e-5,
        max_grad_norm=1.,
        num_train_steps=10000,
        checkpointing_steps=250,
        save_every=5000,
        # mixed_precision="fp16",
        train_batch_size=4,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        num_validation_images=6,
        seed=42,
        **kwargs,
    ):
        fargs = locals().copy()
        
        dels = ["kwargs", "ModuleType", "_python_view_image_mod"]
        for d in dels:
            if d in fargs:
                del fargs[d]
                
        print("kw", kwargs)
        
        actualargs = []
        for k, v in fargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
        
        for k, v in kwargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
                
        args = parse_args(actualargs)
            
        main(args)
        
        
def mainfire_controlnet(
        train_data_dir="/USERSPACE/lukovdg1/coco2017",
        # use_identlin=True,
        output_dir="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_layoutcontrolnet2_attnctrl_noemb_noglobal",
        # resume_from_checkpoint="latest",
        pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        use_controlnet=True,
        omit_global_caption=True,
        control_encoder="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_controlnet/encoder.pth",
        num_control_layers=14,
        # use_adapters=True,
        # control_encoder2="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_simpleadapter/encoder.pth",
        validate_every=250,
        learning_rate=1e-5,
        max_grad_norm=1.,
        num_train_steps=50000,
        checkpointing_steps=250,
        save_every=5000,
        # mixed_precision="fp16",
        train_batch_size=4,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        num_validation_images=6,
        seed=42,
        **kwargs,
    ):
        fargs = locals().copy()
        
        dels = ["kwargs", "ModuleType", "_python_view_image_mod"]
        for d in dels:
            if d in fargs:
                del fargs[d]
                
        print("kw", kwargs)
        
        actualargs = []
        for k, v in fargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
        
        for k, v in kwargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
                
        args = parse_args(actualargs)
            
        main(args)


  
def mainfire_controlnet_bbox(
        train_data_dir="/USERSPACE/lukovdg1/coco2017",
        # use_identlin=True,
        use_bbox=True,
        # no_masked_attention=True,
        output_dir="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_layoutcontrolnet3_l7_bbox_attnmask",
        # resume_from_checkpoint="latest",
        pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        use_controlnet=True,
        omit_global_caption=True,
        control_encoder="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_controlnet/encoder.pth",
        num_control_layers=7,
        # use_adapters=True,
        # control_encoder2="/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_ae_output_v2_simpleadapter/encoder.pth",
        validate_every=250,
        learning_rate=1e-5,
        max_grad_norm=1.,
        num_train_steps=50000,
        checkpointing_steps=250,
        save_every=5000,
        # mixed_precision="fp16",
        train_batch_size=2,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        num_validation_images=6,
        seed=42,
        **kwargs,
    ):
        fargs = locals().copy()
        
        dels = ["kwargs", "ModuleType", "_python_view_image_mod"]
        for d in dels:
            if d in fargs:
                del fargs[d]
                
        print("kw", kwargs)
        
        actualargs = []
        for k, v in fargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
        
        for k, v in kwargs.items():
            if v is True:
                actualargs.append(f"--{k}")
            else:
                actualargs.append(f"--{k}={v}")
                
        args = parse_args(actualargs)
            
        main(args)


# DONE: Implement custom pipeline (PixArtSigmaControlNetPipeline)
# DONE: Implement validator that uses this pipeline
# DONE: Implement saving and loading of model and checkpoints

# DONE: implement simple adapters for the layers not adapted by controlnet adapters and a switch --simple_adapters

# DONE: check if current COCO Instances transform to RGB occludes masks: --> they do
# DONE: check why COCO Instances has more than one polygon per mask sometimes: --> for second part of object
# DONE: write dataloading for COCO Instances that doesn't transform them to RGB and back
# DONE: debug and start training with COCO Instances

# DONE: train simple adapters with total batch size 32 for 50k steps
# DONE: checked simple adapters
# NOTYET: try refine simple adapters with lora and control encoder 1 (basically control-lora + simple adapters)
#    [ ]: load differently pretrained model
#    [ ]: add additional components
#    [ ]: train all additional components (or just the ones added?)

# DONE: train controlnet with total batch size 32 for 50k steps
#    * appears to work already after 10k steps
#    * works pretty well after 20k-30k steps

# NOTYET: try HED controlnet to see how fast it trains 
# DONE: implement attention learning
#    [V]: use F30k entities or caption all COCO objects using LLAVA
#    [V]: write code to load those captions
#    [V]: modify model to classify how to use localized descriptions
#    [V]: train

# DONE: properly train control-lora with batch size 32 and 50k steps
# NOTYET: try simple adapters + lora

# TODO: try layoutcontrolnet with single-prompt and attention mask in the control layers in addition to object layer embeddings

# DONE: trained layoutcontrolnet with single-prompt and attention mask only (no global, no obj layer embeds)  --> train_scripts/control_experiments_v2/pixart_coco_layoutcontrolnet2_attnctrl_noemb_noglobal
# DONE: check if layoutcontrolnet learned to take into account prompts when shapes are ambiguous --> yes!

# DONE: train layoutcontrolnet with bounding boxes --> works but image quality still bad

# TODO: investigate why image quality is bad:
#           MaskControlNet (without text control) produces better quality.
#           

# DONE: train controlnet for bboxes only (no text) to see if bbox conditioning is to blame for poor image quality
#  --> image quality doesn't look much better than when using text conditioning, but maybe it is slightly better
# INFO: try layoutcontrolnet with single-prompt and object layer embeddings (? how to better inform model about which text corresponds to which regions?)  
#    thought: if local specificity of CA is varied across heads and layer but is not content-dependent, then separating into two attention mechanisms with a learnable mixture should work

# DONE: try layoutcontrolnet with attention mix --> TODO: maybe we also need attention mix in backbone?
# TODO: finish training bbox layoutcontrolnet with attention mix
# DONE: check bbox variants with three balls example: bbox + attention-mix gave worse image quality (boring) than bbox only

# BBOX:
# TODO: train bbox + attention-mask with 7 layers
# TODO: try light attention-mix

# BBOX/MASK
# TODO: implement bbox/mask + attention-mask
# TODO: train bbox/mask + attention-mask (with 7 layers?)

# TODO: read MIGC++, DiffX, MS-Diffusion (citations of InstanceDiffusion)


if __name__ == "__main__":
    fire.Fire(mainfire_controlnet_bbox)
    # fire.Fire(mainfire_controllora)