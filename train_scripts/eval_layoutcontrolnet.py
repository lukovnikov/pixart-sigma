from argparse import Namespace
import torch
from train_pixart_layoutcontrolnet_hf2 import load_pretrained, load_data, masktensor_to_colorimage, make_image_grid
import json
from pathlib import Path
from torchvision import transforms
import fire

DEFAULTPATH = "/USERSPACE/lukovdg1/pixart-sigma/train_scripts/control_experiments_v2/pixart_coco_layoutcontrolnet2_attnctrl_noemb_noglobal"
DEFAULTDEVICE = 2

def main(expdir=DEFAULTPATH,
         device=DEFAULTDEVICE):
    pipe, tokenizer, _, args = load_pretrained(Path(expdir), device=torch.device("cuda", device))
    ds, dl = load_data(args, tokenizer, split="val")
    pipe.to(pipe.device)
    pipe.transformer.device

    seed = 42
    maxex = 5
    device = pipe.device


    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    images = []
    cond_images = []
    captions = []


    i = 0
    while i < maxex:
        example = ds[i]
        images.append(
            # pipe(example["captions"][0], control_image=example["cond_image"][None].to(device), 
            #      num_inference_steps=20, generator=generator, height=512, width=512).images[0])
            pipe(prompt=example["captions"][0], obj_prompts=example["seg_captions"], obj_masks=example["cond_image"][None].to(device), 
                 num_inference_steps=20, generator=generator, height=512, width=512).images[0])
        # cond_images.append(transforms.ToPILImage()(masktensor_to_colorimage(example["cond_image"])))
        sumimg = example["cond_image"].float().sum(0, keepdims=True)
        cond_images.append(transforms.ToPILImage()( sumimg / sumimg.max()))
        captions.append(example["captions"][0])
        print(i, captions[-1])
        display(make_image_grid([cond_images[-1], images[-1]], 1, 2))
        i += 1
        if i >= maxex:
            break
if __name__ == "__main__":
    fire.Fire(main)

