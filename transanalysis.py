from typing import OrderedDict
from diffusers import AutoencoderKL
import torch
import numpy as np
from PIL import Image
from itertools import combinations
import pickle
import os 
import fire
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

from huggingface_hub import login as hf_hub_login
from huggingface_hub import hf_hub_download
import tqdm
import json
import rsatoolbox

#you need to log into huggingface to run this code. But please use your own credentials :P


def vae_downloader(model_id, vae_description):
    '''loads all the vaes from huggingface. Does not require the entire model'''
    vae_config_file, vae_file_path, safetensors = vae_description
    hf_hub_download(repo_id=model_id, filename=vae_config_file)
    fileending = ".safetensors" if safetensors else ".bin"
    hf_hub_download(repo_id=model_id, filename="vae/"+vae_file_path+fileending)
    if safetensors:
        return AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors=True)
    else:
        return AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_safetensors = False)
    

def encode_image_old(image, vae, device):
    ''' Convert an image to latent space'''
    vae.to(device)
    image = image.convert("RGB").resize((512, 512))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    image_tensor = image_tensor.to(device)#.half()  # Convert to half precision
    latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    return latent


def encode_image(image:Image, vae, device):
    ''' Convert an image to latent space'''
    vae.to(device)
    image_tensor = transforms.ToTensor()(image).to(device)
    latent = vae.encode(image_tensor[None]).latent_dist.mode() * vae.config.scaling_factor
    return latent


def get_image_files(path_to_folder, amount):
    '''small method to get amount many images from a specific folder'''
    jpg_files = [f for f in os.listdir(path_to_folder) if f.endswith(".jpg")]
    return jpg_files[:amount]



# fantastic 4 channel vaes and where to find them
# huggingface path or name      vae config      weightsfile         is it a safetensors file (true) or bin  
architectures_vaes_meta = { 'stabilityai/stable-diffusion-2-1-base': ("vae/config.json", "diffusion_pytorch_model", True),
                            'stabilityai/stable-diffusion-2-1': ("vae/config.json", "diffusion_pytorch_model", True),
                            'Mitsua/mitsua-diffusion-one':("vae/config.json", "diffusion_pytorch_model", False),
                            'prompthero/openjourney': ("vae/config.json", "diffusion_pytorch_model", True),
                            'Fictiverse/Stable_Diffusion_Microscopic_model': ("vae/config.json", "diffusion_pytorch_model", False),
                            'hakurei/waifu-diffusion': ("vae/config.json", "diffusion_pytorch_model", True),
                            'stabilityai/stable-diffusion-xl-base-1.0': ("vae/config.json", "diffusion_pytorch_model", True),
                            # Common-Canvas-S-C / Mosaic ML
                            'common-canvas/CommonCanvas-S-C': ("vae/config.json", "diffusion_pytorch_model", True),
                            # SD 3
                            'stabilityai/stable-diffusion-3-medium-diffusers': ("vae/config.json","diffusion_pytorch_model",True),
                            # PixArt
                            # Flux
                            'black-forest-labs/FLUX.1-dev': ("vae/config.json","diffusion_pytorch_model",True),
                            # PixArt
                            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS":  ("vae/config.json", "diffusion_pytorch_model", True),
                            # SD1.5
                            "runwayml/stable-diffusion-v1-5": ("vae/config.json", "diffusion_pytorch_model", True)
                }

short_to_full = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": 'stabilityai/stable-diffusion-2-1-base',
    "waifu": 'hakurei/waifu-diffusion',
    "cc": 'common-canvas/CommonCanvas-S-C',
    "mitsua": 'Mitsua/mitsua-diffusion-one',
    "sdxl": 'stabilityai/stable-diffusion-xl-base-1.0',
    "pixart": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "sd3": 'stabilityai/stable-diffusion-3-medium-diffusers',
    "flux1": 'black-forest-labs/FLUX.1-dev',
}


def compute_latents(device = "cuda", 
         whichmodels="sd15,sd21,waifu,cc,mitsua,sdxl,pixart",
         numimg = 100,
        #  include16ch = False, 
         imgpath = "/USERSPACE/lukovdg1/coco2017/val2017",
         savepath = "transferanalysis/vaefunsim/", #it safes the comparison to a dict with pickles at the specific path
    ):
    args = locals().copy()
    imgpath = Path(imgpath)
    savepath = Path(savepath)
    with open(savepath / "args.json", "w") as f:
        json.dump(args, f, indent=4)
    
    whichmodels = whichmodels.split(",")
    whichmodels = [short_to_full[name] for name in whichmodels]
    #this flag states whether to actually load the 2 models using 16 channel latents. Normally I don't as I dont need the results.
    # be careful, evaluation is not fully compatible.

    # select image paths
    usedimagepaths = []
    usedimages = []
    for imgp in imgpath.glob("*.jpg"):
        image = Image.open(imgpath / imgp).convert("RGB")
        image = transforms.CenterCrop((512, 512))(image)
        if all([ss >= 512 for ss in image.size]):
            usedimagepaths.append(imgp)
            usedimages.append(image)
        if len(usedimagepaths) >= numimg:
            break
        
    with open(savepath / "usedimagepaths.json", "w") as f:
        json.dump([str(x) for x in usedimagepaths], f, indent=4)
        
    latentmatrix = OrderedDict()
    
    #load all vaes
    for k in whichmodels:
    # for k, v in architectures_vaes_meta.items():
        v = architectures_vaes_meta[k]
        print(f"loading {k}")
        
        # is16ch = k in ['stabilityai/stable-diffusion-3-medium-diffusers', 'black-forest-labs/FLUX.1-dev']
        # if is16ch and not include16ch:
        #     continue
        
        # load vae
        vae = vae_downloader(k, v)
        
        latentmatrix[k] = []
            
        #for every image
        for imgp, image in tqdm.tqdm(zip(usedimagepaths, usedimages)): 
            # print(imgp, image.size) #to keep track of how long it actually takes

            #generate all latents
            with torch.no_grad():
                latentimage = encode_image(image, vae, device)
            latentmatrix[k].append(latentimage)
            
        latentmatrix[k] = torch.cat(latentmatrix[k], 0)     
    
    with open(savepath / "latentmatrix.pkl", "wb") as f:
        pickle.dump(latentmatrix, f)
        

def compute_funsim(
         path = "transferanalysis/vaefunsim/", #it safes the comparison to a dict with pickles at the specific path
         t = 0.1, #depending on the comparision, one needs a threshold.
    ):
    path = Path(path)
    with open(path / "latentmatrix.pkl", "rb") as f:
        latentmatrix = pickle.load(f)
        
    arch_names = [k for k in latentmatrix]

    funsim_allclose_matrix = np.zeros((len(latentmatrix), len(latentmatrix)))
    funsim_cosine_matrix = np.zeros((len(latentmatrix), len(latentmatrix)))
    for i, fromm in enumerate(arch_names):
        for j, to in enumerate(arch_names):
            from_images, to_images = latentmatrix[fromm], latentmatrix[to]
            from_images, to_images = torch.flatten(from_images, 1, -1), torch.flatten(to_images, 1, -1)
            # if method == "allclose":
            isclose = torch.isclose(from_images, to_images, rtol=t, atol=t)
            allclose = torch.all(isclose, -1)
            avgallclose = torch.mean(allclose.float())
            cosinesims = torch.cosine_similarity(from_images, to_images, -1)
            stdcosinesim, avgcosinesim = torch.std_mean(cosinesims)
            
            funsim_allclose_matrix[i, j] = avgallclose.item()
            funsim_cosine_matrix[i, j] = avgcosinesim.item()
            
    print(arch_names)
    print(f"Cosine sim matrix:\n {funsim_cosine_matrix}")

    np.savetxt(path / "funsim_allclose.csv", funsim_allclose_matrix, delimiter=",", fmt="%.3f")
    np.savetxt(path / "funsim_cosine.csv", funsim_cosine_matrix, delimiter=",", fmt="%.3f")


    # # compariosn 16 channel
    # if include16ch:
    #     results = []
    #     for imgp in path_to_image:
    #         # imgp = path_to_image[0]

    #         image = Image.open(imgpath+imgp)

    #         latents = {}
    #         for k, v in architectures_vaes_16.items():
    #             print(k)
    #             latents[k] = encode_image(image, architectures_vaes_16[k])

    #         for a, b in combinations(architectures_vaes_16.keys(), 2):
    #             close = torch.allclose(latents[a], latents[b], rtol=t, atol=t)
    #             results.append([a, b, imgp, close])

    #     with open(savepath+"comparison_sd3.pkl", "wb") as f1:
    #         pickle.dump(results, f1)
    
    
def compute_rsa(
         path = "transferanalysis/vaefunsim/", #it safes the comparison to a dict with pickles at the specific path
    ):  
    path = Path(path)
    with open(path / "latentmatrix.pkl", "rb") as f:
        latentmatrix = pickle.load(f)
        
    arch_names = [k for k in latentmatrix]
    


if __name__ == "__main__":
    # fire.Fire(compute_latents)
    # fire.Fire(compute_funsim)
    fire.Fire(compute_rsa)
