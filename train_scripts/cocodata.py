from copy import copy, deepcopy
import math
from PIL import Image, ImageDraw
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import tqdm
from transformers import CLIPTokenizer
from torchvision.transforms.functional import to_tensor, to_pil_image
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import colorsys
# from einops import rearrange, repeat
import re


def _tokenize_annotated_prompt(prompt, tokenizer, minimize_length=False):
    prompt = re.split(r"(\{[^\}]+\})", prompt)
    _prompt = []
    _layer_id = []
    for e in prompt:
        m = re.match(r"\{(.+):(\d+)\}", e)
        if m:
            _prompt.append(m.group(1))
            _layer_id.append(int(m.group(2)) + 1)
        else:
            _prompt.append(e)
            _layer_id.append(0)

    for i in range(len(_prompt)):
        if i == len(_prompt) - 1:
            tokenized = tokenizer([_prompt[i]],
                                  padding="max_length" if not minimize_length else "do_not_pad",
                                  max_length=tokenizer.model_max_length,
                                  return_overflowing_tokens=False,
                                  truncation=True,
                                  return_tensors="pt")
        else:
            tokenized = tokenizer([_prompt[i]], return_tensors="pt")
        _prompt[i] = tokenized.input_ids[0, (0 if i == 0 else 1):(-1 if i < len(_prompt) - 1 else None)]
        _layer_id[i] = torch.tensor([_layer_id[i]]).repeat(len(_prompt[i]))

    token_ids = torch.cat(_prompt, 0)
    token_ids = token_ids[:min(len(token_ids), tokenizer.model_max_length)]
    layer_ids = torch.cat(_layer_id, 0)
    layer_ids = layer_ids[:min(len(layer_ids), tokenizer.model_max_length)]

    assert len(token_ids) <= tokenizer.model_max_length
    return token_ids, layer_ids
      
        
def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return torch.nn.functional.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()
    
    
def colorgen(num_colors=100):
    for i in range(1, num_colors):
        r = (i * 53) % 256  # Adjust the prime number for different color patterns
        g = (i * 97) % 256
        b = (i * 163) % 256
        yield [r/256, g/256, b/256]
        

def colorgen_hsv(numhues=36):
    hue = random.randint(0, 360)
    usehues = set()
    huestep = round(360/numhues)
    retries = 0
    while True:
        sat = random.uniform(0.5, 0.9)
        val = random.uniform(0.3, 0.7)
        yield colorsys.hsv_to_rgb(hue/360, sat, val)
        usehues.add(hue)
        # change hue 
        while hue in usehues:
            hue = (hue + huestep * random.randint(0, int(360/huestep))) % 360
            retries += 1
            if retries > numhues:
                usehues = set()
                retries = 0
                continue
            
            
def masktensor_to_colorimage(x, bw=False):
    """ Takes a mask tensor (C x H x W) where C can be > 3 and returns an RGB image tensor (3 x H x W) with random colors assigned to different masks"""
    if not bw:
        randomcolors = torch.tensor([randomcolor_hsv() for _ in range(x.size(0))])     # (C x 3)
        indexes = x.max(0)[1]
        colorimg = randomcolors[indexes].permute(2, 0, 1)
    else:
        colorimg = x.sum(0, keepdim=True)
        colorimg = colorimg / colorimg.max()
    return colorimg
            
            
def randomcolor_hsv():
    hue = random.uniform(0, 360)
    sat = random.uniform(0.4, 0.9)
    val = random.uniform(0.2, 0.8)
    return colorsys.hsv_to_rgb(hue/360, sat, val)


predef_hues = list(np.linspace(0, 360, 50))
predef_vals = list(np.linspace(0, 1, 50))
def randomcolor_predef():
    hue = random.choice(predef_hues)
    val = random.choice(predef_vals)
    sat = 0.75
    return colorsys.hsv_to_rgb(hue/360, sat, val)
    
        
class COCOPanopticExample(object):
    def __init__(self, id=None, img=None, captions=None, seg_img=None, seg_info=None, cropsize=None) -> None:
        super().__init__()
        self.id = id
        self.image_path, self.image_data = None, None
        if isinstance(img, (str, Path)):
            self.image_path = img
        else:
            assert isinstance(img, (Image.Image,))
            self.image_data = img
        assert self.image_data is None or self.image_path is None       # provide either path or data
        self.seg_path, self.seg_data = None, None
        if isinstance(seg_img, (str, Path)):
            self.seg_path = seg_img
        else:
            assert isinstance(seg_img, (Image.Image,))
            self.seg_data = img
        assert self.seg_data is None or self.seg_path is None       # provide either path or data
        
        self.captions = captions
        self.seg_info = seg_info
        self.cropsize = cropsize
        
    def load_image(self):
        if self.image_path is not None:
            img = Image.open(self.image_path).convert("RGB")
        else:
            img = self.image_data
        return img
    
    def load_seg_image(self):
        if self.seg_path is not None:
            img = Image.open(self.seg_path).convert("RGB")
        else:
            img = self.seg_data
        return img        
        
        
class COCOPanopticDataset(Dataset):
    padlimit=1 #5
    min_region_area = 16*16 #-1 # 0.002
    
    def __init__(self, maindir:str=None, split="valid", max_masks=20, min_masks=2, max_samples=None, min_size=350, upscale_to=None,
                 examples=None, mergeregions=False,  # mergeregions?
                 regiondrop=False,           # if False, dropping examples with too many masks, if True: keeping all examples and dropping randomly some masks, if float: acts like True, but also drops some masks with the given number as drop probability
                 casmode=None, simpleencode=False, limitpadding=False,
                 tokenizer="openai/clip-vit-large-patch14",
                 useinstances=False,
                 usescribbles=False, usecanny=False):
        super().__init__()
        assert examples is None or maindir is None      # provide either a directory or a list of already made examples
        self.maindir = maindir
        self.n = 0
        self.load_tokenizer(tokenizer)
        
        self.casmode = casmode
        self.useinstances = useinstances
        
        self.usescribbles = usescribbles
        self.usecanny = usecanny
        
        self.simpleencode = simpleencode
        self.mergeregions = mergeregions
        self.limitpadding = limitpadding
        
        self.max_masks = max_masks
        self.min_masks = min_masks
        self.min_size = min_size
        self.upscale_to = upscale_to
        self.regiondrop = regiondrop if regiondrop != -1. else False
            
        sizestats = {}
        examplespersize = {}
        numtoofewregions = 0
        numtoomanyregions = 0
        numtoosmall = 0
        
        numexamples = 0
        
        if examples is None:        
            
            if split.startswith("v"):
                which = "val"
            elif split.startswith("tr"):
                which = "train"
                
            self.img_dir = Path(self.maindir) / f"{which}2017"
            captionsfile = Path(self.maindir) / "annotations" / f"captions_{which}2017.json"
            
            if self.useinstances:
                instancesfile = Path(self.maindir) / "annotations" / f"inst_{which}2017.json"
                self.panoptic_dir = Path(self.maindir) / "annotations" / f"instances_{which}2017"
            else:
                panopticsfile = Path(self.maindir) / "annotations" / f"panoptic_{which}2017.json"
                self.panoptic_dir = Path(self.maindir) / "annotations" / f"panoptic_{which}2017"
            
            print("loading captions")     
            image_db, captiondb = self.load_captions(captionsfile, img_dir=self.img_dir)        # creates image db and caption db
            print("loading panoptics")
            
            if self.useinstances:
                _, panoptic_db = self.load_instances(instancesfile, panoptic_dir=self.panoptic_dir)      # creates category db and panoptic db
            else:
                _, panoptic_db = self.load_panoptics(panopticsfile, panoptic_dir=self.panoptic_dir)      # creates category db and panoptic db
            
            example_ids = list(image_db.keys())
            
            # filter examples
            print("Creating examples")
            for example_id in tqdm.tqdm(example_ids):
                # captions = [self.tokenize([caption]) for caption in captions]
                frame_size = (image_db[example_id]["width"], image_db[example_id]["height"])
                cropsize = min((min(frame_size) // 64) * 64, 512)
                if cropsize < self.min_size:
                    numtoosmall += 1
                    continue
                
                if cropsize not in sizestats:
                    sizestats[cropsize] = 0
                sizestats[cropsize] += 1
                    
                numregions = len(panoptic_db[example_id]["segments_info"])
                if self.mergeregions:
                    uniqueregioncaptions = set()
                    for _, region in panoptic_db[example_id]["segments_info"].items():
                        uniqueregioncaptions.add(region["caption"])
                    numregions = len(uniqueregioncaptions)
                    
                if numregions > max_masks and self.regiondrop is False:
                    numtoomanyregions += 1
                    continue
                if numregions < min_masks:
                    numtoofewregions += 1
                    continue
                
                if cropsize not in examplespersize:
                    examplespersize[cropsize] = []
                    
                example = COCOPanopticExample(id=example_id, 
                                                img=image_db[example_id]["path"],
                                                seg_img=panoptic_db[example_id]["segments_map"],
                                                seg_info=panoptic_db[example_id]["segments_info"],
                                                captions=captiondb[example_id],
                                                cropsize=cropsize,
                                                )
                examplespersize[cropsize].append(example)
                
                numexamples += 1
                if max_samples is not None and numexamples >= max_samples:
                    break
                
        else:
            print("loading provided examples. maindir and split arguments are ignored.")
            for example in examples:
                cropsize = example.cropsize
                
                if cropsize < self.min_size:
                    continue
                
                if cropsize not in sizestats:
                    sizestats[cropsize] = 0
                sizestats[cropsize] += 1
                
                numregions = len(example.seg_info)
                if numregions > max_masks:
                    numtoomanyregions += 1
                    continue
                if numregions < min_masks:
                    numtoofewregions += 1
                    continue
                
                if cropsize not in examplespersize:
                    examplespersize[cropsize] = []
                    
                examplespersize[cropsize].append(example)
                
                numexamples += 1
                if max_samples is not None and numexamples >= max_samples:
                    break
               
            # self.examples.append(ProcessedCOCOExample(image_path, captions, regions, cropsize=cropsize))     
        
        # self.examples = [(k, v) for k, v in examplespersize.items()]
        # self.examples = sorted(self.examples, key=lambda x: x[0])
        # self.total_n = sum([len(v) for k, v in self.examples])
        
        self.examples = [ve for k, v in examplespersize.items() for ve in v]        
            
        print("Size stats:")
        print(sizestats)
        print(f"Retained examples: {len(self)}")
        print(f"Too many regions: {numtoomanyregions}")
        print(f"Too few regions: {numtoofewregions}")
        print(f"Too small: {numtoosmall}")
        
        self.transforms = []
        
    def filter_ids(self, ids):
        newselfexamples = []
        for res, examples in self.examples:
            newexamples = []
            for example in examples:
                if example.id in ids:
                    newexamples.append(example)
            if len(newexamples) > 0:
                newselfexamples.append((res, newexamples))
        self.examples = newselfexamples
        
    def load_captions(self, captionpath, img_dir=Path("")):
        captions = json.load(open(captionpath))
        # load image db
        image_db = {}
        for imageinfo in captions["images"]:
            image_db[imageinfo["id"]] = {
                "path": img_dir / imageinfo["file_name"],
                "height": imageinfo["height"],
                "width": imageinfo["width"]
            }
        # load caption db
        captiondb = {}   # from image_id to list of captions
        for annotation in captions["annotations"]:
            imgid = annotation["image_id"]
            if imgid not in captiondb:
                captiondb[imgid] = []
            captiondb[imgid].append(annotation["caption"])
            
        return image_db, captiondb
            
    def load_panoptics(self, panopticpath, panoptic_dir=Path("")):
        # load category db
        panoptic_category_db = {}
        panopticsinfo = json.load(open(panopticpath))
        def process_category_name(name):
            if name.endswith("-merged"):
                name = name[:-len("-merged")]
            if name.endswith("-other"):
                name = name[:-len("-other")]
            if name.endswith("-stuff"):
                name = name[:-len("-stuff")]
            name = name.replace("-", " ")
            return name
        for category in panopticsinfo["categories"]:
            panoptic_category_db[category["id"]] = process_category_name(category["name"])
            
        # load panoptics annotations
        panoptic_db = {}
        for annotation in panopticsinfo["annotations"]:
            assert annotation["image_id"] not in panoptic_db
            saveann = {"segments_map": panoptic_dir / annotation["file_name"], "segments_info": {}}
            for segment in annotation["segments_info"]:
                assert segment["id"] not in saveann["segments_info"]
                saveann["segments_info"][segment["id"]] = {"category_id": segment["category_id"],
                                                           "caption": panoptic_category_db[segment["category_id"]]}
            panoptic_db[annotation["image_id"]] = saveann
            
        return panoptic_category_db, panoptic_db
    
    def load_instances(self, panopticpath, panoptic_dir=Path("")):
        # load category db
        panoptic_category_db = {}
        panopticsinfo = json.load(open(panopticpath))
        def process_category_name(name):
            if name.endswith("-merged"):
                name = name[:-len("-merged")]
            if name.endswith("-other"):
                name = name[:-len("-other")]
            if name.endswith("-stuff"):
                name = name[:-len("-stuff")]
            name = name.replace("-", " ")
            return name
        for category in panopticsinfo["categories"]:
            panoptic_category_db[category["id"]] = process_category_name(category["name"])
            
        # load panoptics annotations
        panoptic_db = {}
        for imgid, img in panopticsinfo["imgd"].items():
            assert img["id"] not in panoptic_db
            saveann = {"segments_map": panoptic_dir / (str(Path(img["path"]).stem) + ".png"), "segments_info": {}}
            for segment in img["masks"]:
                assert segment["colorid"] not in saveann["segments_info"]
                saveann["segments_info"][segment["colorid"]] = {"category_id": segment["category_id"],
                                                           "caption": panoptic_category_db[segment["category_id"]]}
            panoptic_db[img["id"]] = saveann
            
        return panoptic_category_db, panoptic_db
                
    def load_tokenizer(self, tokenizer):
        if isinstance(tokenizer, str):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def __getstate__(self):
        ret = copy(self.__dict__)
        del ret["tokenizer"]
        return ret
    
    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.load_tokenizer()
        
    def tokenize(self, x, tokenizer=None, minimize_length=True):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        tokenized = tokenizer(x,  padding="max_length" if not minimize_length else "do_not_pad",
                                  max_length=tokenizer.model_max_length,
                                  return_overflowing_tokens=False,
                                  truncation=True,
                                  return_tensors="pt")
        return tokenized.input_ids, tokenized.attention_mask
    
    def untokenize(self, x, tokenizer=None):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        ret = tokenizer.decode(x)
        return ret
        
    def __getitem__(self, item):
        example = self.examples[item]
    
        return self.materialize_example(example)
        # return example
    
    def __len__(self):
        return len(self.examples)
        #sum([len(v) for k, v in self.examples])
    
    def collate_fn(self, examples):
        # compute size stats
        sizestats = {}
        for example in examples:
            newsize = example["image"].shape[1]
            if newsize not in sizestats:
                sizestats[newsize] = 0
            sizestats[newsize] += 1
        # if sizes are different, throw away those not matching the size of majority
        if len(sizestats) > 1:
            majoritysize, majoritycount = 0, 0
            for s, sc in sizestats.items():
                if sc >= majoritycount:
                    if s > majoritysize:
                        majoritysize, majoritycount = s, sc
                        
            examples = [example for example in examples if example["image"].shape[1] == majoritysize]
        
        # every example is dictionary like specified above
        
        images = []
        cond_images = []
        captions = []
        regionmasks = []
        layerids = []
        encoder_layerids = []
        # regioncounts = []
        image_paths = []
        seg_paths = []
        
        for example in examples:
            images.append(example["image"])   # concat images
            cond_images.append(example["cond_image"])
            
            image_paths.append(example["image_path"])
            seg_paths.append(example["seg_path"])
            
            captions.append(torch.cat(example["captions"], 0))   # batchify captions
            # regioncounts.append(len(example["captions"]))  # keep track of the number of regions per example
            layerids.append(torch.cat(example["layerids"], 0))   # layer ids
            encoder_layerids.append(torch.cat(example["encoder_layerids"], 0))   # layer ids
            materialized_masks = {res: masks[layerids[-1]] for res, masks in example["regionmasks"].items()}
            
            regionmasks.append(materialized_masks)
            
        imagebatch = torch.stack(images, dim=0)
        cond_imagebatch = torch.stack(cond_images, dim=0)
        captionbatch = pad_sequence(captions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        layeridsbatch = pad_sequence(layerids, batch_first=True, padding_value=-1)
        encoder_layeridsbatch = pad_sequence(encoder_layerids, batch_first=True, padding_value=-1)
        captiontypes = [(layerids_i > 0).long() for layerids_i in encoder_layerids]
        captiontypes = pad_sequence(captiontypes, batch_first=True, padding_value=-2)
        captiontypes += 1
        captiontypes[:, 0] = 0
        
        batched_regionmasks = {}
        for regionmask in regionmasks:
            for res, rm in regionmask.items():
                if res not in batched_regionmasks:
                    batched_regionmasks[res] = []
                batched_regionmasks[res].append(rm)
        batched_regionmasks = {res: pad_sequence(v, batch_first=True) for res, v in batched_regionmasks.items()}
        
        # DONE: stack regionmasks to form one tensor (batsize, seqlen, H, W) per mask resolution
        # DONE: passing layer ids: prepare a data structure for converting from current dynamically flat caption format to (batsize, seqlen, hdim)
        # DONE: return (batsize, seqlen) tensor that specifies if the token is part of global description or local description
        # DONE: provide conditioning image for ControlNet
        ret = {"image": rearrange(imagebatch, 'b c h w -> b h w c'), 
                "cond_image": rearrange(cond_imagebatch, 'b c h w -> b h w c'),
                "caption": captionbatch, 
                "layerids": layeridsbatch, 
                "encoder_layerids": encoder_layeridsbatch,
                "regionmasks": batched_regionmasks, 
                "captiontypes": captiontypes,
                "image_paths": image_paths,
                "seg_paths": seg_paths,
                }
        
        
    def materialize_example(self, example):
        def region_code_to_rgb(rcode):
            B = rcode // 256**2
            rcode = rcode % 256**2
            G = rcode // 256
            R = rcode % 256
            return (R, G, B)
            
        # materialize one example
        # 1. load image and segmentation map
        img = example.load_image()   #Image.open(self.image_db[example_id]["path"]).convert("RGB")
        seg_img = example.load_seg_image()   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
        
        if self.upscale_to is not None:
            upscalefactor = self.upscale_to / min(img.size)
            newsize = [math.ceil(s * upscalefactor) for s in img.size]
            img = img.resize(newsize, resample=Image.BILINEAR)
            upscalefactor = self.upscale_to / min(seg_img.size)
            newsize = [math.ceil(s * upscalefactor) for s in seg_img.size]
            seg_img = seg_img.resize(newsize, resample=Image.BOX)
            
        # 2. transform to tensors
        imgtensor = to_tensor(img)
        seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1)
        
        # 3. create conditioning image by randomly swapping out colors
        # cond_imgtensor = torch.ones_like(imgtensor) * torch.tensor(randomcolor_hsv())[:, None, None]
        cond_imgtensor = torch.zeros(self.max_masks + 1, imgtensor.size(1), imgtensor.size(2), dtype=torch.long, device=imgtensor.device)
        
        # 4. pick one caption at random (TODO: or generate one from regions)
        captions = [random.choice(example.captions)]
    
        # 4. load masks
        ids = list(range(0, self.max_masks+1))
        random.shuffle(ids)
        
        for i, (region_code, region_info) in enumerate(example.seg_info.items()):
            rgb = torch.tensor(region_code_to_rgb(region_code))
            region_mask = (seg_imgtensor == rgb[:, None, None]).all(0)
            maskid = ids.pop(0)
            if (region_mask > 0).sum()  < self.min_region_area:
                continue
            cond_imgtensor[maskid] = region_mask
            
        maskid = ids.pop(0)
        bgrmask = cond_imgtensor.long().sum(0) == 0
        # cond_imgtensor[maskid] = bgrmask

        # random square crop of size divisble by 64 and maximum size 512
        cropsize = min((min(imgtensor[0].shape) // 64) * 64, 512)
        crop = (random.randint(0, imgtensor.shape[1] - cropsize), 
                random.randint(0, imgtensor.shape[2] - cropsize))
        # print(cropsize)
        
        imgtensor = imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        cond_imgtensor = cond_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
            
        imgtensor = imgtensor * 2 - 1.
        
        ret = {"image": imgtensor, 
                "cond_image": cond_imgtensor,
                "captions": captions,
                "image_path": example.image_path,
                "seg_path": example.seg_path,
                # "layerids": layerids,
                # "encoder_layerids": encoder_layerids,
                # "regionmasks": downmasktensors
                }
        
        for transform in self.transforms:
            ret = transform(ret)
            
        return ret
    
    
class COCOInstancesExample(object):
    def __init__(self, id=None, img=None, captions=None, seg_info=None, imgsize=None, usecache=False) -> None:
        super().__init__()
        self.id = id
        self.image_path, self.image_data = None, None
        if isinstance(img, (str, Path)):
            self.image_path = img
        else:
            assert isinstance(img, (Image.Image,))
            self.image_data = img
        assert self.image_data is None or self.image_path is None       # provide either path or data
        
        self.captions = captions
        self.seg_info = seg_info
        self.imgsize = imgsize
        
        self.usecache = usecache
        self._img_cache = None
        self._seg_masks_cache = None
        
    def load_image(self, upscale_to=None):
        if self._img_cache is None:
            if self.image_path is not None:
                img = Image.open(self.image_path).convert("RGB")
            else:
                img = self.image_data
            if upscale_to is not None:
                upscalefactor = upscale_to / min(img.size)
                newsize = [math.ceil(s * upscalefactor) for s in img.size]
                img = img.resize(newsize, resample=Image.BILINEAR)
            if self.usecache:
                self._img_cache = img
        else:
            img = self._img_cache
        return img
    
    def load_seg_image(self, upscale_to=None):
        if self._seg_masks_cache is None:
            upscalefactor = upscale_to / min(self.imgsize) if upscale_to is not None else 1.
            newimgsize = [math.ceil(x * upscalefactor) for x in self.imgsize]
            
            masktensors = []
            
            for maskid, mask in enumerate(self.seg_info):
                maskimg = Image.new('L', newimgsize, 'black')
                maskimgdraw = ImageDraw.Draw(maskimg)
                for polygon in mask["segmentation"]:
                    coords = []
                    i = 0
                    while i < len(polygon):
                        coords.append((float(polygon[i]) * upscalefactor, float(polygon[i + 1]) * upscalefactor))
                        i += 2
                    maskimgdraw.polygon(coords, fill="white", outline="white")
                masktensors.append(to_tensor(maskimg).to(torch.bool)[0])
            
            ret = torch.stack(masktensors, 0) if len(masktensors) > 0 else torch.zeros(newimgsize, dtype=torch.bool)[None]
            if self.usecache:
                self._seg_masks_cache = ret 
        else:
            ret = self._seg_masks_cache
        return ret
    

class COCOInstancesDataset(Dataset):
    padlimit=1 #5
    min_region_area = 16*16 #-1 # 0.002
    
    def __init__(self, maindir:str=None, split="valid", max_masks=20, min_masks=2, max_samples=None, min_size=350, upscale_to=None):
        super().__init__()
        self.maindir = maindir
        self.n = 0
        
        self.max_masks = max_masks
        self.min_masks = min_masks
        self.min_size = min_size
        self.upscale_to = upscale_to
            
        sizestats = {}
        examplespersize = {}
        numtoofewregions = 0
        numtoomanyregions = 0
        numtoosmall = 0
        
        numexamples = 0
        
            
        if split.startswith("v"):
            which = "val"
        elif split.startswith("tr"):
            which = "train"
            
        self.img_dir = Path(self.maindir) / f"{which}2017"
        captionsfile = Path(self.maindir) / "annotations" / f"captions_{which}2017.json"
        instancesfile = Path(self.maindir) / "annotations" / f"instances_{which}2017.json"
        instancedescriptionsfile = Path(self.maindir) / "annotations" / f"instance_descriptions_{which}2017.json"
        
        print("loading captions")     
        image_db, caption_db = self.load_captions(captionsfile, img_dir=self.img_dir)        # creates image db and caption db
        
        print("loading instances")
        instance_db = self.load_instances(instancesfile, instancedescriptionsfile)      # creates category db and panoptic db
        
        example_ids = list(image_db.keys())
        
        # filter examples
        print("Creating examples")
        for example_id in tqdm.tqdm(example_ids):
            # captions = [self.tokenize([caption]) for caption in captions]
            frame_size = (image_db[example_id]["width"], image_db[example_id]["height"])
            cropsize = min((min(frame_size) // 64) * 64, 512)
            if cropsize < self.min_size:
                numtoosmall += 1
                continue
            
            if cropsize not in sizestats:
                sizestats[cropsize] = 0
            sizestats[cropsize] += 1
                
            numregions = len(instance_db[example_id]["masks"])
                
            if numregions > max_masks:
                numtoomanyregions += 1
                continue
            if numregions < min_masks:
                numtoofewregions += 1
                continue
            
            if cropsize not in examplespersize:
                examplespersize[cropsize] = []
                
            example = COCOInstancesExample(id=example_id, 
                                            img=image_db[example_id]["path"],
                                            seg_info=instance_db[example_id]["masks"],
                                            captions=caption_db[example_id],
                                            imgsize=instance_db[example_id]["size"],
                                            )
            examplespersize[cropsize].append(example)
            
            numexamples += 1
            if max_samples is not None and numexamples >= max_samples:
                break
                
        
        self.examples = [ve for k, v in examplespersize.items() for ve in v]        
            
        print("Size stats:")
        print(sizestats)
        print(f"Retained examples: {len(self)}")
        print(f"Too many regions: {numtoomanyregions}")
        print(f"Too few regions: {numtoofewregions}")
        print(f"Too small: {numtoosmall}")
        
        self.transforms = []
        
    def filter_ids(self, ids):
        newselfexamples = []
        for res, examples in self.examples:
            newexamples = []
            for example in examples:
                if example.id in ids:
                    newexamples.append(example)
            if len(newexamples) > 0:
                newselfexamples.append((res, newexamples))
        self.examples = newselfexamples
        
    def load_captions(self, captionpath, img_dir=Path("")):
        captions = json.load(open(captionpath))
        # load image db
        image_db = {}
        for imageinfo in captions["images"]:
            image_db[imageinfo["id"]] = {
                "path": img_dir / imageinfo["file_name"],
                "height": imageinfo["height"],
                "width": imageinfo["width"]
            }
        # load caption db
        captiondb = {}   # from image_id to list of captions
        for annotation in captions["annotations"]:
            imgid = annotation["image_id"]
            if imgid not in captiondb:
                captiondb[imgid] = []
            captiondb[imgid].append(annotation["caption"])
            
        return image_db, captiondb
            
    def load_instances(self, instancesfile, instancedescriptionsfile=None):
        # load category db
        category_db = {}
        instancesinfo = json.load(open(instancesfile))
        
        for category in instancesinfo["categories"]:
            category_db[category["id"]] = category["name"]
            
        # load panoptics annotations
        imgd = {}
        
        for image in instancesinfo["images"]:
            imgd[image["id"]] = {"path": image["file_name"], "size": (image["width"], image["height"]), "id": image["id"], "masks": []}
            
        for mask in instancesinfo["annotations"]:
            if "counts" in mask["segmentation"]:
                continue
            if mask["area"] < self.min_region_area: 
                continue
            imgd[mask["image_id"]]["masks"].append(mask)
            
        # load panoptics annotations
        segment_db = {}
        for imgid, img in imgd.items():
            for mask in img["masks"]:
                mask["category_name"] = category_db[mask["category_id"]]
                mask["caption"] = mask["category_name"]
                segment_db[mask["id"]] = mask
                
        if instancedescriptionsfile is not None and Path(instancedescriptionsfile).exists():
            # load descriptions
            instancedescriptions = json.load(open(instancedescriptionsfile))
            for k, v in instancedescriptions.items():
                segment_db[int(k)]["caption"] = v
            
        return imgd

    def __getstate__(self):
        ret = copy(self.__dict__)
        del ret["tokenizer"]
        return ret
    
    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.load_tokenizer()
 
    def __getitem__(self, item):
        example = self.examples[item]
    
        ret = self.materialize_example(example)
        return ret
    
    def __len__(self):
        return len(self.examples)
        #sum([len(v) for k, v in self.examples])
            
    def materialize_example(self, example):   
        # materialize one example
        # 1. load image and segmentation map
        img = example.load_image(upscale_to=self.upscale_to)   #Image.open(self.image_db[example_id]["path"]).convert("RGB")
        seg_imgtensor = example.load_seg_image(upscale_to=self.upscale_to)   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
            
        # 2. transform to tensors
        imgtensor = to_tensor(img)
        
        # 3. create conditioning image by randomly swapping out colors
        # cond_imgtensor = torch.ones_like(imgtensor) * torch.tensor(randomcolor_hsv())[:, None, None]
        cond_imgtensor = torch.zeros(self.max_masks + 1, imgtensor.size(1), imgtensor.size(2), device=imgtensor.device, dtype=torch.float)
        
        # 4. pick one caption at random (TODO: or generate one from regions)
        captions = [random.choice(example.captions)]
        segcaptions = [None for _ in range(self.max_masks + 1)]
    
        # 4. load masks
        ids = list(range(0, self.max_masks+1))
        random.shuffle(ids)
        
        for i, (mask) in enumerate(seg_imgtensor.unbind(0)):
            maskid = ids.pop(0)
            mask = deepcopy(mask)
            cond_imgtensor[maskid] = mask
            segcaptions[maskid] = example.seg_info[i]["caption"]
            
        # cond_imgtensor[0] = torch.where(cond_imgtensor.sum(1) > 0, torch.zeros_like(cond_imgtensor[0]), torch.ones_like(cond_imgtensor[0]))
            
        if False:
            maskid = ids.pop(0)
            bgrmask = cond_imgtensor.long().sum(0) == 0
            cond_imgtensor[maskid] = bgrmask

        imgtensor = imgtensor * 2 - 1.
        
        ret = { "image": imgtensor, 
                "cond_image": cond_imgtensor,
                "captions": captions,
                "seg_captions": segcaptions,
                "image_path": example.image_path,
                }
        
        for transform in self.transforms:
            ret = transform(ret)
            
        return ret
    
    

# !!! IDEA: only use the higher-res images for training later denoising steps
    
def collate_fn(listofdicts):
    ret = {}
    for k in listofdicts[0]:
        ret[k] = []
        
    for d in listofdicts:
        assert set(d.keys()) == set(ret.keys())
        for k, v in d.items():
            ret[k].append(v)
    
    for k in ret:
        if isinstance(ret[k][0], torch.Tensor):
            ret[k] = torch.stack(ret[k], 0)
            
    return ret

    
def main(x=0):
    import pickle
    cocodataset = COCOInstancesDataset(maindir="/USERSPACE/lukovdg1/coco2017", split="v", upscale_to=512)
    
    # example = cocodataset[0]
    
    # for i in tqdm.tqdm(range(len(cocodataset))):
    #     example = cocodataset[i]
        
    # print("iterated over all examples in dataset")
    
    
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=4, collate_fn=collate_fn, num_workers=10)
    
    
    for batch in tqdm.tqdm(dl):
        pass
        
    print("iterated over all examples in dataloader")
    
    
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=4, collate_fn=collate_fn, num_workers=10)
    
    
    for batch in tqdm.tqdm(dl):
        pass
        
    print("iterated again over all examples in dataloader")
        
    
    batch = next(iter(dl))
    # print(batch)
    
    for epoch in range(1):
        i = 0
        for batch in dl:
            print(i, batch["image"].shape)
            i += 1
    

if __name__ == "__main__":
    main()