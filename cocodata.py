from copy import copy
import math
from PIL import Image
import json
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, DataLoader
import tqdm
from transformers import CLIPTokenizer
from torchvision.transforms.functional import to_tensor, to_pil_image
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools
import numpy as np
import colorsys
from einops import rearrange, repeat
import re
import cv2 as cv


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
    min_region_area = -1 # 0.002
    
    def __init__(self, maindir:str=None, split="valid", max_masks=20, min_masks=2, max_samples=None, min_size=350, upscale_to=None,
                 examples=None, mergeregions=False,  # mergeregions?
                 regiondrop=False,           # if False, dropping examples with too many masks, if True: keeping all examples and dropping randomly some masks, if float: acts like True, but also drops some masks with the given number as drop probability
                 casmode=None, simpleencode=False, limitpadding=False,
                 tokenizer_version="openai/clip-vit-large-patch14",
                 usescribbles=False, usecanny=False):
        super().__init__()
        assert examples is None or maindir is None      # provide either a directory or a list of already made examples
        self.maindir = maindir
        self.n = 0
        self.tokenizer_version = tokenizer_version
        self.load_tokenizer()
        
        self.casmode = casmode
        
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
            panopticsfile = Path(self.maindir) / "annotations" / f"panoptic_{which}2017.json"
            self.panoptic_dir = Path(self.maindir) / "annotations" / f"panoptic_{which}2017"
            
            print("loading captions")     
            image_db, captiondb = self.load_captions(captionsfile, img_dir=self.img_dir)        # creates image db and caption db
            print("loading panoptics")
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
                
    def load_tokenizer(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_version)

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
        return tokenized["input_ids"]
    
    def untokenize(self, x, tokenizer=None):
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        ret = tokenizer.decode(x)
        return ret
    
    # def __iter__(self):
    #     self.n = 0
    #     if self.shuffle:
    #         for k, v in self.examples:
    #             random.shuffle(v)
    #     return self
    
    # def __next__(self):
    #     if self.n >= self.total_n:
    #         raise StopIteration()
        
    #     prevc, c = 0, 0
    #     # find bucket
    #     for k, v in self.examples:
    #         prevc = c
    #         c += len(v)
    #         if prevc <= self.n < c:
    #             break
    #     # get example
    #     example = v[self.n - prevc]
    #     # increase count
    #     self.n += 1
        
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
        
        for example in examples:
            images.append(example["image"])   # concat images
            cond_images.append(example["cond_image"])
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
        return {"image": rearrange(imagebatch, 'b c h w -> b h w c'), 
                "cond_image": rearrange(cond_imagebatch, 'b c h w -> b h w c'),
                "caption": captionbatch, 
                "layerids": layeridsbatch, 
                "encoder_layerids": encoder_layeridsbatch,
                "regionmasks": batched_regionmasks, 
                "captiontypes": captiontypes}
        
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
            seg_img = seg_img.resize(newsize, resample=Image.BOX)
            
        # 2. transform to tensors
        imgtensor = to_tensor(img)
        seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1)
        
        # 3. create conditioning image by randomly swapping out colors
        cond_imgtensor = torch.ones_like(imgtensor) * torch.tensor(randomcolor_hsv())[:, None, None]
        
        # 4. pick one caption at random (TODO: or generate one from regions)
        captions = [random.choice(example.captions)]
    
        # 4. load masks
        # masks = [torch.ones_like(imgtensor[0], dtype=torch.bool)]
        # get the captions of the regions and build layer ids
        # region_code_to_layerid = {0: 0}
        # region_code_to_layerid = {}
        
        # region_caption_to_layerid = {}
        # unique_region_captions = set()
        
        # for _, region_info in example.seg_info.items():
        #     unique_region_captions.add(region_info["caption"])
        
        for i, (region_code, region_info) in enumerate(example.seg_info.items()):
            rgb = torch.tensor(region_code_to_rgb(region_code))
            region_mask = (seg_imgtensor == rgb[:, None, None]).all(0)
            if (region_mask > 0).sum() / np.prod(region_mask.shape) < self.min_region_area:
                continue
            # if self.casmode is None or self.casmode.name != "global":
            #     region_caption = region_info["caption"]
            #     if region_caption in unique_region_captions:
            #         if (not self.mergeregions) or (region_caption not in region_caption_to_layerid):
            #             new_layerid = len(masks)
            #             region_caption_to_layerid[region_caption] = new_layerid
            #             captions.append(region_info["caption"])
            #             masks.append(region_mask)
            #         else:
            #             new_layerid = region_caption_to_layerid[region_caption]
            #             masks[new_layerid] = masks[new_layerid] | region_mask        
            #         region_code_to_layerid[region_code] = new_layerid            
            #     else:
            #         pass #continue    # or pass? (if pass, the mask will end up in the conditioning image for controlnet)
            
            randomcolor = torch.tensor(randomcolor_hsv()) #if self.casmode is not None else torch.tensor(randomcolor_predef())
            maskcolor = region_mask.unsqueeze(0).repeat(3, 1, 1) * randomcolor[:, None, None]
        
            cond_imgtensor = torch.where(region_mask.unsqueeze(0) > 0.5, maskcolor, cond_imgtensor)
            
            
        # append extra global prompt
        # extraexpressions = ["This image contains", "In this image are", "In this picture are", "This picture contains"]
        # if (self.casmode.name == "doublecross") and not ("keepprompt" in self.casmode.chunks or "test" in self.casmode.chunks):
        #     # assert not self.simpleencode
        #     captions[0] += ". " + random.choice(extraexpressions) + " " + ", ".join([e for e in captions[1:]]) + "."
            
            
        # minimize_length = False
        
        # if self.casmode.use_global_prompt_only:   # self.casmode in ("posattn-opt", "posattn2-opt"):
        #     caption = captions[0] if not self.casmode.augment_only else ""
        #     if self.casmode.augment_global_caption:       # if training, automatically augment sentences
        #         tojoin = []
        #         for i, capt in enumerate(captions[1:]):
        #             tojoin.append(f"{{{capt}:{i}}}")
        #         caption += " " + random.choice(extraexpressions) + " " + ", ".join(tojoin) + "."
        #     _caption, _layerids = _tokenize_annotated_prompt(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)
        #     # replace region codes with layer ids
        #     for region_code, layerid in region_code_to_layerid.items():
        #         _layerids = torch.masked_fill(_layerids, _layerids == region_code + 1, layerid)
        #     captions, layerids = [_caption], [_layerids]
        #     encoder_layerids = [torch.ones_like(layerids[-1]) * 0]
        # # elif self.simpleencode: #  or (self.casmode in ("posattn-opt", "posattn2-opt") and not ("keepprompt" in self.casmodechunks or "test" in self.casmodechunks)):   # or self.casmode == "doublecross":
        # #     tojoin = []
        # #     for i, capt in enumerate(captions[1:]):
        # #         tojoin.append(f"{{{capt}:{i}}}")
        # #     captions[0] += ". " + random.choice(extraexpressions) + " " + ", ".join(tojoin) + "."
            
        # #     _captions, _layerids = _tokenize_annotated_prompt(captions[0], tokenizer=self.tokenizer, minimize_length=minimize_length)
        # #     captions, layerids = [_captions], [_layerids]
        # #     encoder_layerids = [torch.ones_like(layerids[-1]) * 0]
        # else:
        #     # encode separately
        #     tokenized_captions = []
        #     layerids = []
        #     encoder_layerids = []
        #      #self.casmode != "global"
             
        #     if self.casmode.augment_global_caption:
        #         caption = captions[0] if not self.casmode.augment_only else ""
        #         caption += ". " + random.choice(extraexpressions) + " " + ", ".join([e for e in captions[1:]]) + "."
        #         captions[0] = caption
             
        #     for i, caption in enumerate(captions):
        #         if i == 0:
        #             tokenized_caption, layerids_e = _tokenize_annotated_prompt(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)
        #             # replace region codes with layer ids
        #             for region_code, layerid in region_code_to_layerid.items():
        #                 layerids_e = torch.masked_fill(layerids_e, layerids_e == region_code + 1, layerid)
        #             tokenized_captions.append(tokenized_caption)
        #             layerids.append(layerids_e)    
        #         else:
        #             tokenized_captions.append(self.tokenize(caption, tokenizer=self.tokenizer, minimize_length=minimize_length)[0])
        #             layerids.append(torch.ones_like(tokenized_captions[-1]) * i)    
        #         encoder_layerids.append(torch.ones_like(layerids[-1]) * i)
        #     captions = tokenized_captions
        #     if self.casmode.replace_layerids_with_encoder_layerids:
        #         layerids[0] = encoder_layerids[0]
                
        # if self.limitpadding:
        #     for i in range(len(captions)):
        #         caption = captions[i]
        #         numberpads = (caption == self.tokenizer.pad_token_id).sum()     # assumption: all pads are contiguous and at the end
        #         endidx = -(numberpads - self.padlimit)
        #         captions[i] = caption[:endidx]
        #         layerids[i] = layerids[i][:endidx]
        #         encoder_layerids[i] = encoder_layerids[i][:endidx]

        # random square crop of size divisble by 64 and maximum size 512
        cropsize = min((min(imgtensor[0].shape) // 64) * 64, 512)
        crop = (random.randint(0, imgtensor.shape[1] - cropsize), 
                random.randint(0, imgtensor.shape[2] - cropsize))
        # print(cropsize)
        
        imgtensor = imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        cond_imgtensor = cond_imgtensor[:, crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize]
        # masks = [maske[crop[0]:crop[0]+cropsize, crop[1]:crop[1]+cropsize] for maske in masks]
        
        # # compute downsampled versions of the layer masks
        # downsamples = [cropsize // e for e in [8, 16, 32, 64]]
        # downmaskses = []
        # for mask in masks:
        #     downmasks = {}
        #     for downsample in downsamples:
        #         downsampled = _img_importance_flatten(mask.float(), downsample, downsample)
        #         downmasks[tuple(downsampled.shape)] = downsampled
        #     downmaskses.append(downmasks)
            
        # # concatenate masks in one tensor
        # downmasktensors = {}
        # for downmasks in downmaskses:
        #     for res, downmask in downmasks.items():
        #         if res not in downmasktensors:
        #             downmasktensors[res] = []
        #         downmasktensors[res].append(downmask)
        # downmasktensors = {k: torch.stack(v, 0) for k, v in downmasktensors.items()}
        
        # # if usescribbles, transform cond_imgtensor to scribbles
        # if self.usescribbles:
        #     cond_imgtensor_np = (np.array(cond_imgtensor.permute(1, 2, 0)) * 255).astype("uint8")
        #     canny = cv.Canny(cond_imgtensor_np, 50, 100)
        #     scribbleradius = 5      # TODO: check scribble brush size
        #     brush = np.zeros((scribbleradius*2+1, scribbleradius*2+1))
        #     cv.circle(brush, (scribbleradius, scribbleradius), 0, 1, scribbleradius*2)
        #     cond_scribbles = cv.filter2D(src=canny, ddepth=-1, kernel=brush) > 0
        #     cond_imgtensor = torch.tensor(cond_scribbles)[None].repeat(3, 1, 1).to(cond_imgtensor.dtype).to(cond_imgtensor.device)
            
        # if self.usecanny:
        #     canny = example.canny_data
        #     cond_imgtensor = to_tensor(canny).repeat(3, 1, 1)
            
        imgtensor = imgtensor * 2 - 1.
        
        return {"image": imgtensor, 
                "cond_image": cond_imgtensor,
                "captions": captions,
                # "layerids": layerids,
                # "encoder_layerids": encoder_layerids,
                # "regionmasks": downmasktensors
                }
    
    
class COCODataLoader(object):
    def __init__(self, cocodataset, batch_size=2, shuffle=False, num_workers=0, repeatexample=False) -> None:
        super().__init__()
        self.ds = cocodataset
        self.dataloaders = []
        self.batch_size = min(batch_size.values()) if isinstance(batch_size, dict) else batch_size
        self.repeatexample = repeatexample
            
        for k, cocosubset in self.ds.examples:
            if isinstance(batch_size, dict):
                batsize = batch_size[k]
            else:
                batsize = batch_size
                
            collatefn = cocodataset.collate_fn
            if self.repeatexample:
                collatefn = lambda examples: cocodataset.collate_fn(examples * batch_size)
                batsize = 1
                
            subdl = DataLoader(COCODatasetSubset(cocosubset, self.ds), 
                                       batch_size=batsize, 
                                       collate_fn=collatefn, 
                                       shuffle=shuffle,
                                       num_workers=num_workers)
            self.dataloaders.append(subdl)
        self.subdls_lens = [len(dl) for dl in self.dataloaders]
        # print(self.subdls_lens)
        
    def __iter__(self):
        return itertools.chain(*[iter(subdl) for subdl in self.dataloaders])
        

# !!! IDEA: only use the higher-res images for training later denoising steps
    
    
def main(x=0):
    import pickle
    cocodataset = COCOPanopticDataset(maindir="/USERSPACE/lukovdg1/coco2017", split="train", upscale_to=512)
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=4)
    
    batch = next(iter(dl))
    # print(batch)
    
    for epoch in range(1):
        i = 0
        for batch in dl:
            print(i, batch["image"].shape)
            i += 1
    

if __name__ == "__main__":
    main()