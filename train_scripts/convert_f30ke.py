import re
from pathlib import Path
import sys
import torch
import xmltodict
import json
import pickle as pkl
import copy
import random
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import tqdm
import fire

import numpy as np

import math

from torch.nn.functional import conv2d

from torchvision.transforms.functional import gaussian_blur
from torchvision import transforms
from torch.nn.functional import interpolate
from perlin_noise import perlin_noise

from transformers import CLIPTokenizer, PreTrainedTokenizer

from scipy.ndimage import distance_transform_edt


def get_sam(device=0):
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "f30ke/sam_vit_h.pth"
    model_type = "vit_h"
    
    device = torch.device("cuda", device)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    return predictor


def draw_masks(image, masks):   # mask should be numpy
    overlay = Image.new('RGBA', image.size, (255,255,255,0))
    drawing = ImageDraw.Draw(overlay)
    for mask in masks:
        mask = mask.astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode='L')
        drawing.bitmap((0, 0), mask, fill=tuple([random.randint(0, 255) for _ in range(3)] + [160]))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    return image


def get_objects(annotations):    # returns a dict of object specs
    # print(annotations.keys())
    objs = {}
    for objspec in annotations["annotation"]["object"]:
        name = objspec["name"]
        if "bndbox" not in objspec:
            continue
            
        if not set(objspec.keys()) == {"name", "bndbox"}:
            print(objspec)
        assert set(objspec.keys()) == {"name", "bndbox"}
        
        objspec = objspec["bndbox"]
        if not isinstance(name, list):
            name = [name]
        for namee in name:
            if f"EN#{namee}" not in objs:
                objs[f"EN#{namee}"] = []
            objs[f"EN#{namee}"].append(objspec)
    return objs


def add_bbox_to_spans(spans, objspecs):
    ret = []
    for span in spans:
        start, end, text, spanid, spantype = span
        if spanid == "EN#0":
            continue
        if spanid not in objspecs:
            continue
        objspec = objspecs[spanid]
        # retspan = (start, end, text, spanid, spantype, objspec)
        retspan = (start, end, text, spanid, spantype)
        ret.append(retspan)
    return ret


class Flickr30KCaption:
    def __init__(self, caption=None, annotations=None, objspecs=None, _loaded=False):
        # parse caption
        if not _loaded:
            self.original_caption = caption
            self.puretext, self.spans = self.parse_caption(caption)
            if annotations is not None:
                assert objspecs is None
                objspecs = get_objects(annotations)
            if objspecs is not None:
                self.spans = add_bbox_to_spans(self.spans, objspecs)

    def todict(self):
        keys = ["original_caption", "puretext", "spans"]
        ret = {}
        for k in keys:
            ret[k] = getattr(self, k)
        return ret

    @classmethod
    def fromdict(cls, d):
        ret = cls(_loaded=True)
        for k in d:
            setattr(ret, k, d[k])
        return ret

    @staticmethod
    def parse_caption(caption):
        # get spans, saving their ids and types
        caption = caption.strip()
        pieces = re.split(r"(\[[^\]]+\])", caption)
        spans = []
        puretext = ""
        for piece in pieces:
            m = re.match(r"\[/(EN#\d+)/(\w+)\s([^\]]+)\]", piece)
            if m:
                spanid = m.group(1)
                spantype = m.group(2)
                text = m.group(3)
                spans.append((len(puretext), len(puretext)+len(text), text, spanid, spantype))
            else:
                text = piece
            puretext += str(text)
        return puretext, spans

    def __repr__(self):
        return f"Pure text: {self.puretext} \n\t Spans: {self.spans}"

    def sample(self):
        ret = copy.deepcopy(self)
        # ret.spans = [span[:-1] + ([random.choice(span[-1])],) for span in ret.spans]
        return ret
    
    def spacetokenize(self):
        words = []
        prev_i = 0
        for i, char in enumerate(self.puretext):
            if char == " ":
                words.append((prev_i, i, self.puretext[prev_i:i]))
                prev_i = i+1
            
        objnames = ["GLOBAL" for _ in words]
        
        spaniter = iter(self.spans)
        currentspan = next(spaniter)
        for word_i, (word_start, word_end, word) in enumerate(words):
            if word_end < currentspan[0]:
                pass
            elif word_start >= currentspan[0] and word_end <= currentspan[1]:
                objnames[word_i] = currentspan[3]
                if word_end == currentspan[1]:
                    try:
                        currentspan = next(spaniter)
                    except StopIteration as e:
                        break
        return [w for _, _, w in words], objnames
                
                
        
    

class Flickr30KEntitiesExample:
    def __init__(self, imagepath=None, captionpath=None, annotationpath=None, dictpicklepath=None, image=None, captions=None, objs=None):
        self.imagepath = imagepath
        self.captionpath = captionpath
        self.annotationpath = annotationpath
        self.image = image
        self.captions = captions
        self.objs = objs
        self.dictpicklepath = dictpicklepath

    def todict(self, incl_image=False):
        ret = {}
        keys = ["imagepath", "captionpath", "annotationpath", "captions", "objs", "image"]
        paths = ["imagepath", "captionpath", "annotationpath"]
        for k in keys:
            ret[k] = getattr(self, k)
        for k in paths:
            ret[k] = str(ret[k])
                
        if not incl_image:
            ret["image"] = None 
        dictcaptions = []
        for caption in ret["captions"]:
            dictcaptions.append(caption.todict())
        ret["captions"] = dictcaptions
        return ret
        
    @classmethod
    def fromdict(cls, d, ret=None):
        if ret is None:
            ret = cls()
        for k in d:
            setattr(ret, k, d[k])
        dictcaptions = []
        for caption in ret.captions:
            dictcaptions.append(Flickr30KCaption.fromdict(caption))
        ret.captions = dictcaptions
        return ret

    @property
    def m(self):
        return self.materialize()

    def loadimage(self):
        self.image = Image.open(self.imagepath).convert("RGB")
        return self.image

    def materialize(self):
        if self.dictpicklepath is not None:     # materialize from dictpickle
            with open(self.dictpicklepath, "rb") as f:
                d = pkl.load(f)    
            ret = self.fromdict(d)       # or to keep self: ret = self.fromdict(d, self)
            ret.loadimage()
            return ret
        else:
            # load image:
            self.loadimage()
            captions = open(self.captionpath).readlines()
            annotations = xmltodict.parse(open(self.annotationpath, "rb"))
            objs = get_objects(annotations)

            retcaptions = []
            for caption in captions:
                retcaptions.append(Flickr30KCaption(caption, objspecs=objs))
                
            return type(self)(imagepath=self.imagepath, captionpath=self.captionpath, annotationpath=self.annotationpath,
                            image=self.image, captions=retcaptions, objs=objs)

    def compute_masks_for_boxes(self, sam):
        assert self.image is not None and self.objs is not None    # must be materialized example
        bboxarray = []
        bboxes = []
        for objkey, obj in self.objs.items():
            for bbox in obj:
                bboxes.append(bbox)
                bboxarray.append([int(a) for a in [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]])
                
        bboxarray = torch.tensor(bboxarray, device=sam.device)

        sam.set_image(np.array(self.image.convert("RGB")))
        
        transformed_boxes = sam.transform.apply_boxes_torch(bboxarray, self.image.size)
        
        masks, _, _ = sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.cpu().numpy()
        for mask, obj in zip(list(masks), bboxes):
            obj["mask"] = mask[0, int(obj["ymin"]):int(obj["ymax"]), int(obj["xmin"]):int(obj["xmax"]), ]
            

    def __repr__(self):
        return f"Flickr30KEntitiesExample: \n\timage path: {self.imagepath}, \n\tcaption path: {self.captionpath}, \n\tannotations path: {self.annotationpath}"

    def test_paths(self):
        assert self.imagepath.exists()
        assert self.captionpath.exists()
        assert self.annotationpath.exists()

    def display_notebook(self):
        image, retcaptions = self.materialize()
        display(image)
        print(json.dumps([str(retcaption) for retcaption in retcaptions], indent=4))
        
    def sample(self):
        caption = random.choice(self.captions)
        caption = caption.sample()

        image = self.image
        objectids = set()
        for span in caption.spans:
            objectids.add(span[3])
        objs = {k: v for k, v in self.objs.items() if k in objectids}
        return caption, image, objs
    
    def sample_dualmasks(self):
        caption, image, objs = self.sample()
        masks = objs_to_masks(objs, image.size)
        dualmasks = generate_dual_masks(masks)
        return caption, image, dualmasks
        
    def sample_for_display(self):
        caption = random.choice(self.captions)
        caption = caption.sample()

        # draw = ImageDraw.Draw(self.image.convert("RGBA"))
        image = self.image
        
        maskoverlay = Image.new('RGBA', image.size, (255,255,255,0))
        bboxoverlay = Image.new('RGBA', image.size, (255,255,255,0))
        maskdrawing = ImageDraw.Draw(maskoverlay)
        bboxdrawing = ImageDraw.Draw(bboxoverlay)
        for span in caption.spans:
            for bbox in self.objs[span[3]]:
                bboxdrawing.rectangle(((int(bbox["xmin"]), int(bbox["ymin"])), (int(bbox["xmax"]), int(bbox["ymax"]))), outline="red")
                if "mask" in bbox:
                    mask = bbox["mask"].astype(np.uint8) * 255
                    mask = Image.fromarray(mask, mode='L')
                    maskdrawing.bitmap((int(bbox["xmin"]), int(bbox["ymin"])), mask, fill=tuple([random.randint(100, 255) for _ in range(3)] + [160]))
                    # drawing.bitmap((0, 0), mask, fill=tuple([random.randint(100, 255) for _ in range(3)] + [160]))
        image = Image.alpha_composite(image.convert("RGBA"), maskoverlay)
        image = Image.alpha_composite(image.convert("RGBA"), bboxoverlay)
        return caption, image
        
        
class DualMaskSampler:
    OUTERBOXLEVEL = 0.5
    BBOXLEVEL = 0.75
    POINTLEVEL = 0.25
    INNERBOXLEVEL = 1.
    
    def __init__(self, maskfrac=0.1, bboxfrac=0.2, pointfrac=0.1, outerradius=30, innerradius=20, smoothing=17,
                 return_type="ready", maxobjs=32, tokenizer:PreTrainedTokenizer=None):
        self.maskfrac, self.bboxfrac, self.pointfrac, self.outerradius, self.innerradius, self.smoothing = \
            maskfrac, bboxfrac, pointfrac, outerradius, innerradius, smoothing
            
        self.return_type = return_type
        self.maxobjs = maxobjs
        self.tokenizer = tokenizer  # !!! must be PreTrainedTokenizer from transformers with is_split_into_words support
        
    def sample(self, example):
        caption, image, objs = example.sample()
        masks = objs_to_masks(objs, image.size)
        
        rmap = None
        ret = {}
        for k in masks:
            rr = random.random()
            m = torch.tensor(masks[k])
            if rr < self.bboxfrac:
                # get bbox dualmask for this object
                ret[k] = generate_dualmask_bbox(m, outerboxlevel=self.BBOXLEVEL)
            elif rr < sum([self.bboxfrac, self.pointfrac]):
                # get point dualmask for this object
                ret[k] = generate_dualmask_point(m, outerboxlevel=self.POINTLEVEL, innerboxlevel=self.INNERBOXLEVEL)
            elif rr < sum([self.bboxfrac, self.pointfrac, self.maskfrac]):
                # get precise mask
                ret[k] = m
            else:
                if rmap is None:
                    rmap = generate_random_precision_map(m.shape, clampmin=0.2)
                ret[k] = generate_innerouter_mask(m, rmap, outerradius=self.outerradius, innerradius=self.innerradius, 
                                                  smoothing=self.smoothing, outerboxlevel=self.OUTERBOXLEVEL, innerboxlevel=self.INNERBOXLEVEL)
        
        if self.return_type.lower().startswith("d") or self.return_type is None:
            return caption, image, ret
        
        elif self.return_type.lower().startswith("r"):
            words, wordstoobjnames = caption.spacetokenize()
            
            stackedmasks = []
            objnames = []
            assert len(ret) < self.maxobjs
            for k, v in ret.items():
                objnames.append(k)
                stackedmasks.append(v)
            
            for _ in range(self.maxobjs - len(stackedmasks) - 1):
                objnames.append(None)
                stackedmasks.append(torch.zeros_like(v))
            
            if True:
                nn = list(zip(objnames, stackedmasks))
                random.shuffle(nn)
                objnames, stackedmasks = zip(*nn)
            
            objnames = ["GLOBAL"] + list(objnames)
            stackedmasks = [torch.ones_like(v)] + list(stackedmasks)
            
            objtomaskpos = {k: v for k, v in zip(objnames, list(range(len(objnames))))}
            wordtomaskpos = [objtomaskpos[objname] for objname in wordstoobjnames]
            
            if self.tokenizer is None:
                return image, stackedmasks, words, wordtomaskpos
            
            else:
                tokret = self.tokenizer([words] + [[word] for word in words], is_split_into_words=True, truncation=True)
                tokenized = tokret["input_ids"]
                rectok = []
                retwordtomaskpos = []
                for tokenizedword, wordtomask in zip(tokenized[1:], wordtomaskpos):
                    toksforword = tokenizedword[1:-1]
                    retwordtomaskpos = retwordtomaskpos + [wordtomask] * len(toksforword)
                    rectok = rectok + toksforword
                assert rectok == tokenized[0][1:-1]
                retwordtomaskpos = [0] + retwordtomaskpos + [0]

                stackedmasks = torch.stack(stackedmasks, 0)
                return image, stackedmasks, tokenized[0], retwordtomaskpos
        


class Flickr30KEntitiesDatasetOriginal:
    def __init__(self, datadir="/USERSPACE/lukovdg1/controlnet11/f30ke/", split="train", **kw):
        super().__init__(**kw)
        datadir = Path(datadir)
        self.datadir = datadir
        self.imagedir = datadir / "images"
        self.captiondir = datadir / "sentences"
        self.annotationdir = datadir / "annotations"
        self.split = split
        self.imgids = set([int(x.strip()) for x in open(datadir / f"{split}.txt").readlines()])
        self.examples = {}
        self.exampleids = []
        self.load_data()

    def load_data(self):
        for imgid in self.imgids:
            example = Flickr30KEntitiesExample(self.imagedir / f"{imgid}.jpg",
                                               self.captiondir / f"{imgid}.txt",
                                               self.annotationdir / f"{imgid}.xml")
            example.test_paths()
            self.examples[imgid] = example
            self.exampleids.append(imgid)

    def __getitem__(self, i):
        ret = self.examples[self.exampleids[i]]
        return ret

    def __len__(self):
        return len(self.exampleids)
    

class Flickr30KEntitiesDataset:
    def __init__(self, datadir="/USERSPACE/lukovdg1/controlnet11/f30ke/", split="val", **kw):
        super().__init__(**kw)
        datadir = Path(datadir)
        self.datadir = datadir
        self.anndir = datadir / "dictpickles"
        self.split = split
        self.imgids = set([int(x.strip()) for x in open(datadir / f"{split}.txt").readlines()])
        self.examples = {}
        self.exampleids = []
        self.load_data()

    def load_data(self):
        for imgid in self.imgids:
            example = Flickr30KEntitiesExample(dictpicklepath=self.anndir / f"{imgid}.dict.pkl")
            self.examples[imgid] = example
            self.exampleids.append(imgid)

    def __getitem__(self, i):
        ret = self.examples[self.exampleids[i]]
        return ret

    def __len__(self):
        return len(self.exampleids)


def create_circular_kernel(radius=5):
    kernel = torch.zeros(1, 1, radius * 2 + 1, radius * 2 + 1)
    for i in range(kernel.shape[2]):
        for j in range(kernel.shape[3]):
            x, y = i - radius, j - radius
            if math.sqrt(x**2 + y**2) <= radius:
                kernel[..., i, j] = 1
    return kernel


def variable_hardblur_mask_old(masktensor, 
                      rmap=None,
                      device=torch.device("cpu"),
                      rescale=2,
                      smoothing=11,
                     ):
    img = masktensor.to(device)
    original_shape = img.shape[-2:]
    inner_shape = tuple([x // rescale for x in original_shape])
    
    rmap = rmap.to(device)    
    _rmap = (interpolate(rmap[None, None], inner_shape, mode="bilinear")[0, 0] / rescale).long()
    # _rmap = torch.clamp_min(_rmap, 1)
    
    _current_region_mask = interpolate(img[None, None], inner_shape, mode="nearest")[0, 0]
    kernelsizes = _rmap * _current_region_mask                                               # (H x W)
    # print(kernelsizes.unique())
    for kernelsize in list(kernelsizes.unique().long().cpu().numpy()):
        # print(kernelsize)
        if kernelsize < 1:
            continue
        kernelsize_mask = kernelsizes == kernelsize
        kernel = create_circular_kernel(kernelsize).to(device)
        _expanded_region_mask = (conv2d(kernelsize_mask[None].float(), kernel, padding="same") > 0)
        expanded_region_mask = interpolate(_expanded_region_mask[None].float(), original_shape, mode="bicubic")[0]
        if smoothing > 0:
            expanded_region_mask = gaussian_blur(expanded_region_mask, smoothing)
            expanded_region_mask = expanded_region_mask > 0.5
        expanded_region_mask = expanded_region_mask.bool()[0]
        img = torch.maximum(img, expanded_region_mask)
        
    return img


def gaussian_1d_kernel(size, sigma):
    """Creates a 1D Gaussian kernel using the given sigma."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel


def apply_1d_convolution(input, kernel, axis):
    """Applies 1D convolution along the specified axis (horizontal or vertical)."""
    # Reshape the 1D kernel into the correct shape for convolution
    if axis == 0:  # Vertical convolution (along the height)
        kernel = kernel.view(1, 1, -1, 1)  # Shape [out_channels, in_channels, kernel_height, 1]
    elif axis == 1:  # Horizontal convolution (along the width)
        kernel = kernel.view(1, 1, 1, -1)  # Shape [out_channels, in_channels, 1, kernel_width]

    # Apply convolution, padding is 'reflect' to maintain image size
    return conv2d(input, kernel, padding=(kernel.size(-2) // 2, kernel.size(-1) // 2), groups=1)


def separable_gaussian_blur(image, kernel_size):
    """Applies separable Gaussian blur using a 1D Gaussian kernel."""
    # Create the 1D Gaussian kernel for the given sigma
    # sigma = (size - 1) / 6
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    # kernel_size = int(2 * sigma + 1)
    gaussian_kernel = gaussian_1d_kernel(kernel_size, sigma)

    # Apply the separable Gaussian blur
    # 1D convolution along the horizontal (width) axis
    blurred = apply_1d_convolution(image, gaussian_kernel, axis=1)

    # 1D convolution along the vertical (height) axis
    blurred = apply_1d_convolution(blurred, gaussian_kernel, axis=0)

    return blurred


def variable_hardblur_mask(masktensor, 
                      rmap=None,
                      device=torch.device("cpu"),
                      rescale=2,
                      smoothing=11,
                     ):
    img = masktensor.to(device)
    original_shape = img.shape[-2:]
    inner_shape = tuple([x // rescale for x in original_shape])
    
    rmap = rmap.to(device)    
    _rmap = (interpolate(rmap[None, None], inner_shape, mode="bilinear")[0, 0] / rescale).long()
    # _rmap = torch.clamp_min(_rmap, 1)
    
    _current_region_mask = interpolate(img[None, None], inner_shape, mode="nearest")[0, 0]
    kernelsizes = _rmap * _current_region_mask                                               # (H x W)
    # print(kernelsizes.unique())
    for kernelsize in list(kernelsizes.unique().long().cpu().numpy()):
        # print(kernelsize)
        if kernelsize < 1:
            continue
        kernelsize_mask = kernelsizes == kernelsize
        # kernel = create_circular_kernel(kernelsize).to(device)
        _expanded_region_mask = separable_gaussian_blur(kernelsize_mask[None].float(), kernelsize)
        _expanded_region_mask = _expanded_region_mask > 0
        expanded_region_mask = interpolate(_expanded_region_mask[None].float(), original_shape, mode="bicubic")[0]
        if smoothing > 0:
            expanded_region_mask = separable_gaussian_blur(expanded_region_mask, smoothing)
            expanded_region_mask = expanded_region_mask > 0.5
        expanded_region_mask = expanded_region_mask.bool()[0]
        img = torch.maximum(img, expanded_region_mask)
        
    return img
    
    
def generate_random_precision_map(retshape=(512, 512), weights=(0.3, 0.7, 0.2), rescale=8, clampmin=0., clampmax=1.):
    shape = (512, 512)
    _shape = tuple([x//rescale for x in shape])
    gridsizes = [2**k for k in range(len(weights))]
    noise = None
    for gridsize, weight in zip(gridsizes, weights):
        if noise is None:
            noise = perlin_noise(grid_shape=(gridsize, gridsize), out_shape=_shape) * weight
        else:
            noise += perlin_noise(grid_shape=(gridsize, gridsize), out_shape=_shape) * weight
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = noise.clamp(clampmin, clampmax)
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    retmaxsize = max(retshape)
    
    noise = interpolate(noise[None, None], (retmaxsize, retmaxsize), mode="bilinear")[0, 0]
    noise = noise[:retshape[0], :retshape[1]]
    return noise


def generate_innerouter_mask(mask, rmap=None, outerradius=50, innerradius=40, smoothing=17, outerboxlevel=0.4, innerboxlevel=1., device=torch.device("cpu")):    # input is a binary mask, output is binary mask
    rmap = generate_random_precision_map(mask.shape, clampmin=0.2) if rmap is None else rmap
    rmap2 = generate_random_precision_map(rmap.shape)
    outerrmap = (rmap + (rmap2-0.5) * rmap).clamp(0., 1)
    innerrmap = (rmap - (rmap2-0.5) * rmap).clamp(0., 1)

    # print(outerrmap.shape, mask.shape, outerradius)
    outer_dist = torch.tensor(distance_transform_edt(1 - mask))
    outermask = outer_dist <= (outerrmap * outerradius)
    
    inner_dist = torch.tensor(distance_transform_edt(mask))
    innermask = inner_dist > (innerrmap * innerradius)
    # outermask = variable_hardblur_mask(mask.float(), outerrmap * outerradius, smoothing=smoothing, device=device)
    # rmap = generate_random_precision_map(shape, clampmin=0.2)
    # rmap = rmap * innerradius
    # innermask = 1 - variable_hardblur_mask((~mask.bool()).float(), innerrmap * innerradius, smoothing=smoothing, device=device)
    ret = torch.zeros_like(innermask)
    ret = torch.where(~outermask.bool(), ret, outerboxlevel)
    ret = torch.where(~innermask.bool(), ret, innerboxlevel)
    # ret = innermask + outermask * 0.5
    ret = ret.clamp(0, 1)
    return ret


def generate_dualmask_bbox(m, outerboxlevel=0.75):
    m = torch.tensor(m) if not isinstance(m, torch.Tensor) else m
    ret = torch.zeros_like(m).float()
    nonz = m.nonzero()
    mins, maxs = nonz.min(0)[0], nonz.max(0)[0]
    xmin, ymin = mins[0], mins[1]
    xmax, ymax = maxs[0], maxs[1]
    ret.data[xmin:xmax, ymin:ymax] = outerboxlevel
    return ret


def generate_dualmask_point(m, outerboxlevel=0.25, innerboxlevel=1.):
    m = torch.tensor(m)
    ret = torch.ones_like(m).float() * outerboxlevel
    mblur = transforms.GaussianBlur(35, sigma=5.)(m.float()[None])[0]
    m_ = mblur == mblur.max()
    # display(transforms.ToPILImage()(m_.float()))
    nonz = m_.nonzero()
    idx = random.randint(0, nonz.shape[0]-1)
    point = nonz[idx]
    mins, maxs = nonz.min(0)[0], nonz.max(0)[0]
    xmin, ymin = point[0]-3, point[1]-3
    xmax, ymax = point[0]+4, point[1]+4
    ret.data[xmin:xmax, ymin:ymax] = innerboxlevel
    return ret


def objs_to_masks(objs, imsize):
    ret = {}
    for k, vv in objs.items():
        fullmask = None
        for v in vv:
            mask, xmin, xmax, ymin, ymax = v["mask"], int(v["xmin"]), int(v["xmax"]), int(v["ymin"]), int(v["ymax"])
            if fullmask is None:
                fullmask = np.zeros(imsize[::-1]).astype(mask.dtype)
            fullmask[ymin:ymax, xmin:xmax] = mask
        ret[k] = fullmask
    return ret


def generate_dual_masks(masks, outerradius=30, innerradius=20, smoothing=17):
    rmap = None
    ret = {}
    for k in masks:
        m = torch.tensor(masks[k])
        if rmap is None:
            rmap = generate_random_precision_map(m.shape, clampmin=0.2)
        ret[k] = generate_innerouter_mask(m, rmap, outerradius=outerradius, innerradius=innerradius, smoothing=smoothing)
    return ret


def main_preprocess():
    sam = get_sam()
    datadir = "/USERSPACE/lukovdg1/controlnet11/f30ke/"
    failedids = []
    
    redo = False
    
    for splitname in ("val", "test", "train"):
        ds = Flickr30KEntitiesDatasetOriginal(split=splitname)
        outdir = Path(datadir) / "dictpickles"
        for imgid in tqdm.tqdm(ds.exampleids):
            try:
                filename = outdir / f"{imgid}.dict.pkl"
                if not filename.exists() or redo:
                    x = ds.examples[imgid]
                    x = x.materialize()
                    x.compute_masks_for_boxes(sam)
                    xdict = x.todict()
                    pkl.dump(xdict, open(filename, "wb"))
            except Exception as e:
                print(e)
                failedids.append(imgid)
                print(f"imgid {imgid} failed")
                
    print(f"{len(failedids)} failed")
    json.dump(failedids, open(outdir / "failed_ids.json", "w"))
    

def main():
    datadir = "/USERSPACE/lukovdg1/controlnet11/f30ke/"
    fds = Flickr30KEntitiesDataset(datadir=datadir)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dms = DualMaskSampler(tokenizer=tokenizer)
    example = fds[7].m
    ret = dms.sample(example)
    
    sys.exit()
    examplepath = Path(datadir) / "dictpickles/4045661794.dict.pkl"
    example = Flickr30KEntitiesExample(dictpicklepath=examplepath)
    example = example.materialize()
    print(example)
                
    
        
        
if __name__ == "__main__":
    fire.Fire(main)