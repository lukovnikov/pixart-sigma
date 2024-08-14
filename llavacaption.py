import json
from pathlib import Path
import fire
import torch
import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

from urllib.parse import urlparse

from train_scripts.cocodata import COCOInstancesDataset



LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" 

LLAVA_DEVICE = "cuda:0"
MISTRAL_DEVICE = "cuda:1"


def is_url(url):
    return urlparse(url).scheme != ""


class Captioner:
    def __init__(self, llavamodel=LLAVA_MODEL, mistralmodel=MISTRAL_MODEL, llavadevice=LLAVA_DEVICE, mistraldevice=MISTRAL_DEVICE, loadedmodels=None):
        self.mistraldevice = mistraldevice
        self.llavadevice = llavadevice
        if loadedmodels is None:
            loadedmodels = self.loadmodels(llavamodel, mistralmodel, llavadevice, mistraldevice)
        self.llavaprocessor, self.llavamodel, self.mistralprocessor, self.mistralmodel = loadedmodels
        
    def get_loadedmodels(self):
        return self.llavaprocessor, self.llavamodel, self.mistralprocessor, self.mistralmodel
        
    def loadmodels(self, llavamodel, mistralmodel, llavadevice, mistraldevice):
        llavaprocessor = LlavaNextProcessor.from_pretrained(llavamodel)
        llavamodel = LlavaNextForConditionalGeneration.from_pretrained(llavamodel, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=llavadevice) 
        
        mistralprocessor, mistralmodel = None, None        
        if mistralmodel is not None:
            mistralprocessor = AutoTokenizer.from_pretrained(mistralmodel)
            mistralmodel = AutoModelForCausalLM.from_pretrained(mistralmodel, torch_dtype=torch.float16, device_map=mistraldevice)
        
        return llavaprocessor, llavamodel, mistralprocessor, mistralmodel
    
    def describe_image(self, image, short=False, category_name=None):
        if short:
            prompt = f"[INST] <image> \n Please return a concise description of the object in this image in one sentence. [/INST]"
            prompt = f"[INST] <image> \n What is this object? Describe in one phrase that describes the object and its properties. [/INST]"
            prompt = f"[INST] <image> \n Describe this object in one phrase? [/INST]"
            if category_name is not None:
                # prompt = f"[INST] <image> \n Describe this object and its colors in one phrase given that it's a \"{category_name}\"? [/INST]"
                if category_name == "person":
                    prompt = f"[INST] <image> \n Fully describe the {category_name} in the image in one phrase, including their hair and clothes. [/INST]"
                else:
                    prompt = f"[INST] <image> \n Fully describe the {category_name} in the image in one phrase. [/INST]"
        else:
            prompt = f"[INST] <image> \n Describe this image in one sentences. [/INST]"
        processed = self.llavaprocessor(prompt, image, return_tensors="pt").to(self.llavadevice)
        # print("input: ", prompt)
        
        out = self.llavamodel.generate(**processed, max_new_tokens=100)

        output = self.llavaprocessor.decode(out[0], skip_special_tokens=True)
        # print("output: ", output)
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        return reply
    
    def clean_description(self, text):
        messages = [
            {"role": "user", "content": 
                f"Copy the phrase describing the object in this text: \"{text}\". Return just one phrase. "}
        ]

        model_inputs = self.mistralprocessor.apply_chat_template(messages, return_tensors="pt").to(self.mistraldevice)

        generated_ids = self.mistralmodel.generate(model_inputs, max_new_tokens=100, do_sample=True)
        output = self.mistralprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        splits = output.split("[/INST]")
        assert( len(splits) == 2)
        reply = splits[1].strip()
        
        # print("Output length: ", len(reply.split()))
        return reply
    
    
def add_descriptions(example, captioner):
    img = example.load_image()
    for seg in example.seg_info:
        x, y, xx, yy = seg["bbox"]
        cropped = img.crop([x, y, x + xx, y + yy])
        if xx * yy < 2500:
            descr = seg["category_name"]
        else:
            descr = captioner.describe_image(cropped, short=True, category_name=seg["category_name"])
        seg["description"] = descr.lower()
    return example


def build_seg_dict(dss):
    d = {}
    if not isinstance(dss, list):
        dss = [dss]
    for ds in dss:
        for example in tqdm.tqdm(ds.examples):
            for seg in example.seg_info:
                assert seg["id"] not in d
                d[seg["id"]] = seg
                
    return d

    
def main(split="val", savepath="/USERSPACE/lukovdg1/coco2017/", saveevery=100, device="cuda:0", datafrom=None, datato=None):
    dataset = COCOInstancesDataset(maindir="/USERSPACE/lukovdg1/coco2017", split=split, upscale_to=512, max_masks=21)
    
    # d = build_seg_dict(datasets)
    # print("done build seg dict, no duplicate ids found")
    
    pathsuffix = ""
    if datafrom is not None or datato is not None:
        pathsuffix = f".{datafrom}:{datato}"
        
    path = Path(savepath) / "annotations" / (f"instance_descriptions_{split}2017.json" + pathsuffix)
    
    captioner = Captioner(mistralmodel=None, llavadevice=device)
    
    d = {}
    
    examples = dataset.examples
    if datafrom is not None or datato is not None:
        examples = examples[slice(datafrom, datato, None)]
        
    print(f"Processing {len(examples)} examples for data range {(datafrom, datato)}")
    
    for i, example in enumerate(tqdm.tqdm(examples)):
        example = add_descriptions(example, captioner)
        for seg in example.seg_info:
            assert seg["id"] not in d
            if seg["description"] == seg["category_name"]:
                continue
            d[seg["id"]] = seg["description"]
            
        if i % saveevery == 0:
            with open(path, "w") as f:
                json.dump(d, f, indent=4)
    
    
if __name__ == "__main__":
    fire.Fire(main)