import torch
from diffusers import HunyuanDiTPipeline


def main():
    pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16)
    pipe.to("cuda")

    # You may also use English prompt as HunyuanDiT supports both English and Chinese
    # prompt = "An astronaut riding a horse"
    prompt = "一个宇航员在骑马"
    image = pipe(prompt).images[0]
    
    
if __name__ == "__main__":
    main()
