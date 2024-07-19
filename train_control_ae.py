import colorsys
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from cocodata import COCOPanopticDataset, COCOPanopticExample
from torchvision import transforms
from PIL import Image

import pytorch_lightning as pl

import fire


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlSignalEncoder(torch.nn.Module):
    def __init__(self, inchannels=3, outchannels=4):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
            # torch.nn.SiLU(),
            # torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, outchannels*2, 1, padding=0),
            # torch.nn.Tanh(),
        )
        
    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        mu, logvar = x[:, :self.outchannels], x[:, self.outchannels:]
        return mu, logvar
    

class ControlSignalDecoder(torch.nn.Module):
    def __init__(self, inchannels=4, outchannels=3):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        PAD = 0
        OUTPAD = 0
        KSIZE = 2
        self.layers = torch.nn.Sequential(
            nn.ConvTranspose2d(inchannels, 128, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.SiLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, outchannels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            # nn.Sigmoid(),
        )
        
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    
        
class ControlSignalVAE(pl.LightningModule):
    def __init__(self, pixelchannels=3, latentchannels=4, lamda=0.1):
        super().__init__()
        self.lamda = lamda
        self.encoder = ControlSignalEncoder(pixelchannels, latentchannels)
        self.decoder = ControlSignalDecoder(latentchannels, pixelchannels)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=mu.device, dtype=mu.dtype)
        z = mu + std * esp
        return z
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xhat = self.decode(z)
        return xhat, mu, logvar
    
    def _forward(self, x):
        mu, logvar = self.encoder(x)
        xhat = self.decoder(mu)
        return xhat, mu, logvar
        
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def training_step(self, batch, batch_idx):
        # x = batch["cond_image"]
        x = batch.float()

        xrecons, mu, logvar = self._forward(x)
        reconloss = torch.nn.functional.mse_loss(xrecons, x, reduction="none").mean()
        # kld = -0.5 * torch.mean(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))    
        # kld = kld * self.lamda
        kld = self.lamda * torch.norm(mu, 2, dim=(1, 2, 3)).mean()
        loss = reconloss + kld    
        

        logdict = {
            'loss': loss.mean(),
            'kld': kld,
            'reconloss': reconloss.mean(),
        }
        return logdict
        # self.log_dict(logdict)

        # return loss.mean()
        
        
class AccumulatedDict(object):
    def __init__(self):
        self.sums = {}
        self.counts = {}
        
    def add(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
                if v.dim() == 0:
                    v = v.cpu().item()
            if k not in self.sums:
                self.sums[k] = 0
            self.sums[k] += v
            if k not in self.counts:
                self.counts[k] = 0
            self.counts[k] += 1
            
    def reset(self):
        for k in self.sums:
            self.sums[k] = 0
        for k in self.counts:
            self.counts[k] = 0
            
    def aggregate(self):
        ret = {}
        for k in self.sums:
            ret[k] = self.sums[k] / self.counts[k]
        return ret        
    
    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def randomcolor_hsv():
    hue = random.uniform(0, 360)
    sat = random.uniform(0.4, 0.9)
    val = random.uniform(0.2, 0.8)
    return colorsys.hsv_to_rgb(hue/360, sat, val)


def objectmasks_to_pil(x, colors):
    colors = torch.tensor(colors).to(x.device)
    argmax = torch.argmax(x, 0)
    colormap = colors[argmax]
    colormap = colormap.permute(2, 0, 1)
    ret = transforms.ToPILImage()(colormap)
    return ret
    
    
def main(epochs=100, gpu=0, outdir="control_ae_output", batsize=32, lamda=0.1, plotevery=200):
    NUMOBJ = 20
    model = ControlSignalVAE(pixelchannels=NUMOBJ+1, lamda=lamda)
    
    randomcolors = [randomcolor_hsv() for _ in range(NUMOBJ+1)]
    
    device = torch.device("cuda", gpu)    
    
    cocodataset = COCOPanopticDataset(maindir="/USERSPACE/lukovdg1/coco2017", split="train", upscale_to=512, max_masks=NUMOBJ)
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=batsize, num_workers=10)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) 
    
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    (outpath / "images").mkdir(exist_ok=True)
    
    maximg = 10
    global_step = 0
    
    totalsteps = epochs * len(dl)
    print(f"Total steps: {totalsteps}")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        accdict = AccumulatedDict()
        for i, batch in tqdm.tqdm(enumerate(dl)):
            optimizer.zero_grad()
            x = batch["cond_image"].to(device)
            
            if global_step % plotevery == 0:
                with torch.no_grad():
                    showimgs = []
                    showimgs_r = []
                    x_ = x[:min(len(x), maximg)]
                    xr = model._forward(x_)[0]
                    for j in range(len(x_)):
                        showimgs.append(objectmasks_to_pil(x_[j], randomcolors))
                        showimgs_r.append(objectmasks_to_pil(xr[j], randomcolors))
                    gridimage = image_grid(showimgs + showimgs_r, 2, len(showimgs))
                    gridimage.save(outpath / "images" / f"reconstr_{global_step}.png")
                
            outdict = model.training_step(x, None)
            accdict.add(outdict)
            loss = outdict["loss"]
            loss.backward()
            optimizer.step()
            global_step += 1
            
            
                    
        agg = accdict.aggregate()
        
            
        torch.save(model.state_dict(), outpath / "ae.pth")
        torch.save(model.encoder.state_dict(), outpath / "encoder.pth")
        print(f"Epoch {epoch}: mse={agg['reconloss']}, kld={agg['kld']}")
            
            
            
    
    # trainer = pl.Trainer(max_epochs=20, devices=1, accelerator="gpu")
    # trainer.fit(model, dl)
    
    
    
    
if __name__ == "__main__":
    fire.Fire(main)