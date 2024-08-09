import colorsys
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from cocodata import COCOPanopticDataset, COCOPanopticExample
from torchvision import transforms, models
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
    def __init__(self, inchannels=3, outchannels=4, patchsize=1):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, 32, 3, padding=1),
            # torch.nn.BatchNorm2d(32),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            # torch.nn.BatchNorm2d(32),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1, stride=2),
            # torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            # torch.nn.BatchNorm2d(64),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1, stride=2),
            # torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            # torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1, stride=2),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.SiLU(),
            # torch.nn.Conv2d(128, 128, 3, padding=1),
            # torch.nn.Tanh(),
            torch.nn.SiLU(),
        )
        if patchsize == 2:
            self.layers.append(torch.nn.Conv2d(256, 256, 3, padding=1, stride=2))
            self.layers.append(torch.nn.SiLU())
        
        self.layers.append(torch.nn.Conv2d(256, outchannels*2, 1, padding=0))
        self.zeroconv = None
        
    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        mu, logvar = x[:, :self.outchannels], x[:, self.outchannels:]
        if self.zeroconv is not None:
            mu = self.zeroconv(mu)
        return mu, logvar
    

class ControlSignalDecoder(torch.nn.Module):
    def __init__(self, inchannels=4, outchannels=3, patchsize=1):
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
            # nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.SiLU(),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, outchannels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(outchannels),
            nn.SiLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            # nn.Sigmoid(),
        )
        if patchsize == 2:
            extralayers = torch.nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
                nn.SiLU(),
                nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(inchannels),
                nn.SiLU()
            )
            self.layers = extralayers + self.layers
        
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    


class ControlSignalEncoderV2(torch.nn.Module):
    mult = 2
    def __init__(self, inchannels=3, outchannels=4, patchsize=1):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(inchannels, 32 * self.mult, 3, padding=1),
            torch.nn.BatchNorm2d(32 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32 * self.mult, 32 * self.mult, 3, padding=1),
            torch.nn.BatchNorm2d(32 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32 * self.mult, 64 * self.mult, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64 * self.mult, 64 * self.mult, 3, padding=1),
            torch.nn.BatchNorm2d(64 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64 * self.mult, 128 * self.mult, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(128 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128 * self.mult, 128 * self.mult, 3, padding=1),
            torch.nn.BatchNorm2d(128 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128 * self.mult, 256 * self.mult, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(256 * self.mult),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128 * self.mult, 256 * self.mult, 3, padding=1, stride=2),
            torch.nn.BatchNorm2d(256 * self.mult),
            torch.nn.SiLU(),
        )
        if patchsize == 2:
            self.layers.append(torch.nn.Conv2d(256 * self.mult, 256 * self.mult, 3, padding=1, stride=2))
            self.layers.append(torch.nn.SiLU())
        
        self.layers.append(torch.nn.Conv2d(256 * self.mult, outchannels*2, 1, padding=0))
        self.zeroconv = None
        
    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        mu, logvar = x[:, :self.outchannels], x[:, self.outchannels:]
        if self.zeroconv is not None:
            mu = self.zeroconv(mu)
        return mu, logvar
    

class ControlSignalDecoderV2(torch.nn.Module):
    mult = 2
    
    def __init__(self, inchannels=4, outchannels=3, patchsize=2):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        PAD = 0
        OUTPAD = 0
        KSIZE = 2
        self.layers = torch.nn.Sequential(
            nn.ConvTranspose2d(inchannels, 128 * self.mult, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(128 * self.mult, 128 * self.mult, kernel_size=3, padding=1),
            nn.BatchNorm2d(128 * self.mult),
            nn.SiLU(),
            nn.ConvTranspose2d(128 * self.mult, 64 * self.mult, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(64 * self.mult, 64 * self.mult, kernel_size=3, padding=1),
            nn.BatchNorm2d(64 * self.mult),
            nn.SiLU(),
            nn.ConvTranspose2d(64 * self.mult, 32 * self.mult, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(32 * self.mult, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.SiLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
        )
        if patchsize == 2:
            extralayers = torch.nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
                nn.SiLU(),
                nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannels),
                nn.SiLU()
            )
            self.layers = extralayers + self.layers
        
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    
    
class ControlSignalEncoderV3(torch.nn.Module):
    
    def __init__(self, inchannels=3, outchannels=4, patchsize=1):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        
        self.resnet = create_resnet_encoder(inchannels)
        
        self.layers = torch.nn.Sequential()
        
        if patchsize == 2:
            self.layers.append(torch.nn.Conv2d(512, 512, 3, padding=1, stride=2))
            self.layers.append(torch.nn.SiLU())
        
        self.layers.append(torch.nn.Conv2d(512, outchannels*2, 1, padding=0))
        self.zeroconv = None
        
    def forward(self, x):
        x = self.resnet(x)
        for layer in self.layers:
            x = layer(x)
        mu, logvar = x[:, :self.outchannels], x[:, self.outchannels:]
        if self.zeroconv is not None:
            mu = self.zeroconv(mu)
        return mu, logvar
    
        
class ControlSignalVAE(pl.LightningModule):
    def __init__(self, pixelchannels=3, latentchannels=4, lamda=0.1, patchsize=1):
        super().__init__()
        self.lamda = lamda
        self.encoder = ControlSignalEncoderV3(pixelchannels, latentchannels, patchsize=patchsize)
        self.decoder = ControlSignalDecoderV2(latentchannels, pixelchannels, patchsize=patchsize)
        
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
            ret[k] = self.sums[k] / max(self.counts[k], 1e-3)
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


class MyResNet(models.ResNet):
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


def create_resnet_encoder(inpchannels=21):
    model = models.resnet18()
    # model.conv1 = torch.nn.Conv2d(inpchannels, 64, 7, stride=1, padding=3, bias=False)
    model.conv1 = torch.nn.Conv2d(inpchannels, 64, 1, stride=1, padding=0, bias=False)
    model.avgpool = None
    model.fc = None
    
    model.__class__ = MyResNet
    
    # try:
    x = torch.randn(1, 21, 512, 512)
    y = model(x)
    print(y.shape)
    return model
    
    
    
def main(epochs=2, gpu=0, outdir="control_ae_output_v2_controlnet", batsize=32, lamda=0.00001, plotevery=200, saveevery=200, patchsize=1, latentchannels=4, split="train"):
    
    NUMOBJ = 20
    model = ControlSignalVAE(pixelchannels=NUMOBJ+1, latentchannels=latentchannels, lamda=lamda, patchsize=patchsize)
    
    randomcolors = [randomcolor_hsv() for _ in range(NUMOBJ+1)]
    
    device = torch.device("cuda", gpu)    
    
    cocodataset = COCOPanopticDataset(maindir="/USERSPACE/lukovdg1/coco2017", split=split, upscale_to=512, max_masks=NUMOBJ, useinstances=True)
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=batsize, num_workers=10, collate_fn=collate_fn)
    
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
                    
            if global_step % saveevery == 0 and global_step > 0:
                torch.save(model.state_dict(), outpath / "ae.pth")
                torch.save(model.encoder.state_dict(), outpath / "encoder.pth")
                
                agg = accdict.aggregate()
                print(f"Epoch {epoch}/{epochs}, batch {i}/{len(dl)}: mse={agg['reconloss']:.6f}, kld={agg['kld']:.6f}")
                accdict.reset()
                
            outdict = model.training_step(x, None)
            accdict.add(outdict)
            loss = outdict["loss"]
            loss.backward()
            optimizer.step()
            global_step += 1
            
                    
        agg = accdict.aggregate()
            
        torch.save(model.state_dict(), outpath / "ae.pth")
        torch.save(model.encoder.state_dict(), outpath / "encoder.pth")
        print(f"Epoch {epoch}/{epochs}: mse={agg['reconloss']:.6f}, kld={agg['kld']:.6f}")
        
        accdict.reset()
            
            
            
    
    # trainer = pl.Trainer(max_epochs=20, devices=1, accelerator="gpu")
    # trainer.fit(model, dl)
    
    
    
    
if __name__ == "__main__":
    fire.Fire(main)