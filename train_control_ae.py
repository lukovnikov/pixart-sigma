import torch
from torch import nn
from torch.utils.data import DataLoader

from cocodata import COCOPanopticDataset, COCOPanopticExample

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
            torch.nn.Conv2d(inchannels, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 96, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 96, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(96, 256, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, outchannels*2, 1, padding=0)
        )
        
    def forward(self, x):
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
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=KSIZE, padding=PAD, output_padding=OUTPAD, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, outchannels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z
    
        
class ControlSignalVAE(pl.LightningModule):
    def __init__(self, pixelchannels=3, latentchannels=4):
        super().__init__()
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
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def training_step(self, batch, batch_idx):
        x = batch["cond_image"]

        xrecons, mu, logvar = self.forward(x)
        reconloss = torch.nn.functional.mse_loss(xrecons, x, size_average=False)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())    
        loss = reconloss + kld    

        self.log_dict({
            'loss': loss.mean(),
            'kld': kld,
            'reconloss': reconloss.mean(),
        })

        return loss.mean()
    
    
def main():
    model = ControlSignalVAE()
    
    test_x = torch.randn(2, 3, 512, 512)
    test_x_recon, *_ = model(test_x)
    
    print(torch.nn.functional.mse_loss(test_x_recon, test_x))
    
    
    cocodataset = COCOPanopticDataset(maindir="/USERSPACE/lukovdg1/coco2017", split="val", upscale_to=512)
    print(len(cocodataset))
    dl = DataLoader(cocodataset, batch_size=32, num_workers=10)
    
    trainer = pl.Trainer(max_epochs=20, devices=1, accelerator="gpu")
    trainer.fit(model, dl)
    
    
    
    
if __name__ == "__main__":
    fire.Fire(main)