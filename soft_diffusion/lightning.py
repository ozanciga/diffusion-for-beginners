from torch.nn import functional as F
from inspect import getfullargspec
import pytorch_lightning as pl
from torch import optim
import torch
import numpy as np
from functools import partial
import torchvision.transforms.functional as ttf
from scipy.interpolate import interp1d
import k_diffusion as K   # using crowsonkb's k-diffusion (github.com/crowsonkb/k-diffusion)


def blur(discrete_sigmas, half_kernel=80, image_size=64):
    # gaussian blur, in place of C matrix
    pad_val = int(np.ceil(((half_kernel * 2 + 1)/2 - image_size) / 2 + 1))
    padder = partial(torch.nn.functional.pad, pad=(pad_val, pad_val, pad_val, pad_val), mode='reflect')

    discretize_fn = interp1d(np.linspace(0, 1, len(discrete_sigmas)), discrete_sigmas)

    def fn(image, sigma):
        sigma = discretize_fn(sigma.item())
        image = ttf.gaussian_blur(padder(image), kernel_size=[half_kernel * 2 + 1, half_kernel * 2 + 1], sigma=sigma)
        image = ttf.center_crop(image, [image_size, image_size])
        return image
    return fn


def get_sigmas(sigma_min=0.01, sigma_max=0.10, t_clip=0.25):
    # geometric scheduling
    # roughly at 25%, max noise level, see fig 6
    r = np.exp(np.log(sigma_max / sigma_min) / t_clip) - 1  # y0*(1+r)**t = yt

    def fn(t):
        return sigma_min * (1 + r) ** t
    return fn


class LightningDiffusion(pl.LightningModule):
    def __init__(self, config_path, discrete_sigmas):
        # config path is a json file, e.g.: https://github.com/crowsonkb/k-diffusion/blob/master/configs/config_32x32_small.json
        # may need to update arguments depending on resolution, e.g.,
        # "depths": [2, 2, 4, 4],
        # "channels": [128, 256, 256, 512],
        # "self_attn_depths": [false, false, true, true],

        # see graph_wasserstein.py to get discrete sigmas for your dataset
        super().__init__()

        # Diffusion model
        config = K.config.load_config(open(config_path))
        self.phi_theta = K.config.make_model(config)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.sigma = get_sigmas()
        self.dt = 0.001  # \delta_t
        self.C = blur(discrete_sigmas)
        # gaussian blur C is a linear operator, using the fixed blur in paper - ideally this would be a matrix (C_t * x)

    def training_step(self, batch, batch_idx):
        x0 = batch[0]
        t = self.rng.draw(x0.shape[0])[:, 0].to(x0.device)  # Sample timesteps
        residual = self.sigma(t) * torch.randn_like(x0)
        x = self.C(x0, t) + residual  # eq 4
        noise_pred = self.phi_theta(x, t)  # (r_hat|x_t)
        loss = (self.sigma(t) ** -2) * F.mse_loss(self.C(residual, t), self.C(noise_pred, t), reduction='none')  # Eq 12.
        return loss.mean()

    def configure_optimizers(self):
        opt = optim.Adam(self.phi_theta.parameters(), lr=2E-4, betas=(0.9, 0.999), eps=1E-8)

        def lr_sched(step):
            return min(1, step / 5_000)  # depending on batch size, this should change
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_sched)

        opt_config = {"optimizer": opt}
        if sched is not None:
            opt_config["lr_scheduler"] = {
                "scheduler": sched,
                "interval": "step",
            }
        return opt_config

    def naive_sampler(self):
        x = self.sample_gauss()
        for t in torch.linspace(1, 0, int(1/self.dt)):
            x0_hat = self.phi_theta(x, t) + x
            eta = torch.randn_like(x)
            x = self.C(x0_hat, t - self.dt) + self.sigma(t - self.dt) * eta
        return x

    def momentum_sampler(self):
        x = self.sample_gauss()
        for t in torch.linspace(1, 0, int(1/self.dt)):
            x0_hat = self.phi_theta(x, t) + x
            y_hat = self.C(x0_hat, t)
            eta = torch.randn_like(x)
            eps_hat = y_hat - x
            z = x - ((self.sigma(t - self.dt) / self.sigma(t)) ** 2 - 1) * eps_hat + (self.sigma(t) ** 2 - self.sigma(t - self.dt) ** 2).sqrt() * eta
            y_hat_prev = self.C(x0_hat, t - self.dt)
            x = z + y_hat_prev - y_hat
        return x

    def sample_gauss(self):
        # distribution p_1 not defined in the paper, this is my best guess
        x = torch.rand((1, 3, 64, 64))
        x = self.C(x, 23.)
        return x


