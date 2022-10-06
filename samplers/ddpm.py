from torch import sqrt
import torch
from . import utils


class DDPMSampler:
    # https://arxiv.org/abs/2006.11239

    def __init__(self, num_sample_steps=500, num_train_timesteps=1000, reverse_sample=True):

        beta = utils.get_beta_schedule(num_train_timesteps+1)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.stride = len(alpha) // num_sample_steps
        self.timesteps = torch.arange(num_train_timesteps+1)  # stable diffusion accepts discrete timestep

        # make timestep -> alpha/beta mapping explicit
        self.beta = {t.item(): beta for t, beta in zip(self.timesteps, beta)}
        self.alpha = {t.item(): alpha for t, alpha in zip(self.timesteps, alpha)}
        self.alpha_bar = {t.item(): alpha_bar for t, alpha_bar in zip(self.timesteps, alpha_bar)}

        self.timesteps = self.timesteps[::self.stride]

        if reverse_sample:  # generating samples (T) or training the model (F)
            self.timesteps = reversed(self.timesteps)[:-1]

        self.reverse_sample = reverse_sample

    def __call__(self, eps_theta, x, t):

        t = t.item()
        tprev = t - self.stride if self.reverse_sample else t + self.stride

        beta, alpha = self.beta[t], self.alpha[t]
        alpha_bar, alpha_bar_prev = self.alpha_bar[t], self.alpha_bar[tprev]

        # two alternatives for \sigma, both equally effective according to ddpm paper
        sigma = sqrt(beta)
        beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta  # eqn (7)
        # sigma = sqrt(beta_tilde)
        # algorithm 2
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x_prev = 1 / sqrt(alpha) * (x - (1 - alpha) / (1 - alpha_bar) * eps_theta) + sigma * z

        return x_prev
