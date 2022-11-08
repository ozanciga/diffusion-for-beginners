from torch import sqrt
import torch
from . import utils


class ImprovedDDPMSampler:
    # https://arxiv.org/abs/2102.09672

    def __init__(self, num_sample_steps=500, num_train_timesteps=1000, reverse_sample=True):

        self.timesteps = torch.arange(num_train_timesteps+1)  # stable diffusion accepts discrete timestep

        T, s = num_train_timesteps+1, 0.008
        alpha_bar = torch.cos((self.timesteps/T + s)/(1 + s) * torch.pi/2) ** 2  # (17), f(0) ~ 0.99999 so plug f(t)=\bar{\alpha}
        alpha_bar_prev = torch.hstack([alpha_bar[0], alpha_bar[:-1]])
        beta = 1 - alpha_bar / alpha_bar_prev
        alpha = 1 - beta
        beta = torch.clamp(beta, -torch.inf, 0.999)

        self.stride = len(alpha) // num_sample_steps

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

        # eqn 3
        mu = alpha.rsqrt() * (x - beta * (1 - alpha_bar).rsqrt() * eps_theta)  # eq (13)  reciprocal sqrt should be more numerically stable
        # variance, should be learned but i'm just experimenting w/ plugging (15) directly w/ v=1/2
        beta_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta  # eqn (7)
        v = 0.5
        sigma_theta = torch.exp(v*beta.log() + (1-v)*beta_tilde.log()).sqrt()  # (15)

        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x_prev = mu + sigma_theta * z

        return x_prev
