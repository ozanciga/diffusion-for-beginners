from torch import sqrt
import torch
from . import utils


class DDIMSampler:
    # https://arxiv.org/abs/2010.02502

    def __init__(self, num_sample_steps=50, num_train_timesteps=1000, reverse_sample=True):
        beta = utils.get_beta_schedule(num_train_timesteps+1)
        # alpha in ddim == alpha^bar in ddpm = \prod_0^t (1-beta)
        self.alpha = torch.cumprod(1 - beta, dim=0)

        self.stride = len(self.alpha) // num_sample_steps
        self.timesteps = torch.arange(num_train_timesteps+1)  # stable diffusion accepts discrete timestep
        # make timestep -> alpha mapping explicit to avoid confusion with different sampling steps
        self.alpha = {t.item(): alpha for t, alpha in zip(self.timesteps, self.alpha)}

        self.timesteps = self.timesteps[::self.stride]

        if reverse_sample:  # generating samples (T) or training the model (F)
            self.timesteps = reversed(self.timesteps)[:-1]

        self.reverse_sample = reverse_sample

    def __call__(self, eps_theta, x, t, eta=0):

        t = t.item()
        tprev = t - self.stride if self.reverse_sample else t + self.stride

        alpha, alpha_prev = self.alpha[t], self.alpha[tprev]

        eps = torch.randn_like(x)

        # Eqn. 16
        sigma_tau = (eta * sqrt((1 - alpha_prev) / (1 - alpha)) * sqrt(1 - alpha / alpha_prev)) if eta > 0 else 0
        # sigma_tau interpolates between ddim and ddpm by assigning (0=ddim, 1=ddpm)
        # ddim (song et al.) is notationally simpler than ddpm (ho et al.)
        # (although ddim uses \alpha to refer to \bar{alpha} from ddpm)

        # Eqn. 12
        predicted_x0 = (x - sqrt(1 - alpha) * eps_theta) / sqrt(alpha)
        dp_xt = sqrt(1 - alpha_prev - sigma_tau ** 2)  # direction pointing to xt
        x_prev = sqrt(alpha_prev) * predicted_x0 + dp_xt * eps_theta + sigma_tau * eps

        return x_prev
