import torch
import numpy as np


class DPMSampler:
    # https://arxiv.org/abs/2206.00927

    def __init__(self, denoiser, num_sample_steps=20, num_train_timesteps=1000, schedule='linear'):

        self.schedule = schedule
        T = 1 if self.schedule == 'linear' else 0.9946

        # discrete to continuous- section 3.4, [0,1]->[0,N]
        self.denoiser = denoiser  # lambda x, t, embd: denoiser(x, int(num_train_timesteps * t), embd).sample

        # A simple way for choosing time steps for lambda is uniformly
        # splitting [lambda_t , lambda_eps], which is the setting in our experiments
        (_, _, lmbd_max), (_, _, lmbd_min) = self.lmbd(T), self.lmbd(1E-3)
        lmbds = torch.linspace(lmbd_max, lmbd_min, num_train_timesteps)
        self.timesteps = torch.tensor([self.t_lambda(l, self.schedule) for l in lmbds])

        self.stride = num_train_timesteps // num_sample_steps
        self.timesteps = self.timesteps[::self.stride]
        self.num_train_timesteps = num_train_timesteps

    def eps_theta(self, x, t, embd, guidance_scale=None):
        t = int(max(0, t - 1 / self.num_train_timesteps) * self.num_train_timesteps)
        with torch.inference_mode():
            noise = self.denoiser(x if guidance_scale is None else torch.cat([x] * 2), t, embd).sample
            if guidance_scale is not None:
                noise_pred_uncond, noise_pred_text = noise.chunk(2)
                noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise

    @staticmethod
    def _alpha(t, s=0.008):
        return np.exp(np.log(np.cos(np.pi / 2 * (t + s) / (1 + s))) -
                         np.log(np.cos(np.pi / 2 * (s) / (1 + s))))

    @staticmethod
    def _lmbd(alpha, sigma):
        return np.log(alpha) - np.log(sigma)

    # @staticmethod
    def _sigma(self, alpha):
        # return 0.5 * np.log(1. - np.exp(2. * self.log_alpha(t)))
        return np.sqrt(1 - alpha ** 2)

    @staticmethod
    def t_lambda(lmbd, schedule='linear'):
        # appendix d.4
        if schedule == 'linear':
            b0, b1 = 0.1, 20.
            nom = 2 * np.logaddexp(-2 * lmbd, 0)  # numerically stable log(e**x+e**y)
            denom = np.sqrt(b0 ** 2 + 2 * (b1 - b0) * np.logaddexp(-2 * lmbd, 0)) + b0
            return nom / denom
        elif schedule == 'cosine':
            s = 0.008
            f_lambda = -1 / 2 * np.logaddexp(-2 * lmbd, 0)
            logcos = np.log(np.cos((np.pi * s) / (2 * (1 + s))))
            return 2 * (1 + s) / np.pi * np.arccos(np.exp(f_lambda + logcos)) - s

    def lmbd(self, t):
        log_alpha = self.log_alpha(t, self.schedule)
        # log_sigma = 0.5 * np.log(1. - np.exp(2. * log_alpha))
        sigma = np.sqrt(1. - np.exp(2 * log_alpha))
        log_sigma = np.log(sigma)
        return log_alpha, sigma, (log_alpha-log_sigma)

    def log_alpha(self, t, schedule):
        if schedule == 'linear':
            b0, b1 = 0.1, 20
            return -(b1-b0)/4 * t ** 2 - b0/2*t
        elif schedule == 'cosine':
            s = 0.008
            return np.log(np.cos(np.pi/2*(t+s)/(1+s))) - np.log(np.cos(np.pi/2*(s)/(1+s)))

    def dpm_solver_2(self, x_tilde_prev, t_prev, t, text_embeddings, guidance_scale=None, r1=0.5):
        # algorithm 4
        # numerically _not_ stable
        '''alpha_t, alpha_t_prev = self._alpha(t), self._alpha(t_prev)
        sigma_t, sigma_t_prev = self._sigma(alpha_t), self._sigma(alpha_t_prev)
        lmbd_t, lmbd_t_prev = self._lmbd(alpha_t, sigma_t), self._lmbd(alpha_t_prev, sigma_t_prev)'''

        # numerically stable
        (log_alpha_t, sigma_t, lmbd_t), (log_alpha_t_prev, sigma_t_prev, lmbd_t_prev) = self.lmbd(t), self.lmbd(t_prev)
        h = lmbd_t - lmbd_t_prev
        s = self.t_lambda(lmbd_t_prev + r1 * h, self.schedule)
        log_alpha_s, sigma_s, lmbd_s = self.lmbd(s)

        eps_prev = self.eps_theta(x_tilde_prev, t_prev, text_embeddings, guidance_scale)
        u = np.exp(log_alpha_s - log_alpha_t_prev) * x_tilde_prev - sigma_s * np.expm1(r1 * h) * eps_prev

        x_tilde = (
                + np.exp(log_alpha_t - log_alpha_t_prev) * x_tilde_prev
                - sigma_t * np.expm1(h) * eps_prev
                - sigma_t / (2 * r1) * np.expm1(h) * (self.eps_theta(u, s, text_embeddings, guidance_scale) - eps_prev)
        )
        return x_tilde

    def __call__(self, x, t, text_embeddings, guidance_scale=8.):

        if t == self.timesteps[-1]:
            return x

        t_prev = t.item()
        prev_index = torch.where(self.timesteps < t)[0][0]
        t = self.timesteps[prev_index].item()

        return self.dpm_solver_2(x, t_prev, t, text_embeddings, guidance_scale)
