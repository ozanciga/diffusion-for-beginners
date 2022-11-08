import torch
import numpy as np
import utils


class DPMPlusPlusSampler:
    # https://arxiv.org/abs/2211.01095

    def __init__(self, denoiser, num_sample_steps=20, num_train_timesteps=1000):

        self.timesteps = np.arange(1, num_train_timesteps+1)
        self.t = self.timesteps / num_train_timesteps

        beta = utils.get_beta_schedule(num_train_timesteps+1)
        self.alpha = {t: v for t, v in zip(self.t, torch.cumprod(1 - beta, dim=0).sqrt())}

        self.q = dict()

        self.denoiser = denoiser

        self.stride = int(num_train_timesteps / num_sample_steps)
        self.timesteps = self.timesteps[::self.stride][::-1]
        self.t = self.t[::-1]
        self.num_train_timesteps = num_train_timesteps

    def x_theta(self, x, t, embd, guidance_scale=None):
        # data prediction model
        alpha_t, sigma_t, _ = self.get_coeffs(t)
        timestep = int(max(0, t) * self.num_train_timesteps)
        with torch.inference_mode():
            noise = self.denoiser(x if guidance_scale is None else torch.cat([x] * 2), timestep, embd).sample
            if guidance_scale is not None:
                noise_pred_uncond, noise_pred_text = noise.chunk(2)
                noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            x_pred = (x - sigma_t * noise) / alpha_t  # Parameterization: noise prediction and data prediction
            # print(t, timestep, x_pred.max(), noise.max(), sigma_t, alpha_t)
            return x_pred

    def get_coeffs(self, t):
        # appendix b.1 for alphas, sigmas, lambdas
        '''# below is used for continuous t, uncomment & modify for your needs if you want to
        t_next = t + 1 / self.num_train_timesteps
        t_n, t_n_next = sampler.t[np.digitize(t, sampler.t)], sampler.t[np.digitize(t+1/self.num_train_timesteps, sampler.t)]
        t_n_next = t_n + self.stride
        log_alpha_n, log_alpha_n_next = self.alpha[t].log(), self.alpha[t_next].log()
        log_alpha_t = log_alpha_n + (log_alpha_n_next-log_alpha_n)/(t_n_next-t_n) * (t-t_n)'''
        log_alpha_t = self.alpha[t].log()  # comment this out if above is uncommented
        alpha_t = log_alpha_t.exp()
        sigma_t = (1-alpha_t ** 2).sqrt()
        log_sigma_t = sigma_t.log()
        lambda_t = log_alpha_t - log_sigma_t
        return alpha_t, sigma_t, lambda_t

    def dpm_solver_plus_plus_2m(self, x_tilde_prev, t_prev_prev, t_prev, t, text_embeddings, guidance_scale=None):
        # algorithm 2
        alpha_t, sigma_t, lambda_t = self.get_coeffs(t)
        _, sigma_t_prev, lambda_t_prev = self.get_coeffs(t_prev)
        _, _, lambda_t_prev_prev = self.get_coeffs(t_prev_prev)
        h = lambda_t - lambda_t_prev
        h_prev = lambda_t_prev - lambda_t_prev_prev

        r = h_prev / h
        D = (1 + 1 / (2 * r)) * self.q[t_prev] - 1 / (2 * r) * self.q[t_prev_prev]
        x_tilde = sigma_t / sigma_t_prev * x_tilde_prev - alpha_t * torch.expm1(-h) * D
        self.q[t] = self.x_theta(x_tilde, t, text_embeddings, guidance_scale)

        return x_tilde

    def __call__(self, x, timestep_index, text_embeddings, guidance_scale=8.):

        # the notation used in this paper is really confusing
        # gaussian noise input gets an index of M which maps
        # to t_0, i.e., reversed. so
        # t0->t1->t2 is M->M-1->M-2. in other words t_prev > t

        t = timestep_index.item() / self.num_train_timesteps
        t_prev = (timestep_index.item() + self.stride) / self.num_train_timesteps
        t_prev_prev = (timestep_index.item() + 2 * self.stride) / self.num_train_timesteps

        if timestep_index == self.timesteps[0]:
            # first step (xt0)
            self.q[t] = self.x_theta(x, t, text_embeddings, guidance_scale)
            return x  # typo in text, x0 in algorithm 2

        elif timestep_index == self.timesteps[1]:
            # second step (xt1)
            alpha_t, sigma_t, lambda_t = self.get_coeffs(t)
            _, sigma_t_prev, lambda_t_prev = self.get_coeffs(t_prev)
            h = lambda_t - lambda_t_prev
            x_tilde = sigma_t / sigma_t_prev * x - alpha_t * torch.expm1(-h) * self.q[t_prev]
            self.q[t] = self.x_theta(x_tilde, t, text_embeddings, guidance_scale)
            return x_tilde

        # 2 to M loop
        return self.dpm_solver_plus_plus_2m(x, t_prev_prev, t_prev, t, text_embeddings, guidance_scale)
