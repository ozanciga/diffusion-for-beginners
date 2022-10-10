import torch
from math import sqrt as msqrt


class HeunSampler:
    # https://arxiv.org/abs/2010.02502

    def __init__(
            self,
            num_sample_steps=50,
            num_train_timesteps=1000,
            denoiser=None,
            ddpm=True,
            alpha_bar=None,
    ):
        self.denoiser = denoiser

        self.num_train_timesteps, self.num_sample_steps = num_train_timesteps, num_sample_steps

        self.ddpm_sigmas = ((1-alpha_bar)/alpha_bar).sqrt()
        self.t0 = self.ddpm_sigmas[-1]  # or sigma_max/max noise

        self.stride = len(alpha_bar) // num_sample_steps
        self.timesteps = torch.arange(num_train_timesteps+1)

        # table 1
        sigma_min, sigma_max, rho = self.ddpm_sigmas[0], self.ddpm_sigmas[-1], 7  # stable diff.
        _heun_sigmas = (sigma_max ** (1 / rho) + (self.timesteps / self.num_sample_steps) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.heun_sigmas = _heun_sigmas
        self.sigma_data = 0.5
        self.timesteps = torch.arange(num_sample_steps - 1)

        self.ddpm = ddpm  # model trained with ddpm objective, used to set c_in etc

    def c_skip(self, sigma):
        if self.ddpm:
            return torch.tensor(1.)
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        if self.ddpm:
            return -sigma
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2).rsqrt()

    def c_in(self, sigma):
        if self.ddpm:
            return (1 + sigma ** 2).rsqrt()
        return (sigma ** 2 + self.sigma_data ** 2).rsqrt()

    def c_noise(self, sigma):
        if self.ddpm:
            return torch.abs(self.ddpm_sigmas-sigma).argmin()  # iddpm practical (c.3.4: 3rd bullet)
        return 1 / 4 * sigma.log()

    def d_theta(self, x, sigma, encoder_hidden_states, guidance_scale=None):

        eps_theta = self.predict_noise(
            x if guidance_scale is None else torch.cat([x] * 2), sigma, encoder_hidden_states)
        # hacky bit: if guidance!=None, perform classifier free guidance
        # deviates from original implementation to support minimal code
        # duplication, even though it's not good practice
        if guidance_scale is not None:
            noise_pred_uncond, noise_pred_text = eps_theta.chunk(2)
            eps_theta = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return self.c_skip(sigma) * x + self.c_out(sigma) * eps_theta

    def f_theta(self, x, sigma, encoder_hidden_states):
        with torch.inference_mode():
            return self.denoiser(x, sigma, encoder_hidden_states).sample

    def predict_noise(self, x, sigma, encoder_hidden_states):
        # helper function to predict noise for stable diffusion
        # since heun modifies both input and the stdev, using
        # sd pipeline directly does not yield the correct f_theta output
        return self.f_theta(self.c_in(sigma) * x, self.c_noise(sigma), encoder_hidden_states)

    def stochastic_sampler(self, x, i, encoder_hidden_states, s_churn=80, s_tmin=0.05, s_tmax=50, s_noise=1.003):
        # Algorithm 2 - arg defaults from table 5
        x_next, x_hat, t_next, t_hat, d = self.euler_method(
            x, i, encoder_hidden_states, s_churn, s_tmin, s_tmax, s_noise)  # first order denoising
        x_next = self.heuns_correction(x_next, x_hat, t_next, t_hat, d, encoder_hidden_states)  # second order correction
        return x_next

    def euler_method(self, x, i, encoder_hidden_states=None, guidance_scale=None,  s_churn=80, s_tmin=0.05, s_tmax=50, s_noise=1.003):
        # euler method, omitting second order correction @ lines 9-11 (algorithm 2)
        t, t_next = self.heun_sigmas[i], self.heun_sigmas[i+1]

        eps = torch.randn_like(x) * s_noise
        gamma = min(s_churn / self.num_sample_steps, msqrt(2) - 1) if s_tmin <= t <= s_tmax else 0
        t_hat = t + gamma * t
        if self.ddpm:
            t_hat = self.ddpm_sigmas[torch.abs(self.ddpm_sigmas-t_hat).argmin()]  # iddpm practical, step 3
        x_hat = x + msqrt(max(t_hat ** 2 - t ** 2, 0)) * eps  # gamma < 0 -> negative sqrt
        d = (x_hat - self.d_theta(x_hat, t_hat, encoder_hidden_states, guidance_scale)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d

        return x_next, x_hat, t_next, t_hat, d

    def heuns_correction(self, x_next, x_hat, t_next, t_hat, d, encoder_hidden_states, guidance_scale=None):
        # correction improves differentiation discretization error from o(h^2) -> o(h^3), h = step size
        # correction addresses varying step sizes between two different timesteps
        # see Discretization and higher-order integrators from the paper
        if t_next != 0:
            d_prime = (x_next - self.d_theta(x_next, t_next, encoder_hidden_states, guidance_scale)) / t_next
            x_next = x_hat + (t_next - t_hat) * (1 / 2 * d + 1 / 2 * d_prime)
        return x_next

    def alpha_sampler(self, x, i, encoder_hidden_states, guidance_scale=None, alpha=1):
        # Algorithm 3
        t, t_next = self.heun_sigmas[i], self.heun_sigmas[i+1]
        h = t_next - t
        d = (x - self.d_theta(x, t, encoder_hidden_states, guidance_scale)) / t
        x_prime, t_prime = x + alpha * h * d, t + alpha * h
        if t_prime != 0:
            d_prime = (x_prime - self.d_theta(x_prime, t_prime, encoder_hidden_states, guidance_scale)) / t_prime
            x_next = x + h * ((1 - 0.5 / alpha) * d + 0.5 / alpha * d_prime)
        else:
            x_next = x + h*d

        return x_next

    def __call__(self, *args):
        return self.stochastic_sampler(*args)
