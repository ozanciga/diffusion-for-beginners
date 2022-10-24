import torch
import numpy as np


class ExponentialSampler:
    # https://arxiv.org/abs/2204.13902

    def __init__(
            self,
            denoiser,
            num_sample_steps=50,
            num_train_timesteps=1000,
            r=3,
    ):
        self.r = r
        self.denoiser = denoiser

        self.num_train_timesteps, self.num_sample_steps = num_train_timesteps, num_sample_steps

        self.stride = num_train_timesteps // num_sample_steps
        self.timesteps = reversed(torch.arange(1, num_sample_steps - 1))  # see algo 1, timestep ends at 1 not zero

        self.t = np.linspace(1E-3, 1, self.num_sample_steps+r)
        self.c = self.calc_c()  # calculate \mathbf{C} based on Eq. 15, also see table 1
        self.eps_buffer = dict()

    @staticmethod
    def log_alpha(t, schedule='linear'):
        if schedule == 'linear':
            b0, b1 = 0.1, 20
            return -(b1 - b0) / 4 * t ** 2 - b0 / 2 * t
        elif schedule == 'cosine':
            s = 0.008
            return np.log(np.cos(np.pi / 2 * (t + s) / (1 + s))) - np.log(np.cos(np.pi / 2 * (s) / (1 + s)))

    def calc_c(self):

        from scipy.interpolate import interp1d
        log_alpha_fn = interp1d(
            np.linspace(0, 1+0.01, self.num_train_timesteps+1),
            self.log_alpha(np.linspace(0, 1, self.num_train_timesteps+1), schedule='linear'))

        from scipy.misc import derivative
        # dt = 0.0001
        # G = lambda tau: np.sqrt(-(log_alpha_fn(tau+dt)-log_alpha_fn(tau-dt))/(2*dt))
        G = lambda tau: np.sqrt(-derivative(log_alpha_fn, tau, dx=1e-5))
        L = lambda tau: np.sqrt(1-np.exp(log_alpha_fn(tau)))

        def prod_fn(tau, i, j, r):
            prod = 1.
            for k in range(r+1):
                prod *= (tau - self.t[i + k]) / (self.t[i + j] - self.t[i + k]) if k != j else 1
            return prod

        def get_cij(tau, i, j, r):
            # Eq. 15.  ignoring inverse and transpose since G, L are 1d
            return 1 / 2 * self.psi(self.t[i - 1], tau) * G(tau) ** 2 * 1 / L(tau) * prod_fn(tau, i, j, r)

        cij = dict()
        from scipy import integrate
        for i in range(self.num_sample_steps):
            for j in range(self.r+1):
                cij[i, j] = integrate.quad(get_cij, self.t[i], self.t[i - 1], args=(i, j, self.r), epsrel=1e-4)[0]

        return cij

    def psi(self, t, s):
        return np.sqrt(np.exp((self.log_alpha(t) - self.log_alpha(s))))  # appendix d, PROOF OF PROP 2

    def eps_theta(self, x, i, embd, guidance_scale=None):
        with torch.inference_mode():
            noise = self.denoiser(x if guidance_scale is None else torch.cat([x] * 2), i, embd).sample
            if guidance_scale is not None:
                noise_pred_uncond, noise_pred_text = noise.chunk(2)
                noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise

    def __call__(self, x_hat, i, text_embeddings, guidance_scale=8.):
        # algorithm 1.
        i = int(i)
        self.eps_buffer[self.t[i]] = self.eps_theta(x_hat, i*self.stride, text_embeddings, guidance_scale)
        x_hat_prev = self.psi(self.t[i-1], self.t[i]) * x_hat + sum(
            [self.c[i, j] * self.eps_buffer[self.t[i+j]] for j in range(self.r+1) if self.t[i+j] in self.eps_buffer]
        )  # Eq. 14
        return x_hat_prev
