from torch import sqrt
import torch
from . import utils


class PNDMSampler:
    # https://arxiv.org/abs/2202.09778

    def __init__(self, num_sample_steps=50, num_train_timesteps=1000, reverse_sample=True, ddim=False):

        beta = utils.get_beta_schedule(num_train_timesteps+1)
        alpha = 1 - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)

        self.stride = len(self.alpha_bar) // num_sample_steps
        self.timesteps = torch.arange(num_train_timesteps+1)  # stable diffusion accepts discrete timestep
        self.alpha_bar = {t.item(): alpha for t, alpha in zip(self.timesteps, self.alpha_bar)}

        self.timesteps = self.timesteps[::self.stride]

        if reverse_sample:  # generating samples (T) or training the model (F)
            self.timesteps = reversed(self.timesteps)[:-1]

        self.reverse_sample = reverse_sample

        self.ddim = ddim  # alg. 1 from the paper

        self.et = []

    def prk(self, xt, et, t, tpd):
        '''
        # reuse et 4 times (from the paper):
        we find that the linear multi-step method can reuse
        the result of \eps four times and only compute \eps
        once at every step.
        ---
        so we can just return e_t for each \eps_{\theta}(.,.) call
        '''
        def eps_theta(*args):
            return et
        delta = tpd - t
        tpd2 = t + delta/2  # t plus d/2
        # Eqn 13
        et1 = eps_theta(xt, t)
        xt1 = self.phi(xt, et1, t, tpd2)
        et2 = eps_theta(xt1, tpd2)
        xt2 = self.phi(xt, et2, t, tpd2)
        et3 = eps_theta(xt2, tpd2)
        xt3 = self.phi(xt, et3, t, tpd)  # paper has a typo here
        et4 = eps_theta(xt3, tpd)
        et_d = 1/6 * (et1 + 2*et2 + 2*et3 + et4)   # e_t'
        xtpd = self.phi(xt, et_d, t, tpd)  # another typo, this was defined as xtmd
        return xtpd, et1

    def plms(self, xt, et, t, tpd):
        # Eqn 12
        etm3d, etm2d, etmd = self.et[-3:]
        et_d = 1 / 24 * (55 * et - 59 * etmd + 37 * etm2d - 9 * etm3d)  # e_t'
        xtpd = self.phi(xt, et_d, t, tpd)
        return xtpd, et

    def add_to_error_queue(self, item):
        self.et.append(item)
        if len(self.et) > 3:  # no need to store > 3 e_t (see eqn 12)
            del self.et[0]

    def phi(self, xt, et, t, t_prev):
        # i.e., the transfer part (eqn 11), t_prev = t-\delta
        alpha_bar, alpha_bar_prev = self.alpha_bar[t], self.alpha_bar[t_prev]

        transfer = sqrt(alpha_bar_prev)/sqrt(alpha_bar) * xt  # first term in the eqn.
        # second term of the eqn, split into nominator&denom for clarity:
        nom = alpha_bar_prev-alpha_bar
        denom = sqrt(alpha_bar) * (sqrt((1-alpha_bar_prev)*alpha_bar) + sqrt((1-alpha_bar)*alpha_bar_prev))
        transfer = transfer - nom/denom*et

        return transfer

    def __call__(self, eps_theta, xtp1, t):
        t = t.item()
        tpd = t - self.stride  # \delta = -1 in algorithm 2 (pndms) by inspection
        if self.ddim:  # algorithm 1
            return self.phi(xtp1, eps_theta, t, tpd)
        else:  # algorithm 2
            if len(self.et) < 3:
                xtpd, et = self.prk(xtp1, eps_theta, t, tpd)
            else:
                xtpd, et = self.plms(xtp1, eps_theta, t, tpd)
            self.add_to_error_queue(et)

        return xtpd   # note that xtpd = x_t - 1 since d = -1
