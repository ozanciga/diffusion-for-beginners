import torch


def get_beta_schedule(num_diffusion_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    # beta schedule, start, end are from stable diffusion config
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps) ** 2
    return betas
