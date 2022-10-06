import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from samplers import ddpm
from PIL import Image


def sample_image(pipe, sampler, prompt, init_latents, batch_size=1, guidance_scale=8.):
    # sample image from stable diffusion
    with torch.inference_mode():
        latents = init_latents
        # prompt conditioning
        cond_input = pipe.tokenizer([prompt], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = pipe.text_encoder(cond_input.input_ids.to(pipe.device))[0]
        # unconditional (classifier free guidance)
        uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
        # concat embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        for t in tqdm(sampler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(latent_model_input, t, text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # @crowsonkb found out subtracting noise obtained from unconditioned input (empty prompt)
            # pushes the generated sample toward high density regions given prompt:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = sampler(noise_pred, latents, t)

        with torch.autocast('cuda'):
            images = pipe.vae.decode(latents * 1 / 0.18215).sample

        # tensor to image
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = np.uint8(255 * images)

        return images


if __name__ == '__main__':

    device = 'cuda'
    TOKEN = ''  # add your token here
    model_name = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=TOKEN).to(device)

    sampler = ddpm.DDPMSampler(num_sample_steps=1000)

    prompt = "a man eating an apple sitting on a bench"
    batch_size = 1
    init_latents = torch.randn(batch_size, 4, 64, 64).to(device)
    images = sample_image(pipe, sampler, prompt, init_latents)
    # plot the image
    plt.figure(); plt.imshow(images[0]); plt.show()
    # or save
    Image.fromarray(images[0]).save('images/ddpm.jpg', quality=50)

