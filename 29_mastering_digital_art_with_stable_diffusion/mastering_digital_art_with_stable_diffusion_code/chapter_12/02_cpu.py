from diffusers import StableDiffusionPipeline, DDPMScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
