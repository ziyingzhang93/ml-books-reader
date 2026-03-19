from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")
prompt = "A cat took a fish and running away from the market"
scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                   beta_schedule="scaled_linear")
image = pipe(
    prompt,
    scheduler=scheduler,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("cat.png")
