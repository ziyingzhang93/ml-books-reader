from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info
import torch

# LoRA weights ~3 MB
model_path = "pcuenq/pokemon-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("Green pokemon with menacing face", num_inference_steps=25).images[0]
image.save("green_pokemon.png")
