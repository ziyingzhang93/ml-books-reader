# Stable Diffusion 数字艺术 / Digital Art with SD
## Chapter 15

---

### Check Modules

# 02 — Check Modules / 02 Check Modules

**Chapter 15 — File 1 of 3 / 第15章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Check Modules**.

本脚本演示 **02 Check Modules**。

---
## Step 1 — Step 1

```python
import wandb
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image
from huggingface_hub import model_info
```

---
## Learning Notes / 学习笔记

- **概念**: Check Modules 是机器学习中的常用技术。  
  *Check Modules is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Check Modules / 02 Check Modules
# Complete Code / 完整代码
# ===============================

import wandb
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image
from huggingface_hub import model_info
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Load Lora

# 06 — Load Lora / 06 Load Lora

**Chapter 15 — File 2 of 3 / 第15章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **LoRA weights ~3 MB**.

本脚本演示 **LoRA weights ~3 MB**。

---
## Step 1 — Step 1

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info
import torch
```

---
## Step 2 — LoRA weights ~3 MB

```python
model_path = "pcuenq/pokemon-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("Green pokemon with menacing face", num_inference_steps=25).images[0]
image.save("green_pokemon.png")
```

---
## Learning Notes / 学习笔记

- **概念**: LoRA weights ~3 MB 是机器学习中的常用技术。  
  *LoRA weights ~3 MB is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Lora / 06 Load Lora
# Complete Code / 完整代码
# ===============================

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
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Using Lora

# 07 — Using Lora / 07 Using Lora

**Chapter 15 — File 3 of 3 / 第15章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Using Lora**.

本脚本演示 **07 Using Lora**。

---
## Step 1 — Step 1

```python
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     torch_dtype=torch.float16
                                                    ).to("cuda")
pipeline.load_lora_weights("finetune_lora/pokemon",
                           weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A pokemon with blue eyes").images[0]
image.save("blue_pokemon.png")
```

---
## Learning Notes / 学习笔记

- **概念**: Using Lora 是机器学习中的常用技术。  
  *Using Lora is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Using Lora / 07 Using Lora
# Complete Code / 完整代码
# ===============================

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                     torch_dtype=torch.float16
                                                    ).to("cuda")
pipeline.load_lora_weights("finetune_lora/pokemon",
                           weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A pokemon with blue eyes").images[0]
image.save("blue_pokemon.png")
```

---

### Chapter Summary / 章节总结

# Chapter 15 Summary / 第15章总结

## Theme / 主题: Chapter 15 / Chapter 15

This chapter contains **3 code files** demonstrating chapter 15.

本章包含 **3 个代码文件**，演示Chapter 15。

---
## Evolution / 演化路线

  1. `02_check_modules.ipynb` — Check Modules
  2. `06_load_lora.ipynb` — Load Lora
  3. `07_using_lora.ipynb` — Using Lora

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 15) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 15）是机器学习流水线中的基础构建块。

---
