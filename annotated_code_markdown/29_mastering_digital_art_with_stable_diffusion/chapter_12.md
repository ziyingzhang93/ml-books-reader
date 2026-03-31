# Stable Diffusion 数字艺术 / Digital Art with SD
## Chapter 12

---

### Cuda

# 01 — Cuda / 01 Cuda

**Chapter 12 — File 1 of 4 / 第12章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Cuda**.

本脚本演示 **01 Cuda**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from diffusers import StableDiffusionPipeline, DDPMScheduler
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---
## Learning Notes / 学习笔记

- **概念**: Cuda 是机器学习中的常用技术。  
  *Cuda is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cuda / 01 Cuda
# Complete Code / 完整代码
# ===============================

from diffusers import StableDiffusionPipeline, DDPMScheduler
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Cpu

# 02 — Cpu / 02 Cpu

**Chapter 12 — File 2 of 4 / 第12章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Cpu**.

本脚本演示 **02 Cpu**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from diffusers import StableDiffusionPipeline, DDPMScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---
## Learning Notes / 学习笔记

- **概念**: Cpu 是机器学习中的常用技术。  
  *Cpu is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cpu / 02 Cpu
# Complete Code / 完整代码
# ===============================

from diffusers import StableDiffusionPipeline, DDPMScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
prompt = "A cat took a fish and running in a market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Localmodel

# 03 — Localmodel / 03 Localmodel

**Chapter 12 — File 3 of 4 / 第12章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Localmodel**.

本脚本演示 **03 Localmodel**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from diffusers import StableDiffusionPipeline, DDPMScheduler

model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")
prompt = "A cat took a fish and running away from the market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---
## Learning Notes / 学习笔记

- **概念**: Localmodel 是机器学习中的常用技术。  
  *Localmodel is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Localmodel / 03 Localmodel
# Complete Code / 完整代码
# ===============================

from diffusers import StableDiffusionPipeline, DDPMScheduler

model = "./path/realisticVisionV60B1_v60B1VAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model)
pipe.to("cuda")
prompt = "A cat took a fish and running away from the market"
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
image = pipe(prompt, scheduler=scheduler, num_inference_steps=30, guidance_scale=7.5
            ).images[0]
image.save("cat.png")
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Scheduler

# 04 — Scheduler / 04 Scheduler

**Chapter 12 — File 4 of 4 / 第12章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Scheduler**.

本脚本演示 **04 Scheduler**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
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
```

---
## Learning Notes / 学习笔记

- **概念**: Scheduler 是机器学习中的常用技术。  
  *Scheduler is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scheduler / 04 Scheduler
# Complete Code / 完整代码
# ===============================

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
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **4 code files** demonstrating chapter 12.

本章包含 **4 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_cuda.ipynb` — Cuda
  2. `02_cpu.ipynb` — Cpu
  3. `03_localmodel.ipynb` — Localmodel
  4. `04_scheduler.ipynb` — Scheduler

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
