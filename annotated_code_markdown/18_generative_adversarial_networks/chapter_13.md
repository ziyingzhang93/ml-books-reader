# GAN
## Chapter 13

---

### Fid Numpy

# 01 — Fid Numpy / 01 Fid Numpy

**Chapter 13 — File 1 of 3 / 第13章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of calculating the frechet inception distance**.

本脚本演示 **example of calculating the frechet inception distance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of calculating the frechet inception distance

```python
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
```

---
## Step 2 — calculate frechet inception distance

```python
def calculate_fid(act1, act2):
```

---
## Step 3 — calculate mean and covariance statistics

```python
mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
```

---
## Step 4 — calculate sum squared difference between means

```python
ssdiff = numpy.sum((mu1 - mu2)**2.0)
```

---
## Step 5 — calculate sqrt of product between cov

```python
covmean = sqrtm(sigma1.dot(sigma2))
```

---
## Step 6 — check and correct imaginary numbers from sqrt

```python
if iscomplexobj(covmean):
		covmean = covmean.real
```

---
## Step 7 — calculate score

```python
fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
```

---
## Step 8 — define two collections of activations

```python
act1 = random(10*2048)
act1 = act1.reshape((10,2048))
act2 = random(10*2048)
act2 = act2.reshape((10,2048))
```

---
## Step 9 — fid between act1 and act1

```python
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)
```

---
## Step 10 — fid between act1 and act2

```python
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)
```

---
## Learning Notes / 学习笔记

- **概念**: example of calculating the frechet inception distance 是机器学习中的常用技术。  
  *example of calculating the frechet inception distance is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fid Numpy / 01 Fid Numpy
# Complete Code / 完整代码
# ===============================

# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# define two collections of activations
act1 = random(10*2048)
act1 = act1.reshape((10,2048))
act2 = random(10*2048)
act2 = act2.reshape((10,2048))
# fid between act1 and act1
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)
# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Chapter Summary

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **3 code files** demonstrating chapter 13.

本章包含 **3 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_fid_numpy.ipynb` — Fid Numpy
  2. `02_fid_keras.ipynb` — Fid Keras
  3. `03_fid_cifar10.ipynb` — Fid Cifar10

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
