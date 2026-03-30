# 概率论与机器学习
## Chapter 22

---

### Distributions

# 22 — Divergence Measures / 发散测度

**Chapter 22 — File 1 of 5**

## Summary / 汇总

This notebook plots two probability distributions P and Q as bar charts. These distributions will be used to demonstrate KL divergence and Jensen-Shannon divergence calculations.

本笔记本将两个概率分布P和Q绘制为条形图。这些分布将用于演示KL散度和Jensen-Shannon散度计算。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Define Distributions and Plot / 定义分布并绘制

```python
from matplotlib import pyplot# define distributionsevents = ['red', 'green', 'blue']p = [0.10, 0.40, 0.50]  # Distribution Pq = [0.80, 0.15, 0.05]  # Distribution Qprint('P=%.3f Q=%.3f' % (sum(p), sum(q)))# plot first distribution (P)pyplot.subplot(2, 1, 1)pyplot.bar(events, p)# plot second distribution (Q)pyplot.subplot(2, 1, 2)pyplot.bar(events, q)# show the plotpyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Probability distributions assign mass to outcomes. P has uniform spread while Q concentrates mass on 'red'. Divergence measures quantify how different these distributions are. **概念**: 概率分布将质量分配给结果。P具有均匀分布，而Q在'red'上集中质量。散度测度量化这些分布的不同程度。

- **ML Application**: In machine learning, measuring distribution divergence is essential for comparing learned distributions to targets (GANs), model compression, and domain adaptation. **机器学习应用**: 在机器学习中，测量分布散度对于将学习分布与目标进行比较(GANs)、模型压缩和域适应至关重要。

➡️ **Next**: `02_kl_divergence.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

## Complete Code / 完整代码一览

```python
from matplotlib import pyplotevents = ['red', 'green', 'blue']p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]print('P=%.3f Q=%.3f' % (sum(p), sum(q)))pyplot.subplot(2,1,1)pyplot.bar(events, p)pyplot.subplot(2,1,2)pyplot.bar(events, q)pyplot.show()
```

---

### Kl Divergence

# 22 — Divergence Measures / 发散测度

**Chapter 22 — File 2 of 5**

## Summary / 汇总

This notebook calculates KL divergence from scratch. KL(P||Q) = Σ p*log₂(p/q) measures how different Q is from P when P is the reference distribution.

本笔记本从头开始计算KL散度。KL(P||Q) = Σ p*log₂(p/q)衡量当P是参考分布时Q与P的不同程度。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Define KL Divergence Function and Calculate / 定义KL散度函数并计算

```python
from math import log2# calculate the kl divergence (forward and reverse)def kl_divergence(p, q):    # KL(P||Q) = sum(p[i] * log2(p[i]/q[i]))    # Sum over all outcomes    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))# define distributionsp = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]# calculate (P || Q): How different is Q from the perspective of Pkl_pq = kl_divergence(p, q)print('KL(P || Q): %.3f bits' % kl_pq)# calculate (Q || P): How different is P from the perspective of Qkl_qp = kl_divergence(q, p)print('KL(Q || P): %.3f bits' % kl_qp)
```

## Learning Notes / 学习笔记

- **Concept**: KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P). It measures the information lost when Q approximates P. Forward KL(P||Q) penalizes missing modes; reverse KL(Q||P) penalizes extra modes. **概念**: KL散度是非对称的: KL(P||Q) ≠ KL(Q||P)。它衡量当Q近似P时丢失的信息。前向KL(P||Q)惩罚缺失模式；反向KL(Q||P)惩罚额外模式。

- **ML Application**: In variational autoencoders (VAE), forward KL prevents posterior collapse. In maximum likelihood estimation, reverse KL appears naturally. Understanding the asymmetry is crucial for choosing objectives. **机器学习应用**: 在变分自编码器(VAE)中，前向KL防止后验崩溃。在最大似然估计中，反向KL自然出现。理解非对称性对于选择目标至关重要。

➡️ **Next**: `03_kl_divergence_scipy.ipynb`

## Complete Code / 完整代码一览

```python
from math import log2def kl_divergence(p, q):    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]kl_pq = kl_divergence(p, q)print('KL(P || Q): %.3f bits' % kl_pq)kl_qp = kl_divergence(q, p)print('KL(Q || P): %.3f bits' % kl_qp)
```

---

### Kl Divergence Scipy

# 22 — Divergence Measures / 发散测度

**Chapter 22 — File 3 of 5**

## Summary / 汇总

This notebook calculates KL divergence using scipy.special.rel_entr for numerical stability. The implementation handles edge cases like zero probabilities.

本笔记本使用scipy.special.rel_entr计算KL散度以实现数值稳定性。该实现处理零概率等边界情况。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate KL Divergence Using SciPy / 使用SciPy计算KL散度

```python
from scipy.special import rel_entr# define distributionsp = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]# calculate (P || Q) using scipy's numerically stable implementation# rel_entr returns p*log(p/q) in nats (natural logarithm)kl_pq = rel_entr(p, q)print('KL(P || Q): %.3f nats' % sum(kl_pq))# calculate (Q || P)kl_qp = rel_entr(q, p)print('KL(Q || P): %.3f nats' % sum(kl_qp))
```

## Learning Notes / 学习笔记

- **Concept**: scipy.special.rel_entr uses numerically stable logarithm operations and automatically handles edge cases (p=0, q=0). It returns values in nats (natural log) rather than bits. **概念**: scipy.special.rel_entr使用数值稳定的对数操作，并自动处理边界情况(p=0, q=0)。它以纳特(自然对数)而非比特返回值。

- **ML Application**: The stable implementation is crucial for machine learning where distributions may have near-zero probabilities. TensorFlow/PyTorch use similar approaches internally. **机器学习应用**: 稳定的实现对于机器学习至关重要，其中分布可能具有接近零的概率。TensorFlow/PyTorch内部使用类似的方法。

➡️ **Next**: `04_js_divergence.ipynb`

## Complete Code / 完整代码一览

```python
from scipy.special import rel_entrp = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]kl_pq = rel_entr(p, q)print('KL(P || Q): %.3f nats' % sum(kl_pq))kl_qp = rel_entr(q, p)print('KL(Q || P): %.3f nats' % sum(kl_qp))
```

---

### Js Divergence Scipy

# 22 — Divergence Measures / 发散测度

**Chapter 22 — File 5 of 5**

## Summary / 汇总

This notebook calculates Jensen-Shannon distance using scipy.spatial.distance.jensenshannon. It provides the symmetric distance metric with built-in numerical stability.

本笔记本使用scipy.spatial.distance.jensenshannon计算Jensen-Shannon距离。它提供具有内置数值稳定性的对称距离度量。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Jensen-Shannon Distance Using SciPy / 使用SciPy计算Jensen-Shannon距离

```python
from scipy.spatial.distance import jensenshannonfrom numpy import asarray# define distributionsp = asarray([0.10, 0.40, 0.50])q = asarray([0.80, 0.15, 0.05])# calculate JS distance (base=2 for bits)js_pq = jensenshannon(p, q, base=2)print('JS(P || Q) Distance: %.3f' % js_pq)# calculate JS distance (symmetric, so same result)js_qp = jensenshannon(q, p, base=2)print('JS(Q || P) Distance: %.3f' % js_qp)
```

## Learning Notes / 学习笔记

- **Concept**: scipy's implementation returns the square root of JS divergence, which is a proper metric. The base parameter allows flexible logarithm bases. **概念**: scipy的实现返回JS散度的平方根，这是一个真正的度量。基数参数允许灵活的对数基数。

- **ML Application**: JSD is preferred over KL for comparing distributions in production due to symmetry and stability. It's used in distribution matching, clustering, and model comparison. **机器学习应用**: 由于对称性和稳定性，在生产中比较分布时，JSD优于KL。它用于分布匹配、聚类和模型比较。

➡️ **Next**: `../chapter_23/01_distributions.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Complete Code / 完整代码一览

```python
from scipy.spatial.distance import jensenshannonfrom numpy import asarrayp = asarray([0.10, 0.40, 0.50])q = asarray([0.80, 0.15, 0.05])js_pq = jensenshannon(p, q, base=2)print('JS(P || Q) Distance: %.3f' % js_pq)js_qp = jensenshannon(q, p, base=2)print('JS(Q || P) Distance: %.3f' % js_qp)
```

---

### Chapter Summary

# Chapter 22: Divergence Measures

## Overview
This chapter explores **divergence measures**: metrics for quantifying how different two probability distributions are. The journey progresses from visualization to asymmetric (KL) to symmetric (JS) measures.

## Key Concepts
- **Divergence**: Non-negative measure of dissimilarity between distributions
- **Kullback-Leibler (KL) Divergence**: Asymmetric information-theoretic measure
- **Jensen-Shannon (JS) Divergence**: Symmetric alternative to KL
- **Properties**: What makes a good divergence measure

## Evolution of Examples

### From Visualization to Computation
1. **01_plot_distributions.py**: Visualize two distributions to compare
2. **02_kl_divergence_from_scratch.py**: Implement KL divergence manually
3. **03_kl_divergence_scipy.py**: Use scipy for KL calculation
4. **04_js_divergence_from_scratch.py**: Implement JS divergence manually
5. **05_js_divergence_scipy.py**: Use scipy for JS calculation

## Kullback-Leibler (KL) Divergence

### Definition
```
D_KL(P || Q) = Σₓ P(x) × log(P(x) / Q(x))
              = Σₓ P(x) × [log(P(x)) - log(Q(x))]
```
- P: True distribution (reference)
- Q: Approximation/model distribution
- D_KL ≥ 0, D_KL = 0 iff P = Q

### Interpretation
**Expected surprise of coding P using Q**

If we use an optimal code for distribution Q but data comes from P:
- KL divergence: average extra bits wasted
- Measures inefficiency of approximation Q for true distribution P

### Example
```
Fair coin (P): p(H)=0.5, p(T)=0.5
Biased coin (Q): p(H)=0.3, p(T)=0.7

D_KL(P || Q) = 0.5×log(0.5/0.3) + 0.5×log(0.5/0.7)
             ≈ 0.5×0.737 + 0.5×(-0.470)
             ≈ 0.133 bits
```
On average, coding fair coin data with biased coin code wastes 0.133 bits.

## Properties of KL Divergence

### Non-Negative
```
D_KL(P || Q) ≥ 0
```
Equality only when P = Q everywhere.

### Not Symmetric
```
D_KL(P || Q) ≠ D_KL(Q || P)  [usually]
```

Example:
- D_KL(P || Q): penalizes Q being too peaky where P has mass
- D_KL(Q || P): penalizes Q having mass where P has little

### Not a Distance
- Not symmetric (distance requirement violated)
- Triangle inequality doesn't hold
- Called "divergence", not "distance"

### Forward vs Reverse KL

**Forward KL: D_KL(P || Q)**
- Penalizes Q missing modes of P
- Q under-covers true distribution
- Used in: expectation propagation

**Reverse KL: D_KL(Q || P)**
- Penalizes Q having modes where P doesn't
- Q over-covers true distribution
- Used in: variational inference, GANs

## Jensen-Shannon (JS) Divergence

### Definition
```
D_JS(P || Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)
```
Where M = (P + Q) / 2 (mixture distribution)

Or equivalently:
```
D_JS(P || Q) = sqrt(D_KL(P || M) × D_KL(Q || M))
```

### Symmetry
```
D_JS(P || Q) = D_JS(Q || P)
```
Symmetric!

### Bounded
```
0 ≤ D_JS(P || Q) ≤ log(2) ≈ 0.693 bits
```
KL can be arbitrarily large; JS is bounded.

### Interpretation
**Average KL divergence from each distribution to their mixture**

Balances both directions:
- Neither over-covers nor under-covers
- "Fair" comparison

## Comparison: KL vs JS

| Property | KL | JS |
|----------|----|---------|
| Symmetric | No | Yes |
| Bounded | No | Yes (log 2) |
| Interpretation | Coding inefficiency | Average mutual divergence |
| Use case | Forward/reverse optimization | Symmetric comparison |
| Computational cost | Fast | Medium (two KLs) |
| Zero offset issues | Can be undefined | Never undefined |

## Use Cases

### KL Divergence
- **Variational Inference**: Minimize reverse KL for tractable approximation
- **Information Bottleneck**: Compress while preserving information
- **GANs**: Forward KL encourages mode-covering generator
- **Model Selection**: Compare model fits

### JS Divergence
- **Symmetric Comparison**: When no clear reference distribution
- **Distribution Matching**: GAN loss alternatives
- **Clustering**: Compare point clouds
- **Hypothesis Testing**: Asymptotic statistical test

## Zero Probability Issue

### The Problem
If Q(x) = 0 but P(x) > 0:
- KL: log(P(x) / 0) = ∞
- JS: Well-defined (uses mixture)

### Solutions
1. **Add smoothing**: Q(x) ← Q(x) + ε
2. **Renormalize**: Restrict to support of Q
3. **Use JS**: Symmetric measure avoids this

## Key Takeaways
1. KL divergence measures information loss from using Q instead of P
2. KL is asymmetric: forward vs reverse have different penalties
3. JS divergence fixes asymmetry, providing symmetric comparison
4. KL is unbounded; JS is bounded [0, log 2]
5. Both are used in modern ML: variational inference, GANs, optimization
6. Choose based on whether direction matters (KL) or not (JS)

---
