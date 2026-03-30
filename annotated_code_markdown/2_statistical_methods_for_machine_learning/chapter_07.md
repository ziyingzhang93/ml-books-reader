# 统计方法与机器学习
## Chapter 07

---

### Gaussian

# 07.01 — Population Distribution / 总体分布

**Chapter 07 — File 1 of 2**

## Summary / 摘要

**English:** This notebook visualizes an idealized population distribution—the theoretical Gaussian N(50,5) from which samples are drawn. We plot the probability density function (PDF) to establish the "ground truth" distribution before demonstrating how sample means converge to the true population mean.

**中文:** 本笔记本可视化了理想化的总体分布——从中抽取样本的理论高斯分布N(50,5)。我们绘制概率密度函数(PDF)以建立"真实"分布，然后演示样本均值如何收敛到真实总体均值。

**Formula / 公式:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \text{ with } \mu=50, \sigma=5$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import plotting and statistical libraries / 导入绘图和统计库
from numpy import arange
from matplotlib import pyplot
from scipy.stats import norm
```

## Step 2 — Create X-axis Range / 创建X轴范围

```python
# Create x-axis from 30 to 70 with step 1 / 创建30到70的x轴，步长为1
# This covers the range from μ-4σ to μ+4σ / 这覆盖μ-4σ到μ+4σ的范围
xaxis = arange(30, 70, 1)
```

## Step 3 — Calculate PDF Values / 计算PDF值

```python
# Calculate PDF values for population distribution N(50, 5) / 计算总体分布N(50, 5)的PDF值
# norm.pdf(x, mean, std) computes probability density at each x / norm.pdf()计算每个x处的概率密度
yaxis = norm.pdf(xaxis, 50, 5)
```

## Step 4 — Plot Ideal Population Distribution / 绘制理想总体分布

```python
# Plot the theoretical population distribution / 绘制理论总体分布
pyplot.plot(xaxis, yaxis, linewidth=2)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Probability Density / 概率密度')
pyplot.title('Idealized Population Distribution N(50,5) / 理想化的总体分布 N(50,5)')
pyplot.grid(True, alpha=0.3)
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The population distribution is the theoretical distribution from which all observations are assumed to come. Unlike sample distributions (which vary due to sampling), the population distribution is fixed and describes the entire, often infinite, population. The true population mean $\mu$ is a fixed parameter (though unknown in practice). The Law of Large Numbers states that as sample size increases, the sample mean converges to the true population mean. Visualizing the population distribution provides "ground truth" context for understanding how samples deviate from and eventually approximate the population.

- **ML Application / 机器学习应用:** In machine learning, the true data distribution is often unknown, and we estimate it from samples. Model evaluation assumes training data is drawn from the same distribution as test data. Distribution shift (covariate shift, label shift) occurs when training and test come from different distributions, degrading model performance. Understanding the population distribution's role motivates regularization, domain adaptation, and robust learning techniques. Synthetic data generation aims to approximate the unknown population distribution for training data augmentation.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `02_estimate_mean.ipynb` — Demonstrate convergence of sample means to population mean

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy import arange
from matplotlib import pyplot
from scipy.stats import norm

# ===== Section 2: Set Up X-axis =====
# Create x-axis covering mean ± 4 standard deviations / 创建覆盖均值±4个标准差的x轴
xaxis = arange(30, 70, 1)

# ===== Section 3: Calculate Population PDF =====
# Compute PDF values for N(50, 5) / 计算N(50, 5)的PDF值
yaxis = norm.pdf(xaxis, 50, 5)

# ===== Section 4: Plot Ideal Population Distribution =====
# Visualize the theoretical population distribution / 可视化理论总体分布
pyplot.plot(xaxis, yaxis, linewidth=2)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Probability Density / 概率密度')
pyplot.title('Idealized Population Distribution N(50,5) / 理想化的总体分布 N(50,5)')
pyplot.grid(True, alpha=0.3)
pyplot.show()
```

---
