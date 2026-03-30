# 统计方法与机器学习
## Chapter 04

---

### Gaussian

# 04.01 — Gaussian Distribution / 高斯分布

**Chapter 04 — File 1 of 8**

## Summary / 摘要

**English:** This notebook introduces the Gaussian (Normal) distribution, one of the most fundamental probability distributions in statistics and machine learning. We visualize the idealized probability density function (PDF) of a standard normal distribution N(0,1).

**中文:** 本笔记本介绍了高斯（正态）分布，这是统计学和机器学习中最基础的概率分布之一。我们可视化了标准正态分布 N(0,1) 的理想概率密度函数（PDF）。

**Formula / 公式:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import numerical and visualization libraries / 导入数值和可视化库
from numpy import arange
from matplotlib import pyplot
from scipy.stats import norm
```

## Step 2 — Create X-axis Range / 创建X轴范围

```python
# Generate x-values from -3 to 3 with step size 0.001 / 生成-3到3的x值，步长为0.001
x_axis = arange(-3, 3, 0.001)
```

## Step 3 — Calculate PDF Values / 计算PDF值

```python
# Calculate PDF values for standard normal distribution N(0,1) / 计算标准正态分布的PDF值
# norm.pdf(x, mean=0, std=1) computes probability density at each x / 计算每个x处的概率密度
y_axis = norm.pdf(x_axis, 0, 1)
```

## Step 4 — Plot Gaussian Distribution / 绘制高斯分布

```python
# Create a line plot showing the bell curve / 创建显示钟形曲线的线图
pyplot.plot(x_axis, y_axis)
# Add labels for clarity / 添加标签以便清晰
pyplot.xlabel('Value / 值')
pyplot.ylabel('Probability Density / 概率密度')
pyplot.title('Standard Normal Distribution N(0,1) / 标准正态分布 N(0,1)')
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The Gaussian distribution is characterized by its bell shape and is completely defined by two parameters: mean ($\mu$) and standard deviation ($\sigma$). In this case, we use the standard normal distribution where $\mu = 0$ and $\sigma = 1$. Most real-world data approximately follows this distribution.

- **ML Application / 机器学习应用:** Gaussian distributions are fundamental assumptions in many ML algorithms including linear regression, logistic regression, and Naive Bayes classifiers. The normal distribution's properties make it mathematically tractable for optimization and inference tasks.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `02_dataset.ipynb` — Generate Gaussian samples and create histogram

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy import arange
from matplotlib import pyplot
from scipy.stats import norm

# ===== Section 2: Gaussian Distribution Visualization =====
# Generate x-values from -3 to 3 with high resolution / 生成高分辨率的x值
x_axis = arange(-3, 3, 0.001)

# Calculate PDF for standard normal distribution N(0,1) / 计算标准正态分布的PDF
y_axis = norm.pdf(x_axis, 0, 1)

# Plot the bell curve / 绘制钟形曲线
pyplot.plot(x_axis, y_axis)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Probability Density / 概率密度')
pyplot.title('Standard Normal Distribution N(0,1) / 标准正态分布 N(0,1)')
pyplot.show()
```

---

### Dataset

# 04.02 — Gaussian Sample Dataset / 高斯样本数据集

**Chapter 04 — File 2 of 8**

## Summary / 摘要

**English:** This notebook generates a large sample (10,000 data points) from a Gaussian distribution with mean $\mu=50$ and standard deviation $\sigma=5$. We visualize the resulting histogram to compare real data against the theoretical distribution.

**中文:** 本笔记本从均值 $\mu=50$、标准差 $\sigma=5$ 的高斯分布中生成大样本（10000个数据点）。我们可视化生成的直方图，以将真实数据与理论分布进行比较。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random number generation and plotting libraries / 导入随机数生成和绘图库
from numpy.random import seed, randn
from matplotlib import pyplot
```

## Step 2 — Set Random Seed / 设置随机种子

```python
# Set seed for reproducibility / 设置种子以保证可重现性
# Using seed=1 ensures we get the same random numbers every time / 使用seed=1确保每次得到相同的随机数
seed(1)
```

## Step 3 — Generate Gaussian Samples / 生成高斯样本

```python
# Generate 10000 samples from N(0,1) standard normal / 从标准正态分布N(0,1)生成10000个样本
# Multiply by 5 for std dev, add 50 for mean: N(50, 5) / 乘以5得到标准差，加50得到均值：N(50, 5)
data = 5 * randn(10000) + 50
```

## Step 4 — Plot Histogram / 绘制直方图

```python
# Create histogram to visualize the distribution of generated data / 创建直方图以可视化生成数据的分布
# Default 10 bins; we'll see a rough bell shape / 默认10个箱子；我们会看到粗糙的钟形
pyplot.hist(data)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Histogram of Gaussian Sample N(50,5) / 高斯样本直方图 N(50,5)')
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** Sampling from a theoretical distribution is a core technique in statistics. By setting a seed, we ensure reproducibility—critical for scientific work. The random number generator produces values from N(0,1), which we transform to N($\mu$, $\sigma$) using linear transformation: $X = \sigma Z + \mu$ where $Z \sim N(0,1)$.

- **ML Application / 机器学习应用:** In machine learning, synthetic data generation is used for testing, validation, and in some cases augmentation. Understanding how to generate realistic data distributions helps in algorithm development, simulation studies, and evaluating model robustness under different data conditions.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `03_dataset_more_bins.ipynb` — Refine histogram with more bins for better resolution

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed, randn
from matplotlib import pyplot

# ===== Section 2: Generate Gaussian Samples =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50

# ===== Section 3: Visualize Distribution =====
# Plot histogram with default 10 bins / 用默认10个箱子绘制直方图
pyplot.hist(data)
pyplot.xlabel('Value / 值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Histogram of Gaussian Sample N(50,5) / 高斯样本直方图 N(50,5)')
pyplot.show()
```

---

### Mean

# 04.04 — Sample Mean / 样本均值

**Chapter 04 — File 4 of 8**

## Summary / 摘要

**English:** This notebook calculates the mean (average) of a Gaussian sample. The sample mean $\bar{x}$ is a central tendency measure that estimates the population mean $\mu$. With 10,000 samples from N(50,5), we expect $\bar{x} \approx 50$.

**中文:** 本笔记本计算高斯样本的均值（平均值）。样本均值 $\bar{x}$ 是一个中心趋势测度，用于估计总体均值 $\mu$。从N(50,5)的10000个样本中，我们期望 $\bar{x} \approx 50$。

**Formula / 公式:**
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import numerical operations and random generation / 导入数值运算和随机生成
from numpy import mean
from numpy.random import seed, randn
```

## Step 2 — Generate Gaussian Sample / 生成高斯样本

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50
```

## Step 3 — Calculate Mean / 计算均值

```python
# Calculate the sample mean / 计算样本均值
# numpy.mean() computes the arithmetic average / numpy.mean()计算算术平均值
result = mean(data)

# Print the result with 3 decimal places / 用3位小数打印结果
print('Mean: %.3f' % result)
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The sample mean is an unbiased estimator of the population mean. As sample size increases, the sample mean converges to the true population mean (Law of Large Numbers). The distribution of the sample mean itself is Gaussian with mean $\mu$ and standard deviation $\sigma/\sqrt{n}$, where $n$ is the sample size. This is the foundation of confidence intervals and hypothesis testing.

- **ML Application / 机器学习应用:** In machine learning, the mean is used as a baseline statistic for feature normalization, bias evaluation, and model interpretation. Comparing predicted means to actual means helps diagnose systematic bias in regression models. Many algorithms (like linear regression) directly optimize the mean squared error, making understanding the mean critical to model performance.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `05_median.ipynb` — Calculate and compare median to mean

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy import mean
from numpy.random import seed, randn

# ===== Section 2: Generate Sample Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50

# ===== Section 3: Calculate Mean =====
# Compute sample mean as estimator of population mean / 计算样本均值作为总体均值的估计
result = mean(data)

# Display the result / 显示结果
print('Mean: %.3f' % result)
```

---

### Median

# 04.05 — Sample Median / 样本中位数

**Chapter 04 — File 5 of 8**

## Summary / 摘要

**English:** This notebook calculates the median of a Gaussian sample. The median is the middle value when data is sorted—a robust measure of central tendency less affected by outliers than the mean. For a symmetric distribution like the Gaussian, mean and median coincide.

**中文:** 本笔记本计算高斯样本的中位数。中位数是排序数据中的中间值，是一个鲁棒的中心趋势测度，比均值受异常值影响更小。对于像高斯分布这样的对称分布，均值和中位数重合。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import median calculation and random generation / 导入中位数计算和随机生成
from numpy import median
from numpy.random import seed, randn
```

## Step 2 — Generate Gaussian Sample / 生成高斯样本

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50
```

## Step 3 — Calculate Median / 计算中位数

```python
# Calculate the median (middle value when sorted) / 计算中位数（排序后的中间值）
# For even-sized arrays, median is average of two middle elements / 对于偶数大小的数组，中位数是两个中间元素的平均值
result = median(data)

# Print the result with 3 decimal places / 用3位小数打印结果
print('Median: %.3f' % result)
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The median is the 50th percentile of a distribution. It divides the data into two equal halves. Unlike the mean, the median is robust to outliers—a single extreme value doesn't shift the median much. For symmetric distributions like Gaussian, median $\approx$ mean. For skewed distributions (e.g., income), median is often preferred as a central tendency measure. The median is the value $m$ such that $P(X \leq m) = 0.5$.

- **ML Application / 机器学习应用:** In feature engineering and outlier detection, comparing mean and median reveals data skewness. If mean >> median, the distribution has right tail outliers; if mean << median, left tail outliers exist. This diagnostic is crucial for choosing preprocessing strategies and for robust regression methods that minimize absolute error (L1 norm) rather than squared error (L2 norm).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `06_variance_plots.ipynb` — Visualize impact of variance on distribution shape

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy import median
from numpy.random import seed, randn

# ===== Section 2: Generate Sample Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50

# ===== Section 3: Calculate Median =====
# Compute the robust central tendency measure / 计算鲁棒的中心趋势测度
result = median(data)

# Display the result / 显示结果
print('Median: %.3f' % result)
```

---

### Variance

# 04.07 — Sample Variance / 样本方差

**Chapter 04 — File 7 of 8**

## Summary / 摘要

**English:** This notebook calculates the sample variance, which quantifies the spread of data around the mean. Sample variance $s^2$ is an unbiased estimator of population variance $\sigma^2$. For our N(50,5) sample, we expect variance $\approx 25$ (since $\sigma^2 = 5^2$).

**中文:** 本笔记本计算样本方差，量化数据围绕均值的扩展。样本方差 $s^2$ 是总体方差 $\sigma^2$ 的无偏估计。对于我们的N(50,5)样本，我们期望方差 $\approx 25$（因为 $\sigma^2 = 5^2$）。

**Formula / 公式:**
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import variance calculation and random generation / 导入方差计算和随机生成
from numpy import var
from numpy.random import seed, randn
```

## Step 2 — Generate Gaussian Sample / 生成高斯样本

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50
```

## Step 3 — Calculate Variance / 计算方差

```python
# Calculate sample variance / 计算样本方差
# By default, numpy.var() divides by n (population variance) / 默认情况下，numpy.var()除以n（总体方差）
# For unbiased estimate, we'd divide by (n-1), but this is close enough with large n / 对于无偏估计，我们除以(n-1)，但大n时已足够接近
result = var(data)

# Print the result with 3 decimal places / 用3位小数打印结果
print('Variance: %.3f' % result)
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** Variance is the expected squared deviation from the mean: $\sigma^2 = E[(X - \mu)^2]$. The sample variance $s^2$ estimates this using observed data. An important detail: with $n$ observations, we divide by $(n-1)$ rather than $n$ to get an unbiased estimator—this is Bessel's correction. The variance is measured in squared units, which is why standard deviation $\sigma = \sqrt{\sigma^2}$ is often preferred for interpretation (it's on the same scale as the data).

- **ML Application / 机器学习应用:** Variance estimation drives model selection in cross-validation strategies and ensemble methods. High variance in validation metrics indicates unstable models; techniques like bootstrapping quantify this. In feature selection, high-variance features are more informative, but may lead to overfitting. Variance-stabilizing transformations (e.g., log, square root) are used in preprocessing to meet algorithm assumptions and improve model stability.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `08_stdev.ipynb` — Calculate and interpret standard deviation

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy import var
from numpy.random import seed, randn

# ===== Section 2: Generate Sample Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 10000 samples from N(50, 5) / 从N(50, 5)生成10000个样本
data = 5 * randn(10000) + 50

# ===== Section 3: Calculate Variance =====
# Compute sample variance as measure of spread / 计算样本方差作为扩展的测度
result = var(data)

# Display the result / 显示结果
print('Variance: %.3f' % result)
```

---

### Chapter Summary

# Chapter 4: Gaussian Distribution & Descriptive Statistics
# 第4章：高斯分布与描述性统计

## Theme | 主题
Understanding the bell curve through visual, generative, and quantitative lenses.
通过视觉、生成和定量角度理解钟形曲线。

## Evolution Roadmap | 演变路线图
```
Gaussian PDF (Ideal Shape)
└─ Sample Generation (Empirical Data)
   └─ Histogram with More Bins (Distribution Clarity)
      └─ Mean (Center Point)
         └─ Median (Robust Center)
            └─ Variance Plots (Spread Visualization)
               └─ Variance (Raw Spread)
                  └─ Standard Deviation (Normalized Spread)
```

## Progression Logic | 进度逻辑

### Stage 1: Visualization (Chapters 1-2)
**English:** Plot the theoretical Gaussian PDF to establish the ideal shape and parametric relationship with μ and σ.
**中文:** 绘制理论高斯PDF以建立理想形状和与μ和σ的参数关系。

### Stage 2: Generation (Chapter 3)
**English:** Generate random samples from a Gaussian distribution to create empirical data that mirrors the theoretical curve.
**中文:** 从高斯分布生成随机样本，创建镜像理论曲线的实证数据。

### Stage 3: Measurement - Center (Chapters 4-5)
**English:** Calculate mean (arithmetic average) and median (middle value) to quantify the center of the distribution.
**中文:** 计算均值（算术平均）和中位数（中间值）以量化分布的中心。

### Stage 4: Measurement - Spread (Chapters 6-8)
**English:** Compute variance (squared deviations) and standard deviation (square root of variance) to measure distribution width.
**中文:** 计算方差（偏差平方）和标准差（方差的平方根）以测量分布宽度。

## ML Relevance | ML相关性

1. **Gaussian Assumption (高斯假设)**: Many statistical tests and algorithms assume normality.
2. **Feature Scaling (特征缩放)**: Mean and std dev are used to standardize features (z-score normalization).
3. **Data Profiling (数据分析)**: Mean/median reveal central tendency; variance/stdev reveal data spread and outliers.
4. **Probabilistic Models (概率模型)**: Gaussian distributions form the basis of Naive Bayes, Gaussian Mixture Models, and linear regression residuals.


---
