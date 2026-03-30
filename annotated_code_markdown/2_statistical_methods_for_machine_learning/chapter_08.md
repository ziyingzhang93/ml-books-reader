# 统计方法与机器学习
## Chapter 08

---

### Multiple Simulations

# 08.02 — Central Limit Theorem / 中心极限定理

**Chapter 08 — File 2 of 2**

## Summary / 摘要

**English:** This notebook demonstrates the Central Limit Theorem (CLT): we repeat the dice-rolling simulation 1000 times and plot a histogram of the sample means. Remarkably, despite dice rolls being uniformly distributed on {1,2,3,4,5,6}, the distribution of sample means is approximately Gaussian—a profound result underlying much of statistical inference.

**中文:** 本笔记本演示了中心极限定理(CLT)：我们重复骰子投掷模拟1000次并绘制样本均值的直方图。值得注意的是，尽管骰子投掷在{1,2,3,4,5,6}上均匀分布，样本均值的分布近似为高斯分布——这是一个深刻的结果，支撑了大量的统计推断。

**Formula / 公式:**
$$\sqrt{n}\left(\bar{X}_n - \mu\right) \xrightarrow{d} N(0, \sigma^2) \text{ as } n \to \infty$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random integer generation, mean calculation, and plotting / 导入随机整数生成、均值计算和绘图
from numpy.random import seed
from numpy.random import randint
from numpy import mean
from matplotlib import pyplot
```

## Step 2 — Set Seed / 设置种子

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)
```

## Step 3 — Run Multiple Simulations / 运行多个模拟

```python
# Perform the dice-rolling experiment 1000 times / 执行骰子投掷实验1000次
# Each experiment: roll a die 50 times, calculate the mean / 每个实验：投掷骰子50次，计算均值
# randint(1, 7, 50) generates 50 die rolls / randint(1, 7, 50)生成50次投掷
# mean() calculates the average of those 50 rolls / mean()计算这50次投掷的平均值
means = [mean(randint(1, 7, 50)) for _ in range(1000)]

print(f"Number of simulations: 1000")
print(f"Mean of sample means: {mean(means):.4f}")
print(f"Expected value (theoretical): 3.5")
```

## Step 4 — Plot Distribution of Sample Means / 绘制样本均值的分布

```python
# Create histogram of the sample means / 创建样本均值的直方图
# The Central Limit Theorem: despite dice being discrete and uniform, / 中心极限定理：尽管骰子是离散和均匀分布的，
# the means follow an approximately Gaussian (bell curve) distribution / 均值遵循近似高斯（钟形曲线）分布
pyplot.hist(means, bins=30, edgecolor='black', alpha=0.7)
pyplot.xlabel('Sample Mean / 样本均值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Central Limit Theorem: Distribution of Sample Means from 1000 Dice Experiments / 中心极限定理：1000次骰子实验的样本均值分布')
pyplot.grid(True, alpha=0.3)
pyplot.show()

print("\nNote: Despite individual die rolls being uniformly distributed on {1,2,3,4,5,6},")
print("the distribution of sample means is approximately Gaussian!")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The Central Limit Theorem states that the distribution of sample means approaches a Gaussian distribution as sample size increases, regardless of the underlying population distribution. For a population with mean $\mu$ and variance $\sigma^2$, the sample mean $\bar{X}_n$ is approximately $N(\mu, \sigma^2/n)$ for large $n$. This is one of the most powerful results in statistics: it explains why the Gaussian distribution is ubiquitous and justifies using normal-based inference (confidence intervals, hypothesis tests) even when data isn't Gaussian. The convergence is remarkably fast—even with discrete, non-Gaussian distributions like dice rolls, 50 samples suffice for approximate normality.

- **ML Application / 机器学习应用:** The CLT underpins confidence intervals and uncertainty quantification in ML. Validation curves and cross-validation errors approximately follow Gaussian distributions due to CLT—enabling confidence intervals on model performance. In federated learning and distributed systems, aggregating updates from multiple clients approximates an average that, by CLT, concentrates around the true gradient. Bayesian neural networks use Gaussian approximations to posteriors justified by CLT. Ensemble methods (bagging, boosting) benefit from CLT: averaging predictions reduces variance. Meta-learning and few-shot learning leverage CLT to predict performance from limited evaluations.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Complete**: Chapter 8 concludes the core statistical methods curriculum!

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed
from numpy.random import randint
from numpy import mean
from matplotlib import pyplot

# ===== Section 2: Set Seed =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# ===== Section 3: Run Multiple Simulations =====
# Perform 1000 dice-rolling experiments (50 rolls each) / 执行1000次骰子投掷实验（每次50次投掷）
# Record the mean of each experiment / 记录每个实验的均值
means = [mean(randint(1, 7, 50)) for _ in range(1000)]

print(f"Number of simulations: 1000")
print(f"Mean of sample means: {mean(means):.4f}")
print(f"Expected value (theoretical): 3.5")

# ===== Section 4: Visualize CLT =====
# Plot histogram of sample means / 绘制样本均值的直方图
# CLT: Distribution is approximately Gaussian despite underlying uniform distribution / CLT：分布近似为高斯尽管底层是均匀分布
pyplot.hist(means, bins=30, edgecolor='black', alpha=0.7)
pyplot.xlabel('Sample Mean / 样本均值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Central Limit Theorem: Distribution of Sample Means from 1000 Dice Experiments / 中心极限定理：1000次骰子实验的样本均值分布')
pyplot.grid(True, alpha=0.3)
pyplot.show()

print("\nNote: Despite individual die rolls being uniformly distributed on {1,2,3,4,5,6},")
print("the distribution of sample means is approximately Gaussian!")
```

---
