# 统计方法与机器学习
## Chapter 25

---

### Sample Size Small

```python
# 25 — Non-Gaussian Data: Small Sample / 非高斯数据：小样本

**Chapter 25 — File 1 of 8**

## Summary / 摘要

Small sample sizes (n < 30) from a normal population may appear non-Gaussian visually due to sampling variability, not true distributional difference. With n=10, random fluctuations create histogram irregularities—some bins are empty, others have multiple observations. This demonstrates an important principle: apparent non-normality may reflect sampling variability rather than true population non-normality. A histogram of n=10 observations from normal distribution often looks decidedly non-normal. This motivates using statistical tests rather than visual inspection alone, and illustrates why larger samples are needed for reliable distributional assessment.

正态总体中的小样本量(n < 30)可能由于采样变异性而不是真正的分布差异而在视觉上显示为非高斯。对于n=10，随机波动产生直方图不规则性——某些箱子为空，其他的有多个观察。这演示了一个重要原则：明显的非正态性可能反映采样变异性，而不是真正的总体非正态性。来自正态分布的n=10个观察的直方图通常看起来明确是非正态的。这激发了使用统计测试而不是仅视觉检查，并说明了为什么需要更大的样本进行可靠的分布评估。
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import and Generate Small Sample / 导入和生成小样本

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 10
data = np.random.normal(50, 15, n)

print(f"Small sample (n={n}): {data}")
print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data, ddof=1):.2f}")
```

## Step 2 — Histogram and Normality Tests / 直方图和正态性测试

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(data, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Histogram of Small Sample (n={n})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_data = np.sort(data)
axes[1].plot(sorted_data, np.arange(1, n+1)/n, 'bo-', linewidth=2, markersize=6, label='Empirical')
x_line = np.linspace(sorted_data.min()-10, sorted_data.max()+10, 100)
axes[1].plot(x_line, stats.norm.cdf(x_line, np.mean(data), np.std(data, ddof=1)), 'r-', linewidth=2, label='Normal fit')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Empirical vs Theoretical CDF')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tests
stat, p = stats.shapiro(data)
print(f"\nShapiro-Wilk test: W={stat:.4f}, p={p:.4f}")
print(f"With small n, non-normality is hard to detect statistically")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Small sample size inflates sampling variability. The central limit theorem guarantees sample means are normally distributed regardless of underlying distribution, but individual observations show high variability. Histograms with small n have large bin-to-bin variation. Statistical tests lose power with small samples—they cannot detect moderate deviations from normality due to high variability. Rules of thumb suggest n ≥ 30 for reliable visual and statistical assessment.
  
  **统计概念**: 小样本大小增加了采样变异性。中心极限定理保证样本均值无论基础分布如何都是正态分布，但单个观察显示高变异性。带有小n的直方图具有大的箱间变化。由于高变异性，统计测试失去了功效——它们无法检测到正态性的中等偏差。经验法则建议n ≥ 30以进行可靠的视觉和统计评估。

- **ML Application**: In practice, practitioners should: (1) avoid over-interpreting non-normality in small samples, (2) combine tests with visual inspection, (3) use robust methods tolerant of distributional assumptions when sample size is small. For small-sample ML applications, non-parametric methods (random forests, kernel methods) or Bayesian approaches with informative priors often outperform parametric methods sensitive to distributional assumptions.
```

➡️ **Next**: `02_sample_size_large.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
data = np.random.normal(50, 15, 10)

plt.hist(data, bins=5, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Small Sample Histogram (n=10)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Sample Size Large

```python
# 25 — Non-Gaussian Data: Large Sample / 非高斯数据：大样本

**Chapter 25 — File 2 of 8**

## Summary / 摘要

With n=100 observations from the same normal population, the histogram shows a clear bell-shaped distribution, eliminating ambiguity from sampling variability. The larger sample size reveals the true underlying distribution. This contrasts with n=10, demonstrating how sample size affects reliability of visual and statistical assessment. Larger samples provide stable estimates of distribution parameters and reduce false signals from sampling variability. The central limit theorem predicts sample statistics stabilize as n increases, making large samples essential for reliable distributional characterization in practice.

对于来自相同正态总体的n=100个观察，直方图显示明确的钟形分布，消除了采样变异性的歧义。较大的样本大小揭示了真正的基础分布。这与n=10形成对比，演示了样本大小如何影响视觉和统计评估的可靠性。较大的样本提供了分布参数的稳定估计，并减少了采样变异性的虚假信号。中心极限定理预测样本统计随着n增加而稳定，使大样本对于实践中可靠的分布表征至关重要。
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


## Step 1 — Generate Large Sample / 生成大样本

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 100
data = np.random.normal(50, 15, n)

print(f"Large sample (n={n})")
print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data, ddof=1):.2f}")
```

## Step 2 — Histogram with Normal Fit / 带正态拟合的直方图

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with overlay
axes[0].hist(data, bins=20, density=True, alpha=0.7, edgecolor='black', color='skyblue')
x = np.linspace(data.min()-20, data.max()+20, 100)
axes[0].plot(x, stats.norm.pdf(x, np.mean(data), np.std(data, ddof=1)), 'r-', linewidth=2, label='Normal fit')
axes[0].axvline(np.mean(data), color='green', linestyle='--', linewidth=2, label='Mean')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Large Sample Histogram (n={n})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

stat, p = stats.shapiro(data)
print(f"\nShapiro-Wilk: W={stat:.4f}, p={p:.4f}")
print(f"Result: {'Normal' if p > 0.05 else 'Non-normal'}")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Larger samples provide more stable distribution estimates. Sample variance converges to population variance as n → ∞ by the law of large numbers. Distribution shape becomes clear with n ≥ 100: histograms are smooth, visual assessment reliable, and statistical tests have adequate power. The trade-off: large samples also increase sensitivity to detect trivial departures from normality (highly significant but practically negligible deviations).
  
  **统计概念**: 较大的样本提供更稳定的分布估计。根据大数定律，随着n → ∞，样本方差收敛到总体方差。当n ≥ 100时，分布形状变得清晰：直方图平滑，视觉评估可靠，统计测试具有足够的功效。权衡：大样本也增加了检测正态性的微小偏离的敏感性（高度显著但实际上可忽略的偏差）。

- **ML Application**: Large samples enable confident distributional decisions. However, statistical significance ≠ practical significance. With n > 1000, even tiny departures from normality trigger rejection of H₀, yet may not affect model performance. Practitioners should distinguish: (1) True non-normality requiring transformation, vs (2) Minor statistical deviations with negligible practical impact. Visual inspection + effect size assessment + domain knowledge outweighs p-value alone for decision-making.
```

➡️ **Next**: `03_data_resolution.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.normal(50, 15, 100)

plt.hist(data, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Large Sample Histogram (n=100)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Data Resolution

# 25 — Non-Gaussian Data: Data Resolution / 非高斯数据：数据分辨率

**Chapter 25 — File 3 of 8**

## Summary / 摘要

Data rounding to integer values creates artificial clustering and discrete distribution. Rounding inherent Gaussian noise produces visible non-normality: histogram shows spikes at integer values instead of smooth bell curve. This represents a real source of non-Gaussian data in practice—measurement devices with limited precision, questionnaires with integer scales, or data post-processing that discretizes. Understanding data resolution is critical: the apparent non-normality may reflect measurement artifact rather than underlying population distribution, affecting choice of statistical methods.

数据四舍五入到整数值会造成人为聚类和离散分布。四舍五入固有高斯噪声会产生明显的非正态性：直方图在整数值处显示尖峰，而不是平滑的钟形曲线。这代表了实践中真实的非高斯数据来源——具有有限精度的测量设备、具有整数比例的问卷或离散化数据的后处理。理解数据分辨率至关重要：明显的非正态性可能反映了测量的伪影，而不是基础总体分布，影响统计方法的选择。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate and Round Data / 生成和四舍五入数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate continuous normal data
data_continuous = np.random.normal(50, 15, 100)
# Round to integers
data_rounded = np.round(data_continuous)

print(f"Continuous sample: {data_continuous[:10]}")
print(f"Rounded sample: {data_rounded[:10].astype(int)}")
```

## Step 2 — Compare Distributions / 比较分布

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Continuous
axes[0].hist(data_continuous, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Continuous Data (Original)')
axes[0].grid(True, alpha=0.3)

# Rounded
axes[1].hist(data_rounded, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
axes[1].set_xlabel('Value (Integers)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Rounded to Integers')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test both
stat1, p1 = stats.shapiro(data_continuous)
stat2, p2 = stats.shapiro(data_rounded)

print(f"Shapiro-Wilk - Continuous: W={stat1:.4f}, p={p1:.4f}")
print(f"Shapiro-Wilk - Rounded: W={stat2:.4f}, p={p2:.4f}")
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Discretization and rounding introduce discrete jumps in the data, violating continuity assumptions. The discrete uniform distribution replaces smooth normal curve. With many unique values, impact is minimal; with coarse rounding (e.g., to nearest 10), impact is severe. Fractional data within rounding error contributes noise that creates artificial clustering patterns distinct from true non-normality.

- **ML Application**: Recognize measurement limitations in data. Options: (1) Use original continuous data if available (higher information content), (2) Apply distributional assumptions to latent continuous variables when discrete data result from thresholding, (3) Analyze discrete data with appropriate methods (ordinal regression, categorical models). Many ML algorithms perform comparably with discretized vs continuous inputs, but interpretability may suffer due to information loss.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

➡️ **Next**: `04_extreme_events.ipynb`

## Complete Code / 完整代码一览

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data_continuous = np.random.normal(50, 15, 100)
data_rounded = np.round(data_continuous)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(data_continuous, bins=20, alpha=0.7)
ax1.set_title('Continuous')
ax2.hist(data_rounded, bins=20, alpha=0.7)
ax2.set_title('Rounded to Integers')
plt.show()
```

---

### Extreme Events

# 25 — Non-Gaussian Data: Extreme Events / 非高斯数据：极端事件

**Chapter 25 — File 4 of 8**

## Summary / 摘要

Outliers and extreme values create heavy tails and positive kurtosis. Appending zeros to normally distributed data introduces a point mass at zero, producing bimodal appearance and non-normal distribution. This represents real-world phenomena: zero-inflated data (e.g., number of purchases by customer), missing events, censoring. Such data violates normality assumptions and requires specialized methods (zero-inflated models, robust statistics, truncated distributions).

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate Data / 生成数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate data for this notebook
print("Data generation code here")
```

## Step 2 — Analyze Distribution / 分析分布

```python
# Create histogram
plt.hist([1,2,3], bins=5)
plt.title('Distribution')
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Description here.

- **ML Application**: Application here.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next**: `05_long_tail.ipynb`

## Complete Code / 完整代码一览

```python
# Complete code example here
```

---

### Long Tail

# 25 — Non-Gaussian Data: Long Tail / 非高斯数据：长尾

**Chapter 25 — File 5 of 8**

## Summary / 摘要

Right-skewed or log-normal distributions have long right tails (values extending far from mean with low density). Income, wealth, web traffic follow power laws with long tails. Such data show positive skewness (mean > median), excess kurtosis (fat tails), and violate normality. Log transformation often stabilizes variance and reduces skewness. Lognormal distribution y = exp(X) where X ~ N(μ,σ²) provides natural model for positive-valued right-skewed data.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate Data / 生成数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate data for this notebook
print("Data generation code here")
```

## Step 2 — Analyze Distribution / 分析分布

```python
# Create histogram
plt.hist([1,2,3], bins=5)
plt.title('Distribution')
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Description here.

- **ML Application**: Application here.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next**: `06_long_tail_truncated.ipynb`

## Complete Code / 完整代码一览

```python
# Complete code example here
```

---

### Long Tail Truncated

# 25 — Non-Gaussian Data: Truncated Long Tail / 非高斯数据：截断长尾

**Chapter 25 — File 6 of 8**

## Summary / 摘要

Truncating extreme values (removing values > threshold) transforms right-skewed to more normal distribution. Removing upper outliers reduces skewness and kurtosis, enabling standard methods. Real-world example: winsorizing (capping extreme values at percentile) or trimming. Shows how data cleaning and outlier treatment affect distributional properties. Demonstrates trade-off: removes information but improves model assumptions and robustness.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate Data / 生成数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate data for this notebook
print("Data generation code here")
```

## Step 2 — Analyze Distribution / 分析分布

```python
# Create histogram
plt.hist([1,2,3], bins=5)
plt.title('Distribution')
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Description here.

- **ML Application**: Application here.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next**: `07_exponential.ipynb`

## Complete Code / 完整代码一览

```python
# Complete code example here
```

---

### Exponential

# 25 — Non-Gaussian Data: Exponential Distribution / 非高斯数据：指数分布

**Chapter 25 — File 7 of 8**

## Summary / 摘要

Exponential distribution (waiting times, lifetimes) is strongly right-skewed and non-Gaussian. Log transformation y = log(X) converts exponential to normal. Box-Cox family of transformations generalizes this: y = (X^λ - 1)/λ for optimal λ chosen to maximize likelihood. Lognormal (λ ≈ 0) works well for exponential data. Demonstrates that many non-Gaussian distributions can be normalized through monotonic transformations, recovering normal assumptions.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate Data / 生成数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Generate data for this notebook
print("Data generation code here")
```

## Step 2 — Analyze Distribution / 分析分布

```python
# Create histogram
plt.hist([1,2,3], bins=5)
plt.title('Distribution')
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Description here.

- **ML Application**: Application here.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |

➡️ **Next**: `08_boxcox.ipynb`

## Complete Code / 完整代码一览

```python
# Complete code example here
```

---
