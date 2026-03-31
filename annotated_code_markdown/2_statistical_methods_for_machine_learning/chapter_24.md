# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 24

---

### Dataset

# 24 — Normality Test Dataset / 正态性测试数据集

**Chapter 24 — File 1 of 6**

## Summary / 摘要

This notebook generates a dataset from a Gaussian (normal) distribution and computes basic statistics. A sample of 100 observations is drawn with mean=50 and std=15, forming the foundation for demonstrating various normality tests in subsequent notebooks. Computing mean and standard deviation provides a baseline reference; these values should be close to the theoretical parameters if the data generation is working correctly.

本笔记本从高斯（正态）分布生成数据集并计算基本统计量。从100个观察中进行样本抽取，平均值=50，标准差=15，构成了在后续笔记本中演示各种正态性检验的基础。计算均值和标准差提供了基线参考；如果数据生成工作正确，这些值应接近理论参数。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
# 生成随机数 / Generate random numbers
np.random.seed(42)
```

## Step 2 — Generate Gaussian Data / 生成高斯数据

```python
# Parameters for Gaussian distribution
# 高斯分布的参数
mu = 50  # Mean / 均值
sigma = 15  # Standard deviation / 标准差
n = 100  # Sample size / 样本大小

# Generate data from normal distribution
# 从正态分布生成数据
# 生成随机数 / Generate random numbers
data = np.random.normal(mu, sigma, n)

# Display raw data
# 显示原始数据
# 打印输出 / Print output
print(f"Generated Gaussian Data (first 20 values):")
# 打印输出 / Print output
print(data[:20])
```

## Step 3 — Calculate Descriptive Statistics / 计算描述性统计

```python
# Calculate descriptive statistics
# 计算描述性统计
# 计算均值 / Calculate mean
sample_mean = np.mean(data)
# 计算标准差 / Calculate standard deviation
sample_std = np.std(data, ddof=1)  # Use unbiased estimator (ddof=1)
                                    # 使用无偏估计（ddof=1）
sample_var = np.var(data, ddof=1)

# Additional statistics
# 额外的统计
# 求最小值 / Find minimum value
sample_min = np.min(data)
# 求最大值 / Find maximum value
sample_max = np.max(data)
sample_median = np.median(data)
sample_q25 = np.percentile(data, 25)
sample_q75 = np.percentile(data, 75)
iqr = sample_q75 - sample_q25

# Display statistics
# 显示统计
# 打印输出 / Print output
print(f"\nDescriptive Statistics:")
# 打印输出 / Print output
print(f"  Sample size: {n}")
# 打印输出 / Print output
print(f"  Mean: {sample_mean:.4f} (theoretical: {mu})")
# 打印输出 / Print output
print(f"  Std Dev: {sample_std:.4f} (theoretical: {sigma})")
# 打印输出 / Print output
print(f"  Variance: {sample_var:.4f}")
# 打印输出 / Print output
print(f"\nDistribution bounds:")
# 打印输出 / Print output
print(f"  Min: {sample_min:.4f}")
# 打印输出 / Print output
print(f"  Max: {sample_max:.4f}")
# 打印输出 / Print output
print(f"  Median: {sample_median:.4f}")
# 打印输出 / Print output
print(f"  Q25 (25th percentile): {sample_q25:.4f}")
# 打印输出 / Print output
print(f"  Q75 (75th percentile): {sample_q75:.4f}")
# 打印输出 / Print output
print(f"  IQR (Interquartile Range): {iqr:.4f}")
# 打印输出 / Print output
print(f"  Range: {sample_max - sample_min:.4f}")
```

## Step 4 — Calculate Distributional Moments / 计算分布矩

```python
# Calculate higher moments
# 计算高阶矩
from scipy import stats

# Skewness (asymmetry)
# 偏度（不对称）
skewness = stats.skew(data)

# Kurtosis (tail heaviness)
# 峰度（尾部厚重）
kurtosis = stats.kurtosis(data)  # Excess kurtosis (normal distribution = 0) / 超额峰度（正态分布=0）

# Display moment statistics
# 显示矩统计
# 打印输出 / Print output
print(f"\nDistributional Moments:")
# 打印输出 / Print output
print(f"  Skewness: {skewness:.4f}")
# 打印输出 / Print output
print(f"    Interpretation: {'Left-skewed' if skewness < -0.5 else 'Right-skewed' if skewness > 0.5 else 'Approximately symmetric'}")
# 打印输出 / Print output
print(f"  Kurtosis (excess): {kurtosis:.4f}")
# 打印输出 / Print output
print(f"    Interpretation: {'Light-tailed (platykurtic)' if kurtosis < -0.5 else 'Heavy-tailed (leptokurtic)' if kurtosis > 0.5 else 'Normal-like (mesokurtic)'}")
# 打印输出 / Print output
print(f"\nFor normal distribution: skewness ≈ 0, excess kurtosis ≈ 0")
```

## Step 5 — Summary Table / 摘要表

```python
# Create summary statistics table
# 创建摘要统计表
# 打印输出 / Print output
print(f"\nSummary Statistics Table:")
# 打印输出 / Print output
print(f"{'Statistic':25} | {'Value':>12} | {'Expected for N(50,15)':>20}")
# 打印输出 / Print output
print("-" * 62)
# 打印输出 / Print output
print(f"{'Sample Size':25} | {n:>12d} | {'-':>20}")
# 打印输出 / Print output
print(f"{'Mean':25} | {sample_mean:>12.4f} | {mu:>20.4f}")
# 打印输出 / Print output
print(f"{'Std Dev':25} | {sample_std:>12.4f} | {sigma:>20.4f}")
# 打印输出 / Print output
print(f"{'Variance':25} | {sample_var:>12.4f} | {sigma**2:>20.4f}")
# 打印输出 / Print output
print(f"{'Min':25} | {sample_min:>12.4f} | {'-':>20}")
# 打印输出 / Print output
print(f"{'Max':25} | {sample_max:>12.4f} | {'-':>20}")
# 打印输出 / Print output
print(f"{'Median':25} | {sample_median:>12.4f} | {mu:>20.4f}")
# 打印输出 / Print output
print(f"{'IQR':25} | {iqr:>12.4f} | {1.35*sigma:>20.4f}")
# 打印输出 / Print output
print(f"{'Skewness':25} | {skewness:>12.4f} | {0:>20.4f}")
# 打印输出 / Print output
print(f"{'Kurtosis (excess)':25} | {kurtosis:>12.4f} | {0:>20.4f}")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: The normal distribution is characterized by zero skewness (symmetric) and zero excess kurtosis (not heavy-tailed). Sample statistics will fluctuate around theoretical values due to sampling variability. The standard error of the mean is σ/√n, decreasing with larger samples. Comparing sample statistics to theoretical values provides initial evidence of normality before formal testing.
  
  **统计概念**: 正态分布的特征是零偏度（对称）和零超额峰度（不重尾）。由于采样变异性，样本统计会围绕理论值波动。均值的标准误差为σ/√n，随着样本量增加而减少。在正式检验之前，将样本统计与理论值进行比较提供了正态性的初步证据。

- **ML Application**: Understanding data distribution is fundamental in ML. Non-normal data affects: (1) algorithm assumptions (many algorithms assume normality), (2) statistical inference validity, (3) outlier detection (normal data has ~0.3% beyond 3σ), (4) feature scaling decisions. Reporting descriptive statistics with data enables users to assess appropriateness of downstream methods. Dataset documentation with summary statistics is essential for reproducibility and troubleshooting.
  
  **ML应用**: 理解数据分布在ML中至关重要。非正态数据影响：(1)算法假设（许多算法假设正态性），(2)统计推断的有效性，(3)异常值检测（正态数据在3σ之外有~0.3%），(4)特征缩放决策。使用数据报告描述性统计使用户能够评估下游方法的适当性。带有摘要统计的数据集文档对于可重复性和故障排除至关重要。
```

➡️ **Next**: `02_histogram.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy import stats

# 生成随机数 / Generate random numbers
np.random.seed(42)
mu, sigma, n = 50, 15, 100
# 生成随机数 / Generate random numbers
data = np.random.normal(mu, sigma, n)

# 计算均值 / Calculate mean
sample_mean = np.mean(data)
# 计算标准差 / Calculate standard deviation
sample_std = np.std(data, ddof=1)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# 打印输出 / Print output
print(f"Mean: {sample_mean:.4f}, Std: {sample_std:.4f}")
# 打印输出 / Print output
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
```

---

### Histogram

# 24 — Histogram for Normality / 正态性直方图

**Chapter 24 — File 2 of 6**

## Summary / 摘要

A histogram is a graphical method for visually inspecting the distribution shape. For normally distributed data, the histogram should show a bell-shaped curve, symmetric around the mean. This notebook generates a histogram with an overlaid theoretical normal distribution curve, allowing visual comparison. Histograms are subjective (bin choice affects appearance) but provide intuitive understanding of data distribution before formal statistical tests.

直方图是用于视觉检查分布形状的图形方法。对于正态分布的数据，直方图应显示围绕均值对称的钟形曲线。本笔记本生成一个带有覆盖的理论正态分布曲线的直方图，允许视觉比较。直方图是主观的（箱子选择影响外观），但在正式统计测试之前提供了对数据分布的直观理解。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries and Generate Data / 导入库和生成数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy import stats

# 生成随机数 / Generate random numbers
np.random.seed(42)
mu, sigma, n = 50, 15, 100
# 生成随机数 / Generate random numbers
data = np.random.normal(mu, sigma, n)

# 计算均值 / Calculate mean
sample_mean = np.mean(data)
# 计算标准差 / Calculate standard deviation
sample_std = np.std(data, ddof=1)

# 打印输出 / Print output
print(f"Data: mean={sample_mean:.2f}, std={sample_std:.2f}")
```

## Step 2 — Create Histogram with Normal Curve / 使用正态曲线创建直方图

```python
# Create histogram
# 创建直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram with normal curve
# 图1: 带有正态曲线的直方图
axes[0].hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Overlay normal distribution curve
# 覆盖正态分布曲线
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(data.min() - 2*sample_std, data.max() + 2*sample_std, 100)
normal_curve = stats.norm.pdf(x, sample_mean, sample_std)
axes[0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal fit')

axes[0].axvline(sample_mean, color='green', linestyle='--', linewidth=2, label='Mean')
axes[0].axvline(sample_median := np.median(data), color='orange', linestyle='--', linewidth=2, label='Median')

axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title('Histogram with Normal Distribution Overlay')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Different bin sizes
# 图2: 不同的箱大小
axes[1].hist(data, bins=30, alpha=0.6, color='lightgreen', edgecolor='black', label='30 bins')
axes[1].plot(x, normal_curve * (data.size / 30) * (data.max() - data.min()), 'r-', linewidth=2)
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Histogram (Alternative Binning)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# 打印输出 / Print output
print(f"Visual inspection: Histogram appears bell-shaped and symmetric")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Histograms provide visual assessment of distributional shape, symmetry, and outliers. The normal distribution appears as a bell curve. Deviations (bimodality, skewness, heavy tails) suggest non-normality. Histogram appearance depends on bin number and width—too few bins hide detail, too many create noise. Sturges' rule (k ≈ 1 + log₂(n)) provides a reasonable default.
  
  **统计概念**: 直方图提供了对分布形状、对称性和异常值的视觉评估。正态分布显示为钟形曲线。偏差（双峰、偏度、重尾）表示非正态性。直方图外观取决于箱数和宽度——箱太少隐藏细节，箱太多产生噪声。Sturges规则（k ≈ 1 + log₂(n)）提供了合理的默认值。

- **ML Application**: Exploratory data analysis starts with histograms. Skewed distributions may need transformation before modeling. Multimodal histograms suggest multiple subpopulations (potential data quality issues or need for segmentation). Histograms guide feature engineering decisions—uniform distributions indicate poor discriminative power; normal distributions suit many algorithms. Combined with other plots (violin, box), histograms support comprehensive data characterization.
```

➡️ **Next**: `03_qqplot.ipynb`

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
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy import stats

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
data = np.random.normal(50, 15, 100)

# 计算均值 / Calculate mean
sample_mean = np.mean(data)
# 计算标准差 / Calculate standard deviation
sample_std = np.std(data, ddof=1)

# 绘制直方图 / Draw histogram
plt.hist(data, bins=20, density=True, alpha=0.7, edgecolor='black')
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(data.min() - 20, data.max() + 20, 100)
# 绘制折线图 / Draw line plot
plt.plot(x, stats.norm.pdf(x, sample_mean, sample_std), 'r-', linewidth=2)
# 设置X轴标签 / Set X-axis label
plt.xlabel('Value')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Density')
# 设置图表标题 / Set chart title
plt.title('Histogram with Normal Distribution')
plt.grid(True, alpha=0.3)
# 显示图表 / Display the plot
plt.show()
```

---

### Qqplot



---

### Shapiro Wilk

```python
# 24 — Shapiro-Wilk Test / Shapiro-Wilk测试

**Chapter 24 — File 4 of 6**

## Summary / 摘要

The Shapiro-Wilk test is a formal statistical test for normality with strong power against various non-normal alternatives. The null hypothesis H₀ states the data are normally distributed. The test statistic W measures how well sample quantiles match expected normal quantiles; values close to 1 indicate normality. The p-value from scipy.stats.shapiro quantifies evidence against normality: p > 0.05 (typical significance level) suggests the data are likely normal. The Shapiro-Wilk test is widely recommended and works well for sample sizes up to 5000.

Shapiro-Wilk测试是针对正态性的正式统计测试，对各种非正态替代方案具有很强的功效。零假设H₀指出数据是正态分布的。测试统计量W测量样本分位数与预期正态分位数匹配的程度；接近1的值表示正态性。来自scipy.stats.shapiro的p值量化了反对正态性的证据：p > 0.05（典型显著性水平）表明数据可能是正态分布。Shapiro-Wilk测试被广泛推荐，适用于样本大小高达5000的数据。
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries and Generate Data / 导入库和生成数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import shapiro
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 生成随机数 / Generate random numbers
np.random.seed(42)
# Normal data
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# Non-normal data (exponential)
# 生成随机数 / Generate random numbers
data_exponential = np.random.exponential(2, 100)

# 打印输出 / Print output
print(f"Generated datasets: normal and exponential")
```

## Step 2 — Perform Shapiro-Wilk Test / 执行Shapiro-Wilk测试

```python
# Shapiro-Wilk test for normal data
# Shapiro-Wilk正态数据测试
statistic_normal, pvalue_normal = shapiro(data_normal)

# Shapiro-Wilk test for exponential data
# Shapiro-Wilk指数数据测试
statistic_exp, pvalue_exp = shapiro(data_exponential)

# Display results
# 显示结果
alpha = 0.05  # Significance level / 显著性水平

# 打印输出 / Print output
print(f"Shapiro-Wilk Test Results:")
# 打印输出 / Print output
print(f"\nNormal Data:")
# 打印输出 / Print output
print(f"  Test statistic W: {statistic_normal:.6f}")
# 打印输出 / Print output
print(f"  P-value: {pvalue_normal:.6f}")
# 打印输出 / Print output
print(f"  Conclusion: {'Fail to reject H₀ - Data appears normal' if pvalue_normal > alpha else 'Reject H₀ - Data does not appear normal'}")

# 打印输出 / Print output
print(f"\nExponential Data:")
# 打印输出 / Print output
print(f"  Test statistic W: {statistic_exp:.6f}")
# 打印输出 / Print output
print(f"  P-value: {pvalue_exp:.6f}")
# 打印输出 / Print output
print(f"  Conclusion: {'Fail to reject H₀ - Data appears normal' if pvalue_exp > alpha else 'Reject H₀ - Data does not appear normal'}")

# 打印输出 / Print output
print(f"\nInterpretation Guide:")
# 打印输出 / Print output
print(f"  H₀: Data is normally distributed")
# 打印输出 / Print output
print(f"  H₁: Data is not normally distributed")
# 打印输出 / Print output
print(f"  Significance level (α): {alpha}")
# 打印输出 / Print output
print(f"  If p-value < α: Reject H₀ (evidence of non-normality)")
# 打印输出 / Print output
print(f"  If p-value ≥ α: Fail to reject H₀ (insufficient evidence of non-normality)")
```

## Step 3 — Visualize Comparison / 可视化比较

```python
# Create comparison visualization
# 创建比较可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normal data histogram and test result
axes[0, 0].hist(data_normal, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
axes[0, 0].set_title(f'Normal Data (W={statistic_normal:.4f}, p={pvalue_normal:.4f})')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Exponential data histogram and test result
axes[0, 1].hist(data_exponential, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
axes[0, 1].set_title(f'Exponential Data (W={statistic_exp:.4f}, p={pvalue_exp:.4f})')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Test statistics comparison
axes[1, 0].bar(['Normal', 'Exponential'], [statistic_normal, statistic_exp], 
               color=['skyblue', 'lightcoral'], edgecolor='black', alpha=0.7)
axes[1, 0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Critical region boundary')
axes[1, 0].set_ylabel('Test Statistic W')
axes[1, 0].set_title('Shapiro-Wilk Test Statistic Comparison')
axes[1, 0].set_ylim([0.85, 1.0])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# P-values comparison
axes[1, 1].bar(['Normal', 'Exponential'], [pvalue_normal, pvalue_exp], 
               color=['skyblue', 'lightcoral'], edgecolor='black', alpha=0.7)
axes[1, 1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
axes[1, 1].set_ylabel('P-value')
axes[1, 1].set_title('P-value Comparison')
axes[1, 1].set_ylim([0, max(pvalue_normal, pvalue_exp) * 1.2])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: The Shapiro-Wilk statistic W ∈ [0,1] measures how closely sample quantiles match normal quantiles. W close to 1 indicates normality; small W suggests departure from normality. The test is particularly sensitive to deviations in the tails. Unlike some tests, Shapiro-Wilk is conservative for small samples and powerful for moderate-to-large samples, making it suitable for practical use across sample sizes.
  
  **统计概念**: Shapiro-Wilk统计量W ∈ [0,1]测量样本分位数与正态分位数匹配的紧密程度。W接近1表示正态性；小W表示偏离正态性。该测试对尾部偏差特别敏感。与某些测试不同，Shapiro-Wilk对小样本是保守的，对中等到大样本很强大，使其适合在各种样本大小上的实际使用。

- **ML Application**: Normality tests inform decisions about data preprocessing and algorithm selection. Non-normal data may violate assumptions of t-tests, linear regression, and ANOVA. Options include: (1) data transformation (log, Box-Cox), (2) non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis), (3) robust methods less sensitive to violations. In practice, large samples often show statistical significance even with minor deviations; practitioners combine formal tests with visual inspection and domain knowledge for practical decision-making.
```

➡️ **Next**: `05_dagostinos_test.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import shapiro

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# 生成随机数 / Generate random numbers
data_exponential = np.random.exponential(2, 100)

stat_normal, p_normal = shapiro(data_normal)
stat_exp, p_exp = shapiro(data_exponential)

# 打印输出 / Print output
print(f"Normal: W={stat_normal:.4f}, p={p_normal:.4f}")
# 打印输出 / Print output
print(f"Exponential: W={stat_exp:.4f}, p={p_exp:.4f}")
```

---

### Dagostinos Test

```python
# 24 — D'Agostino-Pearson Test / D'Agostino-Pearson测试

**Chapter 24 — File 5 of 6**

## Summary / 摘要

The D'Agostino-Pearson test combines skewness and kurtosis into a single test statistic. It decomposes non-normality into two components: (1) skewness—asymmetry of the distribution, and (2) kurtosis—heaviness of the tails. The test statistic follows a chi-squared distribution with 2 degrees of freedom. This test is particularly effective for detecting specific types of non-normality (asymmetric vs. heavy-tailed distributions). The scipy.stats.normaltest function provides the implementation, making it easy to diagnose the nature of distributional violations.

D'Agostino-Pearson测试将偏度和峰度组合成单个测试统计量。它将非正态性分解为两个分量：(1)偏度——分布的不对称性，和(2)峰度——尾部厚重。测试统计量遵循具有2个自由度的卡方分布。此测试对于检测特定类型的非正态性（非对称与重尾分布）特别有效。scipy.stats.normaltest函数提供了实现，使诊断分布违反的性质变得容易。
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries and Generate Data / 导入库和生成数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy import stats
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 生成随机数 / Generate random numbers
np.random.seed(42)
# Normal data
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# Skewed data
data_skewed = np.random.gamma(2, 2, 100)  # Right-skewed / 右偏
# Heavy-tailed data
data_heavy = np.random.standard_t(3, 100)  # Student's t / 学生t分布

# 打印输出 / Print output
print(f"Generated three datasets: normal, skewed, heavy-tailed")
```

## Step 2 — Perform D'Agostino-Pearson Test / 执行D'Agostino-Pearson测试

```python
# D'Agostino-Pearson test
# D'Agostino-Pearson测试
k2_normal, p_normal = stats.normaltest(data_normal)
k2_skewed, p_skewed = stats.normaltest(data_skewed)
k2_heavy, p_heavy = stats.normaltest(data_heavy)

# Calculate components for insight
# 计算成分以获得洞察
skew_normal = stats.skew(data_normal)
kurt_normal = stats.kurtosis(data_normal)
skew_skewed = stats.skew(data_skewed)
kurt_skewed = stats.kurtosis(data_skewed)
skew_heavy = stats.skew(data_heavy)
kurt_heavy = stats.kurtosis(data_heavy)

alpha = 0.05

# 打印输出 / Print output
print(f"D'Agostino-Pearson Test Results:")
# 打印输出 / Print output
print(f"\nNormal Data:")
# 打印输出 / Print output
print(f"  Skewness: {skew_normal:.4f}, Kurtosis: {kurt_normal:.4f}")
# 打印输出 / Print output
print(f"  Test statistic K²: {k2_normal:.4f}")
# 打印输出 / Print output
print(f"  P-value: {p_normal:.4f}")
# 打印输出 / Print output
print(f"  Conclusion: {'Normal' if p_normal > alpha else 'Non-normal'}")

# 打印输出 / Print output
print(f"\nSkewed Data:")
# 打印输出 / Print output
print(f"  Skewness: {skew_skewed:.4f}, Kurtosis: {kurt_skewed:.4f}")
# 打印输出 / Print output
print(f"  Test statistic K²: {k2_skewed:.4f}")
# 打印输出 / Print output
print(f"  P-value: {p_skewed:.6f}")
# 打印输出 / Print output
print(f"  Conclusion: {'Normal' if p_skewed > alpha else 'Non-normal (asymmetric)'}")

# 打印输出 / Print output
print(f"\nHeavy-tailed Data:")
# 打印输出 / Print output
print(f"  Skewness: {skew_heavy:.4f}, Kurtosis: {kurt_heavy:.4f}")
# 打印输出 / Print output
print(f"  Test statistic K²: {k2_heavy:.4f}")
# 打印输出 / Print output
print(f"  P-value: {p_heavy:.6f}")
# 打印输出 / Print output
print(f"  Conclusion: {'Normal' if p_heavy > alpha else 'Non-normal (heavy-tailed)'}")
```

## Step 3 — Visualize Distributions and Test Results / 可视化分布和测试结果

```python
# Create comparison visualization
# 创建比较可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Row 1: Histograms
axes[0, 0].hist(data_normal, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
axes[0, 0].set_title(f'Normal (K²={k2_normal:.2f}, p={p_normal:.4f})')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(data_skewed, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
axes[0, 1].set_title(f'Skewed (K²={k2_skewed:.2f}, p={p_skewed:.4f})')

axes[0, 2].hist(data_heavy, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
axes[0, 2].set_title(f'Heavy-tailed (K²={k2_heavy:.2f}, p={p_heavy:.4f})')

# Row 2: Skewness and Kurtosis bars
axes[1, 0].bar(['Skew', 'Kurt'], [skew_normal, kurt_normal], color=['orange', 'purple'], alpha=0.7, edgecolor='black')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_ylabel('Moment Value')
axes[1, 0].set_title('Normal Components')
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].bar(['Skew', 'Kurt'], [skew_skewed, kurt_skewed], color=['orange', 'purple'], alpha=0.7, edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].set_title('Skewed Components')
axes[1, 1].grid(True, alpha=0.3, axis='y')

axes[1, 2].bar(['Skew', 'Kurt'], [skew_heavy, kurt_heavy], color=['orange', 'purple'], alpha=0.7, edgecolor='black')
axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 2].set_title('Heavy-tailed Components')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: D'Agostino-Pearson test decomposes non-normality diagnostically. Skewness component captures asymmetry; kurtosis component captures tail behavior. K² = (skewness²/√6/n) + (kurtosis²/√24/n) combines these under chi-squared distribution. This decomposition helps identify which type of transformation might help (e.g., log transform for skewness, robust methods for heavy tails).
  
  **统计概念**: D'Agostino-Pearson测试通过诊断分解非正态性。偏度分量捕获不对称；峰度分量捕获尾部行为。K² = (skewness²/√6/n) + (kurtosis²/√24/n)在卡方分布下组合这些。此分解有助于识别哪种类型的转换可能会有所帮助（例如，对于偏度的对数转换，对于重尾的稳健方法）。

- **ML Application**: Diagnostic breakdown is invaluable for choosing remedial actions. Right-skewed data (common in financial returns, wait times) benefits from log or Box-Cox transformation. Heavy-tailed data suggests robust regression or quantile regression. The test helps practitioners tailor preprocessing specifically to address identified violations. Combined with Shapiro-Wilk and Q-Q plots, D'Agostino-Pearson provides complete diagnostic toolkit for normality assessment and actionable insights for data transformation.
```

➡️ **Next**: `06_anderson_darling.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy import stats

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# 生成随机数 / Generate random numbers
data_skewed = np.random.gamma(2, 2, 100)

k2_normal, p_normal = stats.normaltest(data_normal)
k2_skewed, p_skewed = stats.normaltest(data_skewed)

# 打印输出 / Print output
print(f"Normal: K²={k2_normal:.4f}, p={p_normal:.4f}")
# 打印输出 / Print output
print(f"Skewed: K²={k2_skewed:.4f}, p={p_skewed:.6f}")
```

---

### Anderson Darling

# 24 — Anderson-Darling Test / Anderson-Darling测试

**Chapter 24 — File 6 of 6**

## Summary / 摘要

The Anderson-Darling test measures the area between the empirical and theoretical cumulative distribution functions (CDFs), emphasizing deviations in the tails. Unlike Shapiro-Wilk and D'Agostino-Pearson which return a single p-value, Anderson-Darling provides the test statistic and critical values at multiple significance levels (25%, 10%, 5%, 2.5%, 1%). This allows practitioners to assess the strength of evidence against normality without committing to a single significance level. The scipy.stats.anderson function provides implementation, supporting multiple distributions beyond the normal.

Anderson-Darling测试测量经验和理论累积分布函数(CDF)之间的面积，强调尾部的偏差。与返回单个p值的Shapiro-Wilk和D'Agostino-Pearson不同，Anderson-Darling在多个显著性水平（25%、10%、5%、2.5%、1%）提供了测试统计量和临界值。这使实践者能够评估反对正态性证据的强度，而无需承诺单个显著性水平。scipy.stats.anderson函数提供了实现，支持超越法线的多个分布。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries and Generate Data / 导入库和生成数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import anderson
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# 生成随机数 / Generate random numbers
data_uniform = np.random.uniform(0, 100, 100)

# 打印输出 / Print output
print(f"Generated datasets: normal and uniform")
```

## Step 2 — Perform Anderson-Darling Test / 执行Anderson-Darling测试

```python
# Anderson-Darling test
# Anderson-Darling测试
result_normal = anderson(data_normal, dist='norm')
result_uniform = anderson(data_uniform, dist='norm')

# Extract results
statistic_normal = result_normal.statistic
critical_values = result_normal.critical_values
significance_levels = result_normal.significance_level

statistic_uniform = result_uniform.statistic

# 打印输出 / Print output
print(f"Anderson-Darling Test Results for Normality:")
# 打印输出 / Print output
print(f"\nNormal Data:")
# 打印输出 / Print output
print(f"  Test statistic: {statistic_normal:.6f}")
# 打印输出 / Print output
print(f"\nComparison with Critical Values:")
# 打印输出 / Print output
print(f"  {'Significance Level':>20} | {'Critical Value':>15} | {'Result':>15}")
# 打印输出 / Print output
print("-" * 55)
# 同时获取索引和值 / Get both index and value
for i, sl in enumerate(significance_levels):
    cv = critical_values[i]
    result = 'Reject H₀' if statistic_normal > cv else 'Fail to reject H₀'
    # 打印输出 / Print output
    print(f"  {sl:>19.0f}% | {cv:>15.4f} | {result:>15}")

# 打印输出 / Print output
print(f"\nUniform Data:")
# 打印输出 / Print output
print(f"  Test statistic: {statistic_uniform:.6f}")
# 同时获取索引和值 / Get both index and value
for i, sl in enumerate(significance_levels):
    cv = critical_values[i]
    result = 'Reject H₀' if statistic_uniform > cv else 'Fail to reject H₀'
    # 打印输出 / Print output
    print(f"  {sl:>19.0f}% critical value: {cv:>15.4f} → {result}")
```

## Step 3 — Visualize Test Statistics / 可视化测试统计

```python
# Create visualization
# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Test statistic vs critical values
# 图1: 测试统计与临界值
sl_labels = [f"{int(sl)}%" for sl in significance_levels]
# 生成等差数组 / Generate array with step
x_pos = np.arange(len(sl_labels))

axes[0].axhline(y=statistic_normal, color='blue', linestyle='-', linewidth=2, label='Normal data')
axes[0].axhline(y=statistic_uniform, color='red', linestyle='-', linewidth=2, label='Uniform data')
axes[0].plot(x_pos, critical_values, 'go-', linewidth=2, markersize=8, label='Critical values')

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(sl_labels)
axes[0].set_ylabel('Test Statistic / Critical Value')
axes[0].set_xlabel('Significance Level')
axes[0].set_title('Anderson-Darling Test: Statistic vs Critical Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Data distributions
# 图2: 数据分布
axes[1].hist(data_normal, bins=20, alpha=0.6, label='Normal', color='blue', edgecolor='black')
axes[1].hist(data_uniform, bins=20, alpha=0.6, label='Uniform', color='red', edgecolor='black')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Data Distributions')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Step 4 — Interpretation Guide / 解释指南

```python
# 打印输出 / Print output
print(f"Anderson-Darling Interpretation Guide:")
# 打印输出 / Print output
print(f"\nH₀: Data follows normal distribution")
# 打印输出 / Print output
print(f"H₁: Data does not follow normal distribution")
# 打印输出 / Print output
print(f"\nDecision Rule:")
# 打印输出 / Print output
print(f"  If test statistic > critical value at significance level α:")
# 打印输出 / Print output
print(f"    → Reject H₀ (evidence of non-normality)")
# 打印输出 / Print output
print(f"  If test statistic ≤ critical value at significance level α:")
# 打印输出 / Print output
print(f"    → Fail to reject H₀ (insufficient evidence of non-normality)")
# 打印输出 / Print output
print(f"\nComparative Sensitivity:")
# 打印输出 / Print output
print(f"  Anderson-Darling emphasizes tail deviations")
# 打印输出 / Print output
print(f"  More sensitive to extremes than other tests")
# 打印输出 / Print output
print(f"  Useful for assessing data for methods sensitive to tail behavior")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: The Anderson-Darling statistic is A² = -n - Σ(2i-1)[ln(F(Y_i)) + ln(1-F(Y_{n-i+1}))]/n, where F is the CDF of normal distribution and Y are ordered data. This gives more weight to tail deviations than Kolmogorov-Smirnov test. The test lacks a simple closed-form p-value but provides critical values at conventional significance levels, enabling flexible hypothesis testing.
  
  **统计概念**: Anderson-Darling统计量是A² = -n - Σ(2i-1)[ln(F(Y_i)) + ln(1-F(Y_{n-i+1}))]/n，其中F是正态分布的CDF，Y是有序数据。这给尾部偏差比Kolmogorov-Smirnov测试更多的权重。该测试缺乏简单的闭式p值，但在常规显著性水平提供临界值，使灵活的假设检验成为可能。

- **ML Application**: Anderson-Darling is preferred when tail behavior is critical (e.g., financial risk modeling, extreme value analysis). Its emphasis on tails makes it detect subtle non-normality missed by other tests. In practice, apply multiple tests (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling) with visual inspection for robust normality assessment. Test selection depends on context: Shapiro-Wilk for overall assessment, D'Agostino-Pearson for identifying violation type, Anderson-Darling for tail-sensitive applications.
```

➡️ **Next**: `../chapter_25/01_sample_size_small.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import anderson

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
data_normal = np.random.normal(50, 15, 100)
# 生成随机数 / Generate random numbers
data_uniform = np.random.uniform(0, 100, 100)

result_normal = anderson(data_normal, dist='norm')
result_uniform = anderson(data_uniform, dist='norm')

# 打印输出 / Print output
print(f"Normal: statistic={result_normal.statistic:.4f}")
# 打印输出 / Print output
print(f"Uniform: statistic={result_uniform.statistic:.4f}")
# 打印输出 / Print output
print(f"Critical values at 5%: {result_normal.critical_values[2]:.4f}")
```

---

### Chapter Summary / 章节总结

# Chapter 24: Normality Tests
# 第24章：正态性检验

## Theme | 主题
From visual to quantitative: assessing whether data deviates from Gaussian normality.
从视觉到定量：评估数据是否偏离高斯正态性。

## Evolution Roadmap | 演变路线图
```
Dataset (observations)
└─ Histogram + Theoretical Curve
   (visual assessment: shape match?)
   └─ Q-Q Plot
      (visual assessment: quantile match?)
      └─ Shapiro-Wilk Test
         (sensitive, especially good for small samples)
         └─ D'Agostino-Pearson Test
            (balanced: skewness + kurtosis)
            └─ Anderson-Darling Test
               (most stringent: heavy weight on tails)
```

## Progression Logic | 进度逻辑

### Stage 1: Visual: Histogram (视觉：直方图)
**English:** Plot histogram with overlaid Gaussian PDF. Visual inspection: does shape match bell curve? Skewness, bimodality, or heavy tails signal non-normality.
**中文:** 绘制带有叠加高斯PDF的直方图。视觉检查：形状是否与钟形曲线匹配？偏斜、双峰或重尾表示非正态性。

### Stage 2: Visual: Q-Q Plot (视觉：Q-Q图)
**English:** Plot sample quantiles vs. theoretical Gaussian quantiles. Perfect normality = straight diagonal line. Deviations indicate non-normality.
**中文:** 绘制样本分位数与理论高斯分位数。完全正态 = 直对角线。偏差表示非正态性。

### Stage 3: Shapiro-Wilk Test (Shapiro-Wilk检验)
**English:** H0: data is normal. W-statistic based on best-fit line to Q-Q plot. Very sensitive, especially powerful for small samples (n < 50).
**中文:** H0：数据正态。W统计基于Q-Q图的最佳拟合线。非常敏感，特别是对小样本(n < 50)很强大。

### Stage 4: D'Agostino-Pearson Test (D'Agostino-Pearson检验)
**English:** K^2-statistic combines skewness and excess kurtosis. Balanced: moderate sensitivity. Good for medium samples (50 < n < 5000).
**中文:** K^2统计结合偏斜和超额峰度。平衡：中等敏感性。适合中等样本(50 < n < 5000)。

### Stage 5: Anderson-Darling Test (Anderson-Darling检验)
**English:** A^2-statistic emphasizes tail fit. Most stringent: detects even small deviations. Good for large samples and critical applications.
**中文:** A^2统计强调尾部拟合。最严格：检测甚至小的偏差。适合大样本和关键应用。

## ML Relevance | ML相关性

1. **Assumptions Validation (假设验证)**: Parametric tests (t-test, ANOVA) assume normality. Normality tests guide test selection.
2. **Data Preprocessing (数据预处理)**: Non-normal data may require transformation (Chapter 25: Box-Cox, log, sqrt) before modeling.
3. **Model Diagnostics (模型诊断)**: Linear regression assumes residuals are normal. Normality tests on residuals assess this.
4. **Robustness Justification (鲁棒性证明)**: If data is non-normal, use nonparametric methods (Spearman, Mann-Whitney) or robust estimation (Huber, M-estimators).


---
