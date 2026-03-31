# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 17

---

### Bootstrap

# 17 — Bootstrap / 自助法

**Chapter 17 — File 1 of 1**

## Summary / 摘要

Bootstrap is a resampling technique that estimates the distribution of a statistic by repeatedly sampling from the observed data with replacement. Each resample (or "bootstrap sample") has the same size as the original dataset but contains duplicates and omissions. The out-of-bag (OOB) samples are data points not selected in a particular bootstrap sample, providing a natural validation set without requiring separate held-out data.

自助法是一种重新采样技术，通过有放回地从观察数据中重复抽样来估计统计量的分布。每个自助样本的大小与原始数据集相同，但包含重复和遗漏。袋外(OOB)样本是在特定自助样本中未被选中的数据点，提供了自然的验证集，无需单独的保留数据。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.utils import resample
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
# 生成随机数 / Generate random numbers
np.random.seed(42)
```

## Step 2 — Generate Original Data / 生成原始数据

```python
# Generate a small original dataset
# 生成一个小的原始数据集
# 创建NumPy数组 / Create NumPy array
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Display original data
# 显示原始数据
# 打印输出 / Print output
print(f"Original data: {data}")
# 打印输出 / Print output
print(f"Original data size: {len(data)}")
# 计算均值 / Calculate mean
print(f"Original mean: {np.mean(data):.2f}")
# 计算标准差 / Calculate standard deviation
print(f"Original std: {np.std(data):.2f}")
```

## Step 3 — Perform Bootstrap Resampling / 执行自助重采样

```python
# Number of bootstrap samples
# 自助样本数
n_iterations = 5

# Store bootstrap statistics
# 存储自助统计量
bootstrap_means = []
bootstrap_samples = []
oob_indices_list = []

# Perform bootstrap resampling
# 执行自助重采样
# 生成整数序列 / Generate integer sequence
for i in range(n_iterations):
    # Resample with replacement
    # 有放回地重采样
    # 生成随机数 / Generate random numbers
    indices = np.random.choice(len(data), len(data), replace=True)
    bootstrap_sample = data[indices]
    
    # Calculate out-of-bag (OOB) indices
    # 计算袋外(OOB)指标
    # 生成等差数组 / Generate array with step
    oob_indices = np.setdiff1d(np.arange(len(data)), np.unique(indices))
    
    # Store results
    # 存储结果
    # 计算均值 / Calculate mean
    bootstrap_means.append(np.mean(bootstrap_sample))
    # 添加元素到列表末尾 / Append element to list end
    bootstrap_samples.append(bootstrap_sample)
    # 添加元素到列表末尾 / Append element to list end
    oob_indices_list.append(oob_indices)
    
    # Print details for each iteration
    # 打印每次迭代的细节
    # 打印输出 / Print output
    print(f"\nBootstrap sample {i+1}:")
    # 打印输出 / Print output
    print(f"  Indices: {sorted(indices)}")
    # 打印输出 / Print output
    print(f"  Sample: {bootstrap_sample}")
    # 计算均值 / Calculate mean
    print(f"  Mean: {np.mean(bootstrap_sample):.2f}")
    # 打印输出 / Print output
    print(f"  OOB indices: {oob_indices}")
    # 打印输出 / Print output
    print(f"  OOB data: {data[oob_indices]}")
```

## Step 4 — Analyze Bootstrap Distribution / 分析自助分布

```python
# Convert to array for analysis
# 转换为数组以进行分析
# 创建NumPy数组 / Create NumPy array
bootstrap_means = np.array(bootstrap_means)

# Calculate bootstrap statistics
# 计算自助统计量
# 打印输出 / Print output
print(f"\nBootstrap Statistics:")
# 计算均值 / Calculate mean
print(f"Mean of bootstrap means: {np.mean(bootstrap_means):.2f}")
# 计算标准差 / Calculate standard deviation
print(f"Std of bootstrap means: {np.std(bootstrap_means):.2f}")
# 求最小值 / Find minimum value
print(f"Min bootstrap mean: {np.min(bootstrap_means):.2f}")
# 求最大值 / Find maximum value
print(f"Max bootstrap mean: {np.max(bootstrap_means):.2f}")
```

## Step 5 — Visualize Bootstrap Distribution / 可视化自助分布

```python
# Create visualization
# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Histogram of bootstrap means
# 图1: 自助均值的直方图
axes[0].hist(bootstrap_means, bins=5, edgecolor='black', alpha=0.7)
# 计算均值 / Calculate mean
axes[0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Original mean')
axes[0].set_xlabel('Mean Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Bootstrap Distribution of Means')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Original data
# 图2: 原始数据
axes[1].plot(data, marker='o', linestyle='-', linewidth=2, markersize=8, label='Original data')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Value')
axes[1].set_title('Original Data')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept**: Bootstrap provides non-parametric estimation of sampling distributions without assuming underlying data distribution. Each resample is drawn with replacement, allowing elements to appear multiple times while others are excluded (OOB samples).
  
  **统计概念**: 自助法提供了无需假设基础数据分布的参数分布非参数估计。每个重采样都有放回地进行，允许元素多次出现，而其他元素被排除（OOB样本）。

- **ML Application**: Bootstrap is fundamental for estimating confidence intervals, assessing model uncertainty, and feature importance (e.g., permutation importance, bagging). Out-of-bag samples provide natural validation without separate test sets, reducing variance in ensemble methods.
  
  **ML应用**: 自助法对估计置信区间、评估模型不确定性和特征重要性（例如排列重要性、装袋法）至关重要。袋外样本提供了自然的验证，无需单独的测试集，减少了集成方法中的方差。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.hist` | 绘制直方图 | Draw histogram |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.utils import resample
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 创建NumPy数组 / Create NumPy array
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 打印输出 / Print output
print(f"Original data: {data}")
# 计算均值 / Calculate mean
print(f"Original mean: {np.mean(data):.2f}")

n_iterations = 100
bootstrap_means = []

# 生成整数序列 / Generate integer sequence
for i in range(n_iterations):
    # 生成随机数 / Generate random numbers
    indices = np.random.choice(len(data), len(data), replace=True)
    bootstrap_sample = data[indices]
    # 计算均值 / Calculate mean
    bootstrap_means.append(np.mean(bootstrap_sample))
    # 生成等差数组 / Generate array with step
    oob_indices = np.setdiff1d(np.arange(len(data)), np.unique(indices))

# 创建NumPy数组 / Create NumPy array
bootstrap_means = np.array(bootstrap_means)
# 计算均值 / Calculate mean
print(f"Bootstrap mean: {np.mean(bootstrap_means):.2f}")
# 计算标准差 / Calculate standard deviation
print(f"Bootstrap std: {np.std(bootstrap_means):.2f}")

# 绘制直方图 / Draw histogram
plt.hist(bootstrap_means, bins=15, edgecolor='black', alpha=0.7)
# 计算均值 / Calculate mean
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label='Original mean')
# 设置X轴标签 / Set X-axis label
plt.xlabel('Mean Value')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('Frequency')
# 设置图表标题 / Set chart title
plt.title('Bootstrap Distribution of Means')
# 显示图例 / Show legend
plt.legend()
plt.grid(True, alpha=0.3)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 17: Bootstrap
# 第17章：自助法

## Theme | 主题
Nonparametric resampling: estimate sampling distributions and confidence intervals without strong distributional assumptions.
非参数重采样：估计采样分布和置信区间，无需强分布假设。

## Evolution Roadmap | 演变路线图
```
Original Sample (observed data)
└─ Bootstrap Sampling with Out-Of-Bag (OOB)
   └─ Compute Statistic (mean, median, slope, etc.)
      └─ Distribution of Statistic emerges
         └─ Confidence Intervals & p-values
```

## Progression Logic | 进度逻辑

### Stage 1: Original Sample (原始样本)
**English:** Start with observed data of size n.
**中文:** 从大小为n的观察数据开始。

### Stage 2: Bootstrap Resampling (自助重采样)
**English:** Draw B bootstrap samples by sampling with replacement from the original sample. Typical: B = 1000 or 10,000.
**中文:** 通过从原始样本中有放回地采样来抽取B个自助样本。典型：B = 1000或10,000。

### Stage 3: Out-Of-Bag (OOB) (袋外)
**English:** Each bootstrap sample omits ~37% of the original observations (on average). These OOB observations can be used for validation without separate test set.
**中文:** 每个自助样本平均遗漏原始观察的~37%。这些OOB观察可用于验证，无需单独的测试集。

### Stage 4: Statistic Distribution (统计分布)
**English:** For each bootstrap sample, compute the statistic of interest (e.g., mean, median, regression slope). Collect B values to form an empirical distribution.
**中文:** 对于每个自助样本，计算感兴趣的统计(例如均值、中位数、回归斜率)。收集B个值以形成经验分布。

### Stage 5: Inference (推断)
**English:** From the empirical distribution, compute confidence intervals (percentile method, BCa method) and p-values without parametric assumptions.
**中文:** 从经验分布中，计算置信区间(百分位方法、BCa方法)和p值，无需参数假设。

## ML Relevance | ML相关性

1. **Nonparametric CI (非参数CI)**: Bootstrap works for any statistic without normality assumption.
2. **Uncertainty Quantification (不确定性量化)**: Bootstrap estimates sampling variability of any model statistic (coefficients, predictions, AUC).
3. **Small Sample Inference (小样本推断)**: Bootstrap is especially powerful when n is small and distributional assumptions are questionable.
4. **Cross-Validation Stability (交叉验证稳定性)**: Bootstrap can assess stability of model estimates and feature importance.


---
