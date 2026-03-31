# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 23

---

### Rank Data

# 23 — Data Ranking / 数据排名

**Chapter 23 — File 1 of 1**

## Summary / 摘要

Data ranking converts numerical values into their ordinal positions, fundamental for non-parametric statistical methods. The scipy.stats.rankdata function assigns ranks from 1 to n, with special handling for tied values. Ranking is essential for distribution-free tests (Mann-Whitney U, Spearman correlation) that don't assume normality. The method='average' parameter assigns the mean rank to tied values, preventing rank bias. Rankings reveal relative ordering independent of scale, enabling robust statistical inference when distributional assumptions fail.

数据排名将数值转换为其序数位置，对非参数统计方法至关重要。scipy.stats.rankdata函数分配从1到n的排名，对平局值有特殊处理。排名对于不假设正态性的无分布测试(Mann-Whitney U、Spearman相关性)至关重要。method='average'参数将平均排名分配给平局值，防止排名偏差。排名独立于规模揭示相对顺序，在分布假设失败时实现稳健的统计推断。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import rankdata
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
# 生成随机数 / Generate random numbers
np.random.seed(42)
```

## Step 2 — Generate Sample Data / 生成样本数据

```python
# Generate random data with some ties
# 生成有一些平局的随机数据
# 创建NumPy数组 / Create NumPy array
data = np.array([23, 15, 8, 42, 15, 30, 8, 45, 28, 15, 35, 22, 38])

# Display original data
# 显示原始数据
# 打印输出 / Print output
print(f"Original data: {data}")
# 打印输出 / Print output
print(f"Data length: {len(data)}")
# 打印输出 / Print output
print(f"Data range: [{data.min()}, {data.max()}]")

# Identify unique values and ties
# 识别唯一值和平局
# 找出唯一值 / Find unique values
unique_vals, counts = np.unique(data, return_counts=True)
ties = unique_vals[counts > 1]

# 打印输出 / Print output
print(f"\nUnique values: {len(unique_vals)}")
# 打印输出 / Print output
print(f"Values with ties: {ties}")
# 获取长度 / Get length
if len(ties) > 0:
    for val in ties:
        # 求和 / Calculate sum
        print(f"  Value {val}: appears {np.sum(data == val)} times")
```

## Step 3 — Perform Data Ranking / 执行数据排名

```python
# Rank data using average method for ties
# 使用平均方法为平局排名数据
ranks_average = rankdata(data, method='average')

# Also demonstrate other ranking methods
# 还演示其他排名方法
ranks_ordinal = rankdata(data, method='ordinal')  # Ordinal ranking (first occurrence gets lower rank)
                                                   # 序数排名（首次出现获得较低排名）
ranks_min = rankdata(data, method='min')  # Minimum rank for ties / 平局的最小排名
ranks_max = rankdata(data, method='max')  # Maximum rank for ties / 平局的最大排名
ranks_dense = rankdata(data, method='dense')  # Dense ranking (no gaps) / 密集排名（无间隙）

# Display ranking results
# 显示排名结果
# 打印输出 / Print output
print(f"\nRanking Methods Comparison:")
# 打印输出 / Print output
print(f"{'Value':>6} | {'Average':>8} | {'Ordinal':>8} | {'Min':>5} | {'Max':>5} | {'Dense':>6}")
# 打印输出 / Print output
print("-" * 50)
# 同时获取索引和值 / Get both index and value
for i, val in enumerate(data):
    # 打印输出 / Print output
    print(f"{val:6d} | {ranks_average[i]:8.1f} | {ranks_ordinal[i]:8d} | {ranks_min[i]:5d} | {ranks_max[i]:5d} | {ranks_dense[i]:6d}")
```

## Step 4 — Analyze Ranking Properties / 分析排名属性

```python
# Ranking statistics (using average method)
# 排名统计（使用平均方法）
# 打印输出 / Print output
print(f"\nRanking Statistics (Average Method):")
# 打印输出 / Print output
print(f"  Number of observations: {len(data)}")
# 打印输出 / Print output
print(f"  Expected rank range: [1, {len(data)}]")
# 打印输出 / Print output
print(f"  Actual rank range: [{ranks_average.min():.1f}, {ranks_average.max():.1f}]")
# 打印输出 / Print output
print(f"  Mean rank: {ranks_average.mean():.2f} (expected: {(len(data)+1)/2:.2f})")
# 打印输出 / Print output
print(f"  Std of ranks: {ranks_average.std():.2f}")

# Verify all ranks are assigned
# 验证所有排名都已分配
# 打印输出 / Print output
print(f"\nRank assignment verification:")
# 打印输出 / Print output
print(f"  All ranks between 1 and n: {(ranks_average.min() >= 1) and (ranks_average.max() <= len(data))}")
# 打印输出 / Print output
print(f"  Ranks sum to n(n+1)/2: {ranks_average.sum()} = {len(data)*(len(data)+1)//2}")
```

## Step 5 — Visualize Data and Ranks / 可视化数据和排名

```python
# Create visualization
# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original data
# 图1: 原始数据
# 生成等差数组 / Generate array with step
indices = np.arange(len(data))
colors = plt.cm.viridis(data / data.max())
axes[0, 0].bar(indices, data, color=colors, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Index')
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('Original Data')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels
# 同时获取索引和值 / Get both index and value
for i, v in enumerate(data):
    axes[0, 0].text(i, v + 1, str(v), ha='center', fontsize=9)

# Plot 2: Rankings (Average method)
# 图2: 排名（平均方法）
axes[0, 1].bar(indices, ranks_average, color=colors, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Rank')
axes[0, 1].set_title('Ranks (Average Method for Ties)')
# 获取长度 / Get length
axes[0, 1].set_ylim([0, len(data) + 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add rank labels
# 同时获取索引和值 / Get both index and value
for i, r in enumerate(ranks_average):
    axes[0, 1].text(i, r + 0.2, f'{r:.1f}', ha='center', fontsize=9)

# Plot 3: Sorted data with ranks
# 图3: 排序数据与排名
sorted_indices = np.argsort(data)
sorted_data = data[sorted_indices]
sorted_ranks = ranks_average[sorted_indices]

ax3 = axes[1, 0]
ax3_twin = ax3.twinx()

bars = ax3.bar(indices, sorted_data, alpha=0.6, color='blue', label='Sorted values')
line = ax3_twin.plot(indices, sorted_ranks, 'ro-', linewidth=2, markersize=6, label='Ranks')

ax3.set_xlabel('Position (after sorting)')
ax3.set_ylabel('Value', color='blue')
ax3_twin.set_ylabel('Rank', color='red')
ax3.set_title('Sorted Data with Corresponding Ranks')
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Comparison of ranking methods
# 图4: 排名方法的比较
axes[1, 1].plot(indices, ranks_average, 'o-', linewidth=2, markersize=6, label='Average')
axes[1, 1].plot(indices, ranks_min, 's--', linewidth=1.5, markersize=5, label='Min')
axes[1, 1].plot(indices, ranks_max, '^--', linewidth=1.5, markersize=5, label='Max')
axes[1, 1].plot(indices, ranks_dense, 'D--', linewidth=1.5, markersize=4, label='Dense')

axes[1, 1].set_xlabel('Index')
axes[1, 1].set_ylabel('Rank')
axes[1, 1].set_title('Comparison of Ranking Methods')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Ranking converts data to ordinal positions, eliminating scale dependence. The average method for ties assigns the mean rank to all tied values, preserving sum of ranks = n(n+1)/2 and ensuring mean rank = (n+1)/2. This approach is essential for non-parametric tests that rely on rank differences rather than absolute values, making tests robust to outliers and distribution shape.
  
  **统计概念**: 排名将数据转换为序数位置，消除规模依赖性。平均平局方法将平均排名分配给所有平局值，保持排名总和 = n(n+1)/2，并确保平均排名 = (n+1)/2。此方法对于依赖于排名差异而非绝对值的非参数检验至关重要，使检验对异常值和分布形状稳健。

- **ML Application**: Ranking is the foundation for non-parametric statistical tests and algorithms. In ML, rankings enable: (1) robust feature engineering (rank-based transformations preserve monotonic relationships), (2) ordinal regression for categorical outcomes, (3) ranking-based ensemble methods (e.g., rank averaging). Ranking also handles outliers naturally—extreme values don't distort ranks. In production systems, ranking metrics like rank-AUC are more interpretable for ranking tasks and less sensitive to score calibration.
  
  **ML应用**: 排名是非参数统计测试和算法的基础。在ML中，排名启用：(1)稳健的特征工程（基于排名的转换保留单调关系），(2)用于分类结果的序数回归，(3)基于排名的集成方法（例如排名平均）。排名也自然处理异常值——极端值不会扭曲排名。在生产系统中，像rank-AUC这样的排名指标对于排名任务更可解释，对分数校准不太敏感。
```

➡️ **Next**: `../chapter_24/01_dataset.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from scipy.stats import rankdata
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# 生成随机数 / Generate random numbers
np.random.seed(42)
# 创建NumPy数组 / Create NumPy array
data = np.array([23, 15, 8, 42, 15, 30, 8, 45, 28, 15, 35, 22, 38])

ranks = rankdata(data, method='average')

# 打印输出 / Print output
print(f"Data: {data}")
# 打印输出 / Print output
print(f"Ranks: {ranks}")
# 打印输出 / Print output
print(f"Mean rank: {ranks.mean():.2f}")
# 打印输出 / Print output
print(f"Expected: {(len(data)+1)/2:.2f}")

# Verify properties
# 打印输出 / Print output
print(f"Sum of ranks: {ranks.sum()} (expected: {len(data)*(len(data)+1)//2})")
```

---

### Chapter Summary / 章节总结

# Chapter 23: Data Ranking
# 第23章：数据排名

## Theme | 主题
From raw values to ordinal position: rank transformation as gateway to nonparametric methods.
从原始值到序数位置：排名转换作为非参数方法的网关。

## Evolution Roadmap | 演变路线图
```
Raw Data (continuous or categorical values)
└─ Rank Transformation
   (assign 1, 2, 3, ... based on sorted order)
   └─ Rank-Based Methods (Spearman, Mann-Whitney, etc.)
```

## Progression Logic | 进度逻辑

### Stage 1: Ranking Mechanism (排名机制)
**English:** Sort data in ascending order, assign ranks 1, 2, ..., n. For ties, assign average rank (e.g., three tied values at position 4 each get rank 5 = (4+5+6)/3).
**中文:** 按升序排序数据，分配等级1、2、...、n。对于平局，分配平均等级(例如，位置4处的三个平局值各获得等级5 = (4+5+6)/3)。

### Stage 2: Properties (属性)
**English:** Ranks are ordinal (preserve order but lose magnitude). Example: [10, 50, 100] and [1, 2, 3] both map to ranks [1, 2, 3].
**中文:** 等级是序数(保留顺序但失去幅度)。例如：[10, 50, 100]和[1, 2, 3]都映射到等级[1, 2, 3]。

### Stage 3: Robustness (鲁棒性)
**English:** Ranks ignore absolute values and outliers. Example: a single extreme value doesn't inflate the rank, only changes its order.
**中文:** 等级忽略绝对值和异常值。例如：单个极端值不会膨胀等级，仅改变其顺序。

### Stage 4: Gateway to Nonparametric Tests (非参数检验的网关)
**English:** Spearman correlation, Mann-Whitney U test, Wilcoxon test, Kruskal-Wallis test all operate on ranks instead of raw values, making them distribution-free.
**中文:** Spearman相关、Mann-Whitney U检验、Wilcoxon检验、Kruskal-Wallis检验都对等级而不是原始值进行操作，使它们不受分布限制。

## ML Relevance | ML相关性

1. **Outlier Robustness (异常值鲁棒性)**: Ranking removes the influence of extreme values, useful for contaminated data.
2. **Nonparametric Methods (非参数方法)**: Many rank-based tests don't assume normality and are more robust than parametric counterparts.
3. **Feature Engineering (特征工程)**: Rank transformation can be used as a preprocessing step for tree-based models (some implementations natively use ranks).
4. **Ordinal Variables (序数变量)**: Ranks are natural for survey ratings, rankings, or any ordinal scale.


---
