# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 12

---

### Test Dataset



---

### Covariance

# 12.2 — Covariance Matrix / 协方差矩阵



**Chapter 12 — File 2 of 3**



## Summary / 摘要



Covariance measures how two variables change together. The covariance matrix shows the covariance between all pairs of variables. Positive covariance indicates variables move in the same direction; negative covariance indicates opposite directions.



协方差衡量两个变量如何一起变化。协方差矩阵显示所有变量对之间的协方差。正协方差表示变量向同一方向移动；负协方差表示相反方向。



**Mathematical Formula / 数学公式:**


$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})$$


Note: Cov(X,X) = Var(X)

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results


## Step 1 — Import and Load Data / 导入并加载数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random, cov, mean, std

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot



# 使用之前生成的相关数据集

# Use the previously generated correlated dataset

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(1)

data1 = 20 * random.randn(1000) + 100

data2 = data1 + (10 * random.randn(1000) + 50)



# 打印输出 / Print output
print(f"Loaded {len(data1)} observations")
```

## Step 2 — Compute Covariance Matrix / 计算协方差矩阵

```python
# 计算协方差矩阵

# Compute covariance matrix

covariance_matrix = cov(data1, data2)



# 打印输出 / Print output
print("Covariance Matrix / 协方差矩阵:")
# 打印输出 / Print output
print(covariance_matrix)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"\nShape: {covariance_matrix.shape}")
```

## Step 3 — Interpret Covariance Matrix Elements / 解释协方差矩阵元素

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# 矩阵的对角线元素是方差

# Diagonal elements are variances

var1 = covariance_matrix[0, 0]

var2 = covariance_matrix[1, 1]



# 非对角线元素是协方差

# Off-diagonal elements are covariances

cov_12 = covariance_matrix[0, 1]

cov_21 = covariance_matrix[1, 0]



# 打印输出 / Print output
print("Covariance Matrix Elements / 协方差矩阵元素:")
# 打印输出 / Print output
print("="*50)

# 打印输出 / Print output
print(f"Var(X) = Cov(X,X) = {var1:.4f}")
# 打印输出 / Print output
print(f"Var(Y) = Cov(Y,Y) = {var2:.4f}")
# 打印输出 / Print output
print(f"Cov(X,Y) = {cov_12:.4f}")
# 打印输出 / Print output
print(f"Cov(Y,X) = {cov_21:.4f}")
# 打印输出 / Print output
print(f"\nNote: Cov(X,Y) = Cov(Y,X) (symmetric matrix)")


# 标准差

std1 = std(data1)

std2 = std(data2)

# 打印输出 / Print output
print(f"\nStandard Deviations:")
# 打印输出 / Print output
print(f"Std(X) = {std1:.4f}")
# 打印输出 / Print output
print(f"Std(Y) = {std2:.4f}")
# 打印输出 / Print output
print(f"\nNote: sqrt(Var) = Std")
# 打印输出 / Print output
print(f"sqrt({var1:.4f}) = {np.sqrt(var1):.4f}")
```

```python
# 协方差的解释
        "# Interpretation of covariance

# 打印输出 / Print output
print("Covariance Interpretation / 协方差解释:")
# 打印输出 / Print output
print("="*50)

# 打印输出 / Print output
print(f"\nCovariance value: {cov_12:.4f}")


if cov_12 > 0:
    # 打印输出 / Print output
    print("Result: POSITIVE covariance")
    # 打印输出 / Print output
    print("Meaning: Variables move in the same direction")
    # 打印输出 / Print output
    print("When X increases, Y tends to increase")

elif cov_12 < 0:
    # 打印输出 / Print output
    print("Result: NEGATIVE covariance")
    # 打印输出 / Print output
    print("Meaning: Variables move in opposite directions")
    # 打印输出 / Print output
    print("When X increases, Y tends to decrease")

else:
    # 打印输出 / Print output
    print("Result: ZERO covariance")
    # 打印输出 / Print output
    print("Meaning: Variables are uncorrelated")


# 协方差的缺点：依赖于量度单位

# Limitation of covariance

# 打印输出 / Print output
print("\n" + "="*50)

# 打印输出 / Print output
print("Limitation / 局限性:")
# 打印输出 / Print output
print("Covariance depends on the scale of variables")

# 打印输出 / Print output
print("Therefore, covariance values are hard to interpret")

# 打印输出 / Print output
print("in absolute terms. We use correlation for")

# 打印输出 / Print output
print("standardized comparison.")
```

```python
# 将协方差矩阵表示为DataFrame以便更好的可读性

# Display covariance matrix as DataFrame for better readability

cov_df = pd.DataFrame(

    covariance_matrix,

    index=['Data1', 'Data2'],

    columns=['Data1', 'Data2']

)



# 打印输出 / Print output
print("\nCovariance Matrix as DataFrame:")
# 打印输出 / Print output
print(cov_df)



# 热力图可视化

# Heatmap visualization

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.patches as mpatches



pyplot.figure(figsize=(6, 5))

pyplot.imshow(covariance_matrix, cmap='RdBu_r', aspect='auto')

pyplot.colorbar(label='Covariance Value')

pyplot.xticks([0, 1], ['Data1', 'Data2'])

pyplot.yticks([0, 1], ['Data1', 'Data2'])

pyplot.title('Covariance Matrix Heatmap / 协方差矩阵热力图')



# 添加数值标签

# 生成整数序列 / Generate integer sequence
for i in range(2):
    # 生成整数序列 / Generate integer sequence
    for j in range(2):
        pyplot.text(j, i, f'{covariance_matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontsize=12)

pyplot.show()
```

## Learning Notes / 学习笔记



- **Statistical Concept / 统计学概念**: The covariance matrix is a fundamental tool in multivariate statistics. The diagonal contains variances (self-covariance), while off-diagonal elements show covariances between pairs. The matrix is always symmetric. Positive covariance indicates positive relationship, negative indicates negative relationship. / 协方差矩阵是多元统计的基础工具。对角线包含方差（自协方差），而非对角线元素显示对之间的协方差。矩阵总是对称的。正协方差表示正关系，负协方差表示负关系。


- **ML Application / 机器学习应用**: Covariance matrices are essential in PCA (Principal Component Analysis), used to identify directions of maximum variance. Used in Gaussian mixture models, Linear Discriminant Analysis, and other multivariate algorithms. Covariance structure helps detect multicollinearity in regression models. / 协方差矩阵在 PCA（主成分分析）中至关重要，用于识别最大方差的方向。用于高斯混合模型、线性判别分析和其他多元算法。协方差结构有助于在回归模型中检测多重共线性。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: `03_correlation.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random, cov, mean, std

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd



# Generate correlated data

# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
random.seed(1)

data1 = 20 * random.randn(1000) + 100

data2 = data1 + (10 * random.randn(1000) + 50)



# Compute covariance matrix

covariance_matrix = cov(data1, data2)

# 打印输出 / Print output
print("Covariance Matrix:")
# 打印输出 / Print output
print(covariance_matrix)



# Extract and interpret elements

var1 = covariance_matrix[0, 0]

var2 = covariance_matrix[1, 1]

cov_12 = covariance_matrix[0, 1]



# 打印输出 / Print output
print(f"\nVar(X) = {var1:.4f}")
# 打印输出 / Print output
print(f"Var(Y) = {var2:.4f}")
# 打印输出 / Print output
print(f"Cov(X,Y) = {cov_12:.4f}")


# Display as DataFrame

cov_df = pd.DataFrame(covariance_matrix,

                      index=['Data1', 'Data2'],

                      columns=['Data1', 'Data2'])

# 打印输出 / Print output
print("\nCovariance Matrix DataFrame:")
# 打印输出 / Print output
print(cov_df)



# Heatmap

pyplot.figure(figsize=(6, 5))

pyplot.imshow(covariance_matrix, cmap='RdBu_r', aspect='auto')

pyplot.colorbar(label='Covariance')

pyplot.xticks([0, 1], ['Data1', 'Data2'])

pyplot.yticks([0, 1], ['Data1', 'Data2'])

pyplot.title('Covariance Matrix Heatmap')

# 生成整数序列 / Generate integer sequence
for i in range(2):
    # 生成整数序列 / Generate integer sequence
    for j in range(2):
        pyplot.text(j, i, f'{covariance_matrix[i, j]:.0f}',
                   ha='center', va='center')

pyplot.show()
```

---

### Correlation

# 12.3 — Pearson's Correlation / Pearson 相关系数\\n\\n**Chapter 12 — File 3 of 3**\\n\\n## Summary / 摘要\\n\\nPearson's correlation coefficient measures the linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive). It is scale-invariant, making it superior to covariance for comparing relationships across different datasets.\\n\\nPearson 相关系数测量两个变量之间的线性关系，范围从 -1（完美负相关）到 +1（完美正相关）。它是尺度不变的，使其在比较不同数据集之间的关系时优于协方差。\\n\\n**Mathematical Formula / 数学公式:**\\n\\n$$r = \\\\frac{\\\\text{Cov}(X,Y)}{\\\\sigma_X \\\\sigma_Y} = \\\\frac{\\\\sum(X_i - \\\\bar{X})(Y_i - \\\\bar{Y})}{\\\\sqrt{\\\\sum(X_i - \\\\bar{X})^2}\\\\sqrt{\\\\sum(Y_i - \\\\bar{Y})^2}}$$

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

## Step 1 — Import and Load Data / 导入并加载数据

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random, corrcoef\\nfrom scipy.stats import pearsonr\\nfrom matplotlib import pyplot\\n\\n# 使用之前生成的相关数据集\\n# Use the previously generated correlated dataset\\nrandom.seed(1)\\ndata1 = 20 * random.randn(1000) + 100\\ndata2 = data1 + (10 * random.randn(1000) + 50)\\n\\nprint(f\\\"Loaded {len(data1)} observations\\\")
```

## Step 2 — Compute Pearson Correlation Coefficient / 计算 Pearson 相关系数

```python
# 方法 1: 使用 scipy.stats.pearsonr\\n# Method 1: Using scipy.stats.pearsonr\\ncorrelation, p_value = pearsonr(data1, data2)\\n\\nprint(f\\\"Pearson Correlation Coefficient: {correlation:.6f}\\\")\\nprint(f\\\"p-value: {p_value:.2e}\\\")\\n\\n# 方法 2: 使用 numpy.corrcoef\\n# Method 2: Using numpy.corrcoef\\ncorrelation_matrix = corrcoef(data1, data2)\\nprint(f\\\"\\\\nUsing corrcoef:\\\")\\nprint(correlation_matrix)
```

## Step 3 — Interpret the Correlation Coefficient / 解释相关系数

```python
# 解释相关系数\\n# Interpret the correlation coefficient\\nprint(\\\"Correlation Coefficient Interpretation / 相关系数解释:\\\")\\nprint(\\\"=\\\"*60)\\nprint(f\\\"\\\\nPearson's r = {correlation:.6f}\\\")\\n\\n# 强度\\nabs_corr = abs(correlation)\\nif abs_corr >= 0.9:\\n    strength = \\\"Very Strong / 非常强\\\"\\nelif abs_corr >= 0.7:\\n    strength = \\\"Strong / 强\\\"\\nelif abs_corr >= 0.5:\\n    strength = \\\"Moderate / 中等\\\"\\nelif abs_corr >= 0.3:\\n    strength = \\\"Weak / 弱\\\"\\nelse:\\n    strength = \\\"Very Weak / 非常弱\\\"\\n\\n# 方向\\ndirection = \\\"Positive / 正\\\" if correlation > 0 else \\\"Negative / 负\\\"\\n\\nprint(f\\\"Strength: {strength}\\\")\\nprint(f\\\"Direction: {direction}\\\")\\n\\n# 显著性\\nalpha = 0.05\\nif p_value < alpha:\\n    print(f\\\"\\\\nSignificant at α = {alpha} level\\\")\\n    print(f\\\"p-value ({p_value:.2e}) < {alpha}\\\")\\n    print(\\\"Reject H0: Correlation is statistically significant\\\")\\nelse:\\n    print(f\\\"\\\\nNot significant at α = {alpha} level\\\")\\n    print(\\\"Fail to reject H0: Correlation is not statistically significant\\\")
```

## Step 4 — Relationship to Covariance / 与协方差的关系

```python
from numpy import cov, std\\n\\n# 计算协方差\\n# Compute covariance\\ncovariance = cov(data1, data2)[0, 1]\\n\\n# 计算标准差\\n# Compute standard deviations\\nstd1 = std(data1, ddof=1)  # 使用样本标准差\\nstd2 = std(data2, ddof=1)  # Use sample standard deviation\\n\\n# 手动计算相关系数\\n# Manually compute correlation\\ncorrelation_manual = covariance / (std1 * std2)\\n\\nprint(\\\"Relationship between Covariance and Correlation / 协方差与相关系数的关系:\\\")\\nprint(\\\"=\\\"*60)\\nprint(f\\\"Covariance: {covariance:.4f}\\\")\\nprint(f\\\"Std(X): {std1:.4f}\\\")\\nprint(f\\\"Std(Y): {std2:.4f}\\\")\\nprint(f\\\"\\\\nManual calculation / 手动计算:\\\")\\nprint(f\\\"r = Cov(X,Y) / (Std(X) * Std(Y))\\\")\\nprint(f\\\"r = {covariance:.4f} / ({std1:.4f} * {std2:.4f})\\\")\\nprint(f\\\"r = {correlation_manual:.6f}\\\")\\nprint(f\\\"\\\\nPearson's r (scipy): {correlation:.6f}\\\")\\nprint(f\\\"Difference: {abs(correlation_manual - correlation):.10f}\\\")
```

## Step 5 — Visualize Correlation / 可视化相关性

```python
import numpy as np\\nfrom scipy import stats as sp_stats\\n\\n# 创建散点图并添加回归线\\n# Create scatter plot with regression line\\npyplot.figure(figsize=(12, 5))\\n\\n# 散点图\\n# Scatter plot\\npyplot.subplot(1, 2, 1)\\npyplot.scatter(data1, data2, alpha=0.4, s=20)\\n\\n# 添加回归线\\n# Add regression line\\nz = np.polyfit(data1, data2, 1)\\np = np.poly1d(z)\\nx_line = np.linspace(data1.min(), data1.max(), 100)\\npyplot.plot(x_line, p(x_line), \\\"r-\\\", linewidth=2, label=f'r = {correlation:.4f}')\\n\\npyplot.xlabel('Variable 1 (data1) / 变量 1')\\npyplot.ylabel('Variable 2 (data2) / 变量 2')\\npyplot.title(f'Scatter Plot with Regression Line (r={correlation:.4f})')\\npyplot.legend()\\npyplot.grid(True, alpha=0.3)\\n\\n# 展示不同相关系数的例子\\n# Show examples of different correlations\\npyplot.subplot(1, 2, 2)\\ncorrelations = [0.0, 0.5, 0.9, -0.5]\\ntitles = ['r=0 (No correlation)', 'r=0.5 (Moderate)', 'r=0.9 (Strong)', 'r=-0.5 (Negative)']\\n\\nfor i, (corr_val, title) in enumerate(zip(correlations, titles)):\\n    print(f\\\"Interpretation example: {title}\\\")\\n\\npyplot.text(0.5, 0.9, \\\"Correlation Strength Scale / 相关强度等级:\\\", \\n           ha='center', va='top', transform=pyplot.gca().transAxes,\\n           fontsize=11, fontweight='bold')\\n\\nscale_text = \\\"\\\"\\\"r = ±1.0 : Perfect correlation (完美相关)\\nr = ±0.7 to ±0.9 : Strong (强)\\nr = ±0.5 to ±0.7 : Moderate (中等)\\nr = ±0.3 to ±0.5 : Weak (弱)\\nr = 0 to ±0.3 : Very weak (非常弱)\\nr = 0 : No correlation (无相关)\\\"\\\"\\\"\\n\\npyplot.text(0.05, 0.75, scale_text, ha='left', va='top', \\n           transform=pyplot.gca().transAxes,\\n           fontsize=10, family='monospace',\\n           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))\\npyplot.axis('off')\\n\\npyplot.tight_layout()\\npyplot.show()
```

## Step 6 — Statistical Significance Testing / 统计显著性检验

```python
# 假设检验的解释\\n# Interpretation of hypothesis test\\nprint(\\\"Hypothesis Test for Pearson Correlation / Pearson相关假设检验:\\\")\\nprint(\\\"=\\\"*60)\\nprint(f\\\"\\\\nNull Hypothesis (H0): ρ = 0 (no correlation)\\\")\\nprint(f\\\"Alternative Hypothesis (H1): ρ ≠ 0 (correlation exists)\\\")\\nprint(f\\\"\\\\nSample size (n): {len(data1)}\\\")\\nprint(f\\\"Pearson's r: {correlation:.6f}\\\")\\nprint(f\\\"p-value: {p_value:.2e}\\\")\\nprint(f\\\"Significance level (α): 0.05\\\")\\nprint(f\\\"\\\\nDecision: \\\", end=\\\"\\\")\\n\\nif p_value < 0.05:\\n    print(f\\\"REJECT H0\\\")\\n    print(f\\\"\\\\nConclusion: There is statistically significant evidence\\\")\\n    print(f\\\"of a linear relationship between the two variables.\\\")\\nelse:\\n    print(f\\\"FAIL TO REJECT H0\\\")\\n    print(f\\\"\\\\nConclusion: There is insufficient evidence\\\")\\n    print(f\\\"of a linear relationship between the two variables.\\\")\\n\\n# 计算 R²（决定系数）\\nr_squared = correlation ** 2\\nprint(f\\\"\\\\n\\\\nCoefficient of Determination (R²) / 决定系数:\\\")\\nprint(f\\\"R² = r² = {r_squared:.6f}\\\")\\nprint(f\\\"Interpretation: {r_squared*100:.2f}% of variance in Y\\\")\\nprint(f\\\"is explained by X (or vice versa).\\\")
```

## Learning Notes / 学习笔记\\n\\n- **Statistical Concept / 统计学概念**: Pearson correlation is a standardized measure of linear relationship, ranging from -1 to +1. It is invariant to scale transformations and more interpretable than covariance. Statistical significance depends on both the correlation strength and sample size. / Pearson相关是线性关系的标准化度量，范围从-1到+1。它对尺度变换不变，比协方差更可解释。统计显著性取决于相关强度和样本大小。\\n\\n- **ML Application / 机器学习应用**: Correlation matrices guide feature selection (removing highly correlated features to reduce multicollinearity), feature engineering, and understanding feature relationships. Used in exploratory data analysis (EDA) to identify important predictors. Correlation analysis helps validate model assumptions and detect spurious relationships. / 相关矩阵指导特征选择（删除高度相关的特征以减少多重共线性）、特征工程和理解特征关系。用于探索性数据分析（EDA）以识别重要的预测变量。相关分析有助于验证模型假设和检测虚假关系。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `../chapter_13/01_test_data.ipynb`

## Complete Code / 完整代码一览

```python
from numpy import random, corrcoef, cov, std\\nfrom scipy.stats import pearsonr\\nfrom matplotlib import pyplot\\nimport numpy as np\\n\\n# Generate data\\nrandom.seed(1)\\ndata1 = 20 * random.randn(1000) + 100\\ndata2 = data1 + (10 * random.randn(1000) + 50)\\n\\n# Compute Pearson correlation\\ncorrelation, p_value = pearsonr(data1, data2)\\nprint(f\\\"Pearson's r: {correlation:.6f}\\\")\\nprint(f\\\"p-value: {p_value:.2e}\\\")\\nprint(f\\\"R² (coefficient of determination): {correlation**2:.6f}\\\")\\n\\n# Verify with corrcoef\\nprint(\\\"\\\\nVerification with corrcoef:\\\")\\nprint(corrcoef(data1, data2))\\n\\n# Relationship to covariance\\ncovariance = cov(data1, data2)[0, 1]\\nstd1 = std(data1, ddof=1)\\nstd2 = std(data2, ddof=1)\\nr_manual = covariance / (std1 * std2)\\nprint(f\\\"\\\\nManual calculation: r = {r_manual:.6f}\\\")\\n\\n# Scatter plot with regression line\\npyplot.figure(figsize=(10, 6))\\npyplot.scatter(data1, data2, alpha=0.4, s=20)\\nz = np.polyfit(data1, data2, 1)\\np = np.poly1d(z)\\nx_line = np.linspace(data1.min(), data1.max(), 100)\\npyplot.plot(x_line, p(x_line), \\\"r-\\\", linewidth=2, label=f'r={correlation:.4f}')\\npyplot.xlabel('Variable 1')\\npyplot.ylabel('Variable 2')\\npyplot.title(f'Scatter Plot with Regression Line')\\npyplot.legend()\\npyplot.grid(True, alpha=0.3)\\npyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 12: Correlation
# 第12章：相关性

## Theme | 主题
From raw covariation to standardized association: measuring strength of linear relationship.
从原始协变到标准化关联：测量线性关系的强度。

## Evolution Roadmap | 演变路线图
```
Test Data (two related variables)
└─ Covariance (raw joint spread)
   └─ Pearson Correlation (standardized [-1,1])
```

## Progression Logic | 进度逻辑

### Stage 1: Paired Data (配对数据)
**English:** Generate or load two continuous variables with known relationship (e.g., height and weight, age and income).
**中文:** 生成或加载两个具有已知关系的连续变量(例如身高和体重、年龄和收入)。

### Stage 2: Covariance (协方差)
**English:** Compute cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]. Measures joint spread but scale depends on units.
**中文:** 计算cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]。测量联合扩展但规模取决于单位。

### Stage 3: Pearson Correlation (皮尔逊相关)
**English:** Normalize covariance by standard deviations: r = cov(X,Y) / (σ_X * σ_Y). Ranges [-1, 1].
**中文:** 通过标准差规范化协方差：r = cov(X,Y) / (σ_X * σ_Y)。范围[-1, 1]。

### Stage 4: Interpretation (解释)
**English:** r = 1 (perfect positive), r = -1 (perfect negative), r = 0 (no linear relationship). p-value tests statistical significance.
**中文:** r = 1(完全正相关)，r = -1(完全负相关)，r = 0(无线性关系)。p值检验统计显著性。

## ML Relevance | ML相关性

1. **Feature Selection (特征选择)**: High correlation between feature and target suggests predictive power.
2. **Multicollinearity Detection (多重共线性检测)**: High correlation between features can degrade model interpretability and estimation.
3. **EDA (探索性数据分析)**: Correlation heatmaps reveal structure and dependencies in high-dimensional data.
4. **Linear Relationships (线性关系)**: Pearson r measures the strength of linear association, guiding model selection (linear vs. nonlinear).


---
