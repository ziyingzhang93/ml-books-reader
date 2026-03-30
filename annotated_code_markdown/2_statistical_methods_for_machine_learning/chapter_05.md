# 统计方法与机器学习
## Chapter 05

---

### Bar Chart

# 05.02 — Bar Chart / 柱状图

**Chapter 05 — File 2 of 5**

## Summary / 摘要

**English:** This notebook creates a bar chart to visualize categorical data. Bar charts are ideal for comparing quantities across discrete categories (e.g., colors, regions, groups). Each bar height represents the value for that category.

**中文:** 本笔记本创建柱状图以可视化分类数据。柱状图适合比较不同类别（例如颜色、区域、组）的数量。每个条形的高度代表该类别的值。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random number generation and plotting library / 导入随机数生成和绘图库
from random import seed
from random import randint
from matplotlib import pyplot
```

## Step 2 — Set Seed and Create Categories / 设置种子并创建类别

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Define category labels / 定义类别标签
# These are discrete categories, not continuous values / 这些是离散类别，不是连续值
x = ['red', 'green', 'blue']
```

## Step 3 — Generate Random Values / 生成随机值

```python
# Generate random quantities for each category / 为每个类别生成随机数量
# randint(0, 100) generates random integers between 0 and 99 / randint(0, 100)生成0到99之间的随机整数
y = [randint(0, 100), randint(0, 100), randint(0, 100)]
```

## Step 4 — Create and Display Bar Chart / 创建并显示柱状图

```python
# Create a bar chart / 创建柱状图
# Each category gets one bar with height equal to its value / 每个类别得到一个条形，高度等于其值
pyplot.bar(x, y)

# Add labels and title / 添加标签和标题
pyplot.xlabel('Color / 颜色')
pyplot.ylabel('Quantity / 数量')
pyplot.title('Bar Chart of Color Quantities / 颜色数量柱状图')

# Display the chart / 显示图表
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** Bar charts are a standard tool for displaying summary statistics across categories. They enable quick visual comparison of group means, counts, or other aggregated measures. The height (or length) of each bar is proportional to the value represented. Bar charts work well with both numerical aggregates (e.g., average sales by region) and counts (e.g., frequency distribution across categories).

- **ML Application / 机器学习应用:** Bar charts are essential in model performance evaluation: comparing accuracy, precision, recall across multiple models or classifiers. Feature importance scores are often displayed as horizontal bar charts. In class imbalance analysis, bar charts reveal the distribution of samples across classes before and after balancing techniques. Confusion matrices and cross-tabulation results are frequently visualized as grouped bar charts for rapid model diagnostics.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

➡️ **Next**: `03_histogram.ipynb` — Visualize distribution of continuous data

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import randint
from matplotlib import pyplot

# ===== Section 2: Prepare Categorical Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Define categories / 定义类别
x = ['red', 'green', 'blue']

# Generate random values for each category / 为每个类别生成随机值
y = [randint(0, 100), randint(0, 100), randint(0, 100)]

# ===== Section 3: Create Bar Chart =====
# Plot bars for categorical data / 为分类数据绘制柱子
pyplot.bar(x, y)

# Add labels and title / 添加标签和标题
pyplot.xlabel('Color / 颜色')
pyplot.ylabel('Quantity / 数量')
pyplot.title('Bar Chart of Color Quantities / 颜色数量柱状图')

# Display chart / 显示图表
pyplot.show()
```

---

### Histogram

# 05.03 — Histogram / 直方图

**Chapter 05 — File 3 of 5**

## Summary / 摘要

**English:** This notebook creates a histogram to visualize the distribution of continuous data. A histogram divides data into bins and shows the frequency of values falling into each bin. Here we visualize 1000 samples from a standard normal distribution.

**中文:** 本笔记本创建直方图以可视化连续数据的分布。直方图将数据分成箱子，并显示落入每个箱子的值的频率。这里我们可视化1000个标准正态分布样本。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random number generation and plotting library / 导入随机数生成和绘图库
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
```

## Step 2 — Set Seed and Generate Gaussian Data / 设置种子并生成高斯数据

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 1000 random samples from standard normal distribution N(0,1) / 从标准正态分布N(0,1)生成1000个样本
x = randn(1000)
```

## Step 3 — Create and Display Histogram / 创建并显示直方图

```python
# Create histogram / 创建直方图
# By default, matplotlib uses 10 bins / 默认情况下，matplotlib使用10个箱子
# Each bin shows how many values fall within that range / 每个箱子显示有多少个值落在该范围内
pyplot.hist(x)

# Add labels and title / 添加标签和标题
pyplot.xlabel('Value / 值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Histogram of Standard Normal Distribution / 标准正态分布直方图')

# Display the histogram / 显示直方图
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** A histogram is a graphical representation of the distribution of a dataset. It partitions the range of data into bins (intervals) and plots the frequency (count) of observations in each bin. Histograms reveal the shape of the distribution: whether it is symmetric, skewed, unimodal, or multimodal. The choice of bin width affects the histogram's appearance—too few bins hide structure; too many create noise. Sturges' rule ($k = \lceil \log_2(n) + 1 \rceil$) provides a heuristic for bin count.

- **ML Application / 机器学mining:** Histograms are fundamental in exploratory data analysis (EDA). They reveal outliers, skewness, and potential multimodality that inform preprocessing decisions. In target variable analysis, histograms expose class imbalance and suggest resampling or cost-weighting strategies. When building regression models, examining histogram of residuals reveals whether the error distribution meets Gaussian assumptions—critical for valid inference and confidence intervals.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `04_boxplot.ipynb` — Compare distributions with box plots

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

# ===== Section 2: Generate Continuous Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate 1000 samples from standard normal N(0,1) / 从标准正态分布N(0,1)生成1000个样本
x = randn(1000)

# ===== Section 3: Create Histogram =====
# Plot histogram showing distribution of values / 绘制显示值分布的直方图
pyplot.hist(x)

# Add labels and title / 添加标签和标题
pyplot.xlabel('Value / 值')
pyplot.ylabel('Frequency / 频数')
pyplot.title('Histogram of Standard Normal Distribution / 标准正态分布直方图')

# Display histogram / 显示直方图
pyplot.show()
```

---

### Boxplot

# 05.04 — Box Plot / 箱线图

**Chapter 05 — File 4 of 5**

## Summary / 摘要

**English:** This notebook creates box plots (box-and-whisker plots) to compare distributions across multiple groups. Box plots show the median, quartiles, and outliers compactly. Here we compare three Gaussian distributions with increasing variance (scales of 1, 5, and 10).

**中文:** 本笔记本创建箱线图以比较多个组的分布。箱线图紧凑地显示中位数、四分位数和异常值。这里我们比较三个高斯分布，方差增加（比例为1、5和10）。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random number generation and plotting library / 导入随机数生成和绘图库
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
```

## Step 2 — Set Seed and Generate Multiple Distributions / 设置种子并生成多个分布

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate three distributions with different scales / 生成三个不同规模的分布
# First: standard normal N(0,1) / 第一个：标准正态分布N(0,1)
# Second: scaled by 5, so N(0,5) / 第二个：缩放5倍，所以N(0,5)
# Third: scaled by 10, so N(0,10) / 第三个：缩放10倍，所以N(0,10)
x = [randn(1000), 5 * randn(1000), 10 * randn(1000)]
```

## Step 3 — Create and Display Box Plot / 创建并显示箱线图

```python
# Create a box plot for comparing multiple distributions / 创建箱线图以比较多个分布
# Box shows interquartile range (IQR), line in box is median / 箱子显示四分位数间距(IQR)，箱子中的线是中位数
# Whiskers extend to 1.5*IQR beyond quartiles, points beyond are outliers / 须线延伸到四分位数外1.5*IQR，超出部分是异常值
pyplot.boxplot(x)

# Add labels and title / 添加标签和标题
pyplot.ylabel('Value / 值')
pyplot.xlabel('Distribution / 分布')
pyplot.title('Box Plot Comparing Distributions with Different Variances / 箱线图比较不同方差的分布')
pyplot.xticks([1, 2, 3], ['N(0,1)', 'N(0,5)', 'N(0,10)'])

# Display the box plot / 显示箱线图
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** A box plot visualizes the five-number summary: minimum, Q1 (25th percentile), median (Q2, 50th percentile), Q3 (75th percentile), and maximum. The "box" spans Q1 to Q3 (the interquartile range, IQR). The median is marked by a line inside the box. Whiskers typically extend to 1.5×IQR beyond the quartiles; points beyond are plotted as potential outliers. Box plots enable rapid visual comparison of distributions across groups, revealing differences in location, spread, and skewness.

- **ML Application / 机器学习应用:** Box plots are invaluable for comparing model performance across validation folds or datasets. They visualize prediction error distributions and robustness. In feature analysis, box plots grouped by class reveal class-conditional distributions and feature importance—features with minimal overlap between class-conditional boxes are more discriminative. When debugging imbalanced datasets, box plots of feature values by class highlight potential distribution shift issues that affect model generalization.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `05_scatter_plot.ipynb` — Analyze bivariate relationships with scatter plot

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

# ===== Section 2: Generate Multiple Distributions =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate three distributions with increasing variance / 生成三个方差递增的分布
x = [randn(1000), 5 * randn(1000), 10 * randn(1000)]

# ===== Section 3: Create Box Plot =====
# Plot box-and-whisker diagram for each distribution / 为每个分布绘制箱须图
pyplot.boxplot(x)

# Add labels and annotations / 添加标签和注释
pyplot.ylabel('Value / 值')
pyplot.xlabel('Distribution / 分布')
pyplot.title('Box Plot Comparing Distributions with Different Variances / 箱线图比较不同方差的分布')
pyplot.xticks([1, 2, 3], ['N(0,1)', 'N(0,5)', 'N(0,10)'])

# Display plot / 显示图表
pyplot.show()
```

---

### Scatter Plot

# 05.05 — Scatter Plot / 散点图

**Chapter 05 — File 5 of 5**

## Summary / 摘要

**English:** This notebook creates a scatter plot to visualize the relationship between two continuous variables. Scatter plots reveal correlations, clusters, and outliers in bivariate data. Here we plot two correlated variables where y is positively related to x with added noise.

**中文:** 本笔记本创建散点图以可视化两个连续变量之间的关系。散点图揭示二元数据中的相关性、聚类和异常值。这里我们绘制两个相关变量，其中y与x呈正相关并加入噪声。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import random number generation and plotting library / 导入随机数生成和绘图库
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
```

## Step 2 — Generate Correlated Variables / 生成相关变量

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# First variable: random normal scaled and centered / 第一个变量：随机正态缩放和居中
# Creates range approximately 60-140 / 创建大约60-140的范围
x = 20 * randn(1000) + 100

# Second variable: positively correlated with x plus noise / 第二个变量：与x呈正相关加上噪声
# y = x + noise, so points scatter around y=x line / y = x + 噪声，所以点围绕y=x线散开
y = x + (10 * randn(1000) + 50)
```

## Step 3 — Create and Display Scatter Plot / 创建并显示散点图

```python
# Create a scatter plot showing relationship between x and y / 创建显示x和y之间关系的散点图
# Each point represents one observation (x_i, y_i) / 每个点代表一个观察(x_i, y_i)
pyplot.scatter(x, y)

# Add labels and title / 添加标签和标题
pyplot.xlabel('X Variable / X变量')
pyplot.ylabel('Y Variable (Correlated with X) / Y变量（与X相关）')
pyplot.title('Scatter Plot of Correlated Variables / 相关变量的散点图')

# Display the scatter plot / 显示散点图
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** A scatter plot displays pairs of numerical values as points in a 2D plane. The pattern of points reveals the relationship between variables: linear positive correlation shows points trending upward; negative correlation shows downward trend; no correlation shows scattered cloud. The tightness of the cloud indicates correlation strength—tight patterns suggest strong correlation; dispersed patterns suggest weak correlation. Scatter plots also reveal outliers (isolated points) and non-linear relationships (curved patterns).

- **ML Application / 机器学习应用:** Scatter plots are crucial for feature engineering and feature selection. Plotting features against the target variable reveals predictive potential—tight scatter indicates strong signal; dispersed patterns suggest weak predictors. In regression diagnostics, scatter plots of predicted vs. actual values assess model fit quality. Residual plots (residuals vs. fitted values) diagnose heteroscedasticity and non-linearity. In dimensionality reduction (PCA, t-SNE), scatter plots of component 1 vs. 2 visualize data structure and class separability.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `../chapter_06/01_python_seed.ipynb` — Begin Chapter 6 on Random Numbers

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

# ===== Section 2: Generate Correlated Data =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Generate first variable / 生成第一个变量
x = 20 * randn(1000) + 100

# Generate second variable with positive correlation to first / 生成与第一个呈正相关的第二个变量
y = x + (10 * randn(1000) + 50)

# ===== Section 3: Create Scatter Plot =====
# Plot points showing relationship between x and y / 绘制显示x和y关系的点
pyplot.scatter(x, y)

# Add labels and title / 添加标签和标题
pyplot.xlabel('X Variable / X变量')
pyplot.ylabel('Y Variable (Correlated with X) / Y变量（与X相关）')
pyplot.title('Scatter Plot of Correlated Variables / 相关变量的散点图')

# Display plot / 显示图表
pyplot.show()
```

---

### Chapter Summary

# Chapter 5: Data Visualization
# 第5章：数据可视化

## Theme | 主题
Progressive exploration of data structure through plot types, from univariate to bivariate.
通过图表类型从单变量到二变量逐步探索数据结构。

## Evolution Roadmap | 演变路线图
```
Line Plot (1D Continuous Sequence)
└─ Bar Chart (1D Categorical)
   └─ Histogram (1D Continuous Distribution)
      └─ Boxplot (1D Comparative Summary)
         └─ Scatter Plot (2D Relationship)
```

## Progression Logic | 进度逻辑

### Stage 1: Time Series (时间序列)
**English:** Line plot shows values over a sequence (time, samples), revealing trends.
**中文:** 线图显示序列上的值（时间、样本），揭示趋势。

### Stage 2: Categorical Data (分类数据)
**English:** Bar chart compares discrete categories via height/color, enabling category-level analysis.
**中文:** 柱状图通过高度/颜色比较离散类别，支持类别级分析。

### Stage 3: Distribution Shape (分布形状)
**English:** Histogram bins continuous values to expose density, center, and spread visually.
**中文:** 直方图将连续值分箱以直观展示密度、中心和扩展。

### Stage 4: Comparative Summary (比较总结)
**English:** Boxplot compresses distribution to quartiles and outliers, enabling side-by-side comparison of groups.
**中文:** 箱线图将分布压缩为四分位数和异常值，支持组的并排比较。

### Stage 5: Bivariate Relationship (二变量关系)
**English:** Scatter plot maps (x,y) pairs to reveal correlation, clusters, and outliers in 2D.
**中文:** 散点图映射(x,y)对以揭示2D中的相关性、聚类和异常值。

## ML Relevance | ML相关性

1. **EDA Foundation (EDA基础)**: Data visualization is the first step in exploratory data analysis.
2. **Distribution Assessment (分布评估)**: Histograms and boxplots identify skewness, multimodality, and outliers.
3. **Feature Relationships (特征关系)**: Scatter plots reveal linear/nonlinear relationships for feature engineering.
4. **Group Comparison (组比较)**: Boxplots and bar charts enable quick visual hypothesis about group differences.


---
