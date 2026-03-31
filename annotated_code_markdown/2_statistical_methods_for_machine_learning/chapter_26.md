# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 26

---

### Summary

# 26 — Five-Number Summary / 五数摘要

**Chapter 26 — File 1 of 1**

## Summary / 摘要

The five-number summary is a descriptive statistic that provides a quick overview of the distribution of a dataset using five key percentiles: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum. This summary is particularly useful for understanding data spread, central tendency, and identifying outliers.

五数摘要是一种描述性统计量，通过五个关键百分位数概括数据集的分布特征：最小值、第一四分位数(Q1)、中位数(Q2)、第三四分位数(Q3)和最大值。这个摘要对于理解数据范围、集中趋势和识别离群值特别有用。

### Key Formula / 关键公式

$$\text{Five-Number Summary} = [\text{min}, Q_1, Q_2 (\text{median}), Q_3, \text{max}]$$

where quartiles divide the sorted data into four equal parts.

## Step 1 — Data Generation / 数据生成

```python
from numpy import percentile
from numpy.random import seed, rand

# Set random seed for reproducibility / 设置随机种子以保证可重现性
seed(1)

# Generate 1000 uniform random samples in [0, 1) / 生成1000个[0,1)范围内的均匀分布样本
data = rand(1000)
```

## Step 2 — Calculate Five-Number Summary / 计算五数摘要

```python
# Calculate quartiles at 25th, 50th (median), and 75th percentiles
# 在第25、50(中位数)和75百分位计算四分位数
quartiles = percentile(data, [25, 50, 75])

# Calculate minimum and maximum values / 计算最小值和最大值
data_min, data_max = data.min(), data.max()

# Display the five-number summary
# 显示五数摘要
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)
```

## Learning Notes / 学习笔记

- **Statistical Concept**: The five-number summary provides a robust description of data distribution that is less sensitive to outliers than mean and standard deviation. The interquartile range (IQR = Q3 - Q1) represents the spread of the middle 50% of data.

  **统计概念**: 五数摘要提供了对数据分布的鲁棒描述，比平均值和标准差对离群值的敏感性更低。四分位距(IQR = Q3 - Q1)表示中间50%数据的分布范围。

- **ML Application**: In machine learning preprocessing, the five-number summary helps identify data quality issues, detect outliers using the IQR method (values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR are flagged), and understand feature distributions before normalization or scaling.

  **ML应用**: 在机器学习的数据预处理中，五数摘要有助于识别数据质量问题、使用四分位距法检测离群值(IQR法)，并在标准化或缩放前理解特征分布。

➡️ **Next**: This is the final notebook for Chapter 26.

## Complete Code / 完整代码一览

```python
from numpy import percentile
from numpy.random import seed, rand

seed(1)
data = rand(1000)
quartiles = percentile(data, [25, 50, 75])
data_min, data_max = data.min(), data.max()

print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)
```

---

### Chapter Summary / 章节总结



---
