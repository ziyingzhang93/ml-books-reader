# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 13

---

### Test Data

# 13.1 — Generate Test Data for Hypothesis Tests / 为假设检验生成测试数据\n\n**Chapter 13 — File 1 of 4**\n\n## Summary / 摘要\n\nBefore performing hypothesis tests, we need sample data with known properties. This notebook generates two independent Gaussian samples with different means to demonstrate t-tests and ANOVA.\n\n在进行假设检验之前，我们需要具有已知属性的样本数据。本笔记本生成两个具有不同均值的独立高斯样本来演示 t 检验和 ANOVA。\n\n**Key Concept / 关键概念:**\n\nSample size, mean difference, and variance all affect hypothesis test power and results.

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random, mean, std, array\nfrom matplotlib import pyplot\n\n# 设置随机种子以确保可重复性\n# Set random seed for reproducibility
```

## Step 2 — Generate Two Independent Groups / 生成两个独立组

## Step 3 — Visualize the Samples / 可视化样本

## Step 4 — Summary Statistics / 汇总统计

## Step 5 — Check Normality / 检验正态性

## Learning Notes / 学习笔记\n\n- **Statistical Concept / 统计学概念**: Sample size affects hypothesis test power. Larger samples provide better estimates and more statistical power. Two groups with identical distributions but different means are ideal for demonstrating difference tests. Normality of samples justifies use of parametric tests. / 样本大小影响假设检验的功效。较大的样本提供更好的估计和更多的统计功效。两个分布相同但均值不同的组非常适合演示差异检验。样本的正态性证明了参数检验的使用的合理性。\n\n- **ML Application / 机器学习应用**: Hypothesis test data generation is crucial for validation and A/B testing in machine learning. Understanding sample properties (size, variance, mean difference) informs experimental design. Proper test data helps validate model performance improvements and detect algorithmic biases. / 假设检验数据生成对机器学习中的验证和 A/B 测试至关重要。理解样本属性（大小、方差、均值差异）指导实验设计。正确的测试数据有助于验证模型性能改进和检测算法偏差。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

➡️ **Next**: `02_ttest.ipynb`

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import random, mean, std\nfrom matplotlib import pyplot\nfrom scipy.stats import shapiro\nimport numpy as np\nimport pandas as pd\n\n# Generate samples\nrandom.seed(1)\nsample1 = random.normal(loc=50, scale=5, size=100)\nsample2 = random.normal(loc=51, scale=5, size=100)\n\nprint(f\"Sample 1: mean={mean(sample1):.4f}, std={std(sample1):.4f}\")\nprint(f\"Sample 2: mean={mean(sample2):.4f}, std={std(sample2):.4f}\")\n\n# Visualize\npyplot.figure(figsize=(12, 5))\npyplot.subplot(1, 2, 1)\npyplot.hist(sample1, bins=20, alpha=0.6, label='Sample 1')\npyplot.hist(sample2, bins=20, alpha=0.6, label='Sample 2')\npyplot.legend()\npyplot.subplot(1, 2, 2)\npyplot.boxplot([sample1, sample2], labels=['Sample 1', 'Sample 2'])\npyplot.tight_layout()\npyplot.show()\n\n# Check normality\nstat1, p1 = shapiro(sample1)\nstat2, p2 = shapiro(sample2)\nprint(f\"\\nNormality test - Sample 1: p={p1:.4f}\")\nprint(f\"Normality test - Sample 2: p={p2:.4f}\")
```

---

### Ttest

# 13.2 — Independent Samples t-test / 独立样本 t 检验\n\n**Chapter 13 — File 2 of 4**\n\n## Summary / 摘要\n\nThe independent samples t-test compares means of two independent groups. It tests whether the population means are equal. This is the most commonly used parametric test in statistics and machine learning.\n\n独立样本 t 检验比较两个独立组的均值。它测试总体均值是否相等。这是统计学和机器学习中最常用的参数检验。\n\n**Mathematical Formula / 数学公式:**\n\n$$t = \\frac{\\bar{X}_1 - \\bar{X}_2}{s_p\\sqrt{2/n}}$$\n\nwhere $s_p = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$ is the pooled standard deviation.

## Step 1 — Import and Load Data / 导入并加载数据

## Step 2 — Perform Independent t-test / 执行独立 t 检验

## Step 3 — Hypothesis Testing Interpretation / 假设检验解释

---

### Paired Ttest

# 13.3 — Paired Samples t-test / 配对样本 t 检验\n\n**Chapter 13 — File 3 of 4**\n\n## Summary / 摘要\n\nThe paired samples t-test compares means from the same subjects measured at different times or under different conditions. It accounts for within-subject correlation, making it more powerful than independent t-tests for paired data.\n\n配对样本 t 检验比较相同受试者在不同时间或不同条件下的测量值。它考虑了受试者内相关性，使其比配对数据的独立 t 检验更强大。\n\n**Mathematical Formula / 数学公式:**\n\n$$t = \\frac{\\bar{D}}{s_D/\\sqrt{n}}$$\n\nwhere $\\bar{D}$ is the mean difference and $s_D$ is the standard deviation of differences.

---

### Anova

# 13.4 — One-Way ANOVA / 单因素方差分析\n\n**Chapter 13 — File 4 of 4**\n\n## Summary / 摘要\n\nOne-way ANOVA (Analysis of Variance) extends the t-test to compare means of three or more independent groups. It tests whether there are significant differences among group means, avoiding the multiple testing problem inherent in multiple pairwise t-tests.\n\n单因素 ANOVA（方差分析）将 t 检验扩展到比较三个或更多独立组的均值。它测试组均值之间是否存在显著差异，避免了多次成对 t 检验中固有的多重测试问题。\n\n**Mathematical Formula / 数学公式:**\n\n$$F = \\frac{\\text{MS}_{\\text{between}}}{\\text{MS}_{\\text{within}}} = \\frac{\\sum_i n_i(\\bar{X}_i - \\bar{X})^2/(k-1)}{\\sum_i \\sum_j (X_{ij} - \\bar{X}_i)^2/(N-k)}$$

---

### Chapter Summary / 章节总结

# Chapter 13: Parametric Hypothesis Tests
# 第13章：参数假设检验

## Theme | 主题
From two independent groups to paired groups to three+ groups: progressive expansion of comparison methods.
从两个独立组到配对组到三个+组：比较方法的逐步扩展。

## Evolution Roadmap | 演变路线图
```
Test Data (two independent samples)
└─ Independent Samples t-Test
   (Are means of two groups different?)
   └─ Paired Samples t-Test
      (Are means of paired observations different?)
      └─ One-Way ANOVA
         (Are means of 3+ groups different?)
```

## Progression Logic | 进度逻辑

### Stage 1: Independent Samples t-Test (独立样本t检验)
**English:** Compare means of two unrelated groups. H0: μ1 = μ2. Assumes equal variances (Welch's variant relaxes this).
**中文:** 比较两个不相关组的均值。H0: μ1 = μ2。假设方差相等(Welch变体放宽此条件)。

### Stage 2: Paired Samples t-Test (配对样本t检验)
**English:** Compare means of two related samples (e.g., before/after). Computes differences, tests if mean(diff) = 0.
**中文:** 比较两个相关样本的均值(例如前/后)。计算差异，检验是否mean(diff) = 0。

### Stage 3: One-Way ANOVA (单因素方差分析)
**English:** Compare means of 3+ independent groups. H0: all means are equal. F-statistic = variance between groups / variance within groups.
**中文:** 比较3个+独立组的均值。H0：所有均值相等。F统计 = 组间方差 / 组内方差。

### Interpretation (解释)
**English:** p-value < α: reject H0, groups differ significantly. p-value ≥ α: fail to reject H0.
**中文:** p值 < α：拒绝H0，组有显著差异。p值 ≥ α：未能拒绝H0。

## ML Relevance | ML相关性

1. **A/B Testing (A/B测试)**: t-test evaluates if two variants (e.g., UI versions) have significantly different outcomes.
2. **Group Comparison (组比较)**: ANOVA determines if multiple conditions/treatments have different effects.
3. **Model Evaluation (模型评估)**: t-test compares performance of two models; ANOVA compares 3+ models.
4. **Experiment Design (实验设计)**: Paired t-test accounts for within-subject variation, increasing statistical power.


---
