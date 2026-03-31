# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 14

---

### Association



---

### Difference



---

### Chapter Summary / 章节总结

# Chapter 14: Effect Size
# 第14章：效应大小

## Theme | 主题
Beyond p-values: quantifying the practical magnitude of association and difference.
超越p值：量化关联和差异的实际幅度。

## Evolution Roadmap | 演变路线图
```
ASSOCIATION STRENGTH (between two variables):
  Pearson r
  (correlation coefficient, [-1, 1])

DIFFERENCE MAGNITUDE (between two groups):
  Cohen's d
  (standardized mean difference, dimensionless)
```

## Progression Logic | 进度逻辑

### Stage 1: Pearson Correlation r (皮尔逊相关r)
**English:** Measures strength of linear association between two continuous variables. Already covered in Chapter 12, here we interpret effect size: |r| < 0.1 (negligible), 0.1-0.3 (small), 0.3-0.5 (medium), > 0.5 (large).
**中文:** 测量两个连续变量之间线性关联的强度。已在第12章中介绍，这里我们解释效应大小：|r| < 0.1(可忽略)，0.1-0.3(小)，0.3-0.5(中等)，> 0.5(大)。

### Stage 2: Cohen's d (科恩's d)
**English:** Standardized mean difference: d = (μ1 - μ2) / σ_pooled. Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), > 0.8 (large).
**中文:** 标准化的平均差：d = (μ1 - μ2) / σ_pooled。解释：|d| < 0.2(可忽略)，0.2-0.5(小)，0.5-0.8(中等)，> 0.8(大)。

## ML Relevance | ML相关性

1. **Practical Significance (实际意义)**: p < 0.05 means statistically significant, but effect size shows if the difference is practically meaningful.
2. **Model Comparison (模型比较)**: Effect size quantifies improvement of one model over another beyond just p-values.
3. **Power Analysis (功率分析)**: Effect size and desired power determine required sample size (Chapter 15).
4. **Publication Standards (出版标准)**: Journals increasingly require reporting effect sizes alongside p-values for transparency.


---
