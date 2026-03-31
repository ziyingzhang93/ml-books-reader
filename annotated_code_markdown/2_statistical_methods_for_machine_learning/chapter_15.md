# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 15

---

### Power



---

### Power Analysis



---

### Power Analysis Fixed



---

### Chapter Summary / 章节总结

# Chapter 15: Statistical Power
# 第15章：统计功率

## Theme | 主题
Planning experiments: balancing sample size, effect size, significance level, and power to detect true effects.
规划实验：平衡样本量、效应大小、显著性水平和检测真实效应的功率。

## Evolution Roadmap | 演变路线图
```
Power Analysis Question: How many samples do I need?
└─ Power Curves
   (visualize tradeoffs: sample size vs. power, effect size, alpha)
```

## Progression Logic | 进度逻辑

### Stage 1: Problem Formulation (问题表述)
**English:** Specify desired significance level (α, typically 0.05), desired power (1 - β, typically 0.80), and effect size (d, from prior research or practical interest). Use power analysis libraries to compute required sample size.
**中文:** 指定所需显著性水平(α，通常为0.05)、所需功率(1 - β，通常为0.80)和效应大小(d，来自先前研究或实际兴趣)。使用功率分析库计算所需样本量。

### Stage 2: Sample Size Estimation (样本量估计)
**English:** Given α, 1-β, and d, calculate n. Larger d allows smaller n; smaller α requires larger n.
**中文:** 给定α、1-β和d，计算n。较大的d允许较小的n；较小的α需要较大的n。

### Stage 3: Power Curves (功率曲线)
**English:** Plot how power changes with sample size. Illustrate sensitivity to α and effect size. Helps choose practical sample size.
**中文:** 绘制功率如何随样本量变化。说明对α和效应大小的敏感性。有助于选择实际样本量。

## ML Relevance | ML相关性

1. **Experiment Design (实验设计)**: Power analysis determines minimum test/validation set size for reliable performance estimation.
2. **A/B Testing (A/B测试)**: How long should I run the experiment? Power analysis answers this before data collection.
3. **Resource Planning (资源规划)**: Trading sample size against effect size and alpha guides data collection budgets.
4. **Robustness (鲁棒性)**: Adequate power protects against false negatives (Type II errors) and ensures consistent findings.


---
