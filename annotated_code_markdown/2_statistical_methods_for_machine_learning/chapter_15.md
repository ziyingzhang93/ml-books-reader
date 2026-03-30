# 统计方法与机器学习 / Statistical Methods for Machine Learning
## Chapter 15

---

### Power



---

### Power Analysis



---

### Power Analysis Fixed

# 15.2 — Power Curves for Different Effect Sizes / 不同效应量的功效曲线

**Chapter 15 — File 2 of 2**

## Summary / 摘要

Power curves visualize the relationship between sample size and statistical power for different effect sizes. They are essential tools for study design, showing how increasing sample size improves the ability to detect effects. This notebook creates comprehensive power curves and provides interpretation guidance.

功效曲线可视化不同效应量的样本量和统计功效之间的关系。它们是研究设计的必不可少工具，显示增加样本量如何提高检测效应的能力。本笔记本创建全面的功效曲线并提供解释指导。

**Relationship / 关系:**

Power increases with: (1) larger effect size, (2) larger sample size, (3) larger significance level α

## Step 1 — Import and Setup / 导入和设置

```python
from statsmodels.stats.power import ttest_ind_solve_power
import numpy as np
from matplotlib import pyplot

# 样本量范围
# Range of sample sizes to examine
sample_sizes = np.linspace(5, 300, 100)
```

## Step 2 — Create Comprehensive Power Curves / 创建全面的功效曲线

```python
# 为不同效应量计算功效
# Compute power for different effect sizes
effect_sizes = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2])
alpha = 0.05

# 计算每个效应量的功效
# Compute power for each effect size
power_curves = {}

for d in effect_sizes:
    powers = []
    for n in sample_sizes:
        power = ttest_ind_solve_power(
            effect_size=d,
            nobs1=n,
            alpha=alpha,
            power=None,
            ratio=1.0,
            alternative='two-sided'
        )
        powers.append(power)
    power_curves[d] = np.array(powers)

print(f"Computed power curves for {len(effect_sizes)} effect sizes")
print(f"Sample sizes range from {sample_sizes[0]:.0f} to {sample_sizes[-1]:.0f}")
```

## Step 3 — Plot Power Curves / 绘制功效曲线

```python
# 创建详细的功效曲线图
# Create detailed power curve plots
fig, axes = pyplot.subplots(1, 2, figsize=(15, 6))

# 左图：所有效应量
# Left plot: All effect sizes
ax = axes[0]
colors = pyplot.cm.viridis(np.linspace(0, 1, len(effect_sizes)))

for i, d in enumerate(effect_sizes):
    ax.plot(sample_sizes, power_curves[d], linewidth=2.5, 
           color=colors[i], label=f'd={d:.1f}', marker='o', 
           markersize=3, markevery=10)

# 标记重要功效水平
# Mark important power levels
ax.axhline(y=0.80, color='red', linestyle='--', linewidth=2, 
          alpha=0.7, label='Power = 0.80 (Standard)')
ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, 
          alpha=0.7, label='Power = 0.90 (High)')

ax.set_xlabel('Sample Size per Group (n)', fontsize=11)
ax.set_ylabel('Statistical Power', fontsize=11)
ax.set_title('Power Curves by Effect Size (α=0.05, two-tailed)', fontsize=12)
ax.legend(loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 300])

# 右图：分组显示
ax = axes[1]
for d in [0.2, 0.5, 0.8]:
    ax.plot(sample_sizes, power_curves[d], linewidth=2.5, label=f'd={d:.1f}')

ax.axhline(y=0.80, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Sample Size per Group (n)', fontsize=11)
ax.set_ylabel('Statistical Power', fontsize=11)
ax.set_title('Effect Size Categories', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 300])

pyplot.tight_layout()
pyplot.show()
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念**: Power curves demonstrate the non-linear relationship between sample size and power. Power increases rapidly initially but plateaus at large sample sizes. The shape of curves depends on effect size: small effects require exponentially larger samples. Power analysis enables researchers to design studies that balance practical feasibility with statistical rigor.

- **ML Application / 机器学习应用**: Power curves guide A/B test planning. Help determine whether observing non-significant results means no effect exists or just insufficient power. Guide meta-analysis planning. Essential for responsible reporting: always disclose power and effect size alongside p-values.

## Complete Code / 完整代码一览

```python
from statsmodels.stats.power import ttest_ind_solve_power
import numpy as np
from matplotlib import pyplot

# Compute power curves
sample_sizes = np.linspace(5, 300, 100)
effect_sizes = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2])
alpha = 0.05

power_curves = {}
for d in effect_sizes:
    powers = [ttest_ind_solve_power(d, n, alpha, None, 1.0, 'two-sided')
              for n in sample_sizes]
    power_curves[d] = np.array(powers)

# Plot power curves
pyplot.figure(figsize=(10, 6))
for d in effect_sizes:
    pyplot.plot(sample_sizes, power_curves[d], linewidth=2, label=f'd={d:.1f}')

pyplot.axhline(y=0.80, color='red', linestyle='--', label='Power=0.80')
pyplot.xlabel('Sample Size per Group')
pyplot.ylabel('Statistical Power')
pyplot.title('Power Curves by Effect Size')
pyplot.legend()
pyplot.grid(True, alpha=0.3)
pyplot.show()
```

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
