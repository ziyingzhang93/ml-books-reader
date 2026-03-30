# 概率论与机器学习
## Chapter 13

---

### Odds

# 01 — Odds / 赔率

**Chapter 13 — File 1 of 3**

## Summary

We convert a probability to odds and back. Odds express the ratio of favorable to unfavorable outcomes. For probability p, odds = p/(1-p). This is fundamental for understanding logistic regression and odds ratios in statistics.

我们将概率转换为赔率，然后转换回去。赔率表示有利结果与不利结果的比率。对于概率p，赔率= p/(1-p)。这对于理解逻辑回归和统计中的赔率比至关重要。

**Formula:**

$$\text{odds} = \frac{p}{1-p}$$

$$p = \frac{\text{odds}}{1 + \text{odds}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Convert Probability to Odds / 将概率转换为赔率

```python
# Convert probability to odds / 将概率转换为赔率
p = 0.8  # Probability / 概率

# Calculate odds / 计算赔率
odds = p / (1 - p)

print(f'Probability to Odds Conversion / 概率到赔率转换')
print(f'=' * 50)
print(f'Probability (p): {p}')
print(f'Odds: {odds:.4f}')
print(f'\nInterpretation / 解释:')
print(f'For every 1 unit of failure, there are {odds:.2f} units of success')
print(f'对于每1个失败单位，有{odds:.2f}个成功单位')
```

## Step 2 — Convert Odds Back to Probability / 将赔率转换回概率

```python
# Convert odds back to probability / 将赔率转换回概率
p_reconstructed = odds / (1 + odds)

print(f'\nOdds to Probability Conversion / 赔率到概率转换')
print(f'=' * 50)
print(f'Odds: {odds:.4f}')
print(f'Reconstructed Probability: {p_reconstructed:.4f}')
print(f'Original Probability: {p:.4f}')
print(f'Match: {abs(p - p_reconstructed) < 1e-10}')
```

## Step 3 — Compare Multiple Probabilities / 比较多个概率

```python
# Create a table of probabilities and odds / 创建概率和赔率的表格
import numpy as np

probabilities = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

print(f'\nProbability to Odds Mapping / 概率到赔率映射')
print(f'=' * 70)
print(f'{"Probability":>12s} {"Odds":>15s} {"Interpretation":>40s}')
print('-' * 70)

for prob in probabilities:
    o = prob / (1 - prob)
    interp = f'1:{(1-prob)/prob:.2f}'
    print(f'{prob:12.2f} {o:15.4f} {interp:>40s}')
```

## Learning Notes / 学习笔记

- **Concept**: Odds are a way to express probability that emphasizes the ratio between events. While probabilities must be between 0 and 1, odds range from 0 to infinity. Odds of 1 means equal likelihood of success and failure.

- **ML Application**: Odds are central to logistic regression interpretation (logit), medical test interpretation (sensitivity/specificity), and calculating odds ratios in causal inference.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Next / 下一步\n\n➡️ **Next**: `02_log_odds.ipynb`

---

### Log Odds

# 02 — Log-Odds / 对数赔率

**Chapter 13 — File 2 of 3**

## Summary

We convert probability to log-odds (logit) and back. Log-odds is the natural logarithm of odds, which linearizes probability in many machine learning models. The inverse function is the sigmoid function.

我们将概率转换为对数赔率（logit），然后转换回去。对数赔率是赔率的自然对数，在许多机器学习模型中使概率线性化。反函数是sigmoid函数。

**Formula:**

$$\text{logit}(p) = \log\frac{p}{1-p}$$

$$\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Convert Probability to Log-Odds / 将概率转换为对数赔率

```python
import numpy as np
import matplotlib.pyplot as plt

# Convert probability to log-odds / 将概率转换为对数赔率
p = 0.8

# Calculate log-odds / 计算对数赔率
log_odds = np.log(p / (1 - p))

print(f'Probability to Log-Odds Conversion / 概率到对数赔率转换')
print(f'=' * 60)
print(f'Probability (p): {p}')
print(f'Odds: {p / (1 - p):.4f}')
print(f'Log-Odds (logit): {log_odds:.4f}')
```

## Step 2 — Convert Log-Odds Back to Probability / 将对数赔率转换回概率

```python
# Sigmoid function (inverse of logit) / Sigmoid函数（logit的反函数）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Convert log-odds back to probability / 将对数赔率转换回概率
p_reconstructed = sigmoid(log_odds)

print(f'\nLog-Odds to Probability Conversion / 对数赔率到概率转换')
print(f'=' * 60)
print(f'Log-Odds: {log_odds:.4f}')
print(f'Reconstructed Probability: {p_reconstructed:.4f}')
print(f'Original Probability: {p:.4f}')
print(f'Match: {abs(p - p_reconstructed) < 1e-10}')
```

## Step 3 — Plot the Relationship / 绘制关系

```python
# Plot logit and sigmoid functions / 绘制logit和sigmoid函数
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot logit: Probability to Log-Odds / 绘制logit：概率到对数赔率
p_vals = np.linspace(0.001, 0.999, 1000)
logit_vals = np.log(p_vals / (1 - p_vals))

axes[0].plot(p_vals, logit_vals, 'b-', linewidth=2)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(0.5, color='k', linestyle='--', alpha=0.3)
axes[0].scatter([p], [log_odds], color='r', s=100, zorder=5, label=f'p={p}')
axes[0].set_xlabel('Probability (p) / 概率')
axes[0].set_ylabel('Log-Odds (logit) / 对数赔率')
axes[0].set_title('Logit Function: p → log-odds')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot sigmoid: Log-Odds to Probability / 绘制sigmoid：对数赔率到概率
z_vals = np.linspace(-10, 10, 1000)
sigmoid_vals = sigmoid(z_vals)

axes[1].plot(z_vals, sigmoid_vals, 'g-', linewidth=2)
axes[1].axhline(0.5, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[1].scatter([log_odds], [p_reconstructed], color='r', s=100, zorder=5, label=f'z={log_odds:.3f}')
axes[1].set_xlabel('Log-Odds (z) / 对数赔率')
axes[1].set_ylabel('Probability (p) / 概率')
axes[1].set_title('Sigmoid Function: z → probability')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: Log-odds linearizes probability, which is why logistic regression uses it. The logit function maps (0,1) to (-∞,∞), and sigmoid does the reverse. This is fundamental to understanding how probabilities are modeled in machine learning.

- **ML Application**: Logistic regression directly models log-odds, neural networks use sigmoid activation functions, and log-odds interpretation helps explain model predictions.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步\n\n➡️ **Next**: `03_likelihood.ipynb`

---

### Likelihood

# 03 — Likelihood / 似然

**Chapter 13 — File 3 of 3**

## Summary

We implement the Bernoulli likelihood function, which measures how well a model's predicted probability matches observed binary outcomes. This is the foundation for maximum likelihood estimation and loss functions in classification.

我们实现伯努利似然函数，它衡量模型的预测概率与观测二元结果的匹配程度。这是最大似然估计和分类中损失函数的基础。

**Formula:**

$$L(y, \hat{y}) = \hat{y}^y (1-\hat{y})^{1-y}$$

$$\log L = y \log \hat{y} + (1-y) \log(1-\hat{y})$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Define Likelihood Function / 定义似然函数

```python
import numpy as np
import matplotlib.pyplot as plt

# Bernoulli likelihood function / 伯努利似然函数
def bernoulli_likelihood(y, y_hat):
    """
    Bernoulli likelihood for binary outcomes.
    L(y, ŷ) = ŷ^y * (1-ŷ)^(1-y)
    
    伯努利似然函数用于二元结果。
    
    Args:
        y: Observed outcome (0 or 1) / 观测结果
        y_hat: Predicted probability (0 to 1) / 预测概率
    
    Returns:
        Likelihood value / 似然值
    """
    return y_hat**y * (1 - y_hat)**(1 - y)

# Log-likelihood (numerically stable) / 对数似然（数值稳定）
def log_bernoulli_likelihood(y, y_hat):
    """
    Log-likelihood (more numerically stable).
    log L = y * log(ŷ) + (1-y) * log(1-ŷ)
    """
    epsilon = 1e-15  # Avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

print('Bernoulli Likelihood Function / 伯努利似然函数')
print('=' * 60)
```

## Step 2 — Calculate Likelihood for Different Predictions / 计算不同预测的似然

## Step 2 — Calculate Likelihood for Different Predictions / 计算不同预测的似然

```python
# Example: y=1 (positive class) / 例子：y=1（正类）
y_true = 1
predictions = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

print(f'\nWhen y (true label) = {y_true}:')
print(f'{"Prediction":>12s} {"Likelihood":>15s} {"Log-Likelihood":>15s} {"Quality":>15s}')
print('-' * 60)

for y_hat in predictions:
    L = bernoulli_likelihood(y_true, y_hat)
    log_L = log_bernoulli_likelihood(y_true, y_hat)
    quality = "Good" if y_hat > 0.7 else ("OK" if y_hat > 0.3 else "Bad")
    print(f'{y_hat:12.2f} {L:15.6f} {log_L:15.6f} {quality:>15s}')

# Example: y=0 (negative class) / 例子：y=0（负类）
y_true = 0
print(f'\n\nWhen y (true label) = {y_true}:')
print(f'{"Prediction":>12s} {"Likelihood":>15s} {"Log-Likelihood":>15s} {"Quality":>15s}')
print('-' * 60)

for y_hat in predictions:
    L = bernoulli_likelihood(y_true, y_hat)
    log_L = log_bernoulli_likelihood(y_true, y_hat)
    quality = "Good" if y_hat < 0.3 else ("OK" if y_hat < 0.7 else "Bad")
    print(f'{y_hat:12.2f} {L:15.6f} {log_L:15.6f} {quality:>15s}')
```

## Step 3 — Visualize Likelihood / 可视化似然

## Step 3 — Visualize Likelihood / 可视化似然

```python
# Plot likelihood for y=0 and y=1 / 绘制y=0和y=1的似然
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_hat_vals = np.linspace(0.001, 0.999, 1000)

# Likelihood for y=1 / y=1的似然
L_y1 = bernoulli_likelihood(1, y_hat_vals)
log_L_y1 = log_bernoulli_likelihood(1, y_hat_vals)

axes[0].plot(y_hat_vals, L_y1, 'b-', linewidth=2, label='y=1 (Positive)')
axes[0].plot(y_hat_vals, bernoulli_likelihood(0, y_hat_vals), 'r-', linewidth=2, label='y=0 (Negative)')
axes[0].set_xlabel('Predicted Probability (ŷ) / 预测概率')
axes[0].set_ylabel('Likelihood / 似然')
axes[0].set_title('Bernoulli Likelihood Function')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Log-likelihood / 对数似然
axes[1].plot(y_hat_vals, log_L_y1, 'b-', linewidth=2, label='y=1 (Positive)')
axes[1].plot(y_hat_vals, log_bernoulli_likelihood(0, y_hat_vals), 'r-', linewidth=2, label='y=0 (Negative)')
axes[1].set_xlabel('Predicted Probability (ŷ) / 预测概率')
axes[1].set_ylabel('Log-Likelihood / 对数似然')
axes[1].set_title('Log-Bernoulli Likelihood Function')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: The Bernoulli likelihood quantifies model quality. Higher likelihood means predictions align better with observations. Log-likelihood is used in practice because it's numerically stable and additive over samples.

- **ML Application**: Likelihood is the foundation for maximum likelihood estimation, logistic regression loss functions, cross-entropy loss in classification, and Bayesian inference.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Chapter 13 Complete / 第13章完成

This chapter covered fundamental probability concepts: odds, log-odds, and likelihood—all essential for understanding classification models.

---
