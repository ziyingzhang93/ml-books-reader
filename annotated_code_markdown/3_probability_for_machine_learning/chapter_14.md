# 概率论与机器学习
## Chapter 14

---

### Bimodal Data Sample

# 01 — Bimodal Data Sample / 双峰数据样本

**Chapter 14 — File 1 of 2**

## Summary

We generate bimodal data from two Gaussian distributions with different means and variances. This creates data that clearly has two distinct clusters, perfect for demonstrating Gaussian Mixture Models.

我们从两个具有不同均值和方差的高斯分布生成双峰数据。这创建了明确有两个不同簇的数据，非常适合演示高斯混合模型。

**Formula:**

$$X \sim p \cdot N(\mu_1, \sigma_1^2) + (1-p) \cdot N(\mu_2, \sigma_2^2)$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Generate Bimodal Data / 生成双峰数据

## Step 1 — Generate Bimodal Data / 生成双峰数据

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate bimodal data / 生成双峰数据
np.random.seed(42)

# Component 1: N(20, 25) / 组件1
n1 = 500
mu1, sigma1 = 20, 5
component1 = np.random.normal(loc=mu1, scale=sigma1, size=n1)

# Component 2: N(40, 25) / 组件2
n2 = 500
mu2, sigma2 = 40, 5
component2 = np.random.normal(loc=mu2, scale=sigma2, size=n2)

# Combine / 组合
bimodal_data = np.concatenate([component1, component2])
labels = np.concatenate([np.zeros(n1), np.ones(n2)])  # True labels / 真实标签

print(f'Bimodal Data Generation / 双峰数据生成')
print(f'Component 1: N({mu1}, {sigma1}²), n={n1}')
print(f'Component 2: N({mu2}, {sigma2}²), n={n2}')
print(f'\nCombined Data Statistics / 组合数据统计:')
print(f'Mean: {np.mean(bimodal_data):.4f}')
print(f'Std: {np.std(bimodal_data):.4f}')
print(f'Min: {np.min(bimodal_data):.4f}')
print(f'Max: {np.max(bimodal_data):.4f}')
```

## Step 2 — Visualize Data / 可视化数据

## Step 2 — Visualize Data / 可视化数据

## Step 2 — Visualize Data / 可视化数据

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram / 直方图
axes[0].hist(bimodal_data, bins=40, alpha=0.7, edgecolor='black', color='steelblue')
axes[0].axvline(mu1, color='r', linestyle='--', linewidth=2, label=f'μ₁={mu1}')
axes[0].axvline(mu2, color='g', linestyle='--', linewidth=2, label=f'μ₂={mu2}')
axes[0].set_xlabel('Value / 值')
axes[0].set_ylabel('Count / 计数')
axes[0].set_title('Bimodal Data Histogram')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Scatter plot / 散点图
axes[1].scatter(bimodal_data[labels == 0], np.random.normal(0, 0.02, n1), 
               alpha=0.5, s=50, color='red', label='Component 1')
axes[1].scatter(bimodal_data[labels == 1], np.random.normal(0, 0.02, n2), 
               alpha=0.5, s=50, color='green', label='Component 2')
axes[1].set_xlabel('Value / 值')
axes[1].set_ylabel('Jitter (for visualization) / 抖动')
axes[1].set_title('Components in Bimodal Data')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim([-0.2, 0.2])

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: Bimodal and multimodal distributions are common in real-world data, often representing different subpopulations or clusters. Single-component models cannot capture their structure.

- **ML Application**: Mixture models are essential for clustering, unsupervised learning, and modeling heterogeneous populations.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.ones` | 全一数组 | Array filled with ones |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `np.zeros` | 全零数组 | Array filled with zeros |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步\n\n➡️ **Next**: `02_gaussian_mixture_em.ipynb`

---

### Gaussian Mixture Em

# 02 — Gaussian Mixture EM / 高斯混合模型

**Chapter 14 — File 2 of 2**

## Summary

We fit a Gaussian Mixture Model (GMM) to bimodal data using the EM algorithm via sklearn. We then predict cluster labels and visualize the fitted model's PDF overlaid on the data.

我们使用sklearn的EM算法将高斯混合模型（GMM）拟合到双峰数据。然后我们预测簇标签并可视化拟合模型的PDF覆盖在数据上。

**Formula:**

$$p(x) = \sum_{k=1}^{K} w_k N(x | \mu_k, \Sigma_k)$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Generate and Fit GMM / 生成和拟合GMM

## Step 1 — Generate and Fit GMM / 生成和拟合GMM

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Generate bimodal data / 生成双峰数据
np.random.seed(42)
component1 = np.random.normal(loc=20, scale=5, size=500)
component2 = np.random.normal(loc=40, scale=5, size=500)
bimodal_data = np.concatenate([component1, component2]).reshape(-1, 1)

# Fit GMM with 2 components / 用2个组件拟合GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(bimodal_data)

print(f'Gaussian Mixture Model / 高斯混合模型')
print(f'=' * 70)
print(f'Number of components: 2')
print(f'\nComponent Parameters / 组件参数:')
for i in range(2):
    print(f'\nComponent {i+1}:')
    print(f'  Weight (π): {gmm.weights_[i]:.4f}')
    print(f'  Mean (μ): {gmm.means_[i][0]:.4f}')
    print(f'  Covariance (σ²): {gmm.covariances_[i][0][0]:.4f}')
    print(f'  Std Dev (σ): {np.sqrt(gmm.covariances_[i][0][0]):.4f}')
```

## Step 2 — Predict Cluster Labels / 预测簇标签

```python
# Predict cluster labels / 预测簇标签
labels = gmm.predict(bimodal_data)

# Get probabilities / 获取概率
probs = gmm.predict_proba(bimodal_data)

print(f'\nPredicted Labels / 预测标签:')
print(f'Cluster 0: {np.sum(labels == 0)} points')
print(f'Cluster 1: {np.sum(labels == 1)} points')
print(f'\nLog-Likelihood / 对数似然: {gmm.score(bimodal_data):.4f}')
```

## Step 3 — Visualize Results / 可视化结果

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data with predicted clusters / 绘图1：带预测簇的数据
axes[0].scatter(bimodal_data[labels == 0], [0]*np.sum(labels == 0), 
               alpha=0.5, s=50, color='red', label='Cluster 0')
axes[0].scatter(bimodal_data[labels == 1], [0]*np.sum(labels == 1), 
               alpha=0.5, s=50, color='blue', label='Cluster 1')
axes[0].set_xlabel('Value / 值')
axes[0].set_title('Predicted Clusters')
axes[0].legend()
axes[0].set_ylim([-0.5, 0.5])
axes[0].grid(alpha=0.3)

# Plot 2: Fitted GMM PDF / 绘图2：拟合的GMM PDF
x = np.linspace(5, 55, 1000).reshape(-1, 1)
density = np.exp(gmm.score_samples(x))
component_density = []
for i in range(2):
    comp_dens = gmm.weights_[i] * norm.pdf(x[:, 0], gmm.means_[i][0], 
                                            np.sqrt(gmm.covariances_[i][0][0]))
    component_density.append(comp_dens)

axes[1].hist(bimodal_data, bins=30, density=True, alpha=0.5, 
            color='gray', edgecolor='black', label='Data')
axes[1].plot(x, density, 'k-', linewidth=2.5, label='GMM PDF')
axes[1].plot(x, component_density[0], 'r--', linewidth=2, label='Component 0')
axes[1].plot(x, component_density[1], 'b--', linewidth=2, label='Component 1')
axes[1].set_xlabel('Value / 值')
axes[1].set_ylabel('Density / 密度')
axes[1].set_title('Fitted GMM PDF')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: Gaussian Mixture Models use the EM algorithm to find the parameters that maximize likelihood. They naturally handle clustering and soft assignment of points to clusters through posterior probabilities.

- **ML Application**: GMMs are used for clustering, density estimation, anomaly detection, and as a foundation for more complex models. The number of components must be chosen (using BIC, AIC, or cross-validation).

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Chapter 14 Complete / 第14章完成\n\nThis chapter covered Gaussian Mixture Models, the foundation for probabilistic clustering.

---

### Chapter Summary

# Chapter 14: Gaussian Mixture Models

## Overview
This chapter explores **Gaussian Mixture Models (GMM)**, a probabilistic model for clustering and density estimation. GMM solves the problem of fitting multiple Normal distributions to data.

## Key Concepts
- **Mixture Model**: Linear combination of K component distributions
- **Gaussian Mixture**: Components are Normal distributions
- **EM Algorithm**: Expectation-Maximization algorithm to fit GMM parameters
- **Latent Variables**: Hidden component membership for each point
- **Bimodal Data**: Data with multiple modes (peaks)

## Evolution of Examples

### Problem and Solution
1. **01_bimodal_data_sample.py**: Generate bimodal synthetic data
2. **02_gaussian_mixture_em.py**: Fit GMM using EM algorithm and visualize components

## The Problem: Bimodal Data
Single Normal distribution cannot represent data with multiple peaks:
- Mean falls between modes
- Variance inflated
- Poor density representation

## The Solution: Gaussian Mixture Model

### Model
```
p(x) = Σ πₖ N(x | μₖ, σₖ²)
```
- K components (Normal distributions)
- πₖ: mixture weights (proportions)
- μₖ, σₖ²: mean and variance of component k

### Learning with EM Algorithm

**E-step**: Calculate responsibility (posterior probability of component for each point)
```
γₖ(xᵢ) = πₖ N(xᵢ | μₖ, σₖ²) / Σⱼ πⱼ N(xᵢ | μⱼ, σⱼ²)
```

**M-step**: Update parameters using responsibilities
```
πₖ ← (Σᵢ γₖ(xᵢ)) / N
μₖ ← (Σᵢ γₖ(xᵢ) xᵢ) / Σᵢ γₖ(xᵢ)
σₖ² ← (Σᵢ γₖ(xᵢ) (xᵢ - μₖ)²) / Σᵢ γₖ(xᵢ)
```

## Advantages
1. **Flexible**: Can model any distribution with enough components
2. **Probabilistic**: Provides component probabilities (soft clustering)
3. **Principled**: EM guarantees improvement at each iteration
4. **Interpretable**: Components are Normal distributions with clear parameters

## Limitations
1. **Model Selection**: Choosing number of components K is difficult
2. **Local Optima**: EM may converge to local maximum
3. **Computational Cost**: O(NK) per iteration for N data points, K components
4. **Initialization Sensitivity**: Results depend on starting point

## Use Cases
- **Clustering**: Soft assignment to K clusters
- **Density Estimation**: Model multimodal distributions
- **Anomaly Detection**: Identify points with low likelihood
- **Missing Data Imputation**: Using component structure

## Key Takeaways
1. GMM extends single Normal to multiple modes
2. EM algorithm learns components iteratively
3. Soft clustering: each point has probability of belonging to each component
4. Number of components must be chosen (via AIC, BIC, or validation)
5. GMM is foundation for advanced topics: hidden Markov models, topic modeling

---
