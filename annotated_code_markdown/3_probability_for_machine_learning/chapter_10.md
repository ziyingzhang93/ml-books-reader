# 概率论与机器学习 / Probability for Machine Learning
## Chapter 10

---

### Histogram 10Bins

# 01 — Histogram with 10 Bins / 10个分箱的直方图

**Chapter 10 — File 1 of 5**

## Summary

We create a histogram with 10 bins from samples drawn from a normal distribution. A histogram is a simple but effective non-parametric density estimation technique that divides the data into equal-width bins and counts observations.

我们从正态分布的样本中创建10个分箱的直方图。直方图是一种简单但有效的非参数密度估计技术，将数据分成等宽的分箱并计算观测值。

**Formula:**

$$\text{Density in bin } i = \frac{\text{Count in bin } i}{n \times \text{bin width}}$$

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

## Step 1 — Import Libraries / 导入库

```python
# Import necessary libraries / 导入必要的库
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import norm
```

## Step 2 — Generate Sample Data / 生成样本数据

```python
# Generate random samples from normal distribution / 从正态分布生成随机样本
np.random.seed(42)  # For reproducibility / 用于可重现性
mu = 50
sigma = 5
num_samples = 1000
# 生成随机数 / Generate random numbers
samples = np.random.normal(loc=mu, scale=sigma, size=num_samples)

# 打印输出 / Print output
print(f'Sample Data / 样本数据')
# 打印输出 / Print output
print(f'Distribution: N(μ={mu}, σ={sigma})')
# 打印输出 / Print output
print(f'Sample count / 样本数: {num_samples}')
# 计算均值 / Calculate mean
print(f'Mean / 均值: {np.mean(samples):.4f}')
# 计算标准差 / Calculate standard deviation
print(f'Std Dev / 标准差: {np.std(samples):.4f}')
```

## Step 3 — Create Histogram with 10 Bins / 创建10个分箱的直方图

```python
# Create histogram with 10 bins / 创建10个分箱的直方图
fig, ax = plt.subplots(figsize=(12, 6))

# Create histogram normalized to density (so area = 1) / 创建归一化到密度的直方图
n, bins, patches = ax.hist(samples, bins=10, density=True, alpha=0.7, 
                            edgecolor='black', color='steelblue', linewidth=1.5)

# Plot true normal distribution / 绘制真实正态分布
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
true_pdf = norm.pdf(x, loc=mu, scale=sigma)
ax.plot(x, true_pdf, 'r-', linewidth=2, label='True Normal PDF')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Density / 密度')
ax.set_title(f'Histogram with 10 Bins (n={num_samples})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# Print bin information / 打印分箱信息
# 打印输出 / Print output
print(f'\nBin Information / 分箱信息:')
# 打印输出 / Print output
print(f'{"Bin":>5s} {"Lower":>10s} {"Upper":>10s} {"Count":>8s} {"Density":>10s}')
# 打印输出 / Print output
print('-' * 50)
# 获取长度 / Get length
for i in range(len(n)):
    # 打印输出 / Print output
    print(f'{i:5d} {bins[i]:10.2f} {bins[i+1]:10.2f} {int(n[i]*num_samples/len(n)):8d} {n[i]:10.6f}')
```

## Step 4 — Analyze Estimation Quality / 分析估计质量

```python
# Calculate Mean Squared Error between histogram and true PDF / 计算直方图和真实PDF之间的MSE
# Evaluate histogram at bin centers / 在分箱中心评估直方图
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
histogram_density = n  # Already normalized

# True density at bin centers / 在分箱中心的真实密度
true_density = norm.pdf(bin_centers, loc=mu, scale=sigma)

# Calculate MSE / 计算MSE
# 计算均值 / Calculate mean
mse = np.mean((histogram_density - true_density)**2)
# 打印输出 / Print output
print(f'\nEstimation Quality / 估计质量:')
# 打印输出 / Print output
print(f'Mean Squared Error (MSE): {mse:.6f}')
# 打印输出 / Print output
print(f'\nBin Center Comparisons / 分箱中心比较:')
# 打印输出 / Print output
print(f'{"Center":>10s} {"Histogram":>15s} {"True PDF":>15s} {"Error":>15s}')
# 打印输出 / Print output
print('-' * 55)
# 将多个序列配对 / Pair multiple sequences
for center, hist_dens, true_dens in zip(bin_centers, histogram_density, true_density):
    error = abs(hist_dens - true_dens)
    # 打印输出 / Print output
    print(f'{center:10.2f} {hist_dens:15.6f} {true_dens:15.6f} {error:15.6f}')
```

## Learning Notes / 学习笔记

- **Concept**: Histograms are the simplest form of density estimation, dividing the range into fixed-width bins and estimating density as count/bin_width. The resolution depends directly on bin width—too few bins loses detail, too many bins creates noise.
  
  **概念**：直方图是最简单的密度估计形式，将范围分成固定宽度的分箱并将密度估计为计数/分箱宽度。分辨率直接取决于分箱宽度——分箱太少会丢失细节，分箱太多会产生噪声。

- **ML Application**: Histograms are used for exploratory data analysis, feature distribution understanding, and as a baseline for more sophisticated density estimation methods like KDE.
  
  **机器学习应用**：直方图用于探索性数据分析、特征分布理解，以及作为更复杂密度估计方法（如KDE）的基准。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步

➡️ **Next**: `02_histogram_3bins.ipynb`

## Complete Code / 完整代码一览

```python
# Complete Histogram Analysis with 10 Bins / 完整的10个分箱直方图分析

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate random samples / 生成随机样本
# 生成随机数 / Generate random numbers
np.random.seed(42)
mu = 50
sigma = 5
num_samples = 1000
# 生成随机数 / Generate random numbers
samples = np.random.normal(loc=mu, scale=sigma, size=num_samples)

# 打印输出 / Print output
print(f'Sample Data')
# 打印输出 / Print output
print(f'Distribution: N(μ={mu}, σ={sigma})')
# 计算均值 / Calculate mean
print(f'Mean: {np.mean(samples):.4f}')
# 计算标准差 / Calculate standard deviation
print(f'Std Dev: {np.std(samples):.4f}')

# Create histogram / 创建直方图
fig, ax = plt.subplots(figsize=(12, 6))

n, bins, patches = ax.hist(samples, bins=10, density=True, alpha=0.7, 
                            edgecolor='black', color='steelblue', linewidth=1.5)

# Plot true distribution / 绘制真实分布
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
true_pdf = norm.pdf(x, loc=mu, scale=sigma)
ax.plot(x, true_pdf, 'r-', linewidth=2, label='True Normal PDF')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Density / 密度')
ax.set_title(f'Histogram with 10 Bins (n={num_samples})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# Print bin information / 打印分箱信息
# 打印输出 / Print output
print(f'\nBin Information:')
# 打印输出 / Print output
print(f'{"Bin":>5s} {"Lower":>10s} {"Upper":>10s} {"Count":>8s} {"Density":>10s}')
# 打印输出 / Print output
print('-' * 50)
# 获取长度 / Get length
for i in range(len(n)):
    # 打印输出 / Print output
    print(f'{i:5d} {bins[i]:10.2f} {bins[i+1]:10.2f} {int(n[i]*num_samples/len(n)):8d} {n[i]:10.6f}')

# Analyze quality / 分析质量
bin_centers = (bins[:-1] + bins[1:]) / 2
histogram_density = n
true_density = norm.pdf(bin_centers, loc=mu, scale=sigma)
# 计算均值 / Calculate mean
mse = np.mean((histogram_density - true_density)**2)

# 打印输出 / Print output
print(f'\nEstimation Quality:')
# 打印输出 / Print output
print(f'Mean Squared Error: {mse:.6f}')
```

---

### Histogram 3Bins



---

### Parametric Estimation



---

### Bimodal Data Sample

# 04 — Bimodal Data Sample / 双峰数据样本

**Chapter 10 — File 4 of 5**

## Summary

We create bimodal data by mixing samples from two Gaussian distributions: N(20,5) and N(40,5). This demonstrates that parametric methods assuming a single normal distribution fail for multimodal data.

我们通过混合两个高斯分布N(20,5)和N(40,5)的样本来创建双峰数据。这演示了假设单一正态分布的参数方法对多峰数据失败。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import norm
```

## Step 2 — Generate Bimodal Data / 生成双峰数据

```python
# Generate data from two normal distributions / 从两个正态分布生成数据
# 生成随机数 / Generate random numbers
np.random.seed(42)
n1 = 500  # Samples from first distribution / 第一个分布的样本
n2 = 500  # Samples from second distribution / 第二个分布的样本

# 生成随机数 / Generate random numbers
samples1 = np.random.normal(loc=20, scale=5, size=n1)
# 生成随机数 / Generate random numbers
samples2 = np.random.normal(loc=40, scale=5, size=n2)

# Combine into bimodal data / 组合成双峰数据
# 拼接数组 / Concatenate arrays
bimodal_samples = np.concatenate([samples1, samples2])

# 打印输出 / Print output
print(f'Bimodal Data Generation / 双峰数据生成')
# 打印输出 / Print output
print(f'Component 1: N(20, 5), n={n1}')
# 打印输出 / Print output
print(f'Component 2: N(40, 5), n={n2}')
# 打印输出 / Print output
print(f'\nCombined Data Statistics / 组合数据统计:')
# 计算均值 / Calculate mean
print(f'Mean: {np.mean(bimodal_samples):.4f}')
# 计算标准差 / Calculate standard deviation
print(f'Std Dev: {np.std(bimodal_samples):.4f}')
# 求最小值 / Find minimum value
print(f'Min: {np.min(bimodal_samples):.4f}')
# 求最大值 / Find maximum value
print(f'Max: {np.max(bimodal_samples):.4f}')
```

## Step 3 — Visualize Bimodal Distribution / 可视化双峰分布

## Step 3 — Visualize Bimodal Distribution / 可视化双峰分布

```python
fig, ax = plt.subplots(figsize=(12, 6))

# Histogram of combined data / 组合数据的直方图
ax.hist(bimodal_samples, bins=30, density=True, alpha=0.6, 
        color='steelblue', edgecolor='black', label='Bimodal Data')

# Individual components / 单个组件
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(0, 60, 1000)
comp1_pdf = norm.pdf(x, loc=20, scale=5) * 0.5  # Weight by proportion / 按比例加权
comp2_pdf = norm.pdf(x, loc=40, scale=5) * 0.5

ax.plot(x, comp1_pdf, 'g--', linewidth=2, label='Component 1: N(20,5)')
ax.plot(x, comp2_pdf, 'orange', linestyle='--', linewidth=2, label='Component 2: N(40,5)')
ax.plot(x, comp1_pdf + comp2_pdf, 'r-', linewidth=2.5, label='True Bimodal PDF')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Density / 密度')
ax.set_title('Bimodal Distribution: Mixture of Two Normals')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Step 4 — Show Parametric Failure / 演示参数方法失败

```python
# Fit single normal to bimodal data / 将单一正态拟合到双峰数据
# 计算均值 / Calculate mean
mu_est = np.mean(bimodal_samples)
# 计算标准差 / Calculate standard deviation
sigma_est = np.std(bimodal_samples, ddof=1)

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(bimodal_samples, bins=30, density=True, alpha=0.6, 
        color='steelblue', edgecolor='black', label='Data')

# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(0, 60, 1000)
true_bimodal = (norm.pdf(x, loc=20, scale=5) + norm.pdf(x, loc=40, scale=5)) / 2
ax.plot(x, true_bimodal, 'g-', linewidth=2.5, label='True Bimodal')

# Fitted single normal (this will be wrong!) / 拟合的单一正态（这会是错的！）
fitted_normal = norm.pdf(x, loc=mu_est, scale=sigma_est)
ax.plot(x, fitted_normal, 'r--', linewidth=2.5, label=f'Fit Single N({mu_est:.1f}, {sigma_est:.1f})')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Density / 密度')
ax.set_title('Parametric Failure: Single Normal Cannot Fit Bimodal Data')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: Bimodal (and multimodal) distributions cannot be represented by a single normal distribution. They require mixture models or more flexible non-parametric methods.

- **ML Application**: Mixture models (Gaussian Mixture Models) are used when data exhibits multiple modes. This is common in clustering and heterogeneous data.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Next / 下一步

➡️ **Next**: `05_kernel_density_estimation.ipynb`

---

### Kernel Density Estimation

# 05 — Kernel Density Estimation / 核密度估计

**Chapter 10 — File 5 of 5**

## Summary

We use Kernel Density Estimation (KDE) from sklearn to estimate the density of bimodal data. KDE is a non-parametric method that can capture complex multimodal distributions without assuming a specific form.

我们使用sklearn的核密度估计（KDE）来估计双峰数据的密度。KDE是一种非参数方法，可以捕获复杂的多峰分布，无需假设特定形式。

**Formula:**

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where $K$ is the kernel and $h$ is the bandwidth

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Import Libraries / 导入库

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import norm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KernelDensity
```

## Step 2 — Generate Bimodal Data / 生成双峰数据

```python
# Generate bimodal data / 生成双峰数据
# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
samples1 = np.random.normal(loc=20, scale=5, size=500)
# 生成随机数 / Generate random numbers
samples2 = np.random.normal(loc=40, scale=5, size=500)
# 拼接数组 / Concatenate arrays
bimodal_samples = np.concatenate([samples1, samples2])
bimodal_samples = bimodal_samples.reshape(-1, 1)  # Reshape for sklearn / 为sklearn重新整形

# 打印输出 / Print output
print(f'Kernel Density Estimation / 核密度估计')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f'Sample shape: {bimodal_samples.shape}')
# 打印输出 / Print output
print(f'Data range: [{bimodal_samples.min():.2f}, {bimodal_samples.max():.2f}]')
```

## Step 3 — Fit KDE / 拟合KDE

```python
# Fit KDE with different bandwidth values / 用不同的带宽值拟合KDE
bandwidths = [0.5, 1.0, 2.0, 5.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 展平为一维数组 / Flatten to 1D array
axes = axes.flatten()

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x = np.linspace(0, 60, 1000).reshape(-1, 1)

# 同时获取索引和值 / Get both index and value
for idx, bw in enumerate(bandwidths):
    # Fit KDE / 拟合KDE
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(bimodal_samples)
    
    # Score samples (log density) / 对样本评分（对数密度）
    log_density = kde.score_samples(x)
    density = np.exp(log_density)
    
    # Plot / 绘制
    ax = axes[idx]
    ax.hist(bimodal_samples, bins=30, density=True, alpha=0.5, 
            color='steelblue', edgecolor='black', label='Data')
    ax.plot(x, density, 'r-', linewidth=2, label='KDE')
    
    ax.set_xlabel('Value / 值')
    ax.set_ylabel('Density / 密度')
    ax.set_title(f'KDE with Bandwidth = {bw}')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Step 4 — Compare Methods / 比较方法

```python
# Compare parametric, histogram, and KDE / 比较参数、直方图和KDE方法
fig, ax = plt.subplots(figsize=(12, 6))

# Data histogram / 数据直方图
ax.hist(bimodal_samples, bins=30, density=True, alpha=0.5, 
        color='steelblue', edgecolor='black', label='Data')

# True bimodal distribution / 真实双峰分布
# 生成等间距数组 / Generate evenly spaced array
x = np.linspace(0, 60, 1000)
true_bimodal = (norm.pdf(x, loc=20, scale=5) + norm.pdf(x, loc=40, scale=5)) / 2
ax.plot(x, true_bimodal, 'g-', linewidth=2.5, label='True Distribution')

# Parametric (single normal) / 参数方法（单一正态）
mu = bimodal_samples.mean()
sigma = bimodal_samples.std()
ax.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'orange', linestyle='--', 
        linewidth=2, label='Parametric (Single Normal)')

# KDE / KDE
kde = KernelDensity(bandwidth=1.5, kernel='gaussian')
kde.fit(bimodal_samples)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
kde_density = np.exp(kde.score_samples(x.reshape(-1, 1)))
ax.plot(x, kde_density, 'r-', linewidth=2.5, label='KDE (Bandwidth=1.5)')

ax.set_xlabel('Value / 值')
ax.set_ylabel('Density / 密度')
ax.set_title('Comparison: Parametric vs Histogram vs KDE')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: KDE is a non-parametric density estimation method that places a kernel (typically Gaussian) at each data point and sums them. The bandwidth controls the smoothness: smaller bandwidth captures more detail but adds noise, while larger bandwidth smooths out details.
  
  **概念**：KDE是一种非参数密度估计方法，在每个数据点处放置一个核（通常是高斯的）并求和。带宽控制平滑度：较小的带宽捕获更多细节但增加噪声，较大的带宽平滑细节。

- **ML Application**: KDE is used for probability density estimation, anomaly detection (identifying low-density regions), and as a preprocessing step in machine learning pipelines for exploratory analysis.
  
  **机器学习应用**：KDE用于概率密度估计、异常检测（识别低密度区域）和作为机器学习管道中用于探索性分析的预处理步骤。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `matplotlib` | 绑图库 | Plotting library |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |

## Chapter 10 Complete / 第10章完成

This chapter covered density estimation techniques from simple histograms to advanced KDE methods.

## Complete Code / 完整代码一览

```python
# Complete KDE Analysis / 完整KDE分析

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
from scipy.stats import norm
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KernelDensity

# Generate bimodal data / 生成双峰数据
# 生成随机数 / Generate random numbers
np.random.seed(42)
# 生成随机数 / Generate random numbers
samples1 = np.random.normal(loc=20, scale=5, size=500)
# 生成随机数 / Generate random numbers
samples2 = np.random.normal(loc=40, scale=5, size=500)
# 拼接数组 / Concatenate arrays
bimodal_samples = np.concatenate([samples1, samples2])
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
bimodal_samples_reshaped = bimodal_samples.reshape(-1, 1)

# Fit KDE with different bandwidths / 用不同带宽拟合KDE
bandwidths = [0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 展平为一维数组 / Flatten to 1D array
axes = axes.flatten()

# 改变数组形状（不改变数据） / Reshape array (data unchanged)
x = np.linspace(0, 60, 1000).reshape(-1, 1)

# 同时获取索引和值 / Get both index and value
for idx, bw in enumerate(bandwidths):
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(bimodal_samples_reshaped)
    log_density = kde.score_samples(x)
    density = np.exp(log_density)
    
    ax = axes[idx]
    ax.hist(bimodal_samples, bins=30, density=True, alpha=0.5, 
            color='steelblue', edgecolor='black', label='Data')
    ax.plot(x[:, 0], density, 'r-', linewidth=2, label='KDE')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'KDE with Bandwidth = {bw}')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# Compare methods / 比较方法
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(bimodal_samples, bins=30, density=True, alpha=0.5, 
        color='steelblue', edgecolor='black', label='Data')

# 生成等间距数组 / Generate evenly spaced array
x_1d = np.linspace(0, 60, 1000)
true_bimodal = (norm.pdf(x_1d, loc=20, scale=5) + norm.pdf(x_1d, loc=40, scale=5)) / 2
ax.plot(x_1d, true_bimodal, 'g-', linewidth=2.5, label='True Distribution')

mu = bimodal_samples.mean()
sigma = bimodal_samples.std()
ax.plot(x_1d, norm.pdf(x_1d, loc=mu, scale=sigma), 'orange', linestyle='--', 
        linewidth=2, label='Parametric')

kde = KernelDensity(bandwidth=1.5, kernel='gaussian')
kde.fit(bimodal_samples_reshaped)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
kde_density = np.exp(kde.score_samples(x_1d.reshape(-1, 1)))
ax.plot(x_1d, kde_density, 'r-', linewidth=2.5, label='KDE')

ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Comparison: Parametric vs KDE')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 10: Density Estimation

## Overview
This chapter explores **estimating probability distributions from data**. The journey progresses from non-parametric (histograms) to parametric (normal fits) to advanced non-parametric (KDE) methods.

## Key Concepts
- **Histogram**: Non-parametric density estimate with fixed bin width
- **Parametric Estimation**: Fit known distribution (e.g., Normal) to data
- **Kernel Density Estimation (KDE)**: Non-parametric method using kernels around each point
- **Bin Width Effect**: Critical hyperparameter in histogram estimation
- **Bimodal Data**: When one distribution fails

## Evolution of Examples

### Histogram Progression (Non-Parametric)
1. **01_histogram_10bins.py**: Generate data and plot with 10 bins
2. **02_histogram_3bins.py**: Plot with 3 bins (under-smoothed)

### Parametric Approach
3. **03_parametric_normal.py**: Fit Normal distribution and plot

### When Parametric Fails
4. **04_bimodal_data.py**: Generate bimodal data showing Normal fit limitation

### Non-Parametric Rescue
5. **05_kernel_density_estimation.py**: Use KDE to capture bimodal structure

## Logic Flow
**Non-Parametric (Histogram) → Adjust Bins → Parametric Fit → Discover Limitation (Bimodal) → Non-Parametric Rescue (KDE)**

## Method Comparison

### Histogram
- **Pros**: Simple, interpretable, fast
- **Cons**: Sensitive to bin width, discontinuous
- **Use When**: Quick visualization, discrete data

### Parametric (Normal Fit)
- **Pros**: Smooth, interpretable parameters (mean, variance), efficient
- **Cons**: Assumes specific distribution, fails on multimodal data
- **Use When**: Data approximately normal, need interpretability

### Kernel Density Estimation (KDE)
- **Pros**: Flexible, captures multiple modes, smooth
- **Cons**: Requires bandwidth selection, computationally expensive for large datasets
- **Use When**: Non-parametric flexibility needed, modes unknown

## Key Takeaways
1. No single method is universally best
2. Histogram bin width critically affects estimation (bias-variance tradeoff)
3. Parametric methods are efficient but make strong assumptions
4. KDE is flexible and can capture complex structures
5. Visual inspection of data guides method selection
6. Understand your data before choosing an estimation method

---
