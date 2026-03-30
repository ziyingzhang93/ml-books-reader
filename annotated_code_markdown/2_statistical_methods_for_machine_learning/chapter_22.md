# 统计方法与机器学习
## Chapter 22

---

### Test Data

# 22 — Test Data for Prediction Intervals / 预测区间的测试数据

**Chapter 22 — File 1 of 3**

## Summary / 摘要

This notebook generates synthetic correlated x-y data for demonstration of prediction intervals. The data follows an underlying linear relationship with random noise, making it suitable for regression modeling. A scatter plot visualizes the data points and their linear trend. This is the foundation for subsequent notebooks that fit a linear regression model and calculate prediction intervals, which quantify uncertainty in individual predictions rather than parameter estimates.

本笔记本为预测区间演示生成合成相关的x-y数据。数据遵循具有随机噪声的基础线性关系，适合回归建模。散点图可视化数据点及其线性趋势。这是后续笔记本的基础，后续笔记本拟合线性回归模型并计算预测区间，预测区间量化了单个预测中的不确定性，而不是参数估计。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 可视化结果 / Visualize results


## Step 1 — Import Libraries / 导入库

```python
# Import required libraries
# 导入所需库
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Generate Correlated Data / 生成相关数据

```python
# Generate independent variable (x)
# 生成独立变量(x)
x = np.arange(0, 10, 0.5)

# Define underlying linear relationship: y = 2x + 1
# 定义基础线性关系: y = 2x + 1
true_slope = 2
true_intercept = 1

# Generate y values with random noise
# 生成带有随机噪声的y值
noise = np.random.normal(0, 2, len(x))  # Gaussian noise with std=2
                                         # 高斯噪声，std=2
y = true_slope * x + true_intercept + noise

# Combine into dataset
# 合并成数据集
data = np.column_stack((x, y))

# Display data properties
# 显示数据属性
print(f"Dataset properties:")
print(f"  Sample size: {len(x)}")
print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
print(f"\nTrue underlying relationship:")
print(f"  y = {true_slope}x + {true_intercept}")
print(f"\nData (first 10 rows):")
print(f"{'x':>6} | {'y':>8}")
print("-" * 16)
for i in range(min(10, len(x))):
    print(f"{x[i]:6.1f} | {y[i]:8.2f}")
```

## Step 3 — Calculate Data Summary Statistics / 计算数据汇总统计

```python
# Calculate summary statistics
# 计算汇总统计
x_mean = np.mean(x)
y_mean = np.mean(y)
x_std = np.std(x, ddof=1)
y_std = np.std(y, ddof=1)

# Calculate correlation
# 计算相关性
correlation = np.corrcoef(x, y)[0, 1]

# Display statistics
# 显示统计
print(f"\nSummary Statistics:")
print(f"  x mean: {x_mean:.4f}, std: {x_std:.4f}")
print(f"  y mean: {y_mean:.4f}, std: {y_std:.4f}")
print(f"  Correlation (r): {correlation:.4f}")
print(f"  R-squared: {correlation**2:.4f}")
```

## Step 4 — Visualize Data with Scatter Plot / 用散点图可视化数据

```python
# Create scatter plot
# 创建散点图
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
# 绘制数据点
ax.scatter(x, y, alpha=0.6, s=50, color='blue', label='Observed data')

# Plot true underlying relationship
# 绘制真实的基础关系
x_line = np.linspace(x.min(), x.max(), 100)
y_line = true_slope * x_line + true_intercept
ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'True: y = {true_slope}x + {true_intercept}')

# Add mean lines
# 添加均值线
ax.axvline(x_mean, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y_mean, color='gray', linestyle=':', alpha=0.5)

# Labels and formatting
# 标签和格式
ax.set_xlabel('X (Independent Variable)', fontsize=12)
ax.set_ylabel('Y (Dependent Variable)', fontsize=12)
ax.set_title('Synthetic Correlated Data with Linear Trend', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text annotation
# 添加文本注释
textstr = f'r = {correlation:.4f}\nn = {len(x)}'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
```

## Step 5 — Analyze Residuals / 分析残差

```python
# Calculate residuals from true relationship
# 从真实关系计算残差
y_true = true_slope * x + true_intercept
residuals = y - y_true

# Calculate residual statistics
# 计算残差统计
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals, ddof=1)

# Display residual analysis
# 显示残差分析
print(f"\nResidual Analysis (from true relationship):")
print(f"  Residual mean: {residuals_mean:.4f} (should be ≈ 0)")
print(f"  Residual std: {residuals_std:.4f}")
print(f"  Min residual: {np.min(residuals):.4f}")
print(f"  Max residual: {np.max(residuals):.4f}")

# Create residual plot
# 创建残差图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Residuals vs. X
# 图1: 残差对X
axes[0].scatter(x, residuals, alpha=0.6, s=50, color='purple')
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0].axhline(residuals_std, color='gray', linestyle=':', alpha=0.7, label='±1 Std')
axes[0].axhline(-residuals_std, color='gray', linestyle=':', alpha=0.7)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Histogram of residuals
# 图2: 残差直方图
axes[1].hist(residuals, bins=10, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Synthetic data with known underlying relationships allows controlled demonstration of statistical methods. The linear model y = βx + α + ε assumes the response is a linear function of predictors plus random noise ε. Strong correlation between x and y indicates the linear model is appropriate, while the residual standard deviation quantifies prediction uncertainty independent of the model fit.
  
  **统计概念**: 具有已知基础关系的合成数据允许对统计方法进行受控演示。线性模型y = βx + α + ε假定响应是预测因子的线性函数加上随机噪声ε。x和y之间的强相关性表示线性模型是合适的，而残差标准差量化了独立于模型拟合的预测不确定性。

- **ML Application**: Understanding the data-generating process is crucial for prediction intervals. In production ML systems, residual analysis reveals model limitations and assumption violations. The residual standard deviation becomes the basis for prediction interval width, directly translating data variability into quantified prediction uncertainty. Systematic residual patterns indicate needed model improvements or variable transformations.
  
  **ML应用**: 理解数据生成过程对于预测区间至关重要。在生产ML系统中，残差分析揭示了模型限制和假设违反。残差标准差成为预测区间宽度的基础，直接将数据变异性转化为量化的预测不确定性。系统性残差模式表示需要改进的模型或变量转换。
```

➡️ **Next**: `02_linear_regression_model.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.arange(0, 10, 0.5)

true_slope, true_intercept = 2, 1
noise = np.random.normal(0, 2, len(x))
y = true_slope * x + true_intercept + noise

print(f"X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")

correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {correlation:.4f}")

plt.scatter(x, y, alpha=0.6, s=50)
y_line = true_slope * x + true_intercept
plt.plot(x, y_line, 'r--', linewidth=2, label=f'True: y = {true_slope}x + {true_intercept}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Synthetic Data with Linear Trend')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Linear Regression Model

# 22 — Linear Regression Model / 线性回归模型

**Chapter 22 — File 2 of 3**

## Summary / 摘要

This notebook fits a linear regression model to the synthetic data using scipy.stats.linregress. The function returns the slope, intercept, r-value (correlation), p-value for significance testing, and standard error. The fitted line is plotted alongside the data, showing how well the model captures the linear trend. The regression equation and fit quality metrics provide the foundation for calculating prediction intervals in the next notebook.

本笔记本使用scipy.stats.linregress将线性回归模型拟合到合成数据。该函数返回斜率、截距、r值（相关性）、显著性检验的p值和标准误差。拟合线与数据一起绘制，显示模型如何很好地捕获线性趋势。回归方程和拟合质量指标为下一个笔记本中计算预测区间提供基础。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


## Step 1 — Import Libraries and Load Data / 导入库和加载数据

```python
# Import required libraries
# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Generate Data / 生成数据

```python
# Generate synthetic data (same as previous notebook)
# 生成合成数据（与前一个笔记本相同）
x = np.arange(0, 10, 0.5)
true_slope = 2
true_intercept = 1
noise = np.random.normal(0, 2, len(x))
y = true_slope * x + true_intercept + noise

# Display data summary
# 显示数据摘要
print(f"Data summary:")
print(f"  Sample size: {len(x)}")
print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
```

## Step 3 — Fit Linear Regression Model / 拟合线性回归模型

```python
# Perform linear regression
# 执行线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Display regression parameters
# 显示回归参数
print(f"\nLinear Regression Results:")
print(f"  Fitted equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"  \nComparison with true relationship:")
print(f"    True: y = {true_slope}x + {true_intercept}")
print(f"    Fitted: y = {slope:.4f}x + {intercept:.4f}")
print(f"  \nModel fit quality:")
print(f"    Correlation (r): {r_value:.4f}")
print(f"    R-squared: {r_value**2:.4f}")
print(f"    Standard error: {std_err:.4f}")
print(f"    p-value: {p_value:.2e}")
```

## Step 4 — Calculate Predictions and Residuals / 计算预测和残差

```python
# Calculate fitted values
# 计算拟合值
y_pred = slope * x + intercept

# Calculate residuals
# 计算残差
residuals = y - y_pred

# Calculate residual statistics
# 计算残差统计
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals, ddof=1)  # Unbiased estimator / 无偏估计
sum_squared_residuals = np.sum(residuals**2)

# Display residual analysis
# 显示残差分析
print(f"\nResidual Analysis:")
print(f"  Mean of residuals: {residuals_mean:.4f} (should be ≈ 0)")
print(f"  Std of residuals: {residuals_std:.4f}")
print(f"  Min residual: {np.min(residuals):.4f}")
print(f"  Max residual: {np.max(residuals):.4f}")
print(f"  Sum of squared residuals (SSE): {sum_squared_residuals:.4f}")
```

## Step 5 — Visualize Fitted Line with Data / 用数据可视化拟合线

```python
# Create visualization
# 创建可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data with fitted line
# 图1: 数据与拟合线
axes[0].scatter(x, y, alpha=0.6, s=50, color='blue', label='Observed data')

# Plot fitted line
# 绘制拟合线
x_line = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_line + intercept
axes[0].plot(x_line, y_fit, 'r-', linewidth=2, label=f'Fitted: y = {slope:.2f}x + {intercept:.2f}')

# Plot true line for reference
# 绘制真实线作为参考
y_true_line = true_slope * x_line + true_intercept
axes[0].plot(x_line, y_true_line, 'g--', linewidth=2, alpha=0.7, label=f'True: y = {true_slope}x + {true_intercept}')

axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y', fontsize=12)
axes[0].set_title('Linear Regression Fit', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals vs. X
# 图2: 残差对X
axes[1].scatter(x, residuals, alpha=0.6, s=50, color='purple')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')

# Add residual standard deviation bands
# 添加残差标准差带
axes[1].axhline(residuals_std, color='gray', linestyle=':', alpha=0.7, label='±1 Std')
axes[1].axhline(-residuals_std, color='gray', linestyle=':', alpha=0.7)
axes[1].fill_between(x_line, -residuals_std, residuals_std, alpha=0.1, color='gray')

axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 6 — Model Evaluation Metrics / 模型评估指标

```python
# Calculate additional metrics
# 计算额外指标
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean)**2)  # Total sum of squares / 总平方和
ss_res = sum_squared_residuals  # Residual sum of squares / 残差平方和
ss_reg = ss_tot - ss_res  # Regression sum of squares / 回归平方和

# R-squared
r_squared = ss_reg / ss_tot

# Adjusted R-squared
n = len(x)
p = 1  # Number of predictors / 预测变量数量
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Mean squared error and RMSE
mse = ss_res / (n - 2)  # Divide by n-2 for unbiased estimate / 除以n-2以获得无偏估计
rmse = np.sqrt(mse)

# Mean absolute error
mae = np.mean(np.abs(residuals))

# Display model evaluation metrics
# 显示模型评估指标
print(f"\nModel Evaluation Metrics:")
print(f"  R-squared: {r_squared:.4f}")
print(f"  Adjusted R-squared: {adj_r_squared:.4f}")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"\nVariance decomposition:")
print(f"  Total SS: {ss_tot:.4f}")
print(f"  Regression SS: {ss_reg:.4f}")
print(f"  Residual SS: {ss_res:.4f}")
print(f"  Proportion explained: {ss_reg/ss_tot*100:.1f}%")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Linear regression estimates slope and intercept by minimizing sum of squared residuals. The residual standard error (square root of MSE) quantifies the typical prediction error and forms the basis for prediction intervals. The R-squared value indicates the proportion of variance explained by the model; higher values (closer to 1) indicate better fit. The p-value tests if the slope is significantly different from zero.
  
  **统计概念**: 线性回归通过最小化残差平方和来估计斜率和截距。残差标准误差（MSE的平方根）量化了典型的预测误差，是预测区间的基础。R-squared值表示模型解释的方差比例；较高的值（接近1）表示更好的拟合。p值测试斜率是否显著不同于零。

- **ML Application**: In production systems, residual analysis is critical for model validation. Non-random residual patterns indicate model assumptions are violated (linearity, homoscedasticity, independence). The residual standard deviation directly affects prediction interval width and uncertainty quantification. Understanding model fit quality helps decide when to use ensemble methods or non-linear models for improved predictions.
  
  **ML应用**: 在生产系统中，残差分析对模型验证至关重要。非随机残差模式表示违反了模型假设（线性性、齐性、独立性）。残差标准差直接影响预测区间宽度和不确定性量化。理解模型拟合质量有助于决定何时使用集成方法或非线性模型以改进预测。
```

➡️ **Next**: `03_prediction_interval.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

np.random.seed(42)
x = np.arange(0, 10, 0.5)
noise = np.random.normal(0, 2, len(x))
y = 2 * x + 1 + noise

slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_pred = slope * x + intercept
residuals = y - y_pred
residuals_std = np.std(residuals, ddof=1)

print(f"Fitted: y = {slope:.4f}x + {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Residual std: {residuals_std:.4f}")
```

---

### Prediction Interval

# 22 — Prediction Interval / 预测区间

**Chapter 22 — File 3 of 3**

## Summary / 摘要

A prediction interval quantifies uncertainty in individual predictions from a regression model. The interval width equals 1.96 times the standard deviation of residuals, which represents the typical prediction error at any x value (for 95% confidence). Unlike confidence intervals (which bound the mean), prediction intervals account for both estimation uncertainty and inherent data variability. Prediction intervals are always wider than confidence intervals because they include the irreducible noise in the data.

预测区间量化了回归模型中单个预测的不确定性。区间宽度等于残差标准差的1.96倍，这代表任何x值处的典型预测误差（95%置信度）。与置信区间（界定均值）不同，预测区间考虑了估计不确定性和固有数据变异性。预测区间总是比置信区间更宽，因为它们包括数据中的不可约噪声。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


## Step 1 — Import Libraries and Prepare Data / 导入库和准备数据

```python
# Import required libraries
# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Set random seed for reproducibility
# 设置随机种子以保证可重复性
np.random.seed(42)
```

## Step 2 — Generate and Fit Data / 生成和拟合数据

```python
# Generate synthetic data
# 生成合成数据
x = np.arange(0, 10, 0.5)
noise = np.random.normal(0, 2, len(x))
y = 2 * x + 1 + noise

# Fit linear regression
# 拟合线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Calculate fitted values and residuals
# 计算拟合值和残差
y_pred = slope * x + intercept
residuals = y - y_pred
residuals_std = np.std(residuals, ddof=1)

# Display model summary
# 显示模型摘要
print(f"Model Summary:")
print(f"  Fitted: y = {slope:.4f}x + {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  Residual std: {residuals_std:.4f}")
print(f"  Sample size: {len(x)}")
```

## Step 3 — Calculate Prediction Interval / 计算预测区间

```python
# Prediction interval parameters
# 预测区间参数
confidence_level = 0.95  # 95% confidence / 95%置信
z_critical = 1.96  # Critical value for 95% / 95%的临界值

# Calculate prediction interval margins
# 计算预测区间余量
# For simple regression, prediction margin = z * residual_std
# 对于简单回归，预测余量 = z * 残差_std
prediction_margin = z_critical * residuals_std

# Calculate PI bounds at each observed x
# 计算每个观察x处的PI边界
pi_lower = y_pred - prediction_margin
pi_upper = y_pred + prediction_margin

# Display prediction interval details
# 显示预测区间详情
print(f"\nPrediction Interval Calculation:")
print(f"  Confidence level: {confidence_level*100:.0f}%")
print(f"  Z-critical value: {z_critical}")
print(f"  Residual std: {residuals_std:.4f}")
print(f"  Margin of error: {prediction_margin:.4f}")
print(f"\nPI width at all points: {2*prediction_margin:.4f}")
print(f"\nSample predictions:")
print(f"{'x':>5} | {'y_pred':>8} | {'PI_lower':>8} | {'PI_upper':>8}")
print("-" * 36)
for i in range(min(5, len(x))):
    print(f"{x[i]:5.1f} | {y_pred[i]:8.2f} | {pi_lower[i]:8.2f} | {pi_upper[i]:8.2f}")
```

## Step 4 — Generate Smooth Prediction Curves / 生成光滑预测曲线

```python
# Create smooth prediction lines for visualization
# 为可视化创建光滑预测线
x_smooth = np.linspace(x.min(), x.max(), 100)

# Predict on smooth x
# 在光滑x上预测
y_smooth_pred = slope * x_smooth + intercept

# Prediction interval bounds on smooth x
# 光滑x上的预测区间边界
pi_smooth_lower = y_smooth_pred - prediction_margin
pi_smooth_upper = y_smooth_pred + prediction_margin

# Calculate confidence interval for reference
# 计算参考的置信区间
# Standard error of the mean prediction
# 均值预测的标准误差
x_mean = np.mean(x)
se_mean = np.sqrt(np.sum((x - x_mean)**2))  # Used for scaling / 用于缩放
se_fit = residuals_std * np.sqrt(1/len(x) + (x_smooth - x_mean)**2 / np.sum((x - x_mean)**2))
ci_margin = z_critical * se_fit
ci_lower = y_smooth_pred - ci_margin
ci_upper = y_smooth_pred + ci_margin

print(f"\nSmooth curves generated for {len(x_smooth)} points")
```

## Step 5 — Visualize Data with Prediction Intervals / 用预测区间可视化数据

```python
# Create comprehensive visualization
# 创建综合可视化
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Prediction interval with data
# 图1: 预测区间与数据
# Shade the prediction interval region
# 给预测区间区域着色
axes[0].fill_between(x_smooth, pi_smooth_lower, pi_smooth_upper, 
                     alpha=0.2, color='red', label='95% Prediction Interval')

# Shade the confidence interval region
# 给置信区间区域着色
axes[0].fill_between(x_smooth, ci_lower, ci_upper, 
                     alpha=0.3, color='blue', label='95% Confidence Interval')

# Plot fitted line
# 绘制拟合线
axes[0].plot(x_smooth, y_smooth_pred, 'g-', linewidth=2, label='Fitted line')

# Plot PI and CI bounds
# 绘制PI和CI边界
axes[0].plot(x_smooth, pi_smooth_lower, 'r--', linewidth=1.5, alpha=0.7)
axes[0].plot(x_smooth, pi_smooth_upper, 'r--', linewidth=1.5, alpha=0.7)
axes[0].plot(x_smooth, ci_lower, 'b--', linewidth=1.5, alpha=0.7)
axes[0].plot(x_smooth, ci_upper, 'b--', linewidth=1.5, alpha=0.7)

# Plot observed data
# 绘制观察数据
axes[0].scatter(x, y, s=50, color='black', alpha=0.6, label='Observed data')

axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y', fontsize=12)
axes[0].set_title('Linear Regression with Prediction Interval', fontsize=14)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: Error bar representation
# 图2: 误差条表示
# Select subset of x for clarity
# 选择x的子集以便清晰
x_subset = x[::3]  # Every 3rd point / 每3个点
y_pred_subset = y_pred[::3]
pi_lower_subset = pi_lower[::3]
pi_upper_subset = pi_upper[::3]

# Calculate errors for error bar
# 计算误差条的误差
errors_pi = np.array([y_pred_subset - pi_lower_subset, pi_upper_subset - y_pred_subset])

axes[1].errorbar(x_subset, y_pred_subset, yerr=errors_pi, fmt='o', 
                markersize=8, capsize=10, capthick=2, linewidth=2, 
                color='red', ecolor='red', label='Prediction Interval')
axes[1].scatter(x, y, s=50, color='black', alpha=0.6, label='Observed data')
axes[1].plot(x_smooth, y_smooth_pred, 'g-', linewidth=2, label='Fitted line')

axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y', fontsize=12)
axes[1].set_title('Prediction Interval as Error Bars', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 6 — Compare Intervals and Analyze Width / 比较区间和分析宽度

```python
# Calculate average widths
# 计算平均宽度
pi_width_avg = np.mean(pi_upper - pi_lower)
ci_width_avg = np.mean(ci_upper - ci_lower)

# Compare intervals
# 比较区间
print(f"\nInterval Comparison:")
print(f"  Prediction Interval (PI) average width: {pi_width_avg:.4f}")
print(f"  Confidence Interval (CI) average width: {ci_width_avg:.4f}")
print(f"  PI/CI ratio: {pi_width_avg/ci_width_avg:.2f}x wider")
print(f"\nReason: PI includes both estimation uncertainty AND inherent data variability")
print(f"  Residual std (data variability): {residuals_std:.4f}")
print(f"  Standard error at center (estimation uncertainty): {se_fit[len(se_fit)//2]:.4f}")
```

```python
## Learning Notes / 学习笔记

- **Statistical Concept**: Prediction intervals are wider than confidence intervals because they account for two sources of uncertainty: (1) uncertainty in estimating the regression line (standard error) and (2) inherent variability of individual observations around the true line (residual std). The prediction margin = z × residual_std captures both components. As sample size increases, the standard error decreases, but the residual std remains constant, so PI width becomes increasingly dominated by residual variability.
  
  **统计概念**: 预测区间比置信区间更宽，因为它们考虑了两种不确定性来源：(1)估计回归线的不确定性（标准误差），和(2)单个观察围绕真实线的固有变异性（残差标准差）。预测余量 = z × 残差_std捕获两个组成部分。随着样本大小增加，标准误差减少，但残差标准差保持不变，因此PI宽度变得越来越由残差变异性主导。

- **ML Application**: In production ML systems, prediction intervals set operational bounds for individual predictions. Wider intervals indicate higher uncertainty and may trigger manual review or alternative processing. For business forecasting, PI widths inform confidence in decisions; narrow intervals allow aggressive strategies while wide intervals require conservative approaches. Monitoring PI width changes over time detects model degradation (increasing residual std suggests distributional shift).
  
  **ML应用**: 在生产ML系统中，预测区间为单个预测设置操作边界。更宽的区间表示更高的不确定性，可能会触发手动审查或替代处理。对于业务预测，PI宽度告知决策的信心；窄区间允许激进策略，而宽区间需要保守方法。随时间监控PI宽度变化可检测模型退化（残差标准差增加表示分布偏移）。
```

➡️ **Next**: `../chapter_23/01_rank_data.ipynb`

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.mean` | 计算均值 | Calculate mean |
| `np.random` | 随机数生成 | Random number generation |
| `np.std` | 计算标准差 | Calculate standard deviation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

np.random.seed(42)
x = np.arange(0, 10, 0.5)
noise = np.random.normal(0, 2, len(x))
y = 2 * x + 1 + noise

slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_pred = slope * x + intercept
residuals = y - y_pred
residuals_std = np.std(residuals, ddof=1)

z_critical = 1.96
prediction_margin = z_critical * residuals_std

pi_lower = y_pred - prediction_margin
pi_upper = y_pred + prediction_margin

print(f"Prediction Interval width: {2*prediction_margin:.4f}")
print(f"Sample PI at x=0: [{pi_lower[0]:.2f}, {pi_upper[0]:.2f}]")
```

---

### Chapter Summary

# Chapter 22: Prediction Intervals
# 第22章：预测区间

## Theme | 主题
From fitted model to individual prediction uncertainty: accounting for both parameter and residual error.
从拟合模型到个体预测不确定性：考虑参数和残差误差。

## Evolution Roadmap | 演变路线图
```
Test Data (x, y pairs)
└─ Fit Regression Model
   └─ Prediction Interval Computation
      (wider than CI because: predict individual, not mean)
      └─ Visualization: data + fitted line + PI band
```

## Progression Logic | 进度逻辑

### Stage 1: Data & Model (数据与模型)
**English:** Generate or load regression data (x, y). Fit a linear model: ŷ = a + bx. Extract residuals and estimate error variance.
**中文:** 生成或加载回归数据(x、y)。拟合线性模型：ŷ = a + bx。提取残差并估计误差方差。

### Stage 2: Confidence Interval for Mean (均值的置信区间)
**English:** For a given x, the 95% CI for E[y|x] is: ŷ ± t_crit * SE_mean, where SE_mean depends on x's distance from mean(x).
**中文:** 对于给定的x，E[y|x]的95% CI为：ŷ ± t_crit * SE_mean，其中SE_mean取决于x与mean(x)的距离。

### Stage 3: Prediction Interval (预测区间)
**English:** PI for one future observation is wider: ŷ ± t_crit * SE_pred, where SE_pred = sqrt(s^2 + SE_mean^2). The s^2 term is residual variance (irreducible error).
**中文:** 一个未来观察的PI更宽：ŷ ± t_crit * SE_pred，其中SE_pred = sqrt(s^2 + SE_mean^2)。s^2项是残差方差(不可约误差)。

### Stage 4: Visualization (可视化)
**English:** Plot data points, fitted line, CI band (narrow, around line), and PI band (wide, encompasses most data). PI widens as x moves away from mean(x).
**中文:** 绘制数据点、拟合线、CI带(狭窄，围绕线)和PI带(宽，包含大部分数据)。当x远离mean(x)时，PI变宽。

## ML Relevance | ML相关性

1. **Individual Prediction (个体预测)**: PI quantifies uncertainty for one new observation, not the mean of many observations.
2. **Interval vs. Point (区间与点)**: Point prediction (ŷ) alone ignores residual variation; PI addresses this.
3. **Risk Quantification (风险量化)**: Wider PI indicates higher residual error, suggesting model limitations or high unexplained variability.
4. **Model Comparison (模型比较)**: Models with smaller residual variance have narrower PIs, all else equal.


---
