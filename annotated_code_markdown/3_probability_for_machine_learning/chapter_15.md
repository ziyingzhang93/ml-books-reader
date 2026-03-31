# 概率论与机器学习 / Probability for Machine Learning
## Chapter 15

---

### Linear Regression

# 01 — Linear Regression / 线性回归

**Chapter 15 — File 1 of 3**

## Summary

We fit a linear regression model and calculate its Mean Squared Error (MSE) and the number of parameters. These values are used to compute information criteria (AIC and BIC) for model selection.

我们拟合线性回归模型并计算其均方误差（MSE）和参数数量。这些值用于计算用于模型选择的信息准则（AIC和BIC）。

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

## Step 1 — Generate Data and Fit Model / 生成数据和拟合模型

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error

# Generate synthetic data / 生成合成数据
# 生成随机数 / Generate random numbers
np.random.seed(42)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.linspace(0, 10, 100).reshape(-1, 1)
# 生成随机数 / Generate random numbers
y = 2.5 * X.ravel() + np.random.normal(0, 2, 100)  # y = 2.5x + noise

# Fit linear regression / 拟合线性回归
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X, y)

# Make predictions / 进行预测
# 用模型做预测 / Make predictions with model
y_pred = model.predict(X)

# Calculate metrics / 计算指标
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)
# 获取长度 / Get length
n_samples = len(y)
n_params = X.shape[1] + 1  # intercept + coefficients / 截距 + 系数

# 打印输出 / Print output
print(f'Linear Regression / 线性回归')
# 打印输出 / Print output
print(f'=' * 60)
# 打印输出 / Print output
print(f'Model: y = {model.intercept_:.4f} + {model.coef_[0]:.4f}*x')
# 打印输出 / Print output
print(f'\nMetrics / 指标:')
# 打印输出 / Print output
print(f'Sample size (n): {n_samples}')
# 打印输出 / Print output
print(f'Number of parameters (k): {n_params}')
# 打印输出 / Print output
print(f'Mean Squared Error (MSE): {mse:.6f}')
# 打印输出 / Print output
print(f'Residual Sum of Squares: {mse * n_samples:.6f}')
```

## Step 2 — Visualize Fit / 可视化拟合

```python
# 创建画布 / Create figure canvas
plt.figure(figsize=(10, 6))
# 绘制散点图 / Draw scatter plot
plt.scatter(X, y, alpha=0.5, s=50, label='Data')
# 绘制折线图 / Draw line plot
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Fit: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
# 设置X轴标签 / Set X-axis label
plt.xlabel('X')
# 设置Y轴标签 / Set Y-axis label
plt.ylabel('y')
# 设置图表标题 / Set chart title
plt.title('Linear Regression Fit')
# 显示图例 / Show legend
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Learning Notes / 学习笔记

- **Concept**: MSE measures average squared difference between predictions and observations. The number of parameters affects model complexity. These form the basis for information criteria.

- **ML Application**: Linear regression is fundamental for regression tasks, and understanding its error metrics is essential for model evaluation.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Next / 下一步\n\n➡️ **Next**: `02_linear_regression_aic.ipynb`

---

### Linear Regression Aic

# 02 — Linear Regression with AIC / 使用AIC的线性回归

**Chapter 15 — File 2 of 3**

## Summary

We implement the Akaike Information Criterion (AIC) formula to evaluate model fit. AIC balances goodness of fit (MSE) against model complexity (number of parameters), penalizing overfitting.

我们实现赤池信息准则（AIC）公式来评估模型拟合。AIC平衡拟合优度（MSE）与模型复杂性（参数数量），对过拟合进行惩罚。

**Formula:**

$$AIC = n \log(MSE) + 2k$$

where $n$ is sample size and $k$ is number of parameters

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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

## Step 1 — Calculate AIC / 计算AIC

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures

# Generate data / 生成数据
# 生成随机数 / Generate random numbers
np.random.seed(42)
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
X = np.linspace(0, 10, 100).reshape(-1, 1)
# 生成随机数 / Generate random numbers
y = 2.5 * X.ravel() + np.random.normal(0, 2, 100)

# Fit linear model and calculate AIC / 拟合线性模型并计算AIC
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X, y)
# 用模型做预测 / Make predictions with model
y_pred = model.predict(X)
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)

# 获取长度 / Get length
n = len(y)
k = 2  # intercept + 1 coefficient
aic = n * np.log(mse) + 2 * k

# 打印输出 / Print output
print(f'AIC Calculation / AIC计算')
# 打印输出 / Print output
print(f'=' * 70)
# 打印输出 / Print output
print(f'Sample size (n): {n}')
# 打印输出 / Print output
print(f'Number of parameters (k): {k}')
# 打印输出 / Print output
print(f'MSE: {mse:.6f}')
# 打印输出 / Print output
print(f'log(MSE): {np.log(mse):.6f}')
# 打印输出 / Print output
print(f'\nAIC = n*log(MSE) + 2k')
# 打印输出 / Print output
print(f'AIC = {n}*{np.log(mse):.6f} + 2*{k}')
# 打印输出 / Print output
print(f'AIC = {n*np.log(mse):.6f} + {2*k}')
# 打印输出 / Print output
print(f'AIC = {aic:.6f}')
```

## Step 2 — Compare Models with Different Complexities / 比较不同复杂度的模型

```python
# Compare models of different polynomial degrees / 比较不同多项式阶数的模型
degrees = range(1, 8)  # polynomial degrees 1-7 / 多项式阶数1-7
mse_vals = []
aic_vals = []
k_vals = []

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    # 拟合并转换数据（一步完成） / Fit and transform data (one step)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    # 训练模型 / Train the model
    model.fit(X_poly, y)
    # 用模型做预测 / Make predictions with model
    y_pred = model.predict(X_poly)
    
    # 计算均方误差 / Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    k = X_poly.shape[1]  # number of features
    aic = n * np.log(mse) + 2 * k
    
    # 添加元素到列表末尾 / Append element to list end
    mse_vals.append(mse)
    # 添加元素到列表末尾 / Append element to list end
    aic_vals.append(aic)
    # 添加元素到列表末尾 / Append element to list end
    k_vals.append(k)
    
    # 打印输出 / Print output
    print(f'Degree {degree}: k={k:2d}, MSE={mse:10.6f}, AIC={aic:10.4f}')

best_degree = degrees[np.argmin(aic_vals)]
# 打印输出 / Print output
print(f'\nBest model (by AIC): Degree {best_degree}')
```

## Step 3 — Visualize AIC / 可视化AIC

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSE vs degree / MSE vs 阶数
axes[0].plot(degrees, mse_vals, 'b-o', linewidth=2, markersize=8)
axes[0].set_xlabel('Polynomial Degree / 多项式阶数')
axes[0].set_ylabel('MSE')
axes[0].set_title('Model Fit (MSE) vs Complexity')
axes[0].grid(alpha=0.3)

# AIC vs degree / AIC vs 阶数
axes[1].plot(degrees, aic_vals, 'r-o', linewidth=2, markersize=8)
axes[1].axvline(best_degree, color='g', linestyle='--', linewidth=2, label=f'Best: Degree {best_degree}')
axes[1].set_xlabel('Polynomial Degree / 多项式阶数')
axes[1].set_ylabel('AIC')
axes[1].set_title('Model Selection (AIC) vs Complexity')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
# 显示图表 / Display the plot
plt.show()
```

## Learning Notes / 学习笔记\n\n- **Concept**: AIC penalizes model complexity, preventing overfitting. Lower AIC indicates better model. The 2k term penalizes each additional parameter.\n\n- **ML Application**: AIC is widely used for model selection, hyperparameter tuning, and comparing different model families.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.random` | 随机数生成 | Random number generation |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Next / 下一步\n\n➡️ **Next**: `03_linear_regression_bic.ipynb`

---

### Linear Regression Bic



---

### Chapter Summary / 章节总结

# Chapter 15: Information Criteria

## Overview
This chapter explores **Information Criteria** for model selection. These metrics balance model fit (likelihood) against model complexity, addressing the fundamental bias-variance tradeoff.

## Key Concepts
- **Likelihood**: How well model fits the data (larger is better)
- **Complexity Penalty**: Discourage overfitting from too many parameters
- **AIC (Akaike Information Criterion)**: Lighter penalty, prefer simpler models moderately
- **BIC (Bayesian Information Criterion)**: Heavier penalty, prefer simpler models more strongly
- **Overfitting**: Model learns noise rather than signal

## Evolution of Examples

### Building Model Selection
1. **01_baseline_linear_regression.py**: Fit baseline linear regression
2. **02_aic_model_selection.py**: Calculate AIC and compare models
3. **03_bic_model_selection.py**: Calculate BIC and compare models

## Formulas

### AIC (Akaike Information Criterion)
```
AIC = 2k - 2ln(L)
```
- k: number of parameters
- L: maximum likelihood
- Prefer model with **lower AIC**
- Asymptotically justified for prediction accuracy

### BIC (Bayesian Information Criterion)
```
BIC = k*ln(n) - 2ln(L)
```
- n: number of observations
- k: number of parameters
- L: maximum likelihood
- Prefer model with **lower BIC**
- Heavier penalty than AIC when n is large

## Key Differences

| Aspect | AIC | BIC |
|--------|-----|-----|
| Penalty coefficient | 2 | ln(n) |
| For large n | 2 < ln(n) | Heavier penalty |
| Philosophy | Prediction error | Model probability |
| Preference | Simpler models (moderate) | Simpler models (strong) |
| Use case | Cross-validation | Bayesian model comparison |

## Intuition

### The Tradeoff
```
Model Quality = Fit - Penalty
           = likelihood - complexity_cost
```

- More parameters → better fit but higher penalty
- Goal: find "sweet spot" where improvement in fit outweighs complexity cost

### Why Penalty?
Extra parameters can always improve training fit, even if they capture noise rather than signal.
Penalty prevents this overfitting by requiring that extra complexity must earn its cost through better fit.

## Practical Application

### Comparing Models
1. Fit candidate models with different complexity (k)
2. Calculate AIC (or BIC) for each
3. Select model with lowest criterion value
4. ΔAIC = AIC_model - AIC_best
   - ΔAIC < 2: substantial support for both models
   - ΔAIC > 10: strong support for best model

## Limitations
1. **Relative comparison only**: AIC value has no absolute meaning
2. **Not for overfitting detection alone**: Use with cross-validation
3. **Assumes proper likelihood**: Requires correct probability model
4. **Small sample bias**: AIC has correction (AICc) for small n

## Key Takeaways
1. Information criteria quantify the bias-variance tradeoff
2. Both AIC and BIC favor simpler models, but BIC more aggressively
3. Use for model comparison, not as absolute quality measures
4. AIC ≈ prediction focus, BIC ≈ model probability focus
5. Complement with cross-validation for robust model selection

---
