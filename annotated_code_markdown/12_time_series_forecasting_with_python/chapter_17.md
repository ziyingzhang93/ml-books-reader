# 时间序列预测 / Time Series Forecasting with Python
## Chapter 17

---

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **5 code files** demonstrating chapter 17.

本章包含 **5 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `forecast_error.ipynb` — Forecast Error
  2. `mean_absolute_error.ipynb` — Mean Absolute Error
  3. `mean_forecast_error.ipynb` — Mean Forecast Error
  4. `mean_squared_error.ipynb` — Mean Squared Error
  5. `root_mean_squared_error.ipynb` — Root Mean Squared Error

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---

### Forecast Error

# 01 — Forecast Error / 预测

**Chapter 17 — File 1 of 5 / 第17章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate forecast error**.

本脚本演示 **calculate forecast error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate forecast error

```python
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 获取长度 / Get length
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
# 打印输出 / Print output
print('Forecast Errors: %s' % forecast_errors)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate forecast error 是机器学习中的常用技术。  
  *calculate forecast error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Forecast Error / 预测
# Complete Code / 完整代码
# ===============================

# calculate forecast error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 获取长度 / Get length
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
# 打印输出 / Print output
print('Forecast Errors: %s' % forecast_errors)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Mean Absolute Error

# 01 — Mean Absolute Error / Mean Absolute Error

**Chapter 17 — File 2 of 5 / 第17章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate mean absolute error**.

本脚本演示 **calculate mean absolute error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate mean absolute error

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mae = mean_absolute_error(expected, predictions)
# 打印输出 / Print output
print('MAE: %f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate mean absolute error 是机器学习中的常用技术。  
  *calculate mean absolute error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mean Absolute Error / Mean Absolute Error
# Complete Code / 完整代码
# ===============================

# calculate mean absolute error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
mae = mean_absolute_error(expected, predictions)
# 打印输出 / Print output
print('MAE: %f' % mae)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Mean Forecast Error

# 01 — Mean Forecast Error / 预测

**Chapter 17 — File 3 of 5 / 第17章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate mean forecast error**.

本脚本演示 **calculate mean forecast error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate mean forecast error

```python
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 获取长度 / Get length
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
# 获取长度 / Get length
bias = sum(forecast_errors) * 1.0/len(expected)
# 打印输出 / Print output
print('Bias: %f' % bias)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate mean forecast error 是机器学习中的常用技术。  
  *calculate mean forecast error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mean Forecast Error / 预测
# Complete Code / 完整代码
# ===============================

# calculate mean forecast error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 获取长度 / Get length
forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
# 获取长度 / Get length
bias = sum(forecast_errors) * 1.0/len(expected)
# 打印输出 / Print output
print('Bias: %f' % bias)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Mean Squared Error

# 01 — Mean Squared Error / Mean Squared Error

**Chapter 17 — File 4 of 5 / 第17章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate mean squared error**.

本脚本演示 **calculate mean squared error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate mean squared error

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(expected, predictions)
# 打印输出 / Print output
print('MSE: %f' % mse)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate mean squared error 是机器学习中的常用技术。  
  *calculate mean squared error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mean Squared Error / Mean Squared Error
# Complete Code / 完整代码
# ===============================

# calculate mean squared error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(expected, predictions)
# 打印输出 / Print output
print('MSE: %f' % mse)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Root Mean Squared Error

# 01 — Root Mean Squared Error / Root Mean Squared Error

**Chapter 17 — File 5 of 5 / 第17章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **calculate root mean squared error**.

本脚本演示 **calculate root mean squared error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate root mean squared error

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(expected, predictions)
rmse = sqrt(mse)
# 打印输出 / Print output
print('RMSE: %f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: calculate root mean squared error 是机器学习中的常用技术。  
  *calculate root mean squared error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Root Mean Squared Error / Root Mean Squared Error
# Complete Code / 完整代码
# ===============================

# calculate root mean squared error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
expected = [0.0, 0.5, 0.0, 0.5, 0.0]
predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
# 计算均方误差 / Calculate Mean Squared Error
mse = mean_squared_error(expected, predictions)
rmse = sqrt(mse)
# 打印输出 / Print output
print('RMSE: %f' % rmse)
```

---
