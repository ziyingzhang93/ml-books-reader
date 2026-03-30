# ML数据准备
## Chapter 25

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 25 — File 1 of 3 / 第25章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **load and summarize the dataset**.

本脚本演示 **load and summarize the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load and summarize the dataset

```python
from numpy import loadtxt
```

---
## Step 2 — load data

```python
dataset = loadtxt('housing.csv', delimiter=",")
```

---
## Step 3 — split into inputs and outputs

```python
X, y = dataset[:, :-1], dataset[:, -1]
```

---
## Step 4 — summarize dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load and summarize the dataset 是机器学习中的常用技术。  
  *load and summarize the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Dataset / 01 Load Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the dataset
from numpy import loadtxt
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# summarize dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Evaluate Model With Target Scaling

# 02 — Evaluate Model With Target Scaling / 模型评估

**Chapter 25 — File 2 of 3 / 第25章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of normalizing input and output variables for regression.**.

本脚本演示 **example of normalizing input and output variables for regression.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of normalizing input and output variables for regression.

```python
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
```

---
## Step 2 — load data

```python
dataset = loadtxt('housing.csv', delimiter=",")
```

---
## Step 3 — split into inputs and outputs

```python
X, y = dataset[:, :-1], dataset[:, -1]
```

---
## Step 4 — prepare the model with input scaling

```python
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
```

---
## Step 5 — prepare the model with target scaling

```python
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
```

---
## Step 6 — evaluate model

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

---
## Step 7 — convert scores to positive

```python
scores = absolute(scores)
```

---
## Step 8 — summarize the result

```python
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

---
## Learning Notes / 学习笔记

- **概念**: example of normalizing input and output variables for regression. 是机器学习中的常用技术。  
  *example of normalizing input and output variables for regression. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Model With Target Scaling / 模型评估
# Complete Code / 完整代码
# ===============================

# example of normalizing input and output variables for regression.
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Evaluate Model With Target Transforms

# 03 — Evaluate Model With Target Transforms / 数据变换

**Chapter 25 — File 3 of 3 / 第25章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **example of power transform input and output variables for regression.**.

本脚本演示 **example of power transform input and output variables for regression.**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of power transform input and output variables for regression.

```python
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
```

---
## Step 2 — load data

```python
dataset = loadtxt('housing.csv', delimiter=",")
```

---
## Step 3 — split into inputs and outputs

```python
X, y = dataset[:, :-1], dataset[:, -1]
```

---
## Step 4 — prepare the model with input scaling and power transform

```python
steps = list()
steps.append(('scale', MinMaxScaler(feature_range=(1e-5,1))))
steps.append(('power', PowerTransformer()))
steps.append(('model', HuberRegressor()))
pipeline = Pipeline(steps=steps)
```

---
## Step 5 — prepare the model with target scaling

```python
model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
```

---
## Step 6 — evaluate model

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

---
## Step 7 — convert scores to positive

```python
scores = absolute(scores)
```

---
## Step 8 — summarize the result

```python
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

---
## Learning Notes / 学习笔记

- **概念**: example of power transform input and output variables for regression. 是机器学习中的常用技术。  
  *example of power transform input and output variables for regression. is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Model With Target Transforms / 数据变换
# Complete Code / 完整代码
# ===============================

# example of power transform input and output variables for regression.
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling and power transform
steps = list()
steps.append(('scale', MinMaxScaler(feature_range=(1e-5,1))))
steps.append(('power', PowerTransformer()))
steps.append(('model', HuberRegressor()))
pipeline = Pipeline(steps=steps)
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

---

### Chapter Summary

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **3 code files** demonstrating chapter 25.

本章包含 **3 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_evaluate_model_with_target_scaling.ipynb` — Evaluate Model With Target Scaling
  3. `03_evaluate_model_with_target_transforms.ipynb` — Evaluate Model With Target Transforms

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
