# 机器学习数据准备 / Data Preparation for ML
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# summarize dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — example of normalizing input and output variables for regression.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import HuberRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 归一化到[0,1]范围 / Normalize to [0,1] range
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
```

---
## Step 5 — prepare the model with target scaling

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
```

---
## Step 6 — evaluate model

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
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
# 打印输出 / Print output
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import HuberRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling
# 归一化到[0,1]范围 / Normalize to [0,1] range
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
# prepare the model with target scaling
# 归一化到[0,1]范围 / Normalize to [0,1] range
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
# 打印输出 / Print output
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
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — example of power transform input and output variables for regression.

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import HuberRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 归一化到[0,1]范围 / Normalize to [0,1] range
steps.append(('scale', MinMaxScaler(feature_range=(1e-5,1))))
# 添加元素到列表末尾 / Append element to list end
steps.append(('power', PowerTransformer()))
# 添加元素到列表末尾 / Append element to list end
steps.append(('model', HuberRegressor()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
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
# 打印输出 / Print output
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import absolute
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import loadtxt
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import HuberRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling and power transform
steps = list()
# 归一化到[0,1]范围 / Normalize to [0,1] range
steps.append(('scale', MinMaxScaler(feature_range=(1e-5,1))))
# 添加元素到列表末尾 / Append element to list end
steps.append(('power', PowerTransformer()))
# 添加元素到列表末尾 / Append element to list end
steps.append(('model', HuberRegressor()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=steps)
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
# evaluate model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
# 打印输出 / Print output
print('Mean MAE: %.3f' % (s_mean))
```

---

### Chapter Summary / 章节总结

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
