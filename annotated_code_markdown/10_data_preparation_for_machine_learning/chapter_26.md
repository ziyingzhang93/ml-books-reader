# ML数据准备
## Chapter 26

---

### Define Dataset

# 01 — Define Dataset / 01 Define Dataset

**Chapter 26 — File 1 of 4 / 第26章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of creating a test dataset and splitting it into train and test sets**.

本脚本演示 **example of creating a test dataset and splitting it into train and test sets**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of creating a test dataset and splitting it into train and test sets

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
```

---
## Step 2 — prepare dataset

```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
```

---
## Step 3 — split data into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — summarize the scale of each input variable

```python
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train[:, i].min(), X_train[:, i].max(),
			X_test[:, i].min(), X_test[:, i].max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of creating a test dataset and splitting it into train and test sets 是机器学习中的常用技术。  
  *example of creating a test dataset and splitting it into train and test sets is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Define Dataset / 01 Define Dataset
# Complete Code / 完整代码
# ===============================

# example of creating a test dataset and splitting it into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train[:, i].min(), X_train[:, i].max(),
			X_test[:, i].min(), X_test[:, i].max()))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Scale Dataset

# 02 — Scale Dataset / 数据缩放

**Chapter 26 — File 2 of 4 / 第26章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of scaling the dataset**.

本脚本演示 **example of scaling the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — example of scaling the dataset

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```

---
## Step 2 — prepare dataset

```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
```

---
## Step 3 — split data into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — define scaler

```python
scaler = MinMaxScaler()
```

---
## Step 5 — fit scaler on the training dataset

```python
scaler.fit(X_train)
```

---
## Step 6 — transform both datasets

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---
## Step 7 — summarize the scale of each input variable

```python
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train_scaled[:, i].min(), X_train_scaled[:, i].max(),
			X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
```

---
## Learning Notes / 学习笔记

- **概念**: example of scaling the dataset 是机器学习中的常用技术。  
  *example of scaling the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scale Dataset / 数据缩放
# Complete Code / 完整代码
# ===============================

# example of scaling the dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)
# transform both datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train_scaled[:, i].min(), X_train_scaled[:, i].max(),
			X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Model Data Save Results

# 03 — Model Data Save Results / 保存/加载模型

**Chapter 26 — File 3 of 4 / 第26章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **example of fitting a model on the scaled dataset**.

本脚本演示 **example of fitting a model on the scaled dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
```

---
## Step 1 — example of fitting a model on the scaled dataset

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump
```

---
## Step 2 — prepare dataset

```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
```

---
## Step 3 — split data into train and test sets

```python
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — define scaler

```python
scaler = MinMaxScaler()
```

---
## Step 5 — fit scaler on the training dataset

```python
scaler.fit(X_train)
```

---
## Step 6 — transform the training dataset

```python
X_train_scaled = scaler.transform(X_train)
```

---
## Step 7 — define model

```python
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)
```

---
## Step 8 — save the model

```python
dump(model, open('model.pkl', 'wb'))
```

---
## Step 9 — save the scaler

```python
dump(scaler, open('scaler.pkl', 'wb'))
```

---
## Learning Notes / 学习笔记

- **概念**: example of fitting a model on the scaled dataset 是机器学习中的常用技术。  
  *example of fitting a model on the scaled dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `model.fit` | 训练模型 | Train the model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model Data Save Results / 保存/加载模型
# Complete Code / 完整代码
# ===============================

# example of fitting a model on the scaled dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=1)
# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)
# transform the training dataset
X_train_scaled = scaler.transform(X_train)
# define model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)
# save the model
dump(model, open('model.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Load Model And Scaler

# 04 — Load Model And Scaler / 数据缩放

**Chapter 26 — File 4 of 4 / 第26章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **load model and scaler and make predictions on new data**.

本脚本演示 **load model and scaler and make predictions on new data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — load model and scaler and make predictions on new data

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
```

---
## Step 2 — prepare dataset

```python
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
```

---
## Step 3 — split data into train and test sets

```python
_, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — load the model

```python
model = load(open('model.pkl', 'rb'))
```

---
## Step 5 — load the scaler

```python
scaler = load(open('scaler.pkl', 'rb'))
```

---
## Step 6 — check scale of the test set before scaling

```python
print('Raw test set range')
for i in range(X_test.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test[:, i].min(), X_test[:, i].max()))
```

---
## Step 7 — transform the test dataset

```python
X_test_scaled = scaler.transform(X_test)
print('Scaled test set range')
for i in range(X_test_scaled.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
```

---
## Step 8 — make predictions on the test set

```python
yhat = model.predict(X_test_scaled)
```

---
## Step 9 — evaluate accuracy

```python
acc = accuracy_score(y_test, yhat)
print('Test Accuracy:', acc)
```

---
## Learning Notes / 学习笔记

- **概念**: load model and scaler and make predictions on new data 是机器学习中的常用技术。  
  *load model and scaler and make predictions on new data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Model And Scaler / 数据缩放
# Complete Code / 完整代码
# ===============================

# load model and scaler and make predictions on new data
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
# check scale of the test set before scaling
print('Raw test set range')
for i in range(X_test.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test[:, i].min(), X_test[:, i].max()))
# transform the test dataset
X_test_scaled = scaler.transform(X_test)
print('Scaled test set range')
for i in range(X_test_scaled.shape[1]):
	print('>%d, min=%.3f, max=%.3f' % (i, X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
# make predictions on the test set
yhat = model.predict(X_test_scaled)
# evaluate accuracy
acc = accuracy_score(y_test, yhat)
print('Test Accuracy:', acc)
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **4 code files** demonstrating chapter 26.

本章包含 **4 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_define_dataset.ipynb` — Define Dataset
  2. `02_scale_dataset.ipynb` — Scale Dataset
  3. `03_model_data_save_results.ipynb` — Model Data Save Results
  4. `04_load_model_and_scaler.ipynb` — Load Model And Scaler

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
