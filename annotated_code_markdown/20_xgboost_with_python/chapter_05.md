# XGBoost
## Chapter 05

---

### Chapter Summary

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **4 code files** demonstrating chapter 05.

本章包含 **4 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `breast_one_hot.ipynb` — Breast One Hot
  2. `horse_colic_missing.ipynb` — Horse Colic Missing
  3. `horse_colic_missing_imputer.ipynb` — Horse Colic Missing Imputer
  4. `iris_label_encode.ipynb` — Iris Label Encode

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---

### Breast One Hot

# 01 — Breast One Hot / Breast One Hot

**Chapter 05 — File 1 of 4 / 第05章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **binary classification, breast cancer dataset, label and one hot encoded**.

本脚本演示 **binary classification, breast cancer dataset, label and one hot encoded**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — binary classification, breast cancer dataset, label and one hot encoded

```python
from numpy import column_stack
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

---
## Step 2 — load data

```python
data = read_csv('datasets-uci-breast-cancer.csv', header=None)
dataset = data.values
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:9]
X = X.astype(str)
Y = dataset[:,9]
```

---
## Step 4 — encode string input values as integers

```python
columns = []
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	feature = onehot_encoder.fit_transform(feature)
	columns.append(feature)
```

---
## Step 5 — collapse columns into array

```python
encoded_x = column_stack(columns)
print("X shape: ", encoded_x.shape)
```

---
## Step 6 — encode string class values as integers

```python
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
```

---
## Step 7 — split data into train and test sets

```python
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
```

---
## Step 8 — fit model on training data

```python
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
```

---
## Step 9 — make predictions for test data

```python
predictions = model.predict(X_test)
```

---
## Step 10 — evaluate predictions

```python
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---
## Learning Notes / 学习笔记

- **概念**: binary classification, breast cancer dataset, label and one hot encoded 是机器学习中的常用技术。  
  *binary classification, breast cancer dataset, label and one hot encoded is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Breast One Hot / Breast One Hot
# Complete Code / 完整代码
# ===============================

# binary classification, breast cancer dataset, label and one hot encoded
from numpy import column_stack
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# load data
data = read_csv('datasets-uci-breast-cancer.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:9]
X = X.astype(str)
Y = dataset[:,9]
# encode string input values as integers
columns = []
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	feature = onehot_encoder.fit_transform(feature)
	columns.append(feature)
# collapse columns into array
encoded_x = column_stack(columns)
print("X shape: ", encoded_x.shape)
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---

➡️ **Next / 下一步**: File 2 of 4

---
