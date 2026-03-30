# XGBoost
## Chapter 06

---

### Chapter Summary

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **3 code files** demonstrating chapter 06.

本章包含 **3 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `cross_validation.ipynb` — Cross Validation
  2. `stratified_cross_validation.ipynb` — Stratified Cross Validation
  3. `train_test_split.ipynb` — Train Test Split

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---

### Stratified Cross Validation

# 01 — Stratified Cross Validation / 交叉验证

**Chapter 06 — File 2 of 3 / 第06章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **stratified k-fold cross validation evaluation of xgboost model**.

本脚本演示 **stratified k-fold cross validation evaluation of xgboost model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — stratified k-fold cross validation evaluation of xgboost model

```python
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — load data

```python
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — CV model

```python
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---
## Learning Notes / 学习笔记

- **概念**: stratified k-fold cross validation evaluation of xgboost model 是机器学习中的常用技术。  
  *stratified k-fold cross validation evaluation of xgboost model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stratified Cross Validation / 交叉验证
# Complete Code / 完整代码
# ===============================

# stratified k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Train Test Split

# 01 — Train Test Split / Train Test Split

**Chapter 06 — File 3 of 3 / 第06章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **train-test split evaluation of xgboost model**.

本脚本演示 **train-test split evaluation of xgboost model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — train-test split evaluation of xgboost model

```python
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load data

```python
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 4 — split data into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
```

---
## Step 5 — fit model on training data

```python
model = XGBClassifier()
model.fit(X_train, y_train)
```

---
## Step 6 — make predictions for test data

```python
predictions = model.predict(X_test)
```

---
## Step 7 — evaluate predictions

```python
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---
## Learning Notes / 学习笔记

- **概念**: train-test split evaluation of xgboost model 是机器学习中的常用技术。  
  *train-test split evaluation of xgboost model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Test Split / Train Test Split
# Complete Code / 完整代码
# ===============================

# train-test split evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---
