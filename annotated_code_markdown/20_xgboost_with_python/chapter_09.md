# XGBoost
## Chapter 09

---

### Feature Selection Fixed

# 01 — Feature Selection Fixed / 特征工程

**Chapter 09 — File 3 of 4 / 第09章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **use feature importance for feature selection, with fix for xgboost 1.0.2**.

本脚本演示 **use feature importance for feature selection, with fix for xgboost 1.0.2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — use feature importance for feature selection, with fix for xgboost 1.0.2

```python
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
```

---
## Step 2 — define custom class to fix bug in xgboost 1.0.2

```python
class MyXGBClassifier(XGBClassifier):
	@property
	def coef_(self):
		return None
```

---
## Step 3 — load data

```python
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
```

---
## Step 4 — split data into X and y

```python
X = dataset[:,0:8]
Y = dataset[:,8]
```

---
## Step 5 — split data into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
```

---
## Step 6 — fit model on all training data

```python
model = MyXGBClassifier()
model.fit(X_train, y_train)
```

---
## Step 7 — make predictions for test data and evaluate

```python
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

---
## Step 8 — Fit model using each importance as a threshold

```python
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
```

---
## Step 9 — select features using threshold

```python
selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
```

---
## Step 10 — train model

```python
selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
```

---
## Step 11 — eval model

```python
select_X_test = selection.transform(X_test)
	predictions = selection_model.predict(select_X_test)
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
```

---
## Learning Notes / 学习笔记

- **概念**: use feature importance for feature selection, with fix for xgboost 1.0.2 是机器学习中的常用技术。  
  *use feature importance for feature selection, with fix for xgboost 1.0.2 is a common technique in machine learning.*

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
# Feature Selection Fixed / 特征工程
# Complete Code / 完整代码
# ===============================

# use feature importance for feature selection, with fix for xgboost 1.0.2
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# define custom class to fix bug in xgboost 1.0.2
class MyXGBClassifier(XGBClassifier):
	@property
	def coef_(self):
		return None

# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# fit model on all training data
model = MyXGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	predictions = selection_model.predict(select_X_test)
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Manual Feature Importance

# 01 — Manual Feature Importance / 特征工程

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **plot feature importance manually**.

本脚本演示 **plot feature importance manually**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
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

---
## Step 1 — plot feature importance manually

```python
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
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
y = dataset[:,8]
```

---
## Step 4 — fit model on training data

```python
model = XGBClassifier()
model.fit(X, y)
```

---
## Step 5 — feature importance

```python
print(model.feature_importances_)
```

---
## Step 6 — plot

```python
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot feature importance manually 是机器学习中的常用技术。  
  *plot feature importance manually is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Manual Feature Importance / 特征工程
# Complete Code / 完整代码
# ===============================

# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
```

---
