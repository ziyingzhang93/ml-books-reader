# 集成学习
## Chapter 09

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 09 — File 1 of 4 / 第09章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **multi-class classification dataset**.

本脚本演示 **multi-class classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — multi-class classification dataset

```python
from collections import Counter
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
```

---
## Step 3 — summarize the dataset

```python
print(X.shape, y.shape)
```

---
## Step 4 — summarize the number of examples in each class

```python
print(Counter(y))
```

---
## Learning Notes / 学习笔记

- **概念**: multi-class classification dataset 是机器学习中的常用技术。  
  *multi-class classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# multi-class classification dataset
from collections import Counter
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# summarize the dataset
print(X.shape, y.shape)
# summarize the number of examples in each class
print(Counter(y))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Ecoc Num Bits

# 04 — Ecoc Num Bits / 04 Ecoc Num Bits

**Chapter 09 — File 4 of 4 / 第09章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **compare the number of bits per class for error-correcting output code classification**.

本脚本演示 **compare the number of bits per class for error-correcting output code classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — compare the number of bits per class for error-correcting output code classification

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — enumerate the number of bits from 1 to 20

```python
for i in range(1,21):
```

---
## Step 5 — create model

```python
model = LogisticRegression()
```

---
## Step 6 — create error correcting output code classifier

```python
models[str(i)] = OutputCodeClassifier(model, code_size=i, random_state=1)
	return models
```

---
## Step 7 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 8 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 9 — evaluate the model and collect the scores

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 10 — define dataset

```python
X, y = get_dataset()
```

---
## Step 11 — get the models to evaluate

```python
models = get_models()
```

---
## Step 12 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models.items():
```

---
## Step 13 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 14 — store the scores

```python
results.append(scores)
	names.append(name)
```

---
## Step 15 — summarize results along the way

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 16 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare the number of bits per class for error-correcting output code classification 是机器学习中的常用技术。  
  *compare the number of bits per class for error-correcting output code classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ecoc Num Bits / 04 Ecoc Num Bits
# Complete Code / 完整代码
# ===============================

# compare the number of bits per class for error-correcting output code classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# enumerate the number of bits from 1 to 20
	for i in range(1,21):
		# create model
		model = LogisticRegression()
		# create error correcting output code classifier
		models[str(i)] = OutputCodeClassifier(model, code_size=i, random_state=1)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the scores
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the scores
	results.append(scores)
	names.append(name)
	# summarize results along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
