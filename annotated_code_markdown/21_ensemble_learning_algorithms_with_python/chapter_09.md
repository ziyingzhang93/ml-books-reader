# 集成学习算法 / Ensemble Learning Algorithms
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
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 4 — summarize the number of examples in each class

```python
# 打印输出 / Print output
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
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# summarize the number of examples in each class
# 打印输出 / Print output
print(Counter(y))
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Ecoc Evaluation



---

### Ecoc Predict



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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.multiclass import OutputCodeClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
# 生成整数序列 / Generate integer sequence
for i in range(1,21):
```

---
## Step 5 — create model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
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
# 获取字典的键值对 / Get dict key-value pairs
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
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 15 — summarize results along the way

```python
# 打印输出 / Print output
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
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.multiclass import OutputCodeClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# enumerate the number of bits from 1 to 20
 # 生成整数序列 / Generate integer sequence
	for i in range(1,21):
		# create model
  # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
		model = LogisticRegression()
		# create error correcting output code classifier
		models[str(i)] = OutputCodeClassifier(model, code_size=i, random_state=1)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the scores
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the scores
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	# summarize results along the way
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

### Chapter Summary / 章节总结



---
