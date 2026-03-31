# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 09

---

### Dataset



---

### Naive Cross Validation



---

### Naive Train Test



---

### Stratified Cv

# 04 — Stratified Cv / 04 Stratified Cv

**Chapter 09 — File 4 of 5 / 第09章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of stratified k-fold cross-validation with an imbalanced dataset**.

本脚本演示 **example of stratified k-fold cross-validation with an imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — example of stratified k-fold cross-validation with an imbalanced dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
```

---
## Step 3 — enumerate the splits and summarize the distributions

```python
for train_ix, test_ix in kfold.split(X, y):
```

---
## Step 4 — select rows

```python
train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
```

---
## Step 5 — summarize train and test composition

```python
# 获取长度 / Get length
train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
 # 获取长度 / Get length
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
 # 打印输出 / Print output
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

---
## Learning Notes / 学习笔记

- **概念**: example of stratified k-fold cross-validation with an imbalanced dataset 是机器学习中的常用技术。  
  *example of stratified k-fold cross-validation with an imbalanced dataset is a common technique in machine learning.*

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
# Stratified Cv / 04 Stratified Cv
# Complete Code / 完整代码
# ===============================

# example of stratified k-fold cross-validation with an imbalanced dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
	# select rows
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# summarize train and test composition
 # 获取长度 / Get length
	train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
 # 获取长度 / Get length
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
 # 打印输出 / Print output
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Stratified Train Test

# 05 — Stratified Train Test / 05 Stratified Train Test

**Chapter 09 — File 5 of 5 / 第09章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of stratified train/test split with an imbalanced dataset**.

本脚本演示 **example of stratified train/test split with an imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


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
```

---
## Step 1 — example of stratified train/test split with an imbalanced dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
```

---
## Step 3 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — summarize

```python
# 获取长度 / Get length
train_0, train_1 = len(trainy[trainy==0]), len(trainy[trainy==1])
# 获取长度 / Get length
test_0, test_1 = len(testy[testy==0]), len(testy[testy==1])
# 打印输出 / Print output
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

---
## Learning Notes / 学习笔记

- **概念**: example of stratified train/test split with an imbalanced dataset 是机器学习中的常用技术。  
  *example of stratified train/test split with an imbalanced dataset is a common technique in machine learning.*

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
# Stratified Train Test / 05 Stratified Train Test
# Complete Code / 完整代码
# ===============================

# example of stratified train/test split with an imbalanced dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# summarize
# 获取长度 / Get length
train_0, train_1 = len(trainy[trainy==0]), len(trainy[trainy==1])
# 获取长度 / Get length
test_0, test_1 = len(testy[testy==0]), len(testy[testy==1])
# 打印输出 / Print output
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

---

### Chapter Summary / 章节总结



---
