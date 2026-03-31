# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 29

---

### Make Classification Dataset

# 01 — Make Classification Dataset / 分类

**Chapter 29 — File 1 of 6 / 第29章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **define a small classification dataset**.

本脚本演示 **define a small classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — define a small classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
```

---
## Step 3 — summarize the shape of the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: define a small classification dataset 是机器学习中的常用技术。  
  *define a small classification dataset is a common technique in machine learning.*

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
# Make Classification Dataset / 分类
# Complete Code / 完整代码
# ===============================

# define a small classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
# summarize the shape of the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Kfold Decision Tree

# 02 — Kfold Decision Tree / 决策树

**Chapter 29 — File 2 of 6 / 第29章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate a decision tree on the entire small dataset**.

本脚本演示 **evaluate a decision tree on the entire small dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate a decision tree on the entire small dataset

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
from sklearn.tree import DecisionTreeClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 3 — define model

```python
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
```

---
## Step 4 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report result

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a decision tree on the entire small dataset 是机器学习中的常用技术。  
  *evaluate a decision tree on the entire small dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kfold Decision Tree / 决策树
# Complete Code / 完整代码
# ===============================

# evaluate a decision tree on the entire small dataset
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
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=1, random_state=1)
# define model
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### All Feature Subset

# 09 — All Feature Subset / 特征工程

**Chapter 29 — File 3 of 6 / 第29章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **feature selection by enumerating all possible subsets of features**.

本脚本演示 **feature selection by enumerating all possible subsets of features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — feature selection by enumerating all possible subsets of features

```python
from itertools import product
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
```

---
## Step 3 — determine the number of columns

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_cols = X.shape[1]
best_subset, best_score = None, 0.0
```

---
## Step 4 — enumerate all combinations of input features

```python
for subset in product([True, False], repeat=n_cols):
```

---
## Step 5 — convert into column indexes

```python
# 同时获取索引和值 / Get both index and value
ix = [i for i, x in enumerate(subset) if x]
```

---
## Step 6 — check for now column (all False)

```python
# 获取长度 / Get length
if len(ix) == 0:
		continue
```

---
## Step 7 — select columns

```python
X_new = X[:, ix]
```

---
## Step 8 — define model

```python
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
```

---
## Step 9 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 10 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 11 — summarize scores

```python
result = mean(scores)
```

---
## Step 12 — report progress

```python
# 打印输出 / Print output
print('>f(%s) = %f ' % (ix, result))
```

---
## Step 13 — check if it is better than the best so far

```python
if best_score is None or result >= best_score:
```

---
## Step 14 — better result

```python
best_subset, best_score = ix, result
```

---
## Step 15 — report best

```python
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best_subset, best_score))
```

---
## Learning Notes / 学习笔记

- **概念**: feature selection by enumerating all possible subsets of features 是机器学习中的常用技术。  
  *feature selection by enumerating all possible subsets of features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# All Feature Subset / 特征工程
# Complete Code / 完整代码
# ===============================

# feature selection by enumerating all possible subsets of features
from itertools import product
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=3, random_state=1)
# determine the number of columns
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_cols = X.shape[1]
best_subset, best_score = None, 0.0
# enumerate all combinations of input features
for subset in product([True, False], repeat=n_cols):
	# convert into column indexes
 # 同时获取索引和值 / Get both index and value
	ix = [i for i, x in enumerate(subset) if x]
	# check for now column (all False)
 # 获取长度 / Get length
	if len(ix) == 0:
		continue
	# select columns
	X_new = X[:, ix]
	# define model
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	model = DecisionTreeClassifier()
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# summarize scores
	result = mean(scores)
	# report progress
 # 打印输出 / Print output
	print('>f(%s) = %f ' % (ix, result))
	# check if it is better than the best so far
	if best_score is None or result >= best_score:
		# better result
		best_subset, best_score = ix, result
# report best
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('f(%s) = %f' % (best_subset, best_score))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Make Large Dataset



---

### Large Decision Tree

# 11 — Large Decision Tree / 决策树

**Chapter 29 — File 5 of 6 / 第29章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate a decision tree on the entire larger dataset**.

本脚本演示 **evaluate a decision tree on the entire larger dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate a decision tree on the entire larger dataset

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
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
```

---
## Step 3 — define model

```python
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
```

---
## Step 4 — define evaluation procedure

```python
cv = StratifiedKFold(n_splits=3)
```

---
## Step 5 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report result

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a decision tree on the entire larger dataset 是机器学习中的常用技术。  
  *evaluate a decision tree on the entire larger dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Large Decision Tree / 决策树
# Complete Code / 完整代码
# ===============================

# evaluate a decision tree on the entire larger dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# define model
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
# define evaluation procedure
cv = StratifiedKFold(n_splits=3)
# evaluate model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report result
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Stochastic Feature

# 19 — Stochastic Feature / 特征工程

**Chapter 29 — File 6 of 6 / 第29章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **stochastic optimization for feature selection**.

本脚本演示 **stochastic optimization for feature selection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — stochastic optimization for feature selection

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
```

---
## Step 2 — objective function

```python
def objective(X, y, subset):
```

---
## Step 3 — convert into column indexes

```python
# 同时获取索引和值 / Get both index and value
ix = [i for i, x in enumerate(subset) if x]
```

---
## Step 4 — check for now column (all False)

```python
# 获取长度 / Get length
if len(ix) == 0:
		return 0.0
```

---
## Step 5 — select columns

```python
X_new = X[:, ix]
```

---
## Step 6 — define model

```python
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
model = DecisionTreeClassifier()
```

---
## Step 7 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=3, n_jobs=-1)
```

---
## Step 8 — summarize scores

```python
result = mean(scores)
	return result, ix
```

---
## Step 9 — mutation operator

```python
def mutate(solution, p_mutate):
```

---
## Step 10 — make a copy

```python
child = solution.copy()
 # 获取长度 / Get length
	for i in range(len(child)):
```

---
## Step 11 — check for a mutation

```python
if rand() < p_mutate:
```

---
## Step 12 — flip the inclusion

```python
child[i] = not child[i]
	return child
```

---
## Step 13 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, n_iter, p_mutate):
```

---
## Step 14 — generate an initial point

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
solution = choice([True, False], size=X.shape[1])
```

---
## Step 15 — evaluate the initial point

```python
solution_eval, ix = objective(X, y, solution)
```

---
## Step 16 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 17 — take a step

```python
candidate = mutate(solution, p_mutate)
```

---
## Step 18 — evaluate candidate point

```python
candidate_eval, ix = objective(X, y, candidate)
```

---
## Step 19 — check if we should keep the new point

```python
if candidate_eval >= solution_eval:
```

---
## Step 20 — store the new point

```python
solution, solution_eval = candidate, candidate_eval
```

---
## Step 21 — report progress

```python
# 打印输出 / Print output
print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
	return solution, solution_eval
```

---
## Step 22 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
```

---
## Step 23 — define the total iterations

```python
n_iter = 100
```

---
## Step 24 — probability of including/excluding a column

```python
p_mut = 10.0 / 500.0
```

---
## Step 25 — perform the hill climbing search

```python
subset, score = hillclimbing(X, y, objective, n_iter, p_mut)
```

---
## Step 26 — convert into column indexes

```python
# 同时获取索引和值 / Get both index and value
ix = [i for i, x in enumerate(subset) if x]
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Best: f(%d) = %f' % (len(ix), score))
```

---
## Learning Notes / 学习笔记

- **概念**: stochastic optimization for feature selection 是机器学习中的常用技术。  
  *stochastic optimization for feature selection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stochastic Feature / 特征工程
# Complete Code / 完整代码
# ===============================

# stochastic optimization for feature selection
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier

# objective function
def objective(X, y, subset):
	# convert into column indexes
 # 同时获取索引和值 / Get both index and value
	ix = [i for i, x in enumerate(subset) if x]
	# check for now column (all False)
 # 获取长度 / Get length
	if len(ix) == 0:
		return 0.0
	# select columns
	X_new = X[:, ix]
	# define model
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	model = DecisionTreeClassifier()
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=3, n_jobs=-1)
	# summarize scores
	result = mean(scores)
	return result, ix

# mutation operator
def mutate(solution, p_mutate):
	# make a copy
	child = solution.copy()
 # 获取长度 / Get length
	for i in range(len(child)):
		# check for a mutation
		if rand() < p_mutate:
			# flip the inclusion
			child[i] = not child[i]
	return child

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, p_mutate):
	# generate an initial point
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	solution = choice([True, False], size=X.shape[1])
	# evaluate the initial point
	solution_eval, ix = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
		candidate = mutate(solution, p_mutate)
		# evaluate candidate point
		candidate_eval, ix = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
		# report progress
  # 打印输出 / Print output
		print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
	return solution, solution_eval

# define dataset
X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
# define the total iterations
n_iter = 100
# probability of including/excluding a column
p_mut = 10.0 / 500.0
# perform the hill climbing search
subset, score = hillclimbing(X, y, objective, n_iter, p_mut)
# convert into column indexes
# 同时获取索引和值 / Get both index and value
ix = [i for i, x in enumerate(subset) if x]
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Best: f(%d) = %f' % (len(ix), score))
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **6 code files** demonstrating chapter 29.

本章包含 **6 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `01_make_classification_dataset.ipynb` — Make Classification Dataset
  2. `02_kfold_decision_tree.ipynb` — Kfold Decision Tree
  3. `09_all_feature_subset.ipynb` — All Feature Subset
  4. `10_make_large_dataset.ipynb` — Make Large Dataset
  5. `11_large_decision_tree.ipynb` — Large Decision Tree
  6. `19_stochastic_feature.ipynb` — Stochastic Feature

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
