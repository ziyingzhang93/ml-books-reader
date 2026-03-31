# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 05

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 05 — File 1 of 2 / 第05章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **define an imbalanced dataset with a 1:100 class ratio**.

本脚本演示 **define an imbalanced dataset with a 1:100 class ratio**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — define an imbalanced dataset with a 1:100 class ratio

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 3 — summarize class distribution

```python
counter = Counter(y)
# 打印输出 / Print output
print(counter)
```

---
## Step 4 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: define an imbalanced dataset with a 1:100 class ratio 是机器学习中的常用技术。  
  *define an imbalanced dataset with a 1:100 class ratio is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# define an imbalanced dataset with a 1:100 class ratio
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# summarize class distribution
counter = Counter(y)
# 打印输出 / Print output
print(counter)
# scatter plot of examples by class label
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Naive Model

# 02 — Naive Model / 02 Naive Model

**Chapter 05 — File 2 of 2 / 第05章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **evaluate a majority class classifier on an 1:100 imbalanced dataset**.

本脚本演示 **evaluate a majority class classifier on an 1:100 imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — evaluate a majority class classifier on an 1:100 imbalanced dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
```

---
## Step 2 — evaluate a model using repeated k-fold cross-validation

```python
def evaluate_model(X, y, model):
```

---
## Step 3 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 4 — evaluate the model on the dataset

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 5 — return scores from each fold and each repeat

```python
return scores
```

---
## Step 6 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 7 — define model

```python
model = DummyClassifier(strategy='most_frequent')
```

---
## Step 8 — evaluate the model

```python
scores = evaluate_model(X, y, model)
```

---
## Step 9 — summarize performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.2f%%' % (mean(scores) * 100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a majority class classifier on an 1:100 imbalanced dataset 是机器学习中的常用技术。  
  *evaluate a majority class classifier on an 1:100 imbalanced dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Naive Model / 02 Naive Model
# Complete Code / 完整代码
# ===============================

# evaluate a majority class classifier on an 1:100 imbalanced dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y, model):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model on the dataset
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# return scores from each fold and each repeat
	return scores

# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = DummyClassifier(strategy='most_frequent')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
# 打印输出 / Print output
print('Mean Accuracy: %.2f%%' % (mean(scores) * 100))
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **2 code files** demonstrating chapter 05.

本章包含 **2 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_naive_model.ipynb` — Naive Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
