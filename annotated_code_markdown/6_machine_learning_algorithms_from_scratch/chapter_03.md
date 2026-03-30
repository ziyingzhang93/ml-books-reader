# 从零实现ML算法
## Chapter 03

---

### Chapter Summary

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **2 code files** demonstrating chapter 03.

本章包含 **2 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `cross_validation_split.ipynb` — Cross Validation Split
  2. `split_train_test.ipynb` — Split Train Test

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---

### Cross Validation Split

# 01 — Cross Validation Split / 交叉验证

**Chapter 03 — File 1 of 2 / 第03章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Example of Creating a Cross Validation Split**.

本脚本演示 **Example of Creating a Cross Validation Split**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Creating a Cross Validation Split

```python
from random import seed
from random import randrange
```

---
## Step 2 — Split a dataset into k folds

```python
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for _ in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
```

---
## Step 3 — test cross validation split

```python
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Creating a Cross Validation Split 是机器学习中的常用技术。  
  *Example of Creating a Cross Validation Split is a common technique in machine learning.*

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
# Cross Validation Split / 交叉验证
# Complete Code / 完整代码
# ===============================

# Example of Creating a Cross Validation Split
from random import seed
from random import randrange

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for _ in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# test cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
folds = cross_validation_split(dataset, 4)
print(folds)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Split Train Test

# 01 — Split Train Test / Split Train Test

**Chapter 03 — File 2 of 2 / 第03章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Example of Splitting a Contrived Dataset into Train and Test**.

本脚本演示 **Example of Splitting a Contrived Dataset into Train and Test**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Splitting a Contrived Dataset into Train and Test

```python
from random import seed
from random import randrange
```

---
## Step 2 — Split a dataset into a train and test set

```python
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
```

---
## Step 3 — test train/test split

```python
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = train_test_split(dataset)
print(train)
print(test)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Splitting a Contrived Dataset into Train and Test 是机器学习中的常用技术。  
  *Example of Splitting a Contrived Dataset into Train and Test is a common technique in machine learning.*

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
# Split Train Test / Split Train Test
# Complete Code / 完整代码
# ===============================

# Example of Splitting a Contrived Dataset into Train and Test
from random import seed
from random import randrange

# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# test train/test split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = train_test_split(dataset)
print(train)
print(test)
```

---
