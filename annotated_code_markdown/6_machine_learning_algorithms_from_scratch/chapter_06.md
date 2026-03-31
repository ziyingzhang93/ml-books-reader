# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 06

---

### Chapter Summary / 章节总结



---

### Harness Cross Validation

# 01 — Harness Cross Validation / 交叉验证

**Chapter 06 — File 1 of 2 / 第06章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Example of a Cross Validation Test Harness**.

本脚本演示 **Example of a Cross Validation Test Harness**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Example of a Cross Validation Test Harness

```python
from random import seed
from random import randrange
from csv import reader
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset
```

---
## Step 3 — Convert string column to float

```python
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
```

---
## Step 4 — Split a dataset into k folds

```python
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
 # 获取长度 / Get length
	fold_size = int(len(dataset) / n_folds)
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_folds):
		fold = list()
  # 获取长度 / Get length
		while len(fold) < fold_size:
   # 获取长度 / Get length
			index = randrange(len(dataset_copy))
   # 添加元素到列表末尾 / Append element to list end
			fold.append(dataset_copy.pop(index))
  # 添加元素到列表末尾 / Append element to list end
		dataset_split.append(fold)
	return dataset_split
```

---
## Step 5 — Calculate accuracy percentage

```python
def accuracy_metric(actual, predicted):
	correct = 0
 # 获取长度 / Get length
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
 # 获取长度 / Get length
	return correct / float(len(actual)) * 100.0
```

---
## Step 6 — Evaluate an algorithm using a cross validation split

```python
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
   # 添加元素到列表末尾 / Append element to list end
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(accuracy)
	return scores
```

---
## Step 7 — zero rule algorithm for classification

```python
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
 # 获取长度 / Get length
	predicted = [prediction for i in range(len(test))]
	return predicted
```

---
## Step 8 — Test cross validation test harness

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 9 — load and prepare data

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
```

---
## Step 10 — evaluate algorithm

```python
n_folds = 5
scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/len(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of a Cross Validation Test Harness 是机器学习中的常用技术。  
  *Example of a Cross Validation Test Harness is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Harness Cross Validation / 交叉验证
# Complete Code / 完整代码
# ===============================

# Example of a Cross Validation Test Harness
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
 # 获取长度 / Get length
	fold_size = int(len(dataset) / n_folds)
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_folds):
		fold = list()
  # 获取长度 / Get length
		while len(fold) < fold_size:
   # 获取长度 / Get length
			index = randrange(len(dataset_copy))
   # 添加元素到列表末尾 / Append element to list end
			fold.append(dataset_copy.pop(index))
  # 添加元素到列表末尾 / Append element to list end
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
 # 获取长度 / Get length
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
 # 获取长度 / Get length
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
   # 添加元素到列表末尾 / Append element to list end
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
  # 添加元素到列表末尾 / Append element to list end
		scores.append(accuracy)
	return scores

# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
 # 获取长度 / Get length
	predicted = [prediction for i in range(len(test))]
	return predicted

# Test cross validation test harness
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, zero_rule_algorithm_classification, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/len(scores)))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Harness Train Test

# 01 — Harness Train Test / Harness Train Test

**Chapter 06 — File 2 of 2 / 第06章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Example of a Train-Test Test Harness**.

本脚本演示 **Example of a Train-Test Test Harness**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Example of a Train-Test Test Harness

```python
from random import seed
from random import randrange
from csv import reader
```

---
## Step 2 — Load a CSV file

```python
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset
```

---
## Step 3 — Convert string column to float

```python
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
```

---
## Step 4 — Split a dataset into a train and test set

```python
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(dataset, split):
	train = list()
 # 获取长度 / Get length
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
 # 获取长度 / Get length
	while len(train) < train_size:
  # 获取长度 / Get length
		index = randrange(len(dataset_copy))
  # 添加元素到列表末尾 / Append element to list end
		train.append(dataset_copy.pop(index))
	return train, dataset_copy
```

---
## Step 5 — Calculate accuracy percentage

```python
def accuracy_metric(actual, predicted):
	correct = 0
 # 获取长度 / Get length
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
 # 获取长度 / Get length
	return correct / float(len(actual)) * 100.0
```

---
## Step 6 — Evaluate an algorithm using a train/test split

```python
def evaluate_algorithm(dataset, algorithm, split, *args):
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
  # 添加元素到列表末尾 / Append element to list end
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	accuracy = accuracy_metric(actual, predicted)
	return accuracy
```

---
## Step 7 — zero rule algorithm for classification

```python
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
 # 获取长度 / Get length
	predicted = [prediction for i in range(len(test))]
	return predicted
```

---
## Step 8 — Test the train/test harness

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 9 — load and prepare data

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
```

---
## Step 10 — evaluate algorithm

```python
split = 0.6
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, split)
# 打印输出 / Print output
print('Accuracy: %.3f%%' % (accuracy))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of a Train-Test Test Harness 是机器学习中的常用技术。  
  *Example of a Train-Test Test Harness is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Harness Train Test / Harness Train Test
# Complete Code / 完整代码
# ===============================

# Example of a Train-Test Test Harness
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
 # 打开文件（自动关闭） / Open file (auto-close)
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
   # 添加元素到列表末尾 / Append element to list end
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset into a train and test set
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(dataset, split):
	train = list()
 # 获取长度 / Get length
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
 # 获取长度 / Get length
	while len(train) < train_size:
  # 获取长度 / Get length
		index = randrange(len(dataset_copy))
  # 添加元素到列表末尾 / Append element to list end
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
 # 获取长度 / Get length
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
 # 获取长度 / Get length
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
  # 添加元素到列表末尾 / Append element to list end
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	accuracy = accuracy_metric(actual, predicted)
	return accuracy

# zero rule algorithm for classification
def zero_rule_algorithm_classification(train, test):
	output_values = [row[-1] for row in train]
	prediction = max(set(output_values), key=output_values.count)
 # 获取长度 / Get length
	predicted = [prediction for i in range(len(test))]
	return predicted

# Test the train/test harness
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm_classification, split)
# 打印输出 / Print output
print('Accuracy: %.3f%%' % (accuracy))
```

---
