# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 18

---

### Chapter Summary / 章节总结



---

### Stacking Sonar

# 01 — Stacking Sonar / 堆叠方法

**Chapter 18 — File 1 of 1 / 第18章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Stacking on the sonar dataset**.

本脚本演示 **Stacking on the sonar dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
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
  🏗️ 定义模型 / Define Model
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Stacking on the sonar dataset

```python
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp
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
## Step 4 — Convert string column to integer

```python
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
 # 同时获取索引和值 / Get both index and value
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
```

---
## Step 5 — Split a dataset into k folds

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
## Step 6 — Calculate accuracy percentage

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
## Step 7 — Evaluate an algorithm using a cross validation split

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
## Step 8 — Calculate the Euclidean distance between two vectors

```python
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
```

---
## Step 9 — Locate neighbors for a new row

```python
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
  # 添加元素到列表末尾 / Append element to list end
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(num_neighbors):
  # 添加元素到列表末尾 / Append element to list end
		neighbors.append(distances[i][0])
	return neighbors
```

---
## Step 10 — Make a prediction with kNN

```python
def knn_predict(model, test_row, num_neighbors=2):
	neighbors = get_neighbors(model, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
```

---
## Step 11 — Prepare the kNN model

```python
def knn_model(train):
	return train
```

---
## Step 12 — Make a prediction with weights

```python
def perceptron_predict(model, row):
	activation = model[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		activation += model[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
```

---
## Step 13 — Estimate Perceptron weights using stochastic gradient descent

```python
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
 # 获取长度 / Get length
	weights = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			prediction = perceptron_predict(weights, row)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
   # 获取长度 / Get length
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
```

---
## Step 14 — Make a prediction with coefficients

```python
def logistic_regression_predict(model, row):
	yhat = model[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += model[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
```

---
## Step 15 — Estimate logistic regression coefficients using stochastic gradient descent

```python
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			yhat = logistic_regression_predict(coef, row)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
```

---
## Step 16 — Make predictions with sub-models and construct a new stacked row

```python
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
 # 获取长度 / Get length
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_row.append(prediction)
 # 添加元素到列表末尾 / Append element to list end
	stacked_row.append(row[-1])
 # 获取长度 / Get length
	return row[0:len(row)-1] + stacked_row
```

---
## Step 17 — Stacked Generalization Algorithm

```python
def stacking(train, test):
	model_list = [knn_model, perceptron_model]
	predict_list = [knn_predict, perceptron_predict]
	models = list()
 # 获取长度 / Get length
	for i in range(len(model_list)):
		model = model_list[i](train)
  # 添加元素到列表末尾 / Append element to list end
		models.append(model)
	stacked_dataset = list()
	for row in train:
		stacked_row = to_stacked_row(models, predict_list, row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_dataset.append(stacked_row)
	stacked_model = logistic_regression_model(stacked_dataset)
	predictions = list()
	for row in test:
		stacked_row = to_stacked_row(models, predict_list, row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_dataset.append(stacked_row)
		prediction = logistic_regression_predict(stacked_model, stacked_row)
		prediction = round(prediction)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(prediction)
	return predictions
```

---
## Step 18 — Test stacking on the sonar dataset

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 19 — load and prepare data

```python
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
```

---
## Step 20 — convert string attributes to integers

```python
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
```

---
## Step 21 — convert class column to integers

```python
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---
## Learning Notes / 学习笔记

- **概念**: Stacking on the sonar dataset 是机器学习中的常用技术。  
  *Stacking on the sonar dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Stacking Sonar / 堆叠方法
# Complete Code / 完整代码
# ===============================

# Stacking on the sonar dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp

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

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
 # 同时获取索引和值 / Get both index and value
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

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

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
  # 添加元素到列表末尾 / Append element to list end
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(num_neighbors):
  # 添加元素到列表末尾 / Append element to list end
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with kNN
def knn_predict(model, test_row, num_neighbors=2):
	neighbors = get_neighbors(model, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Prepare the kNN model
def knn_model(train):
	return train

# Make a prediction with weights
def perceptron_predict(model, row):
	activation = model[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		activation += model[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
 # 获取长度 / Get length
	weights = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			prediction = perceptron_predict(weights, row)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
   # 获取长度 / Get length
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Make a prediction with coefficients
def logistic_regression_predict(model, row):
	yhat = model[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += model[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			yhat = logistic_regression_predict(coef, row)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
 # 获取长度 / Get length
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_row.append(prediction)
 # 添加元素到列表末尾 / Append element to list end
	stacked_row.append(row[-1])
 # 获取长度 / Get length
	return row[0:len(row)-1] + stacked_row

# Stacked Generalization Algorithm
def stacking(train, test):
	model_list = [knn_model, perceptron_model]
	predict_list = [knn_predict, perceptron_predict]
	models = list()
 # 获取长度 / Get length
	for i in range(len(model_list)):
		model = model_list[i](train)
  # 添加元素到列表末尾 / Append element to list end
		models.append(model)
	stacked_dataset = list()
	for row in train:
		stacked_row = to_stacked_row(models, predict_list, row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_dataset.append(stacked_row)
	stacked_model = logistic_regression_model(stacked_dataset)
	predictions = list()
	for row in test:
		stacked_row = to_stacked_row(models, predict_list, row)
  # 添加元素到列表末尾 / Append element to list end
		stacked_dataset.append(stacked_row)
		prediction = logistic_regression_predict(stacked_model, stacked_row)
		prediction = round(prediction)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(prediction)
	return predictions

# Test stacking on the sonar dataset
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---
