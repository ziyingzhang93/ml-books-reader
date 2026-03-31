# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 09

---

### Chapter Summary / 章节总结



---

### Estimate Coefficients

# 01 — Estimate Coefficients / Estimate Coefficients

**Chapter 09 — File 1 of 3 / 第09章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Example of estimating coefficients**.

本脚本演示 **Example of estimating coefficients**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — Example of estimating coefficients

```python
from math import exp
```

---
## Step 2 — Make a prediction with coefficients

```python
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
```

---
## Step 3 — Estimate logistic regression coefficients using stochastic gradient descent

```python
def coefficients_sgd(train, l_rate, n_epoch):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
  # 打印输出 / Print output
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef
```

---
## Step 4 — Calculate coefficients

```python
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.3
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
# 打印输出 / Print output
print(coef)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of estimating coefficients 是机器学习中的常用技术。  
  *Example of estimating coefficients is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Estimate Coefficients / Estimate Coefficients
# Complete Code / 完整代码
# ===============================

# Example of estimating coefficients
from math import exp

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
  # 打印输出 / Print output
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

# Calculate coefficients
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
l_rate = 0.3
n_epoch = 100
coef = coefficients_sgd(dataset, l_rate, n_epoch)
# 打印输出 / Print output
print(coef)
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Logistic Regression Diabetes

# 01 — Logistic Regression Diabetes / 回归

**Chapter 09 — File 2 of 3 / 第09章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Logistic Regression on Diabetes Dataset**.

本脚本演示 **Logistic Regression on Diabetes Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Logistic Regression on Diabetes Dataset

```python
from random import seed
from random import randrange
from csv import reader
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
## Step 4 — Find the min and max values for each column

```python
def dataset_minmax(dataset):
	minmax = list()
 # 获取长度 / Get length
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
  # 添加元素到列表末尾 / Append element to list end
		minmax.append([value_min, value_max])
	return minmax
```

---
## Step 5 — Rescale dataset columns to the range 0-1

```python
def normalize_dataset(dataset, minmax):
	for row in dataset:
  # 获取长度 / Get length
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
```

---
## Step 6 — Split a dataset into k folds

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
## Step 7 — Calculate accuracy percentage

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
## Step 8 — Evaluate an algorithm using a cross validation split

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
## Step 9 — Make a prediction with coefficients

```python
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
```

---
## Step 10 — Estimate logistic regression coefficients using stochastic gradient descent

```python
def coefficients_sgd(train, l_rate, n_epoch):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
```

---
## Step 11 — Logistic Regression Algorithm With Stochastic Gradient Descent

```python
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return(predictions)
```

---
## Step 12 — Test the logistic regression algorithm on the diabetes dataset

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 13 — load and prepare data

```python
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
```

---
## Step 14 — normalize

```python
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
```

---
## Step 15 — evaluate algorithm

```python
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---
## Learning Notes / 学习笔记

- **概念**: Logistic Regression on Diabetes Dataset 是机器学习中的常用技术。  
  *Logistic Regression on Diabetes Dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SGD` | 随机梯度下降 | Stochastic Gradient Descent |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic Regression Diabetes / 回归
# Complete Code / 完整代码
# ===============================

# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
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

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
 # 获取长度 / Get length
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
  # 添加元素到列表末尾 / Append element to list end
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
  # 获取长度 / Get length
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

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

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
 # 获取长度 / Get length
	coef = [0.0 for i in range(len(train[0]))]
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
   # 获取长度 / Get length
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Logistic Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return(predictions)

# Test the logistic regression algorithm on the diabetes dataset
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Predict

# 01 — Predict / Predict

**Chapter 09 — File 3 of 3 / 第09章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Example of making a prediction**.

本脚本演示 **Example of making a prediction**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of making a prediction

```python
from math import exp
```

---
## Step 2 — Make a prediction with coefficients

```python
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
```

---
## Step 3 — test predictions

```python
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
coef = [-0.406605464, 0.852573316, -1.104746259]
for row in dataset:
	yhat = predict(row, coef)
 # 打印输出 / Print output
	print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of making a prediction 是机器学习中的常用技术。  
  *Example of making a prediction is a common technique in machine learning.*

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
# Predict / Predict
# Complete Code / 完整代码
# ===============================

# Example of making a prediction
from math import exp

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
 # 获取长度 / Get length
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# test predictions
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
coef = [-0.406605464, 0.852573316, -1.104746259]
for row in dataset:
	yhat = predict(row, coef)
 # 打印输出 / Print output
	print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))
```

---
