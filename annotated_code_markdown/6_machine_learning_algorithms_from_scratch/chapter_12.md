# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 12

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **6 code files** demonstrating chapter 12.

本章包含 **6 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `gaussian_pdf.ipynb` — Gaussian Pdf
  2. `naive_bayes_iris.ipynb` — Naive Bayes Iris
  3. `probabilities_by_class.ipynb` — Probabilities By Class
  4. `separate_by_class.ipynb` — Separate By Class
  5. `summarize_by_class.ipynb` — Summarize By Class
  6. `summarize_dataset.ipynb` — Summarize Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---

### Gaussian Pdf



---

### Naive Bayes Iris

# 01 — Naive Bayes Iris / Naive Bayes Iris

**Chapter 12 — File 2 of 6 / 第12章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Naive Bayes On The Iris Dataset**.

本脚本演示 **Naive Bayes On The Iris Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Naive Bayes On The Iris Dataset

```python
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
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
## Step 8 — Split the dataset by class values, returns a dictionary

```python
def separate_by_class(dataset):
	separated = dict()
 # 获取长度 / Get length
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
  # 添加元素到列表末尾 / Append element to list end
		separated[class_value].append(vector)
	return separated
```

---
## Step 9 — Calculate the mean of a list of numbers

```python
def mean(numbers):
 # 获取长度 / Get length
	return sum(numbers)/float(len(numbers))
```

---
## Step 10 — Calculate the standard deviation of a list of numbers

```python
def stdev(numbers):
	avg = mean(numbers)
 # 获取长度 / Get length
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
```

---
## Step 11 — Calculate the mean, stdev and count for each column in a dataset

```python
def summarize_dataset(dataset):
 # 获取长度 / Get length
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
```

---
## Step 12 — Split dataset by class then calculate statistics for each row

```python
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
```

---
## Step 13 — Calculate the Gaussian probability distribution function for x

```python
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
```

---
## Step 14 — Calculate the probabilities of predicting each class for a given row

```python
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
  # 获取长度 / Get length
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
```

---
## Step 15 — Predict the class for a given row

```python
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
```

---
## Step 16 — Naive Bayes Algorithm

```python
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(output)
	return(predictions)
```

---
## Step 17 — Test Naive Bayes on Iris Dataset

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
```

---
## Step 18 — convert class column to integers

```python
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
```

---
## Step 19 — evaluate algorithm

```python
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---
## Learning Notes / 学习笔记

- **概念**: Naive Bayes On The Iris Dataset 是机器学习中的常用技术。  
  *Naive Bayes On The Iris Dataset is a common technique in machine learning.*

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
# Naive Bayes Iris / Naive Bayes Iris
# Complete Code / 完整代码
# ===============================

# Naive Bayes On The Iris Dataset
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi

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

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
 # 获取长度 / Get length
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
  # 添加元素到列表末尾 / Append element to list end
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
 # 获取长度 / Get length
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
 # 获取长度 / Get length
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
 # 获取长度 / Get length
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
  # 获取长度 / Get length
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(output)
	return(predictions)

# Test Naive Bayes on Iris Dataset
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Probabilities By Class

# 01 — Probabilities By Class / Probabilities By Class

**Chapter 12 — File 3 of 6 / 第12章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of calculating class probabilities**.

本脚本演示 **Example of calculating class probabilities**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Example of calculating class probabilities

```python
from math import sqrt
from math import pi
from math import exp
```

---
## Step 2 — Split the dataset by class values, returns a dictionary

```python
def separate_by_class(dataset):
	separated = dict()
 # 获取长度 / Get length
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
  # 添加元素到列表末尾 / Append element to list end
		separated[class_value].append(vector)
	return separated
```

---
## Step 3 — Calculate the mean of a list of numbers

```python
def mean(numbers):
 # 获取长度 / Get length
	return sum(numbers)/float(len(numbers))
```

---
## Step 4 — Calculate the standard deviation of a list of numbers

```python
def stdev(numbers):
	avg = mean(numbers)
 # 获取长度 / Get length
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
```

---
## Step 5 — Calculate the mean, stdev and count for each column in a dataset

```python
def summarize_dataset(dataset):
 # 获取长度 / Get length
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
```

---
## Step 6 — Split dataset by class then calculate statistics for each row

```python
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
```

---
## Step 7 — Calculate the Gaussian probability distribution function for x

```python
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
```

---
## Step 8 — Calculate the probabilities of predicting each class for a given row

```python
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
  # 获取长度 / Get length
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
```

---
## Step 9 — Test calculating class probabilities

```python
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]
summaries = summarize_by_class(dataset)
probabilities = calculate_class_probabilities(summaries, dataset[0])
# 打印输出 / Print output
print(probabilities)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of calculating class probabilities 是机器学习中的常用技术。  
  *Example of calculating class probabilities is a common technique in machine learning.*

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
# Probabilities By Class / Probabilities By Class
# Complete Code / 完整代码
# ===============================

# Example of calculating class probabilities
from math import sqrt
from math import pi
from math import exp

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
 # 获取长度 / Get length
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
  # 添加元素到列表末尾 / Append element to list end
		separated[class_value].append(vector)
	return separated

# Calculate the mean of a list of numbers
def mean(numbers):
 # 获取长度 / Get length
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
 # 获取长度 / Get length
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
 # 获取长度 / Get length
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
 # 获取字典的键值对 / Get dict key-value pairs
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
  # 获取长度 / Get length
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# Test calculating class probabilities
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]
summaries = summarize_by_class(dataset)
probabilities = calculate_class_probabilities(summaries, dataset[0])
# 打印输出 / Print output
print(probabilities)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Separate By Class



---

### Summarize By Class



---

### Summarize Dataset



---
