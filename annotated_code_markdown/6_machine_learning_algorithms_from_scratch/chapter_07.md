# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 07

---

### Chapter Summary / 章节总结



---

### Calculate Coefficients

# 01 — Calculate Coefficients / Calculate Coefficients

**Chapter 07 — File 1 of 5 / 第07章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating Coefficients**.

本脚本演示 **Example of Calculating Coefficients**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Calculating Coefficients
Calculate the mean value of a list of numbers

```python
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))
```

---
## Step 2 — Calculate covariance between x and y

```python
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
```

---
## Step 3 — Calculate the variance of a list of numbers

```python
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
```

---
## Step 4 — Calculate coefficients

```python
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
```

---
## Step 5 — calculate coefficients

```python
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
b0, b1 = coefficients(dataset)
# 打印输出 / Print output
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating Coefficients 是机器学习中的常用技术。  
  *Example of Calculating Coefficients is a common technique in machine learning.*

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
# Calculate Coefficients / Calculate Coefficients
# Complete Code / 完整代码
# ===============================

# Example of Calculating Coefficients

# Calculate the mean value of a list of numbers
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# calculate coefficients
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
b0, b1 = coefficients(dataset)
# 打印输出 / Print output
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Covariance

# 01 — Covariance / Covariance

**Chapter 07 — File 2 of 5 / 第07章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating Covariance**.

本脚本演示 **Example of Calculating Covariance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Calculating Covariance
Calculate the mean value of a list of numbers

```python
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))
```

---
## Step 2 — Calculate covariance between x and y

```python
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
```

---
## Step 3 — calculate covariance

```python
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
covar = covariance(x, mean_x, y, mean_y)
# 打印输出 / Print output
print('Covariance: %.3f' % (covar))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating Covariance 是机器学习中的常用技术。  
  *Example of Calculating Covariance is a common technique in machine learning.*

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
# Covariance / Covariance
# Complete Code / 完整代码
# ===============================

# Example of Calculating Covariance

# Calculate the mean value of a list of numbers
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# calculate covariance
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
covar = covariance(x, mean_x, y, mean_y)
# 打印输出 / Print output
print('Covariance: %.3f' % (covar))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Mean And Variance

# 01 — Mean And Variance / Mean And Variance

**Chapter 07 — File 3 of 5 / 第07章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Estimating Mean and Variance**.

本脚本演示 **Example of Estimating Mean and Variance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Estimating Mean and Variance
Calculate the mean value of a list of numbers

```python
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))
```

---
## Step 2 — Calculate the variance of a list of numbers

```python
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
```

---
## Step 3 — calculate mean and variance

```python
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
# 打印输出 / Print output
print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
# 打印输出 / Print output
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Estimating Mean and Variance 是机器学习中的常用技术。  
  *Example of Estimating Mean and Variance is a common technique in machine learning.*

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
# Mean And Variance / Mean And Variance
# Complete Code / 完整代码
# ===============================

# Example of Estimating Mean and Variance

# Calculate the mean value of a list of numbers
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# calculate mean and variance
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
# 打印输出 / Print output
print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
# 打印输出 / Print output
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Simple Linear Regression Contrived

# 01 — Simple Linear Regression Contrived / 线性模型

**Chapter 07 — File 4 of 5 / 第07章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Standalone Simple Linear Regression**.

本脚本演示 **Example of Standalone Simple Linear Regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Standalone Simple Linear Regression

```python
from math import sqrt
```

---
## Step 2 — Calculate root mean squared error

```python
def rmse_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
 # 获取长度 / Get length
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
```

---
## Step 3 — Evaluate regression algorithm on training dataset

```python
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
  # 添加元素到列表末尾 / Append element to list end
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
 # 打印输出 / Print output
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse
```

---
## Step 4 — Calculate the mean value of a list of numbers

```python
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))
```

---
## Step 5 — Calculate covariance between x and y

```python
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
```

---
## Step 6 — Calculate the variance of a list of numbers

```python
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
```

---
## Step 7 — Calculate coefficients

```python
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
```

---
## Step 8 — Simple linear regression algorithm

```python
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return predictions
```

---
## Step 9 — Test simple linear regression

```python
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, simple_linear_regression)
# 打印输出 / Print output
print('RMSE: %.3f' % (rmse))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Standalone Simple Linear Regression 是机器学习中的常用技术。  
  *Example of Standalone Simple Linear Regression is a common technique in machine learning.*

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
# Simple Linear Regression Contrived / 线性模型
# Complete Code / 完整代码
# ===============================

# Example of Standalone Simple Linear Regression
from math import sqrt

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
 # 获取长度 / Get length
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
  # 添加元素到列表末尾 / Append element to list end
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
 # 打印输出 / Print output
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return predictions

# Test simple linear regression
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, simple_linear_regression)
# 打印输出 / Print output
print('RMSE: %.3f' % (rmse))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Simple Linear Regression Insurance

# 01 — Simple Linear Regression Insurance / 线性模型

**Chapter 07 — File 5 of 5 / 第07章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Simple Linear Regression on the Swedish Insurance Dataset**.

本脚本演示 **Example of Simple Linear Regression on the Swedish Insurance Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


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
```

---
## Step 1 — Example of Simple Linear Regression on the Swedish Insurance Dataset

```python
from random import seed
from random import randrange
from csv import reader
from math import sqrt
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
## Step 5 — Calculate root mean squared error

```python
def rmse_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
 # 获取长度 / Get length
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
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
	rmse = rmse_metric(actual, predicted)
	return rmse
```

---
## Step 7 — Calculate the mean value of a list of numbers

```python
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))
```

---
## Step 8 — Calculate covariance between x and y

```python
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
```

---
## Step 9 — Calculate the variance of a list of numbers

```python
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
```

---
## Step 10 — Calculate coefficients

```python
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
```

---
## Step 11 — Simple linear regression algorithm

```python
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return predictions
```

---
## Step 12 — Simple linear regression on insurance dataset

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 13 — load and prepare data

```python
filename = 'insurance.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
```

---
## Step 14 — evaluate algorithm

```python
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
# 打印输出 / Print output
print('RMSE: %.3f' % (rmse))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Simple Linear Regression on the Swedish Insurance Dataset 是机器学习中的常用技术。  
  *Example of Simple Linear Regression on the Swedish Insurance Dataset is a common technique in machine learning.*

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
# Simple Linear Regression Insurance / 线性模型
# Complete Code / 完整代码
# ===============================

# Example of Simple Linear Regression on the Swedish Insurance Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

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

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
 # 获取长度 / Get length
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

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
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
 # 获取长度 / Get length
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
 # 获取长度 / Get length
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
	return predictions

# Simple linear regression on insurance dataset
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'insurance.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
# 打印输出 / Print output
print('RMSE: %.3f' % (rmse))
```

---
