# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 15

---

### Chapter Summary / 章节总结



---

### Backpropagate Error

# 01 — Backpropagate Error / Backpropagate Error

**Chapter 15 — File 1 of 6 / 第15章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of backpropagating error**.

本脚本演示 **Example of backpropagating error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — Example of backpropagating error
Calculate the derivative of an neuron output

```python
def transfer_derivative(output):
	return output * (1.0 - output)
```

---
## Step 2 — Backpropagate error and store in neurons

```python
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```

---
## Step 3 — test backpropagation of error

```python
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of backpropagating error 是机器学习中的常用技术。  
  *Example of backpropagating error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Backpropagate Error / Backpropagate Error
# Complete Code / 完整代码
# ===============================

# Example of backpropagating error

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# test backpropagation of error
network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Backpropagation Seeds

# 01 — Backpropagation Seeds / Backpropagation Seeds

**Chapter 15 — File 2 of 6 / 第15章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Backprop on the Seeds Dataset**.

本脚本演示 **Backprop on the Seeds Dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — Backprop on the Seeds Dataset

```python
from random import seed
from random import randrange
from random import random
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
## Step 5 — Find the min and max values for each column

```python
def dataset_minmax(dataset):
 # 将多个序列配对 / Pair multiple sequences
	return [[min(column), max(column)] for column in zip(*dataset)]
```

---
## Step 6 — Rescale dataset columns to the range 0-1

```python
def normalize_dataset(dataset, minmax):
	for row in dataset:
  # 获取长度 / Get length
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
```

---
## Step 7 — Split a dataset into k folds

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
## Step 8 — Calculate accuracy percentage

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
## Step 9 — Evaluate an algorithm using a cross validation split

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
## Step 10 — Calculate neuron activation for an input

```python
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```

---
## Step 11 — Transfer neuron activation

```python
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```

---
## Step 12 — Forward propagate input to a network output

```python
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```

---
## Step 13 — Calculate the derivative of an neuron output

```python
def transfer_derivative(output):
	return output * (1.0 - output)
```

---
## Step 14 — Backpropagate error and store in neurons

```python
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```

---
## Step 15 — Update network weights with error

```python
def update_weights(network, row, l_rate):
 # 获取长度 / Get length
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
   # 获取长度 / Get length
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
```

---
## Step 16 — Train a network for a fixed number of epochs

```python
def train_network(network, train, l_rate, n_epoch, n_outputs):
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			forward_propagate(network, row)
   # 生成整数序列 / Generate integer sequence
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
```

---
## Step 17 — Initialize a network

```python
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network
```

---
## Step 18 — Make a prediction with a network

```python
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
```

---
## Step 19 — Backpropagation Algorithm With Stochastic Gradient Descent

```python
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
 # 获取长度 / Get length
	n_inputs = len(train[0]) - 1
 # 获取长度 / Get length
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(prediction)
	return(predictions)
```

---
## Step 20 — Test Backprop on Seeds dataset

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 21 — load and prepare data

```python
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
```

---
## Step 22 — convert class column to integers

```python
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
```

---
## Step 23 — normalize input variables

```python
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
```

---
## Step 24 — evaluate algorithm

```python
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---
## Learning Notes / 学习笔记

- **概念**: Backprop on the Seeds Dataset 是机器学习中的常用技术。  
  *Backprop on the Seeds Dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Backpropagation Seeds / Backpropagation Seeds
# Complete Code / 完整代码
# ===============================

# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
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

# Find the min and max values for each column
def dataset_minmax(dataset):
 # 将多个序列配对 / Pair multiple sequences
	return [[min(column), max(column)] for column in zip(*dataset)]

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
  # 获取长度 / Get length
		for i in range(len(row)-1):
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

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
 # 获取长度 / Get length
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
   # 获取长度 / Get length
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
 # 生成整数序列 / Generate integer sequence
	for _ in range(n_epoch):
		for row in train:
			forward_propagate(network, row)
   # 生成整数序列 / Generate integer sequence
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
 # 获取长度 / Get length
	n_inputs = len(train[0]) - 1
 # 获取长度 / Get length
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(prediction)
	return(predictions)

# Test Backprop on Seeds dataset
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
# 获取长度 / Get length
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
# 获取长度 / Get length
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
# 打印输出 / Print output
print('Scores: %s' % scores)
# 打印输出 / Print output
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Forward Propagate

# 01 — Forward Propagate / Forward Propagate

**Chapter 15 — File 3 of 6 / 第15章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of forward propagating input**.

本脚本演示 **Example of forward propagating input**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of forward propagating input

```python
from math import exp
```

---
## Step 2 — Calculate neuron activation for an input

```python
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```

---
## Step 3 — Transfer neuron activation

```python
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```

---
## Step 4 — Forward propagate input to a network output

```python
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```

---
## Step 5 — test forward propagation

```python
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
# 打印输出 / Print output
print(output)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of forward propagating input 是机器学习中的常用技术。  
  *Example of forward propagating input is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Forward Propagate / Forward Propagate
# Complete Code / 完整代码
# ===============================

# Example of forward propagating input
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
row = [1, 0, None]
output = forward_propagate(network, row)
# 打印输出 / Print output
print(output)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Initialize Network

# 01 — Initialize Network / Initialize Network

**Chapter 15 — File 4 of 6 / 第15章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of initializing a network**.

本脚本演示 **Example of initializing a network**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of initializing a network

```python
from random import seed
from random import random
```

---
## Step 2 — Initialize a network

```python
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network
```

---
## Step 3 — Test initializing a network

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of initializing a network 是机器学习中的常用技术。  
  *Example of initializing a network is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Initialize Network / Initialize Network
# Complete Code / 完整代码
# ===============================

# Example of initializing a network
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network

# Test initializing a network
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Predict

# 01 — Predict / Predict

**Chapter 15 — File 5 of 6 / 第15章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of making predictions**.

本脚本演示 **Example of making predictions**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of making predictions

```python
from math import exp
```

---
## Step 2 — Calculate neuron activation for an input

```python
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```

---
## Step 3 — Transfer neuron activation

```python
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```

---
## Step 4 — Forward propagate input to a network output

```python
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```

---
## Step 5 — Make a prediction with a network

```python
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
```

---
## Step 6 — Test making predictions with the network

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
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
 # 打印输出 / Print output
	print('Expected=%d, Got=%d' % (row[-1], prediction))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of making predictions 是机器学习中的常用技术。  
  *Example of making predictions is a common technique in machine learning.*

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

# Example of making predictions
from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
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
network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
for row in dataset:
	prediction = predict(network, row)
 # 打印输出 / Print output
	print('Expected=%d, Got=%d' % (row[-1], prediction))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Train Network

# 01 — Train Network / Train Network

**Chapter 15 — File 6 of 6 / 第15章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Example of training a network by backpropagation**.

本脚本演示 **Example of training a network by backpropagation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model


---
## Step 1 — Example of training a network by backpropagation

```python
from math import exp
from random import seed
from random import random
```

---
## Step 2 — Initialize a network

```python
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network
```

---
## Step 3 — Calculate neuron activation for an input

```python
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
```

---
## Step 4 — Transfer neuron activation

```python
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
```

---
## Step 5 — Forward propagate input to a network output

```python
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
```

---
## Step 6 — Calculate the derivative of an neuron output

```python
def transfer_derivative(output):
	return output * (1.0 - output)
```

---
## Step 7 — Backpropagate error and store in neurons

```python
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
```

---
## Step 8 — Update network weights with error

```python
def update_weights(network, row, l_rate):
 # 获取长度 / Get length
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
   # 获取长度 / Get length
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
```

---
## Step 9 — Train a network for a fixed number of epochs

```python
def train_network(network, train, l_rate, n_epoch, n_outputs):
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
   # 生成整数序列 / Generate integer sequence
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
   # 获取长度 / Get length
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
  # 打印输出 / Print output
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```

---
## Step 10 — Test training backprop algorithm

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
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
# 获取长度 / Get length
n_inputs = len(dataset[0]) - 1
# 获取长度 / Get length
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of training a network by backpropagation 是机器学习中的常用技术。  
  *Example of training a network by backpropagation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `backward` | 反向传播，计算梯度 | Backpropagation: compute gradients |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Network / Train Network
# Complete Code / 完整代码
# ===============================

# Example of training a network by backpropagation
from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
 # 生成整数序列 / Generate integer sequence
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(hidden_layer)
 # 生成整数序列 / Generate integer sequence
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
 # 添加元素到列表末尾 / Append element to list end
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
 # 获取长度 / Get length
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
   # 添加元素到列表末尾 / Append element to list end
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
 # 获取长度 / Get length
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
  # 获取长度 / Get length
		if i != len(network)-1:
   # 获取长度 / Get length
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
    # 添加元素到列表末尾 / Append element to list end
				errors.append(error)
		else:
   # 获取长度 / Get length
			for j in range(len(layer)):
				neuron = layer[j]
    # 添加元素到列表末尾 / Append element to list end
				errors.append(expected[j] - neuron['output'])
  # 获取长度 / Get length
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
 # 获取长度 / Get length
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
   # 获取长度 / Get length
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
 # 生成整数序列 / Generate integer sequence
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
   # 生成整数序列 / Generate integer sequence
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
   # 获取长度 / Get length
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
  # 打印输出 / Print output
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
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
# 获取长度 / Get length
n_inputs = len(dataset[0]) - 1
# 获取长度 / Get length
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
 # 打印输出 / Print output
	print(layer)
```

---
