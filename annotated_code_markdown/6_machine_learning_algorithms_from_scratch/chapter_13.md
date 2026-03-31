# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 13

---

### Chapter Summary / 章节总结



---

### Euclidean Distance

# 01 — Euclidean Distance / Euclidean Distance

**Chapter 13 — File 1 of 5 / 第13章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of calculating Euclidean distance**.

本脚本演示 **Example of calculating Euclidean distance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of calculating Euclidean distance

```python
from math import sqrt
```

---
## Step 2 — calculate the Euclidean distance between two vectors

```python
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
```

---
## Step 3 — Test distance function

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
row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
 # 打印输出 / Print output
	print(distance)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of calculating Euclidean distance 是机器学习中的常用技术。  
  *Example of calculating Euclidean distance is a common technique in machine learning.*

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
# Euclidean Distance / Euclidean Distance
# Complete Code / 完整代码
# ===============================

# Example of calculating Euclidean distance
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Test distance function
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
row0 = dataset[0]
for row in dataset:
	distance = euclidean_distance(row0, row)
 # 打印输出 / Print output
	print(distance)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Get Neighbors

# 01 — Get Neighbors / Get Neighbors

**Chapter 13 — File 2 of 5 / 第13章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of getting neighbours for an instance**.

本脚本演示 **Example of getting neighbours for an instance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of getting neighbours for an instance

```python
from math import sqrt
```

---
## Step 2 — calculate the Euclidean distance between two vectors

```python
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
```

---
## Step 3 — Locate the most similar neighbors

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
## Step 4 — Test distance function

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
neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
 # 打印输出 / Print output
	print(neighbor)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of getting neighbours for an instance 是机器学习中的常用技术。  
  *Example of getting neighbours for an instance is a common technique in machine learning.*

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
# Get Neighbors / Get Neighbors
# Complete Code / 完整代码
# ===============================

# Example of getting neighbours for an instance
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
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

# Test distance function
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
neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
 # 打印输出 / Print output
	print(neighbor)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Knn Classification Abalone



---

### Knn Regression Abalone



---

### Make Predictions

# 01 — Make Predictions / Make Predictions

**Chapter 13 — File 5 of 5 / 第13章 — 第5个文件（共5个）**

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
from math import sqrt
```

---
## Step 2 — calculate the Euclidean distance between two vectors

```python
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
```

---
## Step 3 — Locate the most similar neighbors

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
## Step 4 — Make a classification prediction with neighbors

```python
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
```

---
## Step 5 — Test distance function

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
prediction = predict_classification(dataset, dataset[0], 3)
# 打印输出 / Print output
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
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
# Make Predictions / Make Predictions
# Complete Code / 完整代码
# ===============================

# Example of making predictions
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
 # 获取长度 / Get length
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
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

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# Test distance function
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
prediction = predict_classification(dataset, dataset[0], 3)
# 打印输出 / Print output
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
```

---
