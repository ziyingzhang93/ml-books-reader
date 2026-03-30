# 从零实现ML算法
## Chapter 11

---

### Chapter Summary

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **5 code files** demonstrating chapter 11.

本章包含 **5 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `cart_banknote.ipynb` — Cart Banknote
  2. `create_tree.ipynb` — Create Tree
  3. `gini_split.ipynb` — Gini Split
  4. `make_prediction.ipynb` — Make Prediction
  5. `split_dataset.ipynb` — Split Dataset

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---

### Create Tree

# 01 — Create Tree / 决策树

**Chapter 11 — File 2 of 5 / 第11章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of building a tree**.

本脚本演示 **Example of building a tree**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Example of building a tree
Split a dataset based on an attribute and an attribute value

```python
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
```

---
## Step 2 — Calculate the Gini index for a split dataset

```python
def gini_index(groups, classes):
```

---
## Step 3 — count all samples at split point

```python
n_instances = float(sum([len(group) for group in groups]))
```

---
## Step 4 — sum weighted Gini index for each group

```python
gini = 0.0
	for group in groups:
		size = float(len(group))
```

---
## Step 5 — avoid divide by zero

```python
if size == 0:
			continue
		score = 0.0
```

---
## Step 6 — score the group based on the score for each class

```python
for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
```

---
## Step 7 — weight the group score by its relative size

```python
gini += (1.0 - score) * (size / n_instances)
	return gini
```

---
## Step 8 — Select the best split point for a dataset

```python
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
```

---
## Step 9 — Create a terminal node value

```python
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
```

---
## Step 10 — Create child splits for a node or make terminal

```python
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
```

---
## Step 11 — check for a no split

```python
if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
```

---
## Step 12 — check for max depth

```python
if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
```

---
## Step 13 — process left child

```python
if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
```

---
## Step 14 — process right child

```python
if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
```

---
## Step 15 — Build a decision tree

```python
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
```

---
## Step 16 — Print a decision tree

```python
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of building a tree 是机器学习中的常用技术。  
  *Example of building a tree is a common technique in machine learning.*

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
# Create Tree / 决策树
# Complete Code / 完整代码
# ===============================

# Example of building a tree

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Gini Split

# 01 — Gini Split / Gini Split

**Chapter 11 — File 3 of 5 / 第11章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of calculating Gini index**.

本脚本演示 **Example of calculating Gini index**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Example of calculating Gini index
Calculate the Gini index for a split dataset

```python
def gini_index(groups, classes):
```

---
## Step 2 — count all samples at split point

```python
n_instances = float(sum([len(group) for group in groups]))
```

---
## Step 3 — sum weighted Gini index for each group

```python
gini = 0.0
	for group in groups:
		size = float(len(group))
```

---
## Step 4 — avoid divide by zero

```python
if size == 0:
			continue
		score = 0.0
```

---
## Step 5 — score the group based on the score for each class

```python
for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
```

---
## Step 6 — weight the group score by its relative size

```python
gini += (1.0 - score) * (size / n_instances)
	return gini
```

---
## Step 7 — test Gini values

```python
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of calculating Gini index 是机器学习中的常用技术。  
  *Example of calculating Gini index is a common technique in machine learning.*

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
# Gini Split / Gini Split
# Complete Code / 完整代码
# ===============================

# Example of calculating Gini index

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# test Gini values
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Split Dataset

# 01 — Split Dataset / Split Dataset

**Chapter 11 — File 5 of 5 / 第11章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of getting the best split**.

本脚本演示 **Example of getting the best split**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — Example of getting the best split
Split a dataset based on an attribute and an attribute value

```python
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
```

---
## Step 2 — Calculate the Gini index for a split dataset

```python
def gini_index(groups, classes):
```

---
## Step 3 — count all samples at split point

```python
n_instances = float(sum([len(group) for group in groups]))
```

---
## Step 4 — sum weighted Gini index for each group

```python
gini = 0.0
	for group in groups:
		size = float(len(group))
```

---
## Step 5 — avoid divide by zero

```python
if size == 0:
			continue
		score = 0.0
```

---
## Step 6 — score the group based on the score for each class

```python
for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
```

---
## Step 7 — weight the group score by its relative size

```python
gini += (1.0 - score) * (size / n_instances)
	return gini
```

---
## Step 8 — Select the best split point for a dataset

```python
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
```

---
## Step 9 — Test getting the best split

```python
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
```

---
## Learning Notes / 学习笔记

- **概念**: Example of getting the best split 是机器学习中的常用技术。  
  *Example of getting the best split is a common technique in machine learning.*

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
# Split Dataset / Split Dataset
# Complete Code / 完整代码
# ===============================

# Example of getting the best split

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Test getting the best split
dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
```

---
