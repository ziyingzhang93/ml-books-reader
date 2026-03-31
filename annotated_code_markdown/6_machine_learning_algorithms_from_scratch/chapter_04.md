# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 04

---

### Chapter Summary / 章节总结



---

### Classification Accuracy

# 01 — Classification Accuracy / 分类

**Chapter 04 — File 1 of 5 / 第04章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of calculating classification accuracy**.

本脚本演示 **Example of calculating classification accuracy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — Example of calculating classification accuracy
Calculate accuracy percentage between two lists

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
## Step 2 — Test accuracy

```python
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
# 打印输出 / Print output
print(accuracy)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of calculating classification accuracy 是机器学习中的常用技术。  
  *Example of calculating classification accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Accuracy / 分类
# Complete Code / 完整代码
# ===============================

# Example of calculating classification accuracy

# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
 # 获取长度 / Get length
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
 # 获取长度 / Get length
	return correct / float(len(actual)) * 100.0

# Test accuracy
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
# 打印输出 / Print output
print(accuracy)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Confusion Matrix



---

### Confusion Matrix Pretty

# 01 — Confusion Matrix Pretty / Confusion Matrix Pretty

**Chapter 04 — File 3 of 5 / 第04章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating and Displaying a Pretty Confusion Matrix**.

本脚本演示 **Example of Calculating and Displaying a Pretty Confusion Matrix**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Calculating and Displaying a Pretty Confusion Matrix
calculate a confusion matrix

```python
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
def confusion_matrix(actual, predicted):
	unique = set(actual)
 # 获取长度 / Get length
	matrix = [list() for x in range(len(unique))]
 # 获取长度 / Get length
	for i in range(len(unique)):
  # 获取长度 / Get length
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
 # 同时获取索引和值 / Get both index and value
	for i, value in enumerate(unique):
		lookup[value] = i
 # 获取长度 / Get length
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix
```

---
## Step 2 — pretty print a confusion matrix

```python
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
def print_confusion_matrix(unique, matrix):
 # 打印输出 / Print output
	print('(A)' + ' '.join(str(x) for x in unique))
 # 打印输出 / Print output
	print('(P)---')
 # 同时获取索引和值 / Get both index and value
	for i, x in enumerate(unique):
  # 打印输出 / Print output
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))
```

---
## Step 3 — Test confusion matrix with integers

```python
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
unique, matrix = confusion_matrix(actual, predicted)
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print_confusion_matrix(unique, matrix)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating and Displaying a Pretty Confusion Matrix 是机器学习中的常用技术。  
  *Example of Calculating and Displaying a Pretty Confusion Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Confusion Matrix Pretty / Confusion Matrix Pretty
# Complete Code / 完整代码
# ===============================

# Example of Calculating and Displaying a Pretty Confusion Matrix

# calculate a confusion matrix
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
def confusion_matrix(actual, predicted):
	unique = set(actual)
 # 获取长度 / Get length
	matrix = [list() for x in range(len(unique))]
 # 获取长度 / Get length
	for i in range(len(unique)):
  # 获取长度 / Get length
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
 # 同时获取索引和值 / Get both index and value
	for i, value in enumerate(unique):
		lookup[value] = i
 # 获取长度 / Get length
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix

# pretty print a confusion matrix
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
def print_confusion_matrix(unique, matrix):
 # 打印输出 / Print output
	print('(A)' + ' '.join(str(x) for x in unique))
 # 打印输出 / Print output
	print('(P)---')
 # 同时获取索引和值 / Get both index and value
	for i, x in enumerate(unique):
  # 打印输出 / Print output
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))

# Test confusion matrix with integers
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
unique, matrix = confusion_matrix(actual, predicted)
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
print_confusion_matrix(unique, matrix)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Mean Absolute Error

# 01 — Mean Absolute Error / Mean Absolute Error

**Chapter 04 — File 4 of 5 / 第04章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating Mean Absolute Error**.

本脚本演示 **Example of Calculating Mean Absolute Error**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Example of Calculating Mean Absolute Error
Calculate mean absolute error

```python
def mae_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
 # 获取长度 / Get length
	return sum_error / float(len(actual))
```

---
## Step 2 — Test RMSE

```python
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
# 打印输出 / Print output
print(mae)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating Mean Absolute Error 是机器学习中的常用技术。  
  *Example of Calculating Mean Absolute Error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mean Absolute Error / Mean Absolute Error
# Complete Code / 完整代码
# ===============================

# Example of Calculating Mean Absolute Error

# Calculate mean absolute error
def mae_metric(actual, predicted):
	sum_error = 0.0
 # 获取长度 / Get length
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
 # 获取长度 / Get length
	return sum_error / float(len(actual))

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
# 打印输出 / Print output
print(mae)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Root Mean Squared Error



---
