# 从零实现机器学习算法 / ML Algorithms from Scratch
## Chapter 04

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **5 code files** demonstrating chapter 04.

本章包含 **5 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `classification_accuracy.ipynb` — Classification Accuracy
  2. `confusion_matrix.ipynb` — Confusion Matrix
  3. `confusion_matrix_pretty.ipynb` — Confusion Matrix Pretty
  4. `mean_absolute_error.ipynb` — Mean Absolute Error
  5. `root_mean_squared_error.ipynb` — Root Mean Squared Error

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---

### Classification Accuracy

# 01 — Classification Accuracy / 分类

**Chapter 04 — File 1 of 5 / 第04章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of calculating classification accuracy**.

本脚本演示 **Example of calculating classification accuracy**。

---
## Step 1 — Example of calculating classification accuracy
Calculate accuracy percentage between two lists

```python
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
```

---
## Step 2 — Test accuracy

```python
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of calculating classification accuracy 是机器学习中的常用技术。  
  *Example of calculating classification accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Test accuracy
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Confusion Matrix

# 01 — Confusion Matrix / Confusion Matrix

**Chapter 04 — File 2 of 5 / 第04章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating a Confusion Matrix**.

本脚本演示 **Example of Calculating a Confusion Matrix**。

---
## Step 1 — Example of Calculating a Confusion Matrix
calculate a confusion matrix

```python
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix
```

---
## Step 2 — Test confusion matrix with integers

```python
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print(unique)
print(matrix)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating a Confusion Matrix 是机器学习中的常用技术。  
  *Example of Calculating a Confusion Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Confusion Matrix / Confusion Matrix
# Complete Code / 完整代码
# ===============================

# Example of Calculating a Confusion Matrix

# calculate a confusion matrix
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix

# Test confusion matrix with integers
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print(unique)
print(matrix)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Confusion Matrix Pretty

# 01 — Confusion Matrix Pretty / Confusion Matrix Pretty

**Chapter 04 — File 3 of 5 / 第04章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating and Displaying a Pretty Confusion Matrix**.

本脚本演示 **Example of Calculating and Displaying a Pretty Confusion Matrix**。

---
## Step 1 — Example of Calculating and Displaying a Pretty Confusion Matrix
calculate a confusion matrix

```python
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix
```

---
## Step 2 — pretty print a confusion matrix

```python
def print_confusion_matrix(unique, matrix):
	print('(A)' + ' '.join(str(x) for x in unique))
	print('(P)---')
	for i, x in enumerate(unique):
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))
```

---
## Step 3 — Test confusion matrix with integers

```python
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
print_confusion_matrix(unique, matrix)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating and Displaying a Pretty Confusion Matrix 是机器学习中的常用技术。  
  *Example of Calculating and Displaying a Pretty Confusion Matrix is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
def confusion_matrix(actual, predicted):
	unique = set(actual)
	matrix = [list() for x in range(len(unique))]
	for i in range(len(unique)):
		matrix[i] = [0 for x in range(len(unique))]
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for i in range(len(actual)):
		x = lookup[actual[i]]
		y = lookup[predicted[i]]
		matrix[y][x] += 1
	return unique, matrix

# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
	print('(A)' + ' '.join(str(x) for x in unique))
	print('(P)---')
	for i, x in enumerate(unique):
		print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))

# Test confusion matrix with integers
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,1,0,0,1,0,1,1,1]
unique, matrix = confusion_matrix(actual, predicted)
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
## Step 1 — Example of Calculating Mean Absolute Error
Calculate mean absolute error

```python
def mae_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
	return sum_error / float(len(actual))
```

---
## Step 2 — Test RMSE

```python
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating Mean Absolute Error 是机器学习中的常用技术。  
  *Example of Calculating Mean Absolute Error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
	for i in range(len(actual)):
		sum_error += abs(predicted[i] - actual[i])
	return sum_error / float(len(actual))

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Root Mean Squared Error

# 01 — Root Mean Squared Error / Root Mean Squared Error

**Chapter 04 — File 5 of 5 / 第04章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Example of Calculating the Root Mean Squared Error**.

本脚本演示 **Example of Calculating the Root Mean Squared Error**。

---
## Step 1 — Example of Calculating the Root Mean Squared Error

```python
from math import sqrt
```

---
## Step 2 — Calculate root mean squared error

```python
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)
```

---
## Step 3 — Test RMSE

```python
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
rmse = rmse_metric(actual, predicted)
print(rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: Example of Calculating the Root Mean Squared Error 是机器学习中的常用技术。  
  *Example of Calculating the Root Mean Squared Error is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Root Mean Squared Error / Root Mean Squared Error
# Complete Code / 完整代码
# ===============================

# Example of Calculating the Root Mean Squared Error
from math import sqrt

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
rmse = rmse_metric(actual, predicted)
print(rmse)
```

---
