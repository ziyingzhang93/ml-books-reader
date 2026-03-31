# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 02

---

### Balance

# 01 — Balance / 01 Balance

**Chapter 02 — File 1 of 5 / 第02章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **generate binary classification dataset and plot**.

本脚本演示 **generate binary classification dataset and plot**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — generate binary classification dataset and plot

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — generate dataset

```python
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
```

---
## Step 3 — create scatter plot for samples from each class

```python
# 生成整数序列 / Generate integer sequence
for class_value in range(2):
```

---
## Step 4 — get row indexes for samples with this class

```python
row_ix = where(y == class_value)
```

---
## Step 5 — create scatter of these samples

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
```

---
## Step 6 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: generate binary classification dataset and plot 是机器学习中的常用技术。  
  *generate binary classification dataset and plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Balance / 01 Balance
# Complete Code / 完整代码
# ===============================

# generate binary classification dataset and plot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# generate dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
# create scatter plot for samples from each class
# 生成整数序列 / Generate integer sequence
for class_value in range(2):
	# get row indexes for samples with this class
	row_ix = where(y == class_value)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Controlled Balance

# 02 — Controlled Balance / 02 Controlled Balance

**Chapter 02 — File 2 of 5 / 第02章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **create and plot synthetic dataset with a given class distribution**.

本脚本演示 **create and plot synthetic dataset with a given class distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — create and plot synthetic dataset with a given class distribution

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — create a dataset with a given class distribution

```python
def get_dataset(proportions):
```

---
## Step 3 — determine the number of classes

```python
# 获取长度 / Get length
n_classes = len(proportions)
```

---
## Step 4 — determine the number of examples to generate for each class

```python
# 获取字典的键值对 / Get dict key-value pairs
largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
```

---
## Step 5 — create dataset

```python
X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
```

---
## Step 6 — collect the examples

```python
X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)
```

---
## Step 7 — scatter plot of dataset, different color for each class

```python
def plot_dataset(X, y):
```

---
## Step 8 — create scatter plot for samples from each class

```python
# 获取长度 / Get length
n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
```

---
## Step 9 — get row indexes for samples with this class

```python
row_ix = where(y == class_value)[0]
```

---
## Step 10 — create scatter of these samples

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
```

---
## Step 11 — show a legend

```python
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Step 13 — define the class distribution

```python
proportions = {0:5000, 1:5000}
```

---
## Step 14 — generate dataset

```python
X, y = get_dataset(proportions)
```

---
## Step 15 — plot dataset

```python
plot_dataset(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: create and plot synthetic dataset with a given class distribution 是机器学习中的常用技术。  
  *create and plot synthetic dataset with a given class distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Controlled Balance / 02 Controlled Balance
# Complete Code / 完整代码
# ===============================

# create and plot synthetic dataset with a given class distribution
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
 # 获取长度 / Get length
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
 # 获取字典的键值对 / Get dict key-value pairs
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
 # 获取长度 / Get length
	n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# define the class distribution
proportions = {0:5000, 1:5000}
# generate dataset
X, y = get_dataset(proportions)
# plot dataset
plot_dataset(X, y)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### 1 To 10

# 03 — 1 To 10 / 03 1 To 10

**Chapter 02 — File 3 of 5 / 第02章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **create and plot synthetic dataset with a given class distribution**.

本脚本演示 **create and plot synthetic dataset with a given class distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — create and plot synthetic dataset with a given class distribution

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — create a dataset with a given class distribution

```python
def get_dataset(proportions):
```

---
## Step 3 — determine the number of classes

```python
# 获取长度 / Get length
n_classes = len(proportions)
```

---
## Step 4 — determine the number of examples to generate for each class

```python
# 获取字典的键值对 / Get dict key-value pairs
largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
```

---
## Step 5 — create dataset

```python
X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
```

---
## Step 6 — collect the examples

```python
X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)
```

---
## Step 7 — scatter plot of dataset, different color for each class

```python
def plot_dataset(X, y):
```

---
## Step 8 — create scatter plot for samples from each class

```python
# 获取长度 / Get length
n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
```

---
## Step 9 — get row indexes for samples with this class

```python
row_ix = where(y == class_value)[0]
```

---
## Step 10 — create scatter of these samples

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
```

---
## Step 11 — show a legend

```python
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Step 13 — define the class distribution

```python
proportions = {0:10000, 1:1000}
```

---
## Step 14 — generate dataset

```python
X, y = get_dataset(proportions)
```

---
## Step 15 — plot dataset

```python
plot_dataset(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: create and plot synthetic dataset with a given class distribution 是机器学习中的常用技术。  
  *create and plot synthetic dataset with a given class distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 1 To 10 / 03 1 To 10
# Complete Code / 完整代码
# ===============================

# create and plot synthetic dataset with a given class distribution
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
 # 获取长度 / Get length
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
 # 获取字典的键值对 / Get dict key-value pairs
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
 # 获取长度 / Get length
	n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# define the class distribution
proportions = {0:10000, 1:1000}
# generate dataset
X, y = get_dataset(proportions)
# plot dataset
plot_dataset(X, y)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### 1 To 100

# 04 — 1 To 100 / 04 1 To 100

**Chapter 02 — File 4 of 5 / 第02章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **create and plot synthetic dataset with a given class distribution**.

本脚本演示 **create and plot synthetic dataset with a given class distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — create and plot synthetic dataset with a given class distribution

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — create a dataset with a given class distribution

```python
def get_dataset(proportions):
```

---
## Step 3 — determine the number of classes

```python
# 获取长度 / Get length
n_classes = len(proportions)
```

---
## Step 4 — determine the number of examples to generate for each class

```python
# 获取字典的键值对 / Get dict key-value pairs
largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
```

---
## Step 5 — create dataset

```python
X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
```

---
## Step 6 — collect the examples

```python
X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)
```

---
## Step 7 — scatter plot of dataset, different color for each class

```python
def plot_dataset(X, y):
```

---
## Step 8 — create scatter plot for samples from each class

```python
# 获取长度 / Get length
n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
```

---
## Step 9 — get row indexes for samples with this class

```python
row_ix = where(y == class_value)[0]
```

---
## Step 10 — create scatter of these samples

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
```

---
## Step 11 — show a legend

```python
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Step 13 — define the class distribution

```python
proportions = {0:10000, 1:100}
```

---
## Step 14 — generate dataset

```python
X, y = get_dataset(proportions)
```

---
## Step 15 — plot dataset

```python
plot_dataset(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: create and plot synthetic dataset with a given class distribution 是机器学习中的常用技术。  
  *create and plot synthetic dataset with a given class distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 1 To 100 / 04 1 To 100
# Complete Code / 完整代码
# ===============================

# create and plot synthetic dataset with a given class distribution
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
 # 获取长度 / Get length
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
 # 获取字典的键值对 / Get dict key-value pairs
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
 # 获取长度 / Get length
	n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# define the class distribution
proportions = {0:10000, 1:100}
# generate dataset
X, y = get_dataset(proportions)
# plot dataset
plot_dataset(X, y)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### 1 To 1000

# 05 — 1 To 1000 / 05 1 To 1000

**Chapter 02 — File 5 of 5 / 第02章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **create and plot synthetic dataset with a given class distribution**.

本脚本演示 **create and plot synthetic dataset with a given class distribution**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — create and plot synthetic dataset with a given class distribution

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
```

---
## Step 2 — create a dataset with a given class distribution

```python
def get_dataset(proportions):
```

---
## Step 3 — determine the number of classes

```python
# 获取长度 / Get length
n_classes = len(proportions)
```

---
## Step 4 — determine the number of examples to generate for each class

```python
# 获取字典的键值对 / Get dict key-value pairs
largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
```

---
## Step 5 — create dataset

```python
X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
```

---
## Step 6 — collect the examples

```python
X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)
```

---
## Step 7 — scatter plot of dataset, different color for each class

```python
def plot_dataset(X, y):
```

---
## Step 8 — create scatter plot for samples from each class

```python
# 获取长度 / Get length
n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
```

---
## Step 9 — get row indexes for samples with this class

```python
row_ix = where(y == class_value)[0]
```

---
## Step 10 — create scatter of these samples

```python
pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
```

---
## Step 11 — show a legend

```python
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Step 13 — define the class distribution

```python
proportions = {0:10000, 1:10}
```

---
## Step 14 — generate dataset

```python
X, y = get_dataset(proportions)
```

---
## Step 15 — plot dataset

```python
plot_dataset(X, y)
```

---
## Learning Notes / 学习笔记

- **概念**: create and plot synthetic dataset with a given class distribution 是机器学习中的常用技术。  
  *create and plot synthetic dataset with a given class distribution is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# 1 To 1000 / 05 1 To 1000
# Complete Code / 完整代码
# ===============================

# create and plot synthetic dataset with a given class distribution
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import unique
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs

# create a dataset with a given class distribution
def get_dataset(proportions):
	# determine the number of classes
 # 获取长度 / Get length
	n_classes = len(proportions)
	# determine the number of examples to generate for each class
 # 获取字典的键值对 / Get dict key-value pairs
	largest = max([v for k,v in proportions.items()])
	n_samples = largest * n_classes
	# create dataset
	X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
	# collect the examples
	X_list, y_list = list(), list()
 # 获取字典的键值对 / Get dict key-value pairs
	for k,v in proportions.items():
		row_ix = where(y == k)[0]
		selected = row_ix[:v]
  # 添加元素到列表末尾 / Append element to list end
		X_list.append(X[selected, :])
  # 添加元素到列表末尾 / Append element to list end
		y_list.append(y[selected])
	return vstack(X_list), hstack(y_list)

# scatter plot of dataset, different color for each class
def plot_dataset(X, y):
	# create scatter plot for samples from each class
 # 获取长度 / Get length
	n_classes = len(unique(y))
 # 生成整数序列 / Generate integer sequence
	for class_value in range(n_classes):
		# get row indexes for samples with this class
		row_ix = where(y == class_value)[0]
		# create scatter of these samples
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(class_value))
	# show a legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# define the class distribution
proportions = {0:10000, 1:10}
# generate dataset
X, y = get_dataset(proportions)
# plot dataset
plot_dataset(X, y)
```

---

### Chapter Summary / 章节总结

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **5 code files** demonstrating chapter 02.

本章包含 **5 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `01_balance.ipynb` — Balance
  2. `02_controlled_balance.ipynb` — Controlled Balance
  3. `03_1_to_10.ipynb` — 1 To 10
  4. `04_1_to_100.ipynb` — 1 To 100
  5. `05_1_to_1000.ipynb` — 1 To 1000

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
