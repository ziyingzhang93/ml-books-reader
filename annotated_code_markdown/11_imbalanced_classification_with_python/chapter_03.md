# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 03

---

### Dataset Size

# 01 — Dataset Size / 01 Dataset Size

**Chapter 03 — File 1 of 3 / 第03章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **vary the dataset size for a 1:100 imbalanced dataset**.

本脚本演示 **vary the dataset size for a 1:100 imbalanced dataset**。

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
## Step 1 — vary the dataset size for a 1:100 imbalanced dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — dataset sizes

```python
sizes = [100, 1000, 10000, 100000]
```

---
## Step 3 — create and plot a dataset with each size

```python
# 获取长度 / Get length
for i in range(len(sizes)):
```

---
## Step 4 — determine the dataset size

```python
n = sizes[i]
```

---
## Step 5 — create the dataset

```python
X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
```

---
## Step 6 — summarize class distribution

```python
counter = Counter(y)
 # 打印输出 / Print output
	print('Size=%d, Ratio=%s' % (n, counter))
```

---
## Step 7 — define subplot

```python
pyplot.subplot(2, 2, 1+i)
	pyplot.title('n=%d' % n)
	pyplot.xticks([])
	pyplot.yticks([])
```

---
## Step 8 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
```

---
## Step 9 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: vary the dataset size for a 1:100 imbalanced dataset 是机器学习中的常用技术。  
  *vary the dataset size for a 1:100 imbalanced dataset is a common technique in machine learning.*

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
# Dataset Size / 01 Dataset Size
# Complete Code / 完整代码
# ===============================

# vary the dataset size for a 1:100 imbalanced dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# dataset sizes
sizes = [100, 1000, 10000, 100000]
# create and plot a dataset with each size
# 获取长度 / Get length
for i in range(len(sizes)):
	# determine the dataset size
	n = sizes[i]
	# create the dataset
	X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
	# summarize class distribution
	counter = Counter(y)
 # 打印输出 / Print output
	print('Size=%d, Ratio=%s' % (n, counter))
	# define subplot
	pyplot.subplot(2, 2, 1+i)
	pyplot.title('n=%d' % n)
	pyplot.xticks([])
	pyplot.yticks([])
	# scatter plot of examples by class label
 # 获取字典的键值对 / Get dict key-value pairs
	for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Label Noise

# 02 — Label Noise / 02 Label Noise

**Chapter 03 — File 2 of 3 / 第03章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **vary the label noise for a 1:100 imbalanced dataset**.

本脚本演示 **vary the label noise for a 1:100 imbalanced dataset**。

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
## Step 1 — vary the label noise for a 1:100 imbalanced dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — label noise ratios

```python
noise = [0, 0.01, 0.05, 0.07]
```

---
## Step 3 — create and plot a dataset with different label noise

```python
# 获取长度 / Get length
for i in range(len(noise)):
```

---
## Step 4 — determine the label noise

```python
n = noise[i]
```

---
## Step 5 — create the dataset

```python
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=n, random_state=1)
```

---
## Step 6 — summarize class distribution

```python
counter = Counter(y)
 # 打印输出 / Print output
	print('Noise=%d%%, Ratio=%s' % (int(n*100), counter))
```

---
## Step 7 — define subplot

```python
pyplot.subplot(2, 2, 1+i)
	pyplot.title('noise=%d%%' % int(n*100))
	pyplot.xticks([])
	pyplot.yticks([])
```

---
## Step 8 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
```

---
## Step 9 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: vary the label noise for a 1:100 imbalanced dataset 是机器学习中的常用技术。  
  *vary the label noise for a 1:100 imbalanced dataset is a common technique in machine learning.*

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
# Label Noise / 02 Label Noise
# Complete Code / 完整代码
# ===============================

# vary the label noise for a 1:100 imbalanced dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# label noise ratios
noise = [0, 0.01, 0.05, 0.07]
# create and plot a dataset with different label noise
# 获取长度 / Get length
for i in range(len(noise)):
	# determine the label noise
	n = noise[i]
	# create the dataset
	X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=n, random_state=1)
	# summarize class distribution
	counter = Counter(y)
 # 打印输出 / Print output
	print('Noise=%d%%, Ratio=%s' % (int(n*100), counter))
	# define subplot
	pyplot.subplot(2, 2, 1+i)
	pyplot.title('noise=%d%%' % int(n*100))
	pyplot.xticks([])
	pyplot.yticks([])
	# scatter plot of examples by class label
 # 获取字典的键值对 / Get dict key-value pairs
	for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
# show the figure
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Concepts

# 03 — Concepts / 03 Concepts

**Chapter 03 — File 3 of 3 / 第03章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **vary the number of clusters for a 1:100 imbalanced dataset**.

本脚本演示 **vary the number of clusters for a 1:100 imbalanced dataset**。

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
## Step 1 — vary the number of clusters for a 1:100 imbalanced dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — number of clusters

```python
clusters = [1, 2]
```

---
## Step 3 — create and plot a dataset with different numbers of clusters

```python
# 获取长度 / Get length
for i in range(len(clusters)):
	c = clusters[i]
```

---
## Step 4 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=c, weights=[0.99], flip_y=0, random_state=1)
	counter = Counter(y)
```

---
## Step 5 — define subplot

```python
pyplot.subplot(1, 2, 1+i)
	pyplot.title('Clusters=%d' % c)
	pyplot.xticks([])
	pyplot.yticks([])
```

---
## Step 6 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
```

---
## Step 7 — show the figure

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: vary the number of clusters for a 1:100 imbalanced dataset 是机器学习中的常用技术。  
  *vary the number of clusters for a 1:100 imbalanced dataset is a common technique in machine learning.*

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
# Concepts / 03 Concepts
# Complete Code / 完整代码
# ===============================

# vary the number of clusters for a 1:100 imbalanced dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# number of clusters
clusters = [1, 2]
# create and plot a dataset with different numbers of clusters
# 获取长度 / Get length
for i in range(len(clusters)):
	c = clusters[i]
	# define dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=c, weights=[0.99], flip_y=0, random_state=1)
	counter = Counter(y)
	# define subplot
	pyplot.subplot(1, 2, 1+i)
	pyplot.title('Clusters=%d' % c)
	pyplot.xticks([])
	pyplot.yticks([])
	# scatter plot of examples by class label
 # 获取字典的键值对 / Get dict key-value pairs
	for label, _ in counter.items():
		row_ix = where(y == label)[0]
		pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	pyplot.legend()
# show the figure
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 03 Summary / 第03章总结

## Theme / 主题: Chapter 03 / Chapter 03

This chapter contains **3 code files** demonstrating chapter 03.

本章包含 **3 个代码文件**，演示Chapter 03。

---
## Evolution / 演化路线

  1. `01_dataset_size.ipynb` — Dataset Size
  2. `02_label_noise.ipynb` — Label Noise
  3. `03_concepts.ipynb` — Concepts

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 03) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 03）是机器学习流水线中的基础构建块。

---
