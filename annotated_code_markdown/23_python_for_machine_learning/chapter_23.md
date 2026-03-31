# Python 机器学习 / Python for Machine Learning
## Chapter 23

---

### Iris

# 01 — Iris / 01 Iris

**Chapter 23 — File 1 of 16 / 第23章 — 第1个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Iris**.

本脚本演示 **01 Iris**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
data["target"] = target
# 打印输出 / Print output
print(data)
```

---
## Learning Notes / 学习笔记

- **概念**: Iris 是机器学习中的常用技术。  
  *Iris is a common technique in machine learning.*

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
# Iris / 01 Iris
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
data["target"] = target
# 打印输出 / Print output
print(data)
```

---

➡️ **Next / 下一步**: File 2 of 16

---

### Seaborn

# 02 — Seaborn / 02 Seaborn

**Chapter 23 — File 2 of 16 / 第23章 — 第2个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Seaborn**.

本脚本演示 **02 Seaborn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
data["target"] = target

sns.pairplot(data, kind="scatter", diag_kind="kde", hue="target",
             palette="muted", plot_kws={'alpha':0.7})
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Seaborn 是机器学习中的常用技术。  
  *Seaborn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Seaborn / 02 Seaborn
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
data["target"] = target

sns.pairplot(data, kind="scatter", diag_kind="kde", hue="target",
             palette="muted", plot_kws={'alpha':0.7})
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 16

---

### Seaborn

# 03 — Seaborn / 03 Seaborn

**Chapter 23 — File 3 of 16 / 第23章 — 第3个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Seaborn**.

本脚本演示 **03 Seaborn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset("iris")
sns.pairplot(data, kind="scatter", diag_kind="kde", hue="species",
             palette="muted", plot_kws={'alpha':0.7})
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Seaborn 是机器学习中的常用技术。  
  *Seaborn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Seaborn / 03 Seaborn
# Complete Code / 完整代码
# ===============================

# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset("iris")
sns.pairplot(data, kind="scatter", diag_kind="kde", hue="species",
             palette="muted", plot_kws={'alpha':0.7})
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 16

---

### Datasets

# 04 — Datasets / 04 Datasets

**Chapter 23 — File 4 of 16 / 第23章 — 第4个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Datasets**.

本脚本演示 **04 Datasets**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import seaborn as sns
# 打印输出 / Print output
print(sns.get_dataset_names())
```

---
## Learning Notes / 学习笔记

- **概念**: Datasets 是机器学习中的常用技术。  
  *Datasets is a common technique in machine learning.*

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
# Datasets / 04 Datasets
# Complete Code / 完整代码
# ===============================

import seaborn as sns
# 打印输出 / Print output
print(sns.get_dataset_names())
```

---

➡️ **Next / 下一步**: File 5 of 16

---

### Housing



---

### Diabetes



---

### Openml

# 07 — Openml / 07 Openml

**Chapter 23 — File 7 of 16 / 第23章 — 第7个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Openml**.

本脚本演示 **07 Openml**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets

data = sklearn.datasets.fetch_openml(data_id=42437, return_X_y=False, as_frame=True)
data = data["frame"]
# 打印输出 / Print output
print(data)
```

---
## Learning Notes / 学习笔记

- **概念**: Openml 是机器学习中的常用技术。  
  *Openml is a common technique in machine learning.*

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
# Openml / 07 Openml
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
import sklearn.datasets

data = sklearn.datasets.fetch_openml(data_id=42437, return_X_y=False, as_frame=True)
data = data["frame"]
# 打印输出 / Print output
print(data)
```

---

➡️ **Next / 下一步**: File 8 of 16

---

### Logistic

# 08 — Logistic / 08 Logistic

**Chapter 23 — File 8 of 16 / 第23章 — 第8个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Logistic**.

本脚本演示 **08 Logistic**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=42437, return_X_y=True, as_frame=False)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
clf = LogisticRegression(random_state=0).fit(X, y)
# 打印输出 / Print output
print(clf.score(X,y)) # accuracy
# 打印输出 / Print output
print(clf.coef_)      # coefficient in logistic regression
```

---
## Learning Notes / 学习笔记

- **概念**: Logistic 是机器学习中的常用技术。  
  *Logistic is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic / 08 Logistic
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import fetch_openml

X, y = fetch_openml(data_id=42437, return_X_y=True, as_frame=False)
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
clf = LogisticRegression(random_state=0).fit(X, y)
# 打印输出 / Print output
print(clf.score(X,y)) # accuracy
# 打印输出 / Print output
print(clf.coef_)      # coefficient in logistic regression
```

---

➡️ **Next / 下一步**: File 9 of 16

---

### Tfds

# 09 — Tfds / 09 Tfds

**Chapter 23 — File 9 of 16 / 第23章 — 第9个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Tfds**.

本脚本演示 **09 Tfds**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow_datasets as tfds
# 打印输出 / Print output
print(tfds.list_builders())
```

---
## Learning Notes / 学习笔记

- **概念**: Tfds 是机器学习中的常用技术。  
  *Tfds is a common technique in machine learning.*

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
# Tfds / 09 Tfds
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow_datasets as tfds
# 打印输出 / Print output
print(tfds.list_builders())
```

---

➡️ **Next / 下一步**: File 10 of 16

---

### Mnist

# 10 — Mnist / 10 Mnist

**Chapter 23 — File 10 of 16 / 第23章 — 第10个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Mnist**.

本脚本演示 **10 Mnist**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow_datasets as tfds
ds = tfds.load("mnist", split="train", shuffle_files=True)
# 打印输出 / Print output
print(ds)
```

---
## Learning Notes / 学习笔记

- **概念**: Mnist 是机器学习中的常用技术。  
  *Mnist is a common technique in machine learning.*

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
# Mnist / 10 Mnist
# Complete Code / 完整代码
# ===============================

# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow_datasets as tfds
ds = tfds.load("mnist", split="train", shuffle_files=True)
# 打印输出 / Print output
print(ds)
```

---

➡️ **Next / 下一步**: File 11 of 16

---

### Lenet5



---

### Sklearn

# 12 — Sklearn / 12 Sklearn

**Chapter 23 — File 12 of 16 / 第23章 — 第12个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Sklearn**.

本脚本演示 **12 Sklearn**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_circles(n_samples=500, shuffle=True, factor=0.7, noise=0.1)
# 创建画布 / Create figure canvas
plt.figure(figsize=(6,6))
# 绘制散点图 / Draw scatter plot
plt.scatter(data[:,0], data[:,1], c=target, alpha=0.8, cmap="Set1")
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Sklearn 是机器学习中的常用技术。  
  *Sklearn is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.scatter` | 绘制散点图 | Draw scatter plot |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Sklearn / 12 Sklearn
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_circles
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_circles(n_samples=500, shuffle=True, factor=0.7, noise=0.1)
# 创建画布 / Create figure canvas
plt.figure(figsize=(6,6))
# 绘制散点图 / Draw scatter plot
plt.scatter(data[:,0], data[:,1], c=target, alpha=0.8, cmap="Set1")
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 13 of 16

---

### Make Blobs

# 13 — Make Blobs / 13 Make Blobs

**Chapter 23 — File 13 of 16 / 第23章 — 第13个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Make Blobs**.

本脚本演示 **13 Make Blobs**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_blobs(n_samples=500, n_features=3, centers=4,
                          shuffle=True, random_state=42, cluster_std=2.5)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="Set1")
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Make Blobs 是机器学习中的常用技术。  
  *Make Blobs is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Blobs / 13 Make Blobs
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_blobs
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_blobs(n_samples=500, n_features=3, centers=4,
                          shuffle=True, random_state=42, cluster_std=2.5)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="Set1")
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 14 of 16

---

### Make S Curve

# 14 — Make S Curve / 14 Make S Curve

**Chapter 23 — File 14 of 16 / 第23章 — 第14个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Make S Curve**.

本脚本演示 **14 Make S Curve**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_s_curve, make_swiss_roll
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_s_curve(n_samples=5000, random_state=42)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

data, target = make_swiss_roll(n_samples=5000, random_state=42)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Make S Curve 是机器学习中的常用技术。  
  *Make S Curve is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `plt.figure` | 创建画布 | Create figure |
| `plt.show` | 显示图表 | Display plot |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make S Curve / 14 Make S Curve
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_s_curve, make_swiss_roll
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

data, target = make_s_curve(n_samples=5000, random_state=42)

# 创建画布 / Create figure canvas
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

data, target = make_swiss_roll(n_samples=5000, random_state=42)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=target, alpha=0.7, cmap="viridis")

# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 15 of 16

---

### Regression



---

### Classification

# 16 — Classification / 分类

**Chapter 23 — File 16 of 16 / 第23章 — 第16个文件（共16个）**

---

## Summary / 总结

This script demonstrates **Generate 10-dimensional features and 3-class targets**.

本脚本演示 **Generate 10-dimensional features and 3-class targets**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
```

---
## Step 2 — Generate 10-dimensional features and 3-class targets

```python
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           n_informative=4, n_redundant=2, n_repeated=1,
                           random_state=42)
```

---
## Step 3 — Run SVC on the data

```python
# 支持向量机 / Support Vector Machine
clf = SVC(kernel="rbf")
clf.fit(X, y)
```

---
## Step 4 — Print the accuracy

```python
# 打印输出 / Print output
print(clf.score(X, y))
```

---
## Learning Notes / 学习笔记

- **概念**: Generate 10-dimensional features and 3-class targets 是机器学习中的常用技术。  
  *Generate 10-dimensional features and 3-class targets is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification / 分类
# Complete Code / 完整代码
# ===============================

# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

# Generate 10-dimensional features and 3-class targets
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           n_informative=4, n_redundant=2, n_repeated=1,
                           random_state=42)

# Run SVC on the data
# 支持向量机 / Support Vector Machine
clf = SVC(kernel="rbf")
clf.fit(X, y)

# Print the accuracy
# 打印输出 / Print output
print(clf.score(X, y))
```

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **16 code files** demonstrating chapter 23.

本章包含 **16 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_iris.ipynb` — Iris
  2. `02_seaborn.ipynb` — Seaborn
  3. `03_seaborn.ipynb` — Seaborn
  4. `04_datasets.ipynb` — Datasets
  5. `05_housing.ipynb` — Housing
  6. `06_diabetes.ipynb` — Diabetes
  7. `07_openml.ipynb` — Openml
  8. `08_logistic.ipynb` — Logistic
  9. `09_tfds.ipynb` — Tfds
  10. `10_mnist.ipynb` — Mnist
  11. `11_lenet5.ipynb` — Lenet5
  12. `12_sklearn.ipynb` — Sklearn
  13. `13_make_blobs.ipynb` — Make Blobs
  14. `14_make_s_curve.ipynb` — Make S Curve
  15. `15_regression.ipynb` — Regression
  16. `16_classification.ipynb` — Classification

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
