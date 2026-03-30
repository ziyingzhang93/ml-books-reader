# ML数据准备
## Chapter 16

---

### Dataset Class

# 01 — Dataset Class / 01 Dataset Class

**Chapter 16 — File 1 of 12 / 第16章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **test classification dataset**.

本脚本演示 **test classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — test classification dataset

```python
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — summarize the dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: test classification dataset 是机器学习中的常用技术。  
  *test classification dataset is a common technique in machine learning.*

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
# Dataset Class / 01 Dataset Class
# Complete Code / 完整代码
# ===============================

# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Dataset Reg

# 02 — Dataset Reg / 02 Dataset Reg

**Chapter 16 — File 2 of 12 / 第16章 — 第2个文件（共12个）**

---

## Summary / 总结

This script demonstrates **test regression dataset**.

本脚本演示 **test regression dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — test regression dataset

```python
from sklearn.datasets import make_regression
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
```

---
## Step 3 — summarize the dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: test regression dataset 是机器学习中的常用技术。  
  *test regression dataset is a common technique in machine learning.*

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
# Dataset Reg / 02 Dataset Reg
# Complete Code / 完整代码
# ===============================

# test regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Linear Regression

# 03 — Linear Regression / 线性模型

**Chapter 16 — File 3 of 12 / 第16章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **linear regression feature importance**.

本脚本演示 **linear regression feature importance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — linear regression feature importance

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = LinearRegression()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.coef_
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: linear regression feature importance 是机器学习中的常用技术。  
  *linear regression feature importance is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Linear Regression / 线性模型
# Complete Code / 完整代码
# ===============================

# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Logistic Regression

# 04 — Logistic Regression / 回归

**Chapter 16 — File 4 of 12 / 第16章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **logistic regression for feature importance**.

本脚本演示 **logistic regression for feature importance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — logistic regression for feature importance

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = LogisticRegression()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.coef_[0]
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: logistic regression for feature importance 是机器学习中的常用技术。  
  *logistic regression for feature importance is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Logistic Regression / 回归
# Complete Code / 完整代码
# ===============================

# logistic regression for feature importance
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Cart Regression

# 05 — Cart Regression / 回归

**Chapter 16 — File 5 of 12 / 第16章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **decision tree for feature importance on a regression problem**.

本脚本演示 **decision tree for feature importance on a regression problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — decision tree for feature importance on a regression problem

```python
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = DecisionTreeRegressor()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.feature_importances_
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: decision tree for feature importance on a regression problem 是机器学习中的常用技术。  
  *decision tree for feature importance on a regression problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cart Regression / 回归
# Complete Code / 完整代码
# ===============================

# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Cart Classification

# 06 — Cart Classification / 分类

**Chapter 16 — File 6 of 12 / 第16章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **decision tree for feature importance on a classification problem**.

本脚本演示 **decision tree for feature importance on a classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — decision tree for feature importance on a classification problem

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = DecisionTreeClassifier()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.feature_importances_
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: decision tree for feature importance on a classification problem 是机器学习中的常用技术。  
  *decision tree for feature importance on a classification problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cart Classification / 分类
# Complete Code / 完整代码
# ===============================

# decision tree for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Rf Regression

# 07 — Rf Regression / 回归

**Chapter 16 — File 7 of 12 / 第16章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **random forest for feature importance on a regression problem**.

本脚本演示 **random forest for feature importance on a regression problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — random forest for feature importance on a regression problem

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = RandomForestRegressor()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.feature_importances_
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: random forest for feature importance on a regression problem 是机器学习中的常用技术。  
  *random forest for feature importance on a regression problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rf Regression / 回归
# Complete Code / 完整代码
# ===============================

# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Rf Classification

# 08 — Rf Classification / 分类

**Chapter 16 — File 8 of 12 / 第16章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **random forest for feature importance on a classification problem**.

本脚本演示 **random forest for feature importance on a classification problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — random forest for feature importance on a classification problem

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = RandomForestClassifier()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — get importance

```python
importance = model.feature_importances_
```

---
## Step 6 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 7 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: random forest for feature importance on a classification problem 是机器学习中的常用技术。  
  *random forest for feature importance on a classification problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rf Classification / 分类
# Complete Code / 完整代码
# ===============================

# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Permutation Reg

# 09 — Permutation Reg / 09 Permutation Reg

**Chapter 16 — File 9 of 12 / 第16章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **permutation feature importance with knn for regression**.

本脚本演示 **permutation feature importance with knn for regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — permutation feature importance with knn for regression

```python
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = KNeighborsRegressor()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — perform permutation importance

```python
results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
```

---
## Step 6 — get importance

```python
importance = results.importances_mean
```

---
## Step 7 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 8 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: permutation feature importance with knn for regression 是机器学习中的常用技术。  
  *permutation feature importance with knn for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Permutation Reg / 09 Permutation Reg
# Complete Code / 完整代码
# ===============================

# permutation feature importance with knn for regression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = KNeighborsRegressor()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Permutation Class

# 10 — Permutation Class / 10 Permutation Class

**Chapter 16 — File 10 of 12 / 第16章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **permutation feature importance with knn for classification**.

本脚本演示 **permutation feature importance with knn for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — permutation feature importance with knn for classification

```python
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — define the model

```python
model = KNeighborsClassifier()
```

---
## Step 4 — fit the model

```python
model.fit(X, y)
```

---
## Step 5 — perform permutation importance

```python
results = permutation_importance(model, X, y, scoring='accuracy')
```

---
## Step 6 — get importance

```python
importance = results.importances_mean
```

---
## Step 7 — summarize feature importance

```python
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
```

---
## Step 8 — plot feature importance

```python
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: permutation feature importance with knn for classification 是机器学习中的常用技术。  
  *permutation feature importance with knn for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Permutation Class / 10 Permutation Class
# Complete Code / 完整代码
# ===============================

# permutation feature importance with knn for classification
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### Evaluate All Features

# 11 — Evaluate All Features / 特征工程

**Chapter 16 — File 11 of 12 / 第16章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using all features**.

本脚本演示 **evaluation of a model using all features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — evaluation of a model using all features

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---
## Step 2 — define the dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 3 — split into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — fit the model

```python
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
```

---
## Step 5 — evaluate the model

```python
yhat = model.predict(X_test)
```

---
## Step 6 — evaluate predictions

```python
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using all features 是机器学习中的常用技术。  
  *evaluation of a model using all features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate All Features / 特征工程
# Complete Code / 完整代码
# ===============================

# evaluation of a model using all features
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Evaluate Feature Selection

# 12 — Evaluate Feature Selection / 特征工程

**Chapter 16 — File 12 of 12 / 第16章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using 5 features chosen with random forest importance**.

本脚本演示 **evaluation of a model using 5 features chosen with random forest importance**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌──────────────────────┐
│  训练模型 Train Model  │
└──────────────────────┘
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — evaluation of a model using 5 features chosen with random forest importance

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---
## Step 2 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 3 — configure to select a subset of features

```python
fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=5)
```

---
## Step 4 — learn relationship from training data

```python
fs.fit(X_train, y_train)
```

---
## Step 5 — transform train input data

```python
X_train_fs = fs.transform(X_train)
```

---
## Step 6 — transform test input data

```python
X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 7 — define the dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
```

---
## Step 8 — split into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 9 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
```

---
## Step 10 — fit the model

```python
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
```

---
## Step 11 — evaluate the model

```python
yhat = model.predict(X_test_fs)
```

---
## Step 12 — evaluate predictions

```python
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using 5 features chosen with random forest importance 是机器学习中的常用技术。  
  *evaluation of a model using 5 features chosen with random forest importance is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Feature Selection / 特征工程
# Complete Code / 完整代码
# ===============================

# evaluation of a model using 5 features chosen with random forest importance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=5)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
```

---

### Chapter Summary

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **12 code files** demonstrating chapter 16.

本章包含 **12 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_dataset_class.ipynb` — Dataset Class
  2. `02_dataset_reg.ipynb` — Dataset Reg
  3. `03_linear_regression.ipynb` — Linear Regression
  4. `04_logistic_regression.ipynb` — Logistic Regression
  5. `05_cart_regression.ipynb` — Cart Regression
  6. `06_cart_classification.ipynb` — Cart Classification
  7. `07_rf_regression.ipynb` — Rf Regression
  8. `08_rf_classification.ipynb` — Rf Classification
  9. `09_permutation_reg.ipynb` — Permutation Reg
  10. `10_permutation_class.ipynb` — Permutation Class
  11. `11_evaluate_all_features.ipynb` — Evaluate All Features
  12. `12_evaluate_feature_selection.ipynb` — Evaluate Feature Selection

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
