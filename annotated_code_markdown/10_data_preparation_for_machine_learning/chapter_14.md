# ML数据准备
## Chapter 14

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 14 — File 1 of 9 / 第14章 — 第1个文件（共9个）**

---

## Summary / 总结

This script demonstrates **load and summarize the dataset**.

本脚本演示 **load and summarize the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — load and summarize the dataset

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
```

---
## Step 2 — generate regression dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
```

---
## Step 3 — split into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — summarize

```python
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load and summarize the dataset 是机器学习中的常用技术。  
  *load and summarize the dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Dataset / 01 Load Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the dataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Correlation Selection

# 02 — Correlation Selection / 特征选择

**Chapter 14 — File 2 of 9 / 第14章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **example of correlation feature selection for numerical data**.

本脚本演示 **example of correlation feature selection for numerical data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — example of correlation feature selection for numerical data

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
```

---
## Step 2 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 3 — configure to select all features

```python
fs = SelectKBest(score_func=f_regression, k='all')
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
## Step 7 — load the dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
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
## Step 10 — what are scores for the features

```python
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
```

---
## Step 11 — plot the scores

```python
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of correlation feature selection for numerical data 是机器学习中的常用技术。  
  *example of correlation feature selection for numerical data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Correlation Selection / 特征选择
# Complete Code / 完整代码
# ===============================

# example of correlation feature selection for numerical data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Mutual Information Selection

# 03 — Mutual Information Selection / 特征选择

**Chapter 14 — File 3 of 9 / 第14章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **example of mutual information feature selection for numerical input data**.

本脚本演示 **example of mutual information feature selection for numerical input data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — example of mutual information feature selection for numerical input data

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
```

---
## Step 2 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 3 — configure to select all features

```python
fs = SelectKBest(score_func=mutual_info_regression, k='all')
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
## Step 7 — load the dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
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
## Step 10 — what are scores for the features

```python
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
```

---
## Step 11 — plot the scores

```python
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of mutual information feature selection for numerical input data 是机器学习中的常用技术。  
  *example of mutual information feature selection for numerical input data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mutual Information Selection / 特征选择
# Complete Code / 完整代码
# ===============================

# example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Evaluate All Features

# 04 — Evaluate All Features / 特征工程

**Chapter 14 — File 4 of 9 / 第14章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using all input features**.

本脚本演示 **evaluation of a model using all input features**。

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
## Step 1 — evaluation of a model using all input features

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

---
## Step 2 — load the dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
```

---
## Step 3 — split into train and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — fit the model

```python
model = LinearRegression()
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
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using all input features 是机器学习中的常用技术。  
  *evaluation of a model using all input features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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

# evaluation of a model using all input features
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Evaluate Correlation

# 05 — Evaluate Correlation / 模型评估

**Chapter 14 — File 5 of 9 / 第14章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using 10 features chosen with correlation**.

本脚本演示 **evaluation of a model using 10 features chosen with correlation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — evaluation of a model using 10 features chosen with correlation

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

---
## Step 2 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 3 — configure to select a subset of features

```python
fs = SelectKBest(score_func=f_regression, k=10)
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
## Step 7 — load the dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
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
model = LinearRegression()
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
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using 10 features chosen with correlation 是机器学习中的常用技术。  
  *evaluation of a model using 10 features chosen with correlation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Correlation / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluation of a model using 10 features chosen with correlation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=f_regression, k=10)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Evaluate Correlation More Features

# 06 — Evaluate Correlation More Features / 特征工程

**Chapter 14 — File 6 of 9 / 第14章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using 88 features chosen with correlation**.

本脚本演示 **evaluation of a model using 88 features chosen with correlation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — evaluation of a model using 88 features chosen with correlation

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

---
## Step 2 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 3 — configure to select a subset of features

```python
fs = SelectKBest(score_func=f_regression, k=88)
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
## Step 7 — load the dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
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
model = LinearRegression()
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
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using 88 features chosen with correlation 是机器学习中的常用技术。  
  *evaluation of a model using 88 features chosen with correlation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Correlation More Features / 特征工程
# Complete Code / 完整代码
# ===============================

# evaluation of a model using 88 features chosen with correlation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=f_regression, k=88)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Grid Search Mutual Info

# 08 — Grid Search Mutual Info / 08 Grid Search Mutual Info

**Chapter 14 — File 8 of 9 / 第14章 — 第8个文件（共9个）**

---

## Summary / 总结

This script demonstrates **compare different numbers of features selected using mutual information**.

本脚本演示 **compare different numbers of features selected using mutual information**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Step 1 — compare different numbers of features selected using mutual information

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
```

---
## Step 3 — define the evaluation method

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 4 — define the pipeline to evaluate

```python
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
```

---
## Step 5 — define the grid

```python
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]
```

---
## Step 6 — define the grid search

```python
search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
```

---
## Step 7 — perform the search

```python
results = search.fit(X, y)
```

---
## Step 8 — summarize best

```python
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
```

---
## Step 9 — summarize all

```python
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print('>%.3f with: %r' % (mean, param))
```

---
## Learning Notes / 学习笔记

- **概念**: compare different numbers of features selected using mutual information 是机器学习中的常用技术。  
  *compare different numbers of features selected using mutual information is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Mutual Info / 08 Grid Search Mutual Info
# Complete Code / 完整代码
# ===============================

# compare different numbers of features selected using mutual information
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# define dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print('>%.3f with: %r' % (mean, param))
```

---

➡️ **Next / 下一步**: File 9 of 9

---

### Compare Performance Num Selected

# 09 — Compare Performance Num Selected / 特征选择

**Chapter 14 — File 9 of 9 / 第14章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **compare different numbers of features selected using mutual information**.

本脚本演示 **compare different numbers of features selected using mutual information**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — compare different numbers of features selected using mutual information

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
```

---
## Step 3 — define number of features to evaluate

```python
num_features = [i for i in range(X.shape[1]-19, X.shape[1]+1)]
```

---
## Step 4 — enumerate each number of features

```python
results = list()
for k in num_features:
```

---
## Step 5 — create pipeline

```python
model = LinearRegression()
	fs = SelectKBest(score_func=mutual_info_regression, k=k)
	pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
```

---
## Step 6 — evaluate the model

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	results.append(scores)
```

---
## Step 7 — summarize the results

```python
print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
```

---
## Step 8 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare different numbers of features selected using mutual information 是机器学习中的常用技术。  
  *compare different numbers of features selected using mutual information is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Performance Num Selected / 特征选择
# Complete Code / 完整代码
# ===============================

# compare different numbers of features selected using mutual information
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=1)
# define number of features to evaluate
num_features = [i for i in range(X.shape[1]-19, X.shape[1]+1)]
# enumerate each number of features
results = list()
for k in num_features:
	# create pipeline
	model = LinearRegression()
	fs = SelectKBest(score_func=mutual_info_regression, k=k)
	pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
	# evaluate the model
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	results.append(scores)
	# summarize the results
	print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
```

---

### Chapter Summary

# Chapter 14 Summary / 第14章总结

## Theme / 主题: Chapter 14 / Chapter 14

This chapter contains **9 code files** demonstrating chapter 14.

本章包含 **9 个代码文件**，演示Chapter 14。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_correlation_selection.ipynb` — Correlation Selection
  3. `03_mutual_information_selection.ipynb` — Mutual Information Selection
  4. `04_evaluate_all_features.ipynb` — Evaluate All Features
  5. `05_evaluate_correlation.ipynb` — Evaluate Correlation
  6. `06_evaluate_correlation_more_features.ipynb` — Evaluate Correlation More Features
  7. `07_evaluate_mutual_information.ipynb` — Evaluate Mutual Information
  8. `08_grid_search_mutual_info.ipynb` — Grid Search Mutual Info
  9. `09_compare_performance_num_selected.ipynb` — Compare Performance Num Selected

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 14) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 14）是机器学习流水线中的基础构建块。

---
