# 不平衡分类
## Chapter 30

---

### Load Summarize

# 01 — Load Summarize / 01 Load Summarize

**Chapter 30 — File 1 of 7 / 第30章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **load and summarize the dataset**.

本脚本演示 **load and summarize the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — load and summarize the dataset

```python
from pandas import read_csv
from collections import Counter
```

---
## Step 2 — define the dataset location

```python
filename = 'phoneme.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
dataframe = read_csv(filename, header=None)
```

---
## Step 4 — summarize the shape of the dataset

```python
print(dataframe.shape)
```

---
## Step 5 — summarize the class distribution

```python
target = dataframe.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Summarize / 01 Load Summarize
# Complete Code / 完整代码
# ===============================

# load and summarize the dataset
from pandas import read_csv
from collections import Counter
# define the dataset location
filename = 'phoneme.csv'
# load the csv file as a data frame
dataframe = read_csv(filename, header=None)
# summarize the shape of the dataset
print(dataframe.shape)
# summarize the class distribution
target = dataframe.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Histograms

# 02 — Histograms / 02 Histograms

**Chapter 30 — File 2 of 7 / 第30章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **create histograms of numeric input variables**.

本脚本演示 **create histograms of numeric input variables**。

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
## Step 1 — create histograms of numeric input variables

```python
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'phoneme.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
df = read_csv(filename, header=None)
```

---
## Step 4 — histograms of all variables

```python
df.hist()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create histograms of numeric input variables 是机器学习中的常用技术。  
  *create histograms of numeric input variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Histograms / 02 Histograms
# Complete Code / 完整代码
# ===============================

# create histograms of numeric input variables
from pandas import read_csv
from matplotlib import pyplot
# define the dataset location
filename = 'phoneme.csv'
# load the csv file as a data frame
df = read_csv(filename, header=None)
# histograms of all variables
df.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Scatter Plots

# 03 — Scatter Plots / 03 Scatter Plots

**Chapter 30 — File 3 of 7 / 第30章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **create pairwise scatter plots of numeric input variables**.

本脚本演示 **create pairwise scatter plots of numeric input variables**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — create pairwise scatter plots of numeric input variables

```python
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'phoneme.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
df = read_csv(filename, header=None)
```

---
## Step 4 — define a mapping of class values to colors

```python
color_dict = {0:'blue', 1:'red'}
```

---
## Step 5 — map each row to a color based on the class value

```python
colors = [color_dict[x] for x in df.values[:, -1]]
```

---
## Step 6 — drop the target variable

```python
inputs = DataFrame(df.values[:, :-1])
```

---
## Step 7 — pairwise scatter plots of all numerical variables

```python
scatter_matrix(inputs, diagonal='kde', color=colors)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create pairwise scatter plots of numeric input variables 是机器学习中的常用技术。  
  *create pairwise scatter plots of numeric input variables is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scatter Plots / 03 Scatter Plots
# Complete Code / 完整代码
# ===============================

# create pairwise scatter plots of numeric input variables
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# define the dataset location
filename = 'phoneme.csv'
# load the csv file as a data frame
df = read_csv(filename, header=None)
# define a mapping of class values to colors
color_dict = {0:'blue', 1:'red'}
# map each row to a color based on the class value
colors = [color_dict[x] for x in df.values[:, -1]]
# drop the target variable
inputs = DataFrame(df.values[:, :-1])
# pairwise scatter plots of all numerical variables
scatter_matrix(inputs, diagonal='kde', color=colors)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Baseline

# 04 — Baseline / 04 Baseline

**Chapter 30 — File 4 of 7 / 第30章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **test harness and baseline model evaluation**.

本脚本演示 **test harness and baseline model evaluation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — test harness and baseline model evaluation

```python
from collections import Counter
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
data = data.values
```

---
## Step 5 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
	return X, y
```

---
## Step 6 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 7 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — define the model evaluation metric

```python
metric = make_scorer(geometric_mean_score)
```

---
## Step 9 — evaluate model

```python
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 10 — define the location of the dataset

```python
full_path = 'phoneme.csv'
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 12 — summarize the loaded dataset

```python
print(X.shape, y.shape, Counter(y))
```

---
## Step 13 — define the reference model

```python
model = DummyClassifier(strategy='uniform')
```

---
## Step 14 — evaluate the model

```python
scores = evaluate_model(X, y, model)
```

---
## Step 15 — summarize performance

```python
print('Mean G-Mean: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: test harness and baseline model evaluation 是机器学习中的常用技术。  
  *test harness and baseline model evaluation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Baseline / 04 Baseline
# Complete Code / 完整代码
# ===============================

# test harness and baseline model evaluation
from collections import Counter
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	return X, y

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(geometric_mean_score)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define the location of the dataset
full_path = 'phoneme.csv'
# load the dataset
X, y = load_dataset(full_path)
# summarize the loaded dataset
print(X.shape, y.shape, Counter(y))
# define the reference model
model = DummyClassifier(strategy='uniform')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
print('Mean G-Mean: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Make Predictions

# 07 — Make Predictions / 07 Make Predictions

**Chapter 30 — File 7 of 7 / 第30章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **fit a model and make predictions for the phoneme dataset**.

本脚本演示 **fit a model and make predictions for the phoneme dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
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
## Step 1 — fit a model and make predictions for the phoneme dataset

```python
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
data = data.values
```

---
## Step 5 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
	return X, y
```

---
## Step 6 — define the location of the dataset

```python
full_path = 'phoneme.csv'
```

---
## Step 7 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 8 — define the model

```python
model = ExtraTreesClassifier(n_estimators=1000)
```

---
## Step 9 — define the pipeline steps

```python
steps = [('s', MinMaxScaler()), ('o', ADASYN()), ('m', model)]
```

---
## Step 10 — define the pipeline

```python
pipeline = Pipeline(steps=steps)
```

---
## Step 11 — fit the model

```python
pipeline.fit(X, y)
```

---
## Step 12 — evaluate on some nasal cases (known class 0)

```python
print('Nasal:')
data = [[1.24,0.875,-0.205,-0.078,0.067],
	[0.268,1.352,1.035,-0.332,0.217],
	[1.567,0.867,1.3,1.041,0.559]]
for row in data:
```

---
## Step 13 — make prediction

```python
yhat = pipeline.predict([row])
```

---
## Step 14 — get the label

```python
label = yhat[0]
```

---
## Step 15 — summarize

```python
print('>Predicted=%d (expected 0)' % (label))
```

---
## Step 16 — evaluate on some oral cases (known class 1)

```python
print('Oral:')
data = [[0.125,0.548,0.795,0.836,0.0],
	[0.318,0.811,0.818,0.821,0.86],
	[0.151,0.642,1.454,1.281,-0.716]]
for row in data:
```

---
## Step 17 — make prediction

```python
yhat = pipeline.predict([row])
```

---
## Step 18 — get the label

```python
label = yhat[0]
```

---
## Step 19 — summarize

```python
print('>Predicted=%d (expected 1)' % (label))
```

---
## Learning Notes / 学习笔记

- **概念**: fit a model and make predictions for the phoneme dataset 是机器学习中的常用技术。  
  *fit a model and make predictions for the phoneme dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Predictions / 07 Make Predictions
# Complete Code / 完整代码
# ===============================

# fit a model and make predictions for the phoneme dataset
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	return X, y

# define the location of the dataset
full_path = 'phoneme.csv'
# load the dataset
X, y = load_dataset(full_path)
# define the model
model = ExtraTreesClassifier(n_estimators=1000)
# define the pipeline steps
steps = [('s', MinMaxScaler()), ('o', ADASYN()), ('m', model)]
# define the pipeline
pipeline = Pipeline(steps=steps)
# fit the model
pipeline.fit(X, y)
# evaluate on some nasal cases (known class 0)
print('Nasal:')
data = [[1.24,0.875,-0.205,-0.078,0.067],
	[0.268,1.352,1.035,-0.332,0.217],
	[1.567,0.867,1.3,1.041,0.559]]
for row in data:
	# make prediction
	yhat = pipeline.predict([row])
	# get the label
	label = yhat[0]
	# summarize
	print('>Predicted=%d (expected 0)' % (label))
# evaluate on some oral cases (known class 1)
print('Oral:')
data = [[0.125,0.548,0.795,0.836,0.0],
	[0.318,0.811,0.818,0.821,0.86],
	[0.151,0.642,1.454,1.281,-0.716]]
for row in data:
	# make prediction
	yhat = pipeline.predict([row])
	# get the label
	label = yhat[0]
	# summarize
	print('>Predicted=%d (expected 1)' % (label))
```

---

### Chapter Summary

# Chapter 30 Summary / 第30章总结

## Theme / 主题: Chapter 30 / Chapter 30

This chapter contains **7 code files** demonstrating chapter 30.

本章包含 **7 个代码文件**，演示Chapter 30。

---
## Evolution / 演化路线

  1. `01_load_summarize.ipynb` — Load Summarize
  2. `02_histograms.ipynb` — Histograms
  3. `03_scatter_plots.ipynb` — Scatter Plots
  4. `04_baseline.ipynb` — Baseline
  5. `05_evaluate_models.ipynb` — Evaluate Models
  6. `06_data_sampling_models.ipynb` — Data Sampling Models
  7. `07_make_predictions.ipynb` — Make Predictions

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 30) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 30）是机器学习流水线中的基础构建块。

---
