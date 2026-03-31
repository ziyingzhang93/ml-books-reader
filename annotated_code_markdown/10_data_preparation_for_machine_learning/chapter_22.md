# 机器学习数据准备 / Data Preparation for ML
## Chapter 22

---

### Demo Discretization Transform



---

### Load Dataset



---

### Model Evaluation



---

### Uniform Discretization Transform



---

### Uniform Discretization Model Eval



---

### Kmeans Discretization Transform



---

### Kmeans Discretization Model Eval

# 07 — Kmeans Discretization Model Eval / 模型评估

**Chapter 22 — File 7 of 10 / 第22章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the sonar dataset with k-means ordinal discretization transform**.

本脚本演示 **evaluate knn on the sonar dataset with k-means ordinal discretization transform**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — evaluate knn on the sonar dataset with k-means ordinal discretization transform

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
```

---
## Step 3 — separate into input and output columns

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 4 — ensure inputs are floats and output is an integer label

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y.astype('str'))
```

---
## Step 5 — define the pipeline

```python
trans = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
```

---
## Step 6 — evaluate the pipeline

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — report pipeline performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate knn on the sonar dataset with k-means ordinal discretization transform 是机器学习中的常用技术。  
  *evaluate knn on the sonar dataset with k-means ordinal discretization transform is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `KMeans` | K均值聚类 | K-Means clustering |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Kmeans Discretization Model Eval / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the sonar dataset with k-means ordinal discretization transform
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report pipeline performance
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Quantile Discretization Transform

# 08 — Quantile Discretization Transform / 数据变换

**Chapter 22 — File 8 of 10 / 第22章 — 第8个文件（共10个）**

---

## Summary / 总结

This script demonstrates **visualize a quantile ordinal discretization transform of the sonar dataset**.

本脚本演示 **visualize a quantile ordinal discretization transform of the sonar dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — visualize a quantile ordinal discretization transform of the sonar dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
```

---
## Step 3 — retrieve just the numeric input values

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
```

---
## Step 4 — perform a quantile discretization transform of the dataset

```python
trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
```

---
## Step 5 — convert the array back to a dataframe

```python
dataset = DataFrame(data)
```

---
## Step 6 — histograms of the variables

```python
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
```

---
## Step 7 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: visualize a quantile ordinal discretization transform of the sonar dataset 是机器学习中的常用技术。  
  *visualize a quantile ordinal discretization transform of the sonar dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Quantile Discretization Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# visualize a quantile ordinal discretization transform of the sonar dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a quantile discretization transform of the dataset
trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 10

---

### Quantile Discretization Model Eval



---

### Compare N Bins

# 10 — Compare N Bins / 10 Compare N Bins

**Chapter 22 — File 10 of 10 / 第22章 — 第10个文件（共10个）**

---

## Summary / 总结

This script demonstrates **explore number of discrete bins on classification accuracy**.

本脚本演示 **explore number of discrete bins on classification accuracy**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — explore number of discrete bins on classification accuracy

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
```

---
## Step 3 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
 # 转换为NumPy数组 / Convert to NumPy array
	data = dataset.values
```

---
## Step 4 — separate into input and output columns

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 5 — ensure inputs are floats and output is an integer label

```python
# 转换数据类型 / Convert data type
X = X.astype('float32')
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y.astype('str'))
	return X, y
```

---
## Step 6 — get a list of models to evaluate

```python
def get_models():
	models = dict()
 # 生成整数序列 / Generate integer sequence
	for i in range(2,11):
```

---
## Step 7 — define the pipeline

```python
trans = KBinsDiscretizer(n_bins=i, encode='ordinal', strategy='quantile')
  # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
		model = KNeighborsClassifier()
  # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
		models[str(i)] = Pipeline(steps=[('t', trans), ('m', model)])
	return models
```

---
## Step 8 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 9 — get the dataset

```python
X, y = get_dataset()
```

---
## Step 10 — get the models to evaluate

```python
models = get_models()
```

---
## Step 11 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 12 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore number of discrete bins on classification accuracy 是机器学习中的常用技术。  
  *explore number of discrete bins on classification accuracy is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare N Bins / 10 Compare N Bins
# Complete Code / 完整代码
# ===============================

# explore number of discrete bins on classification accuracy
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import KBinsDiscretizer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	# load dataset
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataset = read_csv('sonar.csv', header=None)
 # 转换为NumPy数组 / Convert to NumPy array
	data = dataset.values
	# separate into input and output columns
	X, y = data[:, :-1], data[:, -1]
	# ensure inputs are floats and output is an integer label
 # 转换数据类型 / Convert data type
	X = X.astype('float32')
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y.astype('str'))
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
 # 生成整数序列 / Generate integer sequence
	for i in range(2,11):
		# define the pipeline
		trans = KBinsDiscretizer(n_bins=i, encode='ordinal', strategy='quantile')
  # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
		model = KNeighborsClassifier()
  # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
		models[str(i)] = Pipeline(steps=[('t', trans), ('m', model)])
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# get the dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **10 code files** demonstrating chapter 22.

本章包含 **10 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_demo_discretization_transform.ipynb` — Demo Discretization Transform
  2. `02_load_dataset.ipynb` — Load Dataset
  3. `03_model_evaluation.ipynb` — Model Evaluation
  4. `04_uniform_discretization_transform.ipynb` — Uniform Discretization Transform
  5. `05_uniform_discretization_model_eval.ipynb` — Uniform Discretization Model Eval
  6. `06_kmeans_discretization_transform.ipynb` — Kmeans Discretization Transform
  7. `07_kmeans_discretization_model_eval.ipynb` — Kmeans Discretization Model Eval
  8. `08_quantile_discretization_transform.ipynb` — Quantile Discretization Transform
  9. `09_quantile_discretization_model_eval.ipynb` — Quantile Discretization Model Eval
  10. `10_compare_n_bins.ipynb` — Compare N Bins

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
