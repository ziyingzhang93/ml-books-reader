# 机器学习数据准备 / Data Preparation for ML
## Chapter 17

---

### Demo Normalization

# 01 — Demo Normalization / 01 Demo Normalization

**Chapter 17 — File 1 of 8 / 第17章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a normalization**.

本脚本演示 **example of a normalization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — example of a normalization

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
```

---
## Step 2 — define data

```python
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
# 打印输出 / Print output
print(data)
```

---
## Step 3 — define min max scaler

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
```

---
## Step 4 — transform data

```python
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
scaled = scaler.fit_transform(data)
# 打印输出 / Print output
print(scaled)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a normalization 是机器学习中的常用技术。  
  *example of a normalization is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Demo Normalization / 01 Demo Normalization
# Complete Code / 完整代码
# ===============================

# example of a normalization
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
# 打印输出 / Print output
print(data)
# define min max scaler
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# transform data
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
scaled = scaler.fit_transform(data)
# 打印输出 / Print output
print(scaled)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Demo Standardization

# 02 — Demo Standardization / 02 Demo Standardization

**Chapter 17 — File 2 of 8 / 第17章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a standardization**.

本脚本演示 **example of a standardization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — example of a standardization

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — define data

```python
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
# 打印输出 / Print output
print(data)
```

---
## Step 3 — define standard scaler

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
```

---
## Step 4 — transform data

```python
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
scaled = scaler.fit_transform(data)
# 打印输出 / Print output
print(scaled)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a standardization 是机器学习中的常用技术。  
  *example of a standardization is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Demo Standardization / 02 Demo Standardization
# Complete Code / 完整代码
# ===============================

# example of a standardization
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# define data
data = asarray([[100, 0.001],
				[8, 0.05],
				[50, 0.005],
				[88, 0.07],
				[4, 0.1]])
# 打印输出 / Print output
print(data)
# define standard scaler
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
scaler = StandardScaler()
# transform data
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
scaled = scaler.fit_transform(data)
# 打印输出 / Print output
print(scaled)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Load Dataset

# 03 — Load Dataset / 03 Load Dataset

**Chapter 17 — File 3 of 8 / 第17章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load and summarize the diabetes dataset**.

本脚本演示 **load and summarize the diabetes dataset**。

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
## Step 1 — load and summarize the diabetes dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — summarize the shape of the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 4 — summarize each variable

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 5 — histograms of the variables

```python
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
```

---
## Step 6 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and summarize the diabetes dataset 是机器学习中的常用技术。  
  *load and summarize the diabetes dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `describe()` | 统计摘要信息 | Statistical summary |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Dataset / 03 Load Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the diabetes dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# summarize the shape of the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# summarize each variable
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Model Evaluate

# 04 — Model Evaluate / 模型评估

**Chapter 17 — File 4 of 8 / 第17章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the raw diabetes dataset**.

本脚本演示 **evaluate knn on the raw diabetes dataset**。

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
## Step 1 — evaluate knn on the raw diabetes dataset

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
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
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
## Step 5 — define and configure the model

```python
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
```

---
## Step 6 — evaluate the model

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — report model performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate knn on the raw diabetes dataset 是机器学习中的常用技术。  
  *evaluate knn on the raw diabetes dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
# Model Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the raw diabetes dataset
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
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
# 转换数据类型 / Convert data type
X = X.astype('float32')
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y.astype('str'))
# define and configure the model
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
model = KNeighborsClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report model performance
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Normalize Data

# 05 — Normalize Data / 05 Normalize Data

**Chapter 17 — File 5 of 8 / 第17章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **visualize a minmax scaler transform of the diabetes dataset**.

本脚本演示 **visualize a minmax scaler transform of the diabetes dataset**。

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
## Step 1 — visualize a minmax scaler transform of the diabetes dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — retrieve just the numeric input values

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
```

---
## Step 4 — perform a robust scaler transform of the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
trans = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
```

---
## Step 5 — convert the array back to a dataframe

```python
dataset = DataFrame(data)
```

---
## Step 6 — summarize

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 7 — histograms of the variables

```python
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: visualize a minmax scaler transform of the diabetes dataset 是机器学习中的常用技术。  
  *visualize a minmax scaler transform of the diabetes dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `describe()` | 统计摘要信息 | Statistical summary |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normalize Data / 05 Normalize Data
# Complete Code / 完整代码
# ===============================

# visualize a minmax scaler transform of the diabetes dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a robust scaler transform of the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
trans = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Normalize Evaluate

# 06 — Normalize Evaluate / 模型评估

**Chapter 17 — File 6 of 8 / 第17章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the diabetes dataset with minmax scaler transform**.

本脚本演示 **evaluate knn on the diabetes dataset with minmax scaler transform**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — evaluate knn on the diabetes dataset with minmax scaler transform

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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
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
# 归一化到[0,1]范围 / Normalize to [0,1] range
trans = MinMaxScaler()
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

- **概念**: evaluate knn on the diabetes dataset with minmax scaler transform 是机器学习中的常用技术。  
  *evaluate knn on the diabetes dataset with minmax scaler transform is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
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
# Normalize Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the diabetes dataset with minmax scaler transform
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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
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
# 归一化到[0,1]范围 / Normalize to [0,1] range
trans = MinMaxScaler()
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

➡️ **Next / 下一步**: File 7 of 8

---

### Standardize Data

# 07 — Standardize Data / 07 Standardize Data

**Chapter 17 — File 7 of 8 / 第17章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **visualize a standard scaler transform of the diabetes dataset**.

本脚本演示 **visualize a standard scaler transform of the diabetes dataset**。

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
## Step 1 — visualize a standard scaler transform of the diabetes dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — retrieve just the numeric input values

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
```

---
## Step 4 — perform a robust scaler transform of the dataset

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
trans = StandardScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
```

---
## Step 5 — convert the array back to a dataframe

```python
dataset = DataFrame(data)
```

---
## Step 6 — summarize

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---
## Step 7 — histograms of the variables

```python
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: visualize a standard scaler transform of the diabetes dataset 是机器学习中的常用技术。  
  *visualize a standard scaler transform of the diabetes dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `describe()` | 统计摘要信息 | Statistical summary |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardize Data / 07 Standardize Data
# Complete Code / 完整代码
# ===============================

# visualize a standard scaler transform of the diabetes dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a robust scaler transform of the dataset
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
trans = StandardScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
# histograms of the variables
fig = dataset.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Standardize Evaluate

# 08 — Standardize Evaluate / 模型评估

**Chapter 17 — File 8 of 8 / 第17章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the diabetes dataset with standard scaler transform**.

本脚本演示 **evaluate knn on the diabetes dataset with standard scaler transform**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — evaluate knn on the diabetes dataset with standard scaler transform

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
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
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
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
trans = StandardScaler()
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

- **概念**: evaluate knn on the diabetes dataset with standard scaler transform 是机器学习中的常用技术。  
  *evaluate knn on the diabetes dataset with standard scaler transform is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Standardize Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the diabetes dataset with standard scaler transform
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
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
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
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
trans = StandardScaler()
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

### Chapter Summary / 章节总结

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **8 code files** demonstrating chapter 17.

本章包含 **8 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `01_demo_normalization.ipynb` — Demo Normalization
  2. `02_demo_standardization.ipynb` — Demo Standardization
  3. `03_load_dataset.ipynb` — Load Dataset
  4. `04_model_evaluate.ipynb` — Model Evaluate
  5. `05_normalize_data.ipynb` — Normalize Data
  6. `06_normalize_evaluate.ipynb` — Normalize Evaluate
  7. `07_standardize_data.ipynb` — Standardize Data
  8. `08_standardize_evaluate.ipynb` — Standardize Evaluate

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
