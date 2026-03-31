# 机器学习数据准备 / Data Preparation for ML
## Chapter 21

---

### Demo Quantile Transform



---

### Load Dataset



---

### Model Evaluation

# 03 — Model Evaluation / 模型评估

**Chapter 21 — File 3 of 8 / 第21章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the raw sonar dataset**.

本脚本演示 **evaluate knn on the raw sonar dataset**。

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
## Step 1 — evaluate knn on the raw sonar dataset

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

- **概念**: evaluate knn on the raw sonar dataset 是机器学习中的常用技术。  
  *evaluate knn on the raw sonar dataset is a common technique in machine learning.*

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
# Model Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the raw sonar dataset
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

➡️ **Next / 下一步**: File 4 of 8

---

### Normal Quantile Transform

# 04 — Normal Quantile Transform / 数据变换

**Chapter 21 — File 4 of 8 / 第21章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **visualize a normal quantile transform of the sonar dataset**.

本脚本演示 **visualize a normal quantile transform of the sonar dataset**。

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
## Step 1 — visualize a normal quantile transform of the sonar dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer
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
## Step 4 — perform a normal quantile transform of the dataset

```python
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
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

- **概念**: visualize a normal quantile transform of the sonar dataset 是机器学习中的常用技术。  
  *visualize a normal quantile transform of the sonar dataset is a common technique in machine learning.*

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
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normal Quantile Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# visualize a normal quantile transform of the sonar dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a normal quantile transform of the dataset
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
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

➡️ **Next / 下一步**: File 5 of 8

---

### Normal Quantile Model Evaluation

# 05 — Normal Quantile Model Evaluation / 模型评估

**Chapter 21 — File 5 of 8 / 第21章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the sonar dataset with normal quantile transform**.

本脚本演示 **evaluate knn on the sonar dataset with normal quantile transform**。

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
## Step 1 — evaluate knn on the sonar dataset with normal quantile transform

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
from sklearn.preprocessing import QuantileTransformer
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
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
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

- **概念**: evaluate knn on the sonar dataset with normal quantile transform 是机器学习中的常用技术。  
  *evaluate knn on the sonar dataset with normal quantile transform is a common technique in machine learning.*

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
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normal Quantile Model Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the sonar dataset with normal quantile transform
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
from sklearn.preprocessing import QuantileTransformer
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
trans = QuantileTransformer(n_quantiles=100, output_distribution='normal')
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

➡️ **Next / 下一步**: File 6 of 8

---

### Uniform Quantile Transform

# 06 — Uniform Quantile Transform / 数据变换

**Chapter 21 — File 6 of 8 / 第21章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **visualize a uniform quantile transform of the sonar dataset**.

本脚本演示 **visualize a uniform quantile transform of the sonar dataset**。

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
## Step 1 — visualize a uniform quantile transform of the sonar dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer
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
## Step 4 — perform a uniform quantile transform of the dataset

```python
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
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

- **概念**: visualize a uniform quantile transform of the sonar dataset 是机器学习中的常用技术。  
  *visualize a uniform quantile transform of the sonar dataset is a common technique in machine learning.*

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
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Uniform Quantile Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# visualize a uniform quantile transform of the sonar dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import QuantileTransformer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a uniform quantile transform of the dataset
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
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

➡️ **Next / 下一步**: File 7 of 8

---

### Uniform Quantile Model Evaluation

# 07 — Uniform Quantile Model Evaluation / 模型评估

**Chapter 21 — File 7 of 8 / 第21章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the sonar dataset with uniform quantile transform**.

本脚本演示 **evaluate knn on the sonar dataset with uniform quantile transform**。

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
## Step 1 — evaluate knn on the sonar dataset with uniform quantile transform

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
from sklearn.preprocessing import QuantileTransformer
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
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
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

- **概念**: evaluate knn on the sonar dataset with uniform quantile transform 是机器学习中的常用技术。  
  *evaluate knn on the sonar dataset with uniform quantile transform is a common technique in machine learning.*

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
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Uniform Quantile Model Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the sonar dataset with uniform quantile transform
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
from sklearn.preprocessing import QuantileTransformer
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
trans = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
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

➡️ **Next / 下一步**: File 8 of 8

---

### Compare Num Quantiles

# 08 — Compare Num Quantiles / 08 Compare Num Quantiles

**Chapter 21 — File 8 of 8 / 第21章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **explore number of quantiles on classification accuracy**.

本脚本演示 **explore number of quantiles on classification accuracy**。

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
## Step 1 — explore number of quantiles on classification accuracy

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
from sklearn.preprocessing import QuantileTransformer
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
def get_dataset(filename):
```

---
## Step 3 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(filename, header=None)
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
	for i in range(1,100):
```

---
## Step 7 — define the pipeline

```python
trans = QuantileTransformer(n_quantiles=i, output_distribution='uniform')
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
## Step 9 — define dataset

```python
X, y = get_dataset('sonar.csv')
```

---
## Step 10 — get the models to evaluate

```python
models = get_models()
```

---
## Step 11 — evaluate the models and store results

```python
results = list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(mean(scores))
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 12 — plot model performance for comparison

```python
pyplot.plot(results)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore number of quantiles on classification accuracy 是机器学习中的常用技术。  
  *explore number of quantiles on classification accuracy is a common technique in machine learning.*

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
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Num Quantiles / 08 Compare Num Quantiles
# Complete Code / 完整代码
# ===============================

# explore number of quantiles on classification accuracy
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
from sklearn.preprocessing import QuantileTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset(filename):
	# load dataset
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataset = read_csv(filename, header=None)
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
	for i in range(1,100):
		# define the pipeline
		trans = QuantileTransformer(n_quantiles=i, output_distribution='uniform')
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

# define dataset
X, y = get_dataset('sonar.csv')
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results = list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(mean(scores))
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.plot(results)
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **8 code files** demonstrating chapter 21.

本章包含 **8 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_demo_quantile_transform.ipynb` — Demo Quantile Transform
  2. `02_load_dataset.ipynb` — Load Dataset
  3. `03_model_evaluation.ipynb` — Model Evaluation
  4. `04_normal_quantile_transform.ipynb` — Normal Quantile Transform
  5. `05_normal_quantile_model_evaluation.ipynb` — Normal Quantile Model Evaluation
  6. `06_uniform_quantile_transform.ipynb` — Uniform Quantile Transform
  7. `07_uniform_quantile_model_evaluation.ipynb` — Uniform Quantile Model Evaluation
  8. `08_compare_num_quantiles.ipynb` — Compare Num Quantiles

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
