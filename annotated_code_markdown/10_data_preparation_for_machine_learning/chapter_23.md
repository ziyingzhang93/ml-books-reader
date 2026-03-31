# 机器学习数据准备 / Data Preparation for ML
## Chapter 23

---

### Example Transform

# 01 — Example Transform / 数据变换

**Chapter 23 — File 1 of 7 / 第23章 — 第1个文件（共7个）**

---

## Summary / 总结

This script demonstrates **demonstrate the types of features created**.

本脚本演示 **demonstrate the types of features created**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — demonstrate the types of features created

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
```

---
## Step 2 — define the dataset

```python
data = asarray([[2,3],[2,3],[2,3]])
# 打印输出 / Print output
print(data)
```

---
## Step 3 — perform a polynomial features transform of the dataset

```python
trans = PolynomialFeatures(degree=2)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# 打印输出 / Print output
print(data)
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate the types of features created 是机器学习中的常用技术。  
  *demonstrate the types of features created is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Example Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# demonstrate the types of features created
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
# define the dataset
data = asarray([[2,3],[2,3],[2,3]])
# 打印输出 / Print output
print(data)
# perform a polynomial features transform of the dataset
trans = PolynomialFeatures(degree=2)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# 打印输出 / Print output
print(data)
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Load Dataset



---

### Model Evaluation

# 03 — Model Evaluation / 模型评估

**Chapter 23 — File 3 of 7 / 第23章 — 第3个文件（共7个）**

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

➡️ **Next / 下一步**: File 4 of 7

---

### Polynomial Transform

# 04 — Polynomial Transform / 数据变换

**Chapter 23 — File 4 of 7 / 第23章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **visualize a polynomial features transform of the sonar dataset**.

本脚本演示 **visualize a polynomial features transform of the sonar dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — visualize a polynomial features transform of the sonar dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
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
## Step 4 — perform a polynomial features transform of the dataset

```python
trans = PolynomialFeatures(degree=3)
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: visualize a polynomial features transform of the sonar dataset 是机器学习中的常用技术。  
  *visualize a polynomial features transform of the sonar dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Polynomial Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# visualize a polynomial features transform of the sonar dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import DataFrame
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
# load dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('sonar.csv', header=None)
# retrieve just the numeric input values
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values[:, :-1]
# perform a polynomial features transform of the dataset
trans = PolynomialFeatures(degree=3)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Polynomial Model Evaluation

# 05 — Polynomial Model Evaluation / 模型评估

**Chapter 23 — File 5 of 7 / 第23章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluate knn on the sonar dataset with polynomial features transform**.

本脚本演示 **evaluate knn on the sonar dataset with polynomial features transform**。

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
## Step 1 — evaluate knn on the sonar dataset with polynomial features transform

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
from sklearn.preprocessing import PolynomialFeatures
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
trans = PolynomialFeatures(degree=3)
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

- **概念**: evaluate knn on the sonar dataset with polynomial features transform 是机器学习中的常用技术。  
  *evaluate knn on the sonar dataset with polynomial features transform is a common technique in machine learning.*

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
# Polynomial Model Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate knn on the sonar dataset with polynomial features transform
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
from sklearn.preprocessing import PolynomialFeatures
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
trans = PolynomialFeatures(degree=3)
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

➡️ **Next / 下一步**: File 6 of 7

---

### Compare Degree Num Features

# 06 — Compare Degree Num Features / 特征工程

**Chapter 23 — File 6 of 7 / 第23章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **compare the effect of the degree on the number of created features**.

本脚本演示 **compare the effect of the degree on the number of created features**。

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
## Step 1 — compare the effect of the degree on the number of created features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
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
## Step 6 — define dataset

```python
X, y = get_dataset('sonar.csv')
```

---
## Step 7 — calculate change in number of features

```python
num_features = list()
# 生成整数序列 / Generate integer sequence
degrees = [i for i in range(1, 6)]
for d in degrees:
```

---
## Step 8 — create transform

```python
trans = PolynomialFeatures(degree=d)
```

---
## Step 9 — fit and transform

```python
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
data = trans.fit_transform(X)
```

---
## Step 10 — record number of features

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_features.append(data.shape[1])
```

---
## Step 11 — summarize

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Degree: %d, Features: %d' % (d, data.shape[1]))
```

---
## Step 12 — plot degree vs number of features

```python
pyplot.plot(degrees, num_features)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare the effect of the degree on the number of created features 是机器学习中的常用技术。  
  *compare the effect of the degree on the number of created features is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
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
# Compare Degree Num Features / 特征工程
# Complete Code / 完整代码
# ===============================

# compare the effect of the degree on the number of created features
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PolynomialFeatures
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

# define dataset
X, y = get_dataset('sonar.csv')
# calculate change in number of features
num_features = list()
# 生成整数序列 / Generate integer sequence
degrees = [i for i in range(1, 6)]
for d in degrees:
	# create transform
	trans = PolynomialFeatures(degree=d)
	# fit and transform
 # 拟合并转换数据（一步完成） / Fit and transform data (one step)
	data = trans.fit_transform(X)
	# record number of features
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	num_features.append(data.shape[1])
	# summarize
 # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
	print('Degree: %d, Features: %d' % (d, data.shape[1]))
# plot degree vs number of features
pyplot.plot(degrees, num_features)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Compare Performace With Degree



---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **7 code files** demonstrating chapter 23.

本章包含 **7 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_example_transform.ipynb` — Example Transform
  2. `02_load_dataset.ipynb` — Load Dataset
  3. `03_model_evaluation.ipynb` — Model Evaluation
  4. `04_polynomial_transform.ipynb` — Polynomial Transform
  5. `05_polynomial_model_evaluation.ipynb` — Polynomial Model Evaluation
  6. `06_compare_degree_num_features.ipynb` — Compare Degree Num Features
  7. `07_compare_performace_with_degree.ipynb` — Compare Performace With Degree

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
