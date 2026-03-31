# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 28

---

### Load Summarize

# 01 — Load Summarize / 01 Load Summarize

**Chapter 28 — File 1 of 6 / 第28章 — 第1个文件（共6个）**

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
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🏗️ 定义模型 / Define Model
```

---
## Step 1 — load and summarize the dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
```

---
## Step 2 — define the dataset location

```python
filename = 'german.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, header=None)
```

---
## Step 4 — summarize the shape of the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataframe.shape)
```

---
## Step 5 — summarize the class distribution

```python
# 转换为NumPy数组 / Convert to NumPy array
target = dataframe.values[:,-1]
counter = Counter(target)
# 获取字典的键值对 / Get dict key-value pairs
for k,v in counter.items():
 # 获取长度 / Get length
	per = v / len(target) * 100
 # 打印输出 / Print output
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# define the dataset location
filename = 'german.csv'
# load the csv file as a data frame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, header=None)
# summarize the shape of the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataframe.shape)
# summarize the class distribution
# 转换为NumPy数组 / Convert to NumPy array
target = dataframe.values[:,-1]
counter = Counter(target)
# 获取字典的键值对 / Get dict key-value pairs
for k,v in counter.items():
 # 获取长度 / Get length
	per = v / len(target) * 100
 # 打印输出 / Print output
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Histograms

# 02 — Histograms / 02 Histograms

**Chapter 28 — File 2 of 6 / 第28章 — 第2个文件（共6个）**

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'german.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv(filename, header=None)
```

---
## Step 4 — select columns with numerical data types

```python
# 获取列名 / Get column names
num_ix = df.select_dtypes(include=['int64', 'float64']).columns
```

---
## Step 5 — select a subset of the dataframe with the chosen columns

```python
subset = df[num_ix]
```

---
## Step 6 — create a histogram plot of each numeric variable

```python
ax = subset.hist()
```

---
## Step 7 — disable axis labels to avoid the clutter

```python
# 展平为一维数组 / Flatten to 1D array
for axis in ax.flatten():
	axis.set_xticklabels([])
	axis.set_yticklabels([])
```

---
## Step 8 — show the plot

```python
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# define the dataset location
filename = 'german.csv'
# load the csv file as a data frame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv(filename, header=None)
# select columns with numerical data types
# 获取列名 / Get column names
num_ix = df.select_dtypes(include=['int64', 'float64']).columns
# select a subset of the dataframe with the chosen columns
subset = df[num_ix]
# create a histogram plot of each numeric variable
ax = subset.hist()
# disable axis labels to avoid the clutter
# 展平为一维数组 / Flatten to 1D array
for axis in ax.flatten():
	axis.set_xticklabels([])
	axis.set_yticklabels([])
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Baseline



---

### Evaluate Models

# 04 — Evaluate Models / 模型评估

**Chapter 28 — File 4 of 6 / 第28章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **spot check machine learning algorithms on the german credit dataset**.

本脚本演示 **spot check machine learning algorithms on the german credit dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — spot check machine learning algorithms on the german credit dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import fbeta_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.gaussian_process import GaussianProcessClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(full_path, header=None)
```

---
## Step 4 — split into inputs and outputs

```python
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
 # 删除指定列或行 / Drop specified columns or rows
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
```

---
## Step 5 — select categorical and numerical features

```python
# 获取列名 / Get column names
cat_ix = X.select_dtypes(include=['object', 'bool']).columns
 # 获取列名 / Get column names
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
```

---
## Step 6 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
 # 转换为NumPy数组 / Convert to NumPy array
	return X.values, y, cat_ix, num_ix
```

---
## Step 7 — calculate f2-measure

```python
def f2_measure(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=2)
```

---
## Step 8 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 9 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 10 — define the model evaluation metric

```python
metric = make_scorer(f2_measure)
```

---
## Step 11 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 12 — define models to test

```python
def get_models():
	models, names = list(), list()
```

---
## Step 13 — LR

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(LogisticRegression(solver='liblinear'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
```

---
## Step 14 — LDA

```python
# 添加元素到列表末尾 / Append element to list end
models.append(LinearDiscriminantAnalysis())
 # 添加元素到列表末尾 / Append element to list end
	names.append('LDA')
```

---
## Step 15 — NB

```python
# 添加元素到列表末尾 / Append element to list end
models.append(GaussianNB())
 # 添加元素到列表末尾 / Append element to list end
	names.append('NB')
```

---
## Step 16 — GPC

```python
# 添加元素到列表末尾 / Append element to list end
models.append(GaussianProcessClassifier())
 # 添加元素到列表末尾 / Append element to list end
	names.append('GPC')
```

---
## Step 17 — SVM

```python
# 支持向量机 / Support Vector Machine
models.append(SVC(gamma='scale'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('SVM')
	return models, names
```

---
## Step 18 — define the location of the dataset

```python
full_path = 'german.csv'
```

---
## Step 19 — load the dataset

```python
X, y, cat_ix, num_ix = load_dataset(full_path)
```

---
## Step 20 — define models

```python
models, names = get_models()
results = list()
```

---
## Step 21 — evaluate each model

```python
# 获取长度 / Get length
for i in range(len(models)):
```

---
## Step 22 — one hot encode categorical, normalize numerical

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
```

---
## Step 23 — wrap the model in a pipeline

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t',ct),('m',models[i])])
```

---
## Step 24 — evaluate the model and store results

```python
scores = evaluate_model(X, y, pipeline)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
```

---
## Step 25 — summarize and store

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
```

---
## Step 26 — plot the results

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: spot check machine learning algorithms on the german credit dataset 是机器学习中的常用技术。  
  *spot check machine learning algorithms on the german credit dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `SVM` | 支持向量机 | Support Vector Machine |
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
# Evaluate Models / 模型评估
# Complete Code / 完整代码
# ===============================

# spot check machine learning algorithms on the german credit dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import fbeta_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.gaussian_process import GaussianProcessClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(full_path, header=None)
	# split into inputs and outputs
 # 获取列名 / Get column names
	last_ix = len(dataframe.columns) - 1
 # 删除指定列或行 / Drop specified columns or rows
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	# select categorical and numerical features
 # 获取列名 / Get column names
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
 # 获取列名 / Get column names
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y)
 # 转换为NumPy数组 / Convert to NumPy array
	return X.values, y, cat_ix, num_ix

# calculate f2-measure
def f2_measure(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=2)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(f2_measure)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# LR
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='liblinear'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
	# LDA
 # 添加元素到列表末尾 / Append element to list end
	models.append(LinearDiscriminantAnalysis())
 # 添加元素到列表末尾 / Append element to list end
	names.append('LDA')
	# NB
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianNB())
 # 添加元素到列表末尾 / Append element to list end
	names.append('NB')
	# GPC
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianProcessClassifier())
 # 添加元素到列表末尾 / Append element to list end
	names.append('GPC')
	# SVM
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('SVM')
	return models, names

# define the location of the dataset
full_path = 'german.csv'
# load the dataset
X, y, cat_ix, num_ix = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
# 获取长度 / Get length
for i in range(len(models)):
	# one hot encode categorical, normalize numerical
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
	# wrap the model in a pipeline
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	pipeline = Pipeline(steps=[('t',ct),('m',models[i])])
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
	# summarize and store
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Data Sampling Models

# 05 — Data Sampling Models / 05 Data Sampling Models

**Chapter 28 — File 5 of 6 / 第28章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate undersampling with logistic regression on the imbalanced german credit dataset**.

本脚本演示 **evaluate undersampling with logistic regression on the imbalanced german credit dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
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
## Step 1 — evaluate undersampling with logistic regression on the imbalanced german credit dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import fbeta_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(full_path, header=None)
```

---
## Step 4 — split into inputs and outputs

```python
# 获取列名 / Get column names
last_ix = len(dataframe.columns) - 1
 # 删除指定列或行 / Drop specified columns or rows
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
```

---
## Step 5 — select categorical and numerical features

```python
# 获取列名 / Get column names
cat_ix = X.select_dtypes(include=['object', 'bool']).columns
 # 获取列名 / Get column names
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
```

---
## Step 6 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
 # 转换为NumPy数组 / Convert to NumPy array
	return X.values, y, cat_ix, num_ix
```

---
## Step 7 — calculate f2-measure

```python
def f2_measure(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=2)
```

---
## Step 8 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 9 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 10 — define the model evaluation metric

```python
metric = make_scorer(f2_measure)
```

---
## Step 11 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 12 — define undersampling models to test

```python
def get_models():
	models, names = list(), list()
```

---
## Step 13 — TL

```python
# 添加元素到列表末尾 / Append element to list end
models.append(TomekLinks())
 # 添加元素到列表末尾 / Append element to list end
	names.append('TL')
```

---
## Step 14 — ENN

```python
# 添加元素到列表末尾 / Append element to list end
models.append(EditedNearestNeighbours())
 # 添加元素到列表末尾 / Append element to list end
	names.append('ENN')
```

---
## Step 15 — RENN

```python
# 添加元素到列表末尾 / Append element to list end
models.append(RepeatedEditedNearestNeighbours())
 # 添加元素到列表末尾 / Append element to list end
	names.append('RENN')
```

---
## Step 16 — OSS

```python
# 添加元素到列表末尾 / Append element to list end
models.append(OneSidedSelection())
 # 添加元素到列表末尾 / Append element to list end
	names.append('OSS')
```

---
## Step 17 — NCR

```python
# 添加元素到列表末尾 / Append element to list end
models.append(NeighbourhoodCleaningRule())
 # 添加元素到列表末尾 / Append element to list end
	names.append('NCR')
	return models, names
```

---
## Step 18 — define the location of the dataset

```python
full_path = 'german.csv'
```

---
## Step 19 — load the dataset

```python
X, y, cat_ix, num_ix = load_dataset(full_path)
```

---
## Step 20 — define models

```python
models, names = get_models()
results = list()
```

---
## Step 21 — evaluate each model

```python
# 获取长度 / Get length
for i in range(len(models)):
```

---
## Step 22 — define model to evaluate

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear', class_weight='balanced')
```

---
## Step 23 — one hot encode categorical, normalize numerical

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
```

---
## Step 24 — scale, then undersample, then fit model

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t',ct), ('s', models[i]), ('m',model)])
```

---
## Step 25 — evaluate the model and store results

```python
scores = evaluate_model(X, y, pipeline)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
```

---
## Step 26 — summarize and store

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
```

---
## Step 27 — plot the results

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate undersampling with logistic regression on the imbalanced german credit dataset 是机器学习中的常用技术。  
  *evaluate undersampling with logistic regression on the imbalanced german credit dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
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
# Data Sampling Models / 05 Data Sampling Models
# Complete Code / 完整代码
# ===============================

# evaluate undersampling with logistic regression on the imbalanced german credit dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.compose import ColumnTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import fbeta_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	dataframe = read_csv(full_path, header=None)
	# split into inputs and outputs
 # 获取列名 / Get column names
	last_ix = len(dataframe.columns) - 1
 # 删除指定列或行 / Drop specified columns or rows
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	# select categorical and numerical features
 # 获取列名 / Get column names
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
 # 获取列名 / Get column names
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y)
 # 转换为NumPy数组 / Convert to NumPy array
	return X.values, y, cat_ix, num_ix

# calculate f2-measure
def f2_measure(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=2)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(f2_measure)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define undersampling models to test
def get_models():
	models, names = list(), list()
	# TL
 # 添加元素到列表末尾 / Append element to list end
	models.append(TomekLinks())
 # 添加元素到列表末尾 / Append element to list end
	names.append('TL')
	# ENN
 # 添加元素到列表末尾 / Append element to list end
	models.append(EditedNearestNeighbours())
 # 添加元素到列表末尾 / Append element to list end
	names.append('ENN')
	# RENN
 # 添加元素到列表末尾 / Append element to list end
	models.append(RepeatedEditedNearestNeighbours())
 # 添加元素到列表末尾 / Append element to list end
	names.append('RENN')
	# OSS
 # 添加元素到列表末尾 / Append element to list end
	models.append(OneSidedSelection())
 # 添加元素到列表末尾 / Append element to list end
	names.append('OSS')
	# NCR
 # 添加元素到列表末尾 / Append element to list end
	models.append(NeighbourhoodCleaningRule())
 # 添加元素到列表末尾 / Append element to list end
	names.append('NCR')
	return models, names

# define the location of the dataset
full_path = 'german.csv'
# load the dataset
X, y, cat_ix, num_ix = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
# 获取长度 / Get length
for i in range(len(models)):
	# define model to evaluate
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear', class_weight='balanced')
	# one hot encode categorical, normalize numerical
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
	# scale, then undersample, then fit model
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	pipeline = Pipeline(steps=[('t',ct), ('s', models[i]), ('m',model)])
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
	# summarize and store
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Make Predictions



---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **6 code files** demonstrating chapter 28.

本章包含 **6 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `01_load_summarize.ipynb` — Load Summarize
  2. `02_histograms.ipynb` — Histograms
  3. `03_baseline.ipynb` — Baseline
  4. `04_evaluate_models.ipynb` — Evaluate Models
  5. `05_data_sampling_models.ipynb` — Data Sampling Models
  6. `06_make_predictions.ipynb` — Make Predictions

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
