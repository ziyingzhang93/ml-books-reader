# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 27

---

### Load Summarize

# 01 — Load Summarize / 01 Load Summarize

**Chapter 27 — File 1 of 7 / 第27章 — 第1个文件（共7个）**

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
filename = 'oil-spill.csv'
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
filename = 'oil-spill.csv'
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

➡️ **Next / 下一步**: File 2 of 7

---

### Histograms

# 02 — Histograms / 02 Histograms

**Chapter 27 — File 2 of 7 / 第27章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **create histograms of each variable**.

本脚本演示 **create histograms of each variable**。

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
## Step 1 — create histograms of each variable

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'oil-spill.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, header=None)
```

---
## Step 4 — create a histogram plot of each variable

```python
ax = dataframe.hist()
```

---
## Step 5 — disable axis labels

```python
# 展平为一维数组 / Flatten to 1D array
for axis in ax.flatten():
	axis.set_title('')
	axis.set_xticklabels([])
	axis.set_yticklabels([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: create histograms of each variable 是机器学习中的常用技术。  
  *create histograms of each variable is a common technique in machine learning.*

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

# create histograms of each variable
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# define the dataset location
filename = 'oil-spill.csv'
# load the csv file as a data frame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, header=None)
# create a histogram plot of each variable
ax = dataframe.hist()
# disable axis labels
# 展平为一维数组 / Flatten to 1D array
for axis in ax.flatten():
	axis.set_title('')
	axis.set_xticklabels([])
	axis.set_yticklabels([])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Baseline

# 03 — Baseline / 03 Baseline

**Chapter 27 — File 3 of 7 / 第27章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **test harness and baseline model evaluation**.

本脚本演示 **test harness and baseline model evaluation**。

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
## Step 1 — test harness and baseline model evaluation

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(full_path, header=None)
```

---
## Step 4 — drop unused columns

```python
# 删除指定列或行 / Drop specified columns or rows
data.drop(22, axis=1, inplace=True)
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(0, axis=1, inplace=True)
```

---
## Step 5 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = data.values
```

---
## Step 6 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 7 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
	return X, y
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
## Step 10 — define the model evaluation the metric

```python
metric = make_scorer(geometric_mean_score)
```

---
## Step 11 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 12 — define the location of the dataset

```python
full_path = 'oil-spill.csv'
```

---
## Step 13 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 14 — summarize the loaded dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape, Counter(y))
```

---
## Step 15 — define the reference model

```python
model = DummyClassifier(strategy='uniform')
```

---
## Step 16 — evaluate the model

```python
scores = evaluate_model(X, y, model)
```

---
## Step 17 — summarize performance

```python
# 打印输出 / Print output
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
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Baseline / 03 Baseline
# Complete Code / 完整代码
# ===============================

# test harness and baseline model evaluation
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.metrics import geometric_mean_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(full_path, header=None)
	# drop unused columns
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(22, axis=1, inplace=True)
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(0, axis=1, inplace=True)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y)
	return X, y

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation the metric
	metric = make_scorer(geometric_mean_score)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define the location of the dataset
full_path = 'oil-spill.csv'
# load the dataset
X, y = load_dataset(full_path)
# summarize the loaded dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape, Counter(y))
# define the reference model
model = DummyClassifier(strategy='uniform')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
# 打印输出 / Print output
print('Mean G-Mean: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Probabilistic Models



---

### Balanced Models



---

### Data Sampling Models

# 06 — Data Sampling Models / 06 Data Sampling Models

**Chapter 27 — File 6 of 7 / 第27章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **compare data sampling with logistic regression on the oil spill dataset**.

本脚本演示 **compare data sampling with logistic regression on the oil spill dataset**。

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
## Step 1 — compare data sampling with logistic regression on the oil spill dataset

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
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
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
data = read_csv(full_path, header=None)
```

---
## Step 4 — drop unused columns

```python
# 删除指定列或行 / Drop specified columns or rows
data.drop(22, axis=1, inplace=True)
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(0, axis=1, inplace=True)
```

---
## Step 5 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = data.values
```

---
## Step 6 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 7 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
	return X, y
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
## Step 10 — define the model evaluation the metric

```python
metric = make_scorer(geometric_mean_score)
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
## Step 13 — SMOTEENN

```python
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
	steps = [('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
```

---
## Step 14 — SMOTEENN + Norm

```python
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	steps = [('t', MinMaxScaler()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Norm')
```

---
## Step 15 — SMOTEENN + Std

```python
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
	steps = [('t', StandardScaler()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Std')
```

---
## Step 16 — SMOTEENN + Power

```python
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Power')
	return models, names
```

---
## Step 17 — define the location of the dataset

```python
full_path = 'oil-spill.csv'
```

---
## Step 18 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 19 — define models

```python
models, names = get_models()
```

---
## Step 20 — evaluate each model

```python
results = list()
# 获取长度 / Get length
for i in range(len(models)):
```

---
## Step 21 — evaluate the model and store results

```python
scores = evaluate_model(X, y, models[i])
```

---
## Step 22 — summarize and store

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
```

---
## Step 23 — plot the results

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare data sampling with logistic regression on the oil spill dataset 是机器学习中的常用技术。  
  *compare data sampling with logistic regression on the oil spill dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
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
# Data Sampling Models / 06 Data Sampling Models
# Complete Code / 完整代码
# ===============================

# compare data sampling with logistic regression on the oil spill dataset
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
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import make_scorer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(full_path, header=None)
	# drop unused columns
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(22, axis=1, inplace=True)
 # 删除指定列或行 / Drop specified columns or rows
	data.drop(0, axis=1, inplace=True)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y)
	return X, y

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation the metric
	metric = make_scorer(geometric_mean_score)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# SMOTEENN
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
	steps = [('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
	# SMOTEENN + Norm
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	steps = [('t', MinMaxScaler()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Norm')
	# SMOTEENN + Std
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
	steps = [('t', StandardScaler()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Std')
	# SMOTEENN + Power
	sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 归一化到[0,1]范围 / Normalize to [0,1] range
	steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()), ('e', sampling), ('m', model)]
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	models.append(Pipeline(steps=steps))
 # 添加元素到列表末尾 / Append element to list end
	names.append('Power')
	return models, names

# define the location of the dataset
full_path = 'oil-spill.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
# evaluate each model
results = list()
# 获取长度 / Get length
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
	# summarize and store
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Make Prediction



---

### Chapter Summary / 章节总结

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **7 code files** demonstrating chapter 27.

本章包含 **7 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `01_load_summarize.ipynb` — Load Summarize
  2. `02_histograms.ipynb` — Histograms
  3. `03_baseline.ipynb` — Baseline
  4. `04_probabilistic_models.ipynb` — Probabilistic Models
  5. `05_balanced_models.ipynb` — Balanced Models
  6. `06_data_sampling_models.ipynb` — Data Sampling Models
  7. `07_make_prediction.ipynb` — Make Prediction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---
