# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 29

---

### Load Summarize

# 01 — Load Summarize / 01 Load Summarize

**Chapter 29 — File 1 of 7 / 第29章 — 第1个文件（共7个）**

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
filename = 'mammography.csv'
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# define the dataset location
filename = 'mammography.csv'
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
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

---

➡️ **Next / 下一步**: File 2 of 7

---

### Histograms

# 02 — Histograms / 02 Histograms

**Chapter 29 — File 2 of 7 / 第29章 — 第2个文件（共7个）**

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
filename = 'mammography.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# define the dataset location
filename = 'mammography.csv'
# load the csv file as a data frame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv(filename, header=None)
# histograms of all variables
df.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Scatterplots

# 03 — Scatterplots / 03 Scatterplots

**Chapter 29 — File 3 of 7 / 第29章 — 第3个文件（共7个）**

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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'mammography.csv'
```

---
## Step 3 — load the csv file as a data frame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv(filename, header=None)
```

---
## Step 4 — define a mapping of class values to colors

```python
color_dict = {"'-1'":'blue', "'1'":'red'}
```

---
## Step 5 — map each row to a color based on the class value

```python
# 转换为NumPy数组 / Convert to NumPy array
colors = [color_dict[str(x)] for x in df.values[:, -1]]
```

---
## Step 6 — pairwise scatter plots of all numerical variables

```python
scatter_matrix(df, diagonal='kde', color=colors)
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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Scatterplots / 03 Scatterplots
# Complete Code / 完整代码
# ===============================

# create pairwise scatter plots of numeric input variables
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import scatter_matrix
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# define the dataset location
filename = 'mammography.csv'
# load the csv file as a data frame
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv(filename, header=None)
# define a mapping of class values to colors
color_dict = {"'-1'":'blue', "'1'":'red'}
# map each row to a color based on the class value
# 转换为NumPy数组 / Convert to NumPy array
colors = [color_dict[str(x)] for x in df.values[:, -1]]
# pairwise scatter plots of all numerical variables
scatter_matrix(df, diagonal='kde', color=colors)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Baseline

# 04 — Baseline / 04 Baseline

**Chapter 29 — File 4 of 7 / 第29章 — 第4个文件（共7个）**

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
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = data.values
```

---
## Step 5 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 6 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 8 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 9 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 10 — define the location of the dataset

```python
full_path = 'mammography.csv'
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 12 — summarize the loaded dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape, Counter(y))
```

---
## Step 13 — define the reference model

```python
model = DummyClassifier(strategy='stratified')
```

---
## Step 14 — evaluate the model

```python
scores = evaluate_model(X, y, model)
```

---
## Step 15 — summarize performance

```python
# 打印输出 / Print output
print('Mean ROC AUC: %.3f (%.3f)' % (mean(scores), std(scores)))
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
# Baseline / 04 Baseline
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
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(full_path, header=None)
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
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	return scores

# define the location of the dataset
full_path = 'mammography.csv'
# load the dataset
X, y = load_dataset(full_path)
# summarize the loaded dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape, Counter(y))
# define the reference model
model = DummyClassifier(strategy='stratified')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
# 打印输出 / Print output
print('Mean ROC AUC: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Evaluate Models

# 05 — Evaluate Models / 模型评估

**Chapter 29 — File 5 of 7 / 第29章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **spot check machine learning algorithms on the mammography dataset**.

本脚本演示 **spot check machine learning algorithms on the mammography dataset**。

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
## Step 1 — spot check machine learning algorithms on the mammography dataset

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
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
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
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = data.values
```

---
## Step 5 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 6 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 8 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 9 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 10 — define models to test

```python
def get_models():
	models, names = list(), list()
```

---
## Step 11 — LR

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
models.append(LogisticRegression(solver='lbfgs'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
```

---
## Step 12 — SVM

```python
# 支持向量机 / Support Vector Machine
models.append(SVC(gamma='scale'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('SVM')
```

---
## Step 13 — Bagging

```python
# 添加元素到列表末尾 / Append element to list end
models.append(BaggingClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('BAG')
```

---
## Step 14 — RF

```python
# 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
models.append(RandomForestClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('RF')
```

---
## Step 15 — GBM

```python
# 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
models.append(GradientBoostingClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('GBM')
	return models, names
```

---
## Step 16 — define the location of the dataset

```python
full_path = 'mammography.csv'
```

---
## Step 17 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 18 — define models

```python
models, names = get_models()
results = list()
```

---
## Step 19 — evaluate each model

```python
# 获取长度 / Get length
for i in range(len(models)):
```

---
## Step 20 — evaluate the model and store results

```python
scores = evaluate_model(X, y, models[i])
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
```

---
## Step 21 — summarize and store

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
```

---
## Step 22 — plot the results

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: spot check machine learning algorithms on the mammography dataset 是机器学习中的常用技术。  
  *spot check machine learning algorithms on the mammography dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Models / 模型评估
# Complete Code / 完整代码
# ===============================

# spot check machine learning algorithms on the mammography dataset
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
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import GradientBoostingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(full_path, header=None)
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
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# LR
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='lbfgs'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('LR')
	# SVM
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale'))
 # 添加元素到列表末尾 / Append element to list end
	names.append('SVM')
	# Bagging
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('BAG')
	# RF
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('RF')
	# GBM
 # 梯度提升：逐步修正前一棵树的错误 / Gradient Boosting: iteratively correct errors
	models.append(GradientBoostingClassifier(n_estimators=1000))
 # 添加元素到列表末尾 / Append element to list end
	names.append('GBM')
	return models, names

# define the location of the dataset
full_path = 'mammography.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
# 获取长度 / Get length
for i in range(len(models)):
	# evaluate the model and store results
	scores = evaluate_model(X, y, models[i])
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

➡️ **Next / 下一步**: File 6 of 7

---

### Cost Sensitive Models



---

### Make Predictions

# 07 — Make Predictions / 07 Make Predictions

**Chapter 29 — File 7 of 7 / 第29章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **fit a model and make predictions for the mammography dataset**.

本脚本演示 **fit a model and make predictions for the mammography dataset**。

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
## Step 1 — fit a model and make predictions for the mammography dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
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
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = data.values
```

---
## Step 5 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 6 — label encode the target variable to have the classes 0 and 1

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — define the location of the dataset

```python
full_path = 'mammography.csv'
```

---
## Step 8 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 9 — define model to evaluate

```python
# 支持向量机 / Support Vector Machine
model = SVC(gamma='scale', class_weight='balanced')
```

---
## Step 10 — power transform then fit model

```python
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t',PowerTransformer()), ('m',model)])
```

---
## Step 11 — fit the model

```python
pipeline.fit(X, y)
```

---
## Step 12 — evaluate on some no cancer cases (known class 0)

```python
# 打印输出 / Print output
print('No Cancer:')
data = [[0.23001961,5.0725783,-0.27606055,0.83244412,-0.37786573,0.4803223],
	[0.15549112,-0.16939038,0.67065219,-0.85955255,-0.37786573,-0.94572324],
	[-0.78441482,-0.44365372,5.6747053,-0.85955255,-0.37786573,-0.94572324]]
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
# 打印输出 / Print output
print('>Predicted=%d (expected 0)' % (label))
```

---
## Step 16 — evaluate on some cancer (known class 1)

```python
# 打印输出 / Print output
print('Cancer:')
data = [[2.0158239,0.15353258,-0.32114211,2.1923706,-0.37786573,0.96176503],
	[2.3191888,0.72860087,-0.50146835,-0.85955255,-0.37786573,-0.94572324],
	[0.19224721,-0.2003556,-0.230979,1.2003796,2.2620867,1.132403]]
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
# 打印输出 / Print output
print('>Predicted=%d (expected 1)' % (label))
```

---
## Learning Notes / 学习笔记

- **概念**: fit a model and make predictions for the mammography dataset 是机器学习中的常用技术。  
  *fit a model and make predictions for the mammography dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Predictions / 07 Make Predictions
# Complete Code / 完整代码
# ===============================

# fit a model and make predictions for the mammography dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import PowerTransformer
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(full_path, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	y = LabelEncoder().fit_transform(y)
	return X, y

# define the location of the dataset
full_path = 'mammography.csv'
# load the dataset
X, y = load_dataset(full_path)
# define model to evaluate
# 支持向量机 / Support Vector Machine
model = SVC(gamma='scale', class_weight='balanced')
# power transform then fit model
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('t',PowerTransformer()), ('m',model)])
# fit the model
pipeline.fit(X, y)
# evaluate on some no cancer cases (known class 0)
# 打印输出 / Print output
print('No Cancer:')
data = [[0.23001961,5.0725783,-0.27606055,0.83244412,-0.37786573,0.4803223],
	[0.15549112,-0.16939038,0.67065219,-0.85955255,-0.37786573,-0.94572324],
	[-0.78441482,-0.44365372,5.6747053,-0.85955255,-0.37786573,-0.94572324]]
for row in data:
	# make prediction
	yhat = pipeline.predict([row])
	# get the label
	label = yhat[0]
	# summarize
 # 打印输出 / Print output
	print('>Predicted=%d (expected 0)' % (label))
# evaluate on some cancer (known class 1)
# 打印输出 / Print output
print('Cancer:')
data = [[2.0158239,0.15353258,-0.32114211,2.1923706,-0.37786573,0.96176503],
	[2.3191888,0.72860087,-0.50146835,-0.85955255,-0.37786573,-0.94572324],
	[0.19224721,-0.2003556,-0.230979,1.2003796,2.2620867,1.132403]]
for row in data:
	# make prediction
	yhat = pipeline.predict([row])
	# get the label
	label = yhat[0]
	# summarize
 # 打印输出 / Print output
	print('>Predicted=%d (expected 1)' % (label))
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **7 code files** demonstrating chapter 29.

本章包含 **7 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `01_load_summarize.ipynb` — Load Summarize
  2. `02_histograms.ipynb` — Histograms
  3. `03_scatterplots.ipynb` — Scatterplots
  4. `04_baseline.ipynb` — Baseline
  5. `05_evaluate_models.ipynb` — Evaluate Models
  6. `06_cost_sensitive_models.ipynb` — Cost Sensitive Models
  7. `07_make_predictions.ipynb` — Make Predictions

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
