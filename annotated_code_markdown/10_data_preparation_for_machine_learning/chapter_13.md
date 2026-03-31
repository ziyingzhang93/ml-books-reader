# 机器学习数据准备 / Data Preparation for ML
## Chapter 13

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 13 — File 1 of 8 / 第13章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load and summarize the dataset**.

本脚本演示 **load and summarize the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data

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
## Step 1 — load and summarize the dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 7 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 8 — summarize

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X_train.shape, y_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X_train.shape, y_train.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Test', X_test.shape, y_test.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Anova Selection

# 02 — Anova Selection / 特征选择

**Chapter 13 — File 2 of 8 / 第13章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of anova f-test feature selection for numerical data**.

本脚本演示 **example of anova f-test feature selection for numerical data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
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
## Step 1 — example of anova f-test feature selection for numerical data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 7 — configure to select all features

```python
fs = SelectKBest(score_func=f_classif, k='all')
```

---
## Step 8 — learn relationship from training data

```python
fs.fit(X_train, y_train)
```

---
## Step 9 — transform train input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train_fs = fs.transform(X_train)
```

---
## Step 10 — transform test input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 12 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 13 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
```

---
## Step 14 — what are scores for the features

```python
# 获取长度 / Get length
for i in range(len(fs.scores_)):
 # 打印输出 / Print output
	print('Feature %d: %f' % (i, fs.scores_[i]))
```

---
## Step 15 — plot the scores

```python
# 获取长度 / Get length
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of anova f-test feature selection for numerical data 是机器学习中的常用技术。  
  *example of anova f-test feature selection for numerical data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Anova Selection / 特征选择
# Complete Code / 完整代码
# ===============================

# example of anova f-test feature selection for numerical data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
	# transform test input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
# 获取长度 / Get length
for i in range(len(fs.scores_)):
 # 打印输出 / Print output
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
# 获取长度 / Get length
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Mutual Information Selection

# 03 — Mutual Information Selection / 特征选择

**Chapter 13 — File 3 of 8 / 第13章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of mutual information feature selection for numerical input data**.

本脚本演示 **example of mutual information feature selection for numerical input data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 7 — configure to select all features

```python
fs = SelectKBest(score_func=mutual_info_classif, k='all')
```

---
## Step 8 — learn relationship from training data

```python
fs.fit(X_train, y_train)
```

---
## Step 9 — transform train input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train_fs = fs.transform(X_train)
```

---
## Step 10 — transform test input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 12 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 13 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
```

---
## Step 14 — what are scores for the features

```python
# 获取长度 / Get length
for i in range(len(fs.scores_)):
 # 打印输出 / Print output
	print('Feature %d: %f' % (i, fs.scores_[i]))
```

---
## Step 15 — plot the scores

```python
# 获取长度 / Get length
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
	# transform test input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
# 获取长度 / Get length
for i in range(len(fs.scores_)):
 # 打印输出 / Print output
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
# 获取长度 / Get length
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Evaluate All Features

# 04 — Evaluate All Features / 特征工程

**Chapter 13 — File 4 of 8 / 第13章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using all input features**.

本脚本演示 **evaluation of a model using all input features**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 训练模型 / Train the model
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 7 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 8 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 9 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 10 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
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
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
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
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Evaluate Anova

# 05 — Evaluate Anova / 模型评估

**Chapter 13 — File 5 of 8 / 第13章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using 4 features chosen with anova f-test**.

本脚本演示 **evaluation of a model using 4 features chosen with anova f-test**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
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
## Step 1 — evaluation of a model using 4 features chosen with anova f-test

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 7 — configure to select a subset of features

```python
fs = SelectKBest(score_func=f_classif, k=4)
```

---
## Step 8 — learn relationship from training data

```python
fs.fit(X_train, y_train)
```

---
## Step 9 — transform train input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train_fs = fs.transform(X_train)
```

---
## Step 10 — transform test input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 12 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 13 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
```

---
## Step 14 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train)
```

---
## Step 15 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
```

---
## Step 16 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using 4 features chosen with anova f-test 是机器学习中的常用技术。  
  *evaluation of a model using 4 features chosen with anova f-test is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Anova / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluation of a model using 4 features chosen with anova f-test
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=f_classif, k=4)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
	# transform test input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Evaluate Mutual Information

# 06 — Evaluate Mutual Information / 模型评估

**Chapter 13 — File 6 of 8 / 第13章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using 4 features chosen with mutual information**.

本脚本演示 **evaluation of a model using 4 features chosen with mutual information**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model
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
## Step 1 — evaluation of a model using 4 features chosen with mutual information

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — feature selection

```python
def select_features(X_train, y_train, X_test):
```

---
## Step 7 — configure to select a subset of features

```python
fs = SelectKBest(score_func=mutual_info_classif, k=4)
```

---
## Step 8 — learn relationship from training data

```python
fs.fit(X_train, y_train)
```

---
## Step 9 — transform train input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train_fs = fs.transform(X_train)
```

---
## Step 10 — transform test input data

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 11 — load the dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 12 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 13 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
```

---
## Step 14 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train)
```

---
## Step 15 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
```

---
## Step 16 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model using 4 features chosen with mutual information 是机器学习中的常用技术。  
  *evaluation of a model using 4 features chosen with mutual information is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Mutual Information / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluation of a model using 4 features chosen with mutual information
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=mutual_info_classif, k=4)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
	# transform test input data
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Grid Search Anova

# 07 — Grid Search Anova / 07 Grid Search Anova

**Chapter 13 — File 7 of 8 / 第13章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare different numbers of features selected using anova f-test**.

本脚本演示 **compare different numbers of features selected using anova f-test**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
```

---
## Step 1 — compare different numbers of features selected using anova f-test

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — define dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 7 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — define the pipeline to evaluate

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
fs = SelectKBest(score_func=f_classif)
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
```

---
## Step 9 — define the grid

```python
grid = dict()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
grid['anova__k'] = [i+1 for i in range(X.shape[1])]
```

---
## Step 10 — define the grid search

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
```

---
## Step 11 — perform the search

```python
results = search.fit(X, y)
```

---
## Step 12 — summarize best

```python
# 打印输出 / Print output
print('Best Mean Accuracy: %.3f' % results.best_score_)
# 打印输出 / Print output
print('Best Config: %s' % results.best_params_)
```

---
## Learning Notes / 学习笔记

- **概念**: compare different numbers of features selected using anova f-test 是机器学习中的常用技术。  
  *compare different numbers of features selected using anova f-test is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Anova / 07 Grid Search Anova
# Complete Code / 完整代码
# ===============================

# compare different numbers of features selected using anova f-test
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# define dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
fs = SelectKBest(score_func=f_classif)
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
# define the grid
grid = dict()
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
grid['anova__k'] = [i+1 for i in range(X.shape[1])]
# define the grid search
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
# 打印输出 / Print output
print('Best Mean Accuracy: %.3f' % results.best_score_)
# 打印输出 / Print output
print('Best Config: %s' % results.best_params_)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Compare Performance Num Features

# 08 — Compare Performance Num Features / 特征工程

**Chapter 13 — File 8 of 8 / 第13章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare different numbers of features selected using anova f-test**.

本脚本演示 **compare different numbers of features selected using anova f-test**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — compare different numbers of features selected using anova f-test

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
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load the dataset

```python
def load_dataset(filename):
```

---
## Step 3 — load the dataset as a pandas DataFrame

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve numpy array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input (X) and output (y) variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y
```

---
## Step 6 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 7 — define dataset

```python
X, y = load_dataset('pima-indians-diabetes.csv')
```

---
## Step 8 — define number of features to evaluate

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_features = [i+1 for i in range(X.shape[1])]
```

---
## Step 9 — enumerate each number of features

```python
results = list()
for k in num_features:
```

---
## Step 10 — create pipeline

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='liblinear')
	fs = SelectKBest(score_func=f_classif, k=k)
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(pipeline, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
```

---
## Step 12 — summarize the results

```python
# 打印输出 / Print output
print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
```

---
## Step 13 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare different numbers of features selected using anova f-test 是机器学习中的常用技术。  
  *compare different numbers of features selected using anova f-test is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Performance Num Features / 特征工程
# Complete Code / 完整代码
# ===============================

# compare different numbers of features selected using anova f-test
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
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import f_classif
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# load the dataset
def load_dataset(filename):
	# load the dataset as a pandas DataFrame
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve numpy array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input (X) and output (y) variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	return X, y

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = load_dataset('pima-indians-diabetes.csv')
# define number of features to evaluate
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
num_features = [i+1 for i in range(X.shape[1])]
# enumerate each number of features
results = list()
for k in num_features:
	# create pipeline
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
	fs = SelectKBest(score_func=f_classif, k=k)
 # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
	pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
	# evaluate the model
	scores = evaluate_model(pipeline, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
	# summarize the results
 # 打印输出 / Print output
	print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 13 Summary / 第13章总结

## Theme / 主题: Chapter 13 / Chapter 13

This chapter contains **8 code files** demonstrating chapter 13.

本章包含 **8 个代码文件**，演示Chapter 13。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_anova_selection.ipynb` — Anova Selection
  3. `03_mutual_information_selection.ipynb` — Mutual Information Selection
  4. `04_evaluate_all_features.ipynb` — Evaluate All Features
  5. `05_evaluate_anova.ipynb` — Evaluate Anova
  6. `06_evaluate_mutual_information.ipynb` — Evaluate Mutual Information
  7. `07_grid_search_anova.ipynb` — Grid Search Anova
  8. `08_compare_performance_num_features.ipynb` — Compare Performance Num Features

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 13) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 13）是机器学习流水线中的基础构建块。

---
