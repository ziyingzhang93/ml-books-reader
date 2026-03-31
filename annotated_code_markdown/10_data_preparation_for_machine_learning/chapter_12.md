# 机器学习数据准备 / Data Preparation for ML
## Chapter 12

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 12 — File 1 of 7 / 第12章 — 第1个文件（共7个）**

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
## Step 3 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv(filename, header=None)
```

---
## Step 4 — retrieve array

```python
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 5 — split into input and output variables

```python
X = dataset[:, :-1]
	y = dataset[:,-1]
```

---
## Step 6 — format all fields as string

```python
# 转换数据类型 / Convert data type
X = X.astype(str)
	return X, y
```

---
## Step 7 — load the dataset

```python
X, y = load_dataset('breast-cancer.csv')
```

---
## Step 8 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 9 — summarize

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
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
	# load the dataset
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	data = read_csv(filename, header=None)
	# retrieve array
 # 转换为NumPy数组 / Convert to NumPy array
	dataset = data.values
	# split into input and output variables
	X = dataset[:, :-1]
	y = dataset[:,-1]
	# format all fields as string
 # 转换数据类型 / Convert data type
	X = X.astype(str)
	return X, y

# load the dataset
X, y = load_dataset('breast-cancer.csv')
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

➡️ **Next / 下一步**: File 2 of 7

---

### Load And Encode



---

### Chi Squared



---

### Mutual Information

# 04 — Mutual Information / 04 Mutual Information

**Chapter 12 — File 4 of 7 / 第12章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of mutual information feature selection for categorical data**.

本脚本演示 **example of mutual information feature selection for categorical data**。

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
## Step 1 — example of mutual information feature selection for categorical data

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
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
```

---
## Step 6 — format all fields as string

```python
# 转换数据类型 / Convert data type
X = X.astype(str)
	return X, y
```

---
## Step 7 — prepare input data

```python
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
```

---
## Step 8 — prepare target

```python
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
```

---
## Step 9 — feature selection

```python
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
```

---
## Step 10 — load the dataset

```python
X, y = load_dataset('breast-cancer.csv')
```

---
## Step 11 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 12 — prepare input data

```python
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
```

---
## Step 13 — prepare output data

```python
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
```

---
## Step 14 — feature selection

```python
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
```

---
## Step 15 — what are scores for the features

```python
# 获取长度 / Get length
for i in range(len(fs.scores_)):
 # 打印输出 / Print output
	print('Feature %d: %f' % (i, fs.scores_[i]))
```

---
## Step 16 — plot the scores

```python
# 获取长度 / Get length
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of mutual information feature selection for categorical data 是机器学习中的常用技术。  
  *example of mutual information feature selection for categorical data is a common technique in machine learning.*

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
# Mutual Information / 04 Mutual Information
# Complete Code / 完整代码
# ===============================

# example of mutual information feature selection for categorical data
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
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
	# format all fields as string
 # 转换数据类型 / Convert data type
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
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

➡️ **Next / 下一步**: File 5 of 7

---

### Evaluate All Features

# 05 — Evaluate All Features / 特征工程

**Chapter 12 — File 5 of 7 / 第12章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model using all input features**.

本脚本演示 **evaluation of a model using all input features**。

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
## Step 1 — evaluation of a model using all input features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
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
```

---
## Step 6 — format all fields as string

```python
# 转换数据类型 / Convert data type
X = X.astype(str)
	return X, y
```

---
## Step 7 — prepare input data

```python
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
```

---
## Step 8 — prepare target

```python
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
```

---
## Step 9 — load the dataset

```python
X, y = load_dataset('breast-cancer.csv')
```

---
## Step 10 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 11 — prepare input data

```python
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
```

---
## Step 12 — prepare output data

```python
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
```

---
## Step 13 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_enc, y_train_enc)
```

---
## Step 14 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_enc)
```

---
## Step 15 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
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
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
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
	# format all fields as string
 # 转换数据类型 / Convert data type
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_enc, y_train_enc)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_enc)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Evaluate Chi Squared

# 06 — Evaluate Chi Squared / 模型评估

**Chapter 12 — File 6 of 7 / 第12章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model fit using chi squared input features**.

本脚本演示 **evaluation of a model fit using chi squared input features**。

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
## Step 1 — evaluation of a model fit using chi squared input features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import chi2
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
```

---
## Step 6 — format all fields as string

```python
# 转换数据类型 / Convert data type
X = X.astype(str)
	return X, y
```

---
## Step 7 — prepare input data

```python
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
```

---
## Step 8 — prepare target

```python
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
```

---
## Step 9 — feature selection

```python
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k=4)
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs
```

---
## Step 10 — load the dataset

```python
X, y = load_dataset('breast-cancer.csv')
```

---
## Step 11 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 12 — prepare input data

```python
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
```

---
## Step 13 — prepare output data

```python
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
```

---
## Step 14 — feature selection

```python
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
```

---
## Step 15 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train_enc)
```

---
## Step 16 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
```

---
## Step 17 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model fit using chi squared input features 是机器学习中的常用技术。  
  *evaluation of a model fit using chi squared input features is a common technique in machine learning.*

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
# Evaluate Chi Squared / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluation of a model fit using chi squared input features
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import chi2
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
	# format all fields as string
 # 转换数据类型 / Convert data type
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k=4)
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train_enc)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Evaluate Mututal Information

# 07 — Evaluate Mututal Information / 模型评估

**Chapter 12 — File 7 of 7 / 第12章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluation of a model fit using mutual information input features**.

本脚本演示 **evaluation of a model fit using mutual information input features**。

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
## Step 1 — evaluation of a model fit using mutual information input features

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
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
```

---
## Step 6 — format all fields as string

```python
# 转换数据类型 / Convert data type
X = X.astype(str)
	return X, y
```

---
## Step 7 — prepare input data

```python
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
```

---
## Step 8 — prepare target

```python
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
```

---
## Step 9 — feature selection

```python
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k=4)
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs
```

---
## Step 10 — load the dataset

```python
X, y = load_dataset('breast-cancer.csv')
```

---
## Step 11 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 12 — prepare input data

```python
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
```

---
## Step 13 — prepare output data

```python
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
```

---
## Step 14 — feature selection

```python
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
```

---
## Step 15 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train_enc)
```

---
## Step 16 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
```

---
## Step 17 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluation of a model fit using mutual information input features 是机器学习中的常用技术。  
  *evaluation of a model fit using mutual information input features is a common technique in machine learning.*

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
# Evaluate Mututal Information / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluation of a model fit using mutual information input features
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import mutual_info_classif
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
	# format all fields as string
 # 转换数据类型 / Convert data type
	X = X.astype(str)
	return X, y

# prepare input data
def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_enc = oe.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
 # 将类别标签编码为数字 / Encode categorical labels to numbers
	le = LabelEncoder()
	le.fit(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_train_enc = le.transform(y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_classif, k=4)
	fs.fit(X_train, y_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_train_fs = fs.transform(X_train)
 # 用已拟合的模型转换数据 / Transform data with fitted model
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# feature selection
X_train_fs, X_test_fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(X_train_fs, y_train_enc)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test_fs)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test_enc, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

### Chapter Summary / 章节总结

# Chapter 12 Summary / 第12章总结

## Theme / 主题: Chapter 12 / Chapter 12

This chapter contains **7 code files** demonstrating chapter 12.

本章包含 **7 个代码文件**，演示Chapter 12。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_load_and_encode.ipynb` — Load And Encode
  3. `03_chi_squared.ipynb` — Chi Squared
  4. `04_mutual_information.ipynb` — Mutual Information
  5. `05_evaluate_all_features.ipynb` — Evaluate All Features
  6. `06_evaluate_chi_squared.ipynb` — Evaluate Chi Squared
  7. `07_evaluate_mututal_information.ipynb` — Evaluate Mututal Information

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 12) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 12）是机器学习流水线中的基础构建块。

---
