# 机器学习数据准备 / Data Preparation for ML
## Chapter 19

---

### Demo Ordinal Encoding



---

### Demo Onehot Encoding



---

### Demo Dummy Encoding



---

### Load Dataset

# 04 — Load Dataset / 04 Load Dataset

**Chapter 19 — File 4 of 8 / 第19章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load and summarize the dataset**.

本脚本演示 **load and summarize the dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load and summarize the dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
```

---
## Step 3 — retrieve the array of data

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
```

---
## Step 4 — separate into input and output columns

```python
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
```

---
## Step 5 — summarize

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Input', X.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Output', y.shape)
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

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Dataset / 04 Load Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
# retrieve the array of data
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
# summarize
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Input', X.shape)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Output', y.shape)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Ordinal Encode Dataset

# 05 — Ordinal Encode Dataset / 数据编码

**Chapter 19 — File 5 of 8 / 第19章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **ordinal encode the breast cancer dataset**.

本脚本演示 **ordinal encode the breast cancer dataset**。

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
## Step 1 — ordinal encode the breast cancer dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
```

---
## Step 3 — retrieve the array of data

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
```

---
## Step 4 — separate into input and output columns

```python
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
```

---
## Step 5 — ordinal encode input variables

```python
ordinal_encoder = OrdinalEncoder()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = ordinal_encoder.fit_transform(X)
```

---
## Step 6 — ordinal encode target variable

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
y = label_encoder.fit_transform(y)
```

---
## Step 7 — summarize the transformed data

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Input', X.shape)
# 打印输出 / Print output
print(X[:5, :])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Output', y.shape)
# 打印输出 / Print output
print(y[:5])
```

---
## Learning Notes / 学习笔记

- **概念**: ordinal encode the breast cancer dataset 是机器学习中的常用技术。  
  *ordinal encode the breast cancer dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ordinal Encode Dataset / 数据编码
# Complete Code / 完整代码
# ===============================

# ordinal encode the breast cancer dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
# retrieve the array of data
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = ordinal_encoder.fit_transform(X)
# ordinal encode target variable
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
y = label_encoder.fit_transform(y)
# summarize the transformed data
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Input', X.shape)
# 打印输出 / Print output
print(X[:5, :])
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Output', y.shape)
# 打印输出 / Print output
print(y[:5])
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Ordinal Encode Evaluate Model

# 06 — Ordinal Encode Evaluate Model / 模型评估

**Chapter 19 — File 6 of 8 / 第19章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate logistic regression on the breast cancer dataset with an ordinal encoding**.

本脚本演示 **evaluate logistic regression on the breast cancer dataset with an ordinal encoding**。

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
## Step 1 — evaluate logistic regression on the breast cancer dataset with an ordinal encoding

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
```

---
## Step 3 — retrieve the array of data

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
```

---
## Step 4 — separate into input and output columns

```python
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
```

---
## Step 5 — split the dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 6 — ordinal encode input variables

```python
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = ordinal_encoder.transform(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = ordinal_encoder.transform(X_test)
```

---
## Step 7 — ordinal encode target variable

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_train = label_encoder.transform(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_test = label_encoder.transform(y_test)
```

---
## Step 8 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
```

---
## Step 9 — fit on the training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 10 — predict on test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 11 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate logistic regression on the breast cancer dataset with an ordinal encoding 是机器学习中的常用技术。  
  *evaluate logistic regression on the breast cancer dataset with an ordinal encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ordinal Encode Evaluate Model / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate logistic regression on the breast cancer dataset with an ordinal encoding
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OrdinalEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
# retrieve the array of data
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
# split the dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = ordinal_encoder.transform(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = ordinal_encoder.transform(X_test)
# ordinal encode target variable
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_train = label_encoder.transform(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_test = label_encoder.transform(y_test)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# fit on the training set
# 训练模型 / Train the model
model.fit(X_train, y_train)
# predict on test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Onehot Encode Dataset



---

### Onehot Encode Evaluate Model

# 08 — Onehot Encode Evaluate Model / 模型评估

**Chapter 19 — File 8 of 8 / 第19章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate logistic regression on the breast cancer dataset with a one-hot encoding**.

本脚本演示 **evaluate logistic regression on the breast cancer dataset with a one-hot encoding**。

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
## Step 1 — evaluate logistic regression on the breast cancer dataset with a one-hot encoding

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
```

---
## Step 3 — retrieve the array of data

```python
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
```

---
## Step 4 — separate into input and output columns

```python
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
```

---
## Step 5 — split the dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 6 — one-hot encode input variables

```python
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = onehot_encoder.transform(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = onehot_encoder.transform(X_test)
```

---
## Step 7 — ordinal encode target variable

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_train = label_encoder.transform(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_test = label_encoder.transform(y_test)
```

---
## Step 8 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
```

---
## Step 9 — fit on the training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 10 — predict on test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 11 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate logistic regression on the breast cancer dataset with a one-hot encoding 是机器学习中的常用技术。  
  *evaluate logistic regression on the breast cancer dataset with a one-hot encoding is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Onehot Encode Evaluate Model / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate logistic regression on the breast cancer dataset with a one-hot encoding
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import OneHotEncoder
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('breast-cancer.csv', header=None)
# retrieve the array of data
# 转换为NumPy数组 / Convert to NumPy array
data = dataset.values
# separate into input and output columns
# 转换数据类型 / Convert data type
X = data[:, :-1].astype(str)
# 转换数据类型 / Convert data type
y = data[:, -1].astype(str)
# split the dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# one-hot encode input variables
# 独热编码：每个类别变成0/1向量 / One-hot encode: each category becomes 0/1 vector
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = onehot_encoder.transform(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = onehot_encoder.transform(X_test)
# ordinal encode target variable
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_train = label_encoder.transform(y_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
y_test = label_encoder.transform(y_test)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# fit on the training set
# 训练模型 / Train the model
model.fit(X_train, y_train)
# predict on test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.2f' % (accuracy*100))
```

---

### Chapter Summary / 章节总结



---
