# 机器学习数据准备 / Data Preparation for ML
## Chapter 06

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 06 — File 1 of 6 / 第06章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **generate gaussian data**.

本脚本演示 **generate gaussian data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — generate gaussian data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
```

---
## Step 2 — seed the random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 3 — generate univariate observations

```python
data = 5 * randn(10000) + 50
```

---
## Step 4 — summarize

```python
# 打印输出 / Print output
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
```

---
## Learning Notes / 学习笔记

- **概念**: generate gaussian data 是机器学习中的常用技术。  
  *generate gaussian data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# generate gaussian data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# seed the random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# summarize
# 打印输出 / Print output
print('mean=%.3f stdv=%.3f' % (mean(data), std(data)))
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Outliers Stdev

# 02 — Outliers Stdev / 异常值检测

**Chapter 06 — File 2 of 6 / 第06章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **identify outliers with standard deviation**.

本脚本演示 **identify outliers with standard deviation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — identify outliers with standard deviation

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
```

---
## Step 2 — seed the random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 3 — generate univariate observations

```python
data = 5 * randn(10000) + 50
```

---
## Step 4 — calculate summary statistics

```python
data_mean, data_std = mean(data), std(data)
```

---
## Step 5 — define outliers

```python
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
```

---
## Step 6 — identify outliers

```python
outliers = [x for x in data if x < lower or x > upper]
# 打印输出 / Print output
print('Identified outliers: %d' % len(outliers))
```

---
## Step 7 — remove outliers

```python
outliers_removed = [x for x in data if x >= lower and x <= upper]
# 打印输出 / Print output
print('Non-outlier observations: %d' % len(outliers_removed))
```

---
## Learning Notes / 学习笔记

- **概念**: identify outliers with standard deviation 是机器学习中的常用技术。  
  *identify outliers with standard deviation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Outliers Stdev / 异常值检测
# Complete Code / 完整代码
# ===============================

# identify outliers with standard deviation
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# seed the random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate summary statistics
data_mean, data_std = mean(data), std(data)
# define outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
# 打印输出 / Print output
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
# 打印输出 / Print output
print('Non-outlier observations: %d' % len(outliers_removed))
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Outliers Iqr

# 03 — Outliers Iqr / 异常值检测

**Chapter 06 — File 3 of 6 / 第06章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **identify outliers with interquartile range**.

本脚本演示 **identify outliers with interquartile range**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — identify outliers with interquartile range

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import percentile
```

---
## Step 2 — seed the random number generator

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
```

---
## Step 3 — generate univariate observations

```python
data = 5 * randn(10000) + 50
```

---
## Step 4 — calculate interquartile range

```python
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
# 打印输出 / Print output
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
```

---
## Step 5 — calculate the outlier cutoff

```python
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
```

---
## Step 6 — identify outliers

```python
outliers = [x for x in data if x < lower or x > upper]
# 打印输出 / Print output
print('Identified outliers: %d' % len(outliers))
```

---
## Step 7 — remove outliers

```python
outliers_removed = [x for x in data if x >= lower and x <= upper]
# 打印输出 / Print output
print('Non-outlier observations: %d' % len(outliers_removed))
```

---
## Learning Notes / 学习笔记

- **概念**: identify outliers with interquartile range 是机器学习中的常用技术。  
  *identify outliers with interquartile range is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Outliers Iqr / 异常值检测
# Complete Code / 完整代码
# ===============================

# identify outliers with interquartile range
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import seed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import percentile
# seed the random number generator
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
# generate univariate observations
data = 5 * randn(10000) + 50
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
# 打印输出 / Print output
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
# 打印输出 / Print output
print('Identified outliers: %d' % len(outliers))
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]
# 打印输出 / Print output
print('Non-outlier observations: %d' % len(outliers_removed))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Load Housing Dataset

# 04 — Load Housing Dataset / 04 Load Housing Dataset

**Chapter 06 — File 4 of 6 / 第06章 — 第4个文件（共6个）**

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
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
```

---
## Step 3 — retrieve the array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
```

---
## Step 4 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 5 — summarize the shape of the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Step 6 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 7 — summarize the shape of the train and test sets

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
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
# Load Housing Dataset / 04 Load Housing Dataset
# Complete Code / 完整代码
# ===============================

# load and summarize the dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
# retrieve the array
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# summarize the shape of the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the train and test sets
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Evaluate Model

# 05 — Evaluate Model / 模型评估

**Chapter 06 — File 5 of 6 / 第06章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate model on the raw dataset**.

本脚本演示 **evaluate model on the raw dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — evaluate model on the raw dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
```

---
## Step 3 — retrieve the array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
```

---
## Step 4 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 5 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 6 — fit the model

```python
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 8 — evaluate predictions

```python
mae = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('MAE: %.3f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate model on the raw dataset 是机器学习中的常用技术。  
  *evaluate model on the raw dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
# Evaluate Model / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate model on the raw dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
# retrieve the array
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('MAE: %.3f' % mae)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Evaluate Remove Outliers

# 06 — Evaluate Remove Outliers / 模型评估

**Chapter 06 — File 6 of 6 / 第06章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **evaluate model on training dataset with outliers removed**.

本脚本演示 **evaluate model on training dataset with outliers removed**。

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
## Step 1 — evaluate model on training dataset with outliers removed

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import LocalOutlierFactor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
```

---
## Step 3 — retrieve the array

```python
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
```

---
## Step 4 — split into input and output elements

```python
X, y = data[:, :-1], data[:, -1]
```

---
## Step 5 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 6 — summarize the shape of the training dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
```

---
## Step 7 — identify outliers in the training dataset

```python
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
```

---
## Step 8 — select all rows that are not outliers

```python
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
```

---
## Step 9 — summarize the shape of the updated training dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
```

---
## Step 10 — fit the model

```python
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 11 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 12 — evaluate predictions

```python
mae = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('MAE: %.3f' % mae)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate model on training dataset with outliers removed 是机器学习中的常用技术。  
  *evaluate model on training dataset with outliers removed is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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
# Evaluate Remove Outliers / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate model on training dataset with outliers removed
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import LocalOutlierFactor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
df = read_csv('housing.csv', header=None)
# retrieve the array
# 转换为NumPy数组 / Convert to NumPy array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('MAE: %.3f' % mae)
```

---

### Chapter Summary / 章节总结

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **6 code files** demonstrating chapter 06.

本章包含 **6 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_outliers_stdev.ipynb` — Outliers Stdev
  3. `03_outliers_iqr.ipynb` — Outliers Iqr
  4. `04_load_housing_dataset.ipynb` — Load Housing Dataset
  5. `05_evaluate_model.ipynb` — Evaluate Model
  6. `06_evaluate_remove_outliers.ipynb` — Evaluate Remove Outliers

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
