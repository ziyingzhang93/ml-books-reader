# 机器学习数据准备 / Data Preparation for ML
## Chapter 07

---

### Load Dataset

# 01 — Load Dataset / 01 Load Dataset

**Chapter 07 — File 1 of 8 / 第07章 — 第1个文件（共8个）**

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
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — summarize the dataset

```python
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
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
| `describe()` | 统计摘要信息 | Statistical summary |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

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
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# summarize the dataset
# 生成统计摘要（均值、标准差等） / Generate statistical summary (mean, std, etc.)
print(dataset.describe())
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Load Show Rows

# 02 — Load Show Rows / 02 Load Show Rows

**Chapter 07 — File 2 of 8 / 第07章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load the dataset and review rows**.

本脚本演示 **load the dataset and review rows**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load the dataset and review rows

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — summarize the first 20 rows of data

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---
## Learning Notes / 学习笔记

- **概念**: load the dataset and review rows 是机器学习中的常用技术。  
  *load the dataset and review rows is a common technique in machine learning.*

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
# Load Show Rows / 02 Load Show Rows
# Complete Code / 完整代码
# ===============================

# load the dataset and review rows
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# summarize the first 20 rows of data
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Num Missing

# 03 — Num Missing / 缺失值处理

**Chapter 07 — File 3 of 8 / 第07章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of summarizing the number of missing values for each variable**.

本脚本演示 **example of summarizing the number of missing values for each variable**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of summarizing the number of missing values for each variable

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — count the number of missing values for each column

```python
num_missing = (dataset[[1,2,3,4,5]] == 0).sum()
```

---
## Step 4 — report the results

```python
# 打印输出 / Print output
print(num_missing)
```

---
## Learning Notes / 学习笔记

- **概念**: example of summarizing the number of missing values for each variable 是机器学习中的常用技术。  
  *example of summarizing the number of missing values for each variable is a common technique in machine learning.*

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
# Num Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# example of summarizing the number of missing values for each variable
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# count the number of missing values for each column
num_missing = (dataset[[1,2,3,4,5]] == 0).sum()
# report the results
# 打印输出 / Print output
print(num_missing)
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Mark Missing

# 04 — Mark Missing / 缺失值处理

**Chapter 07 — File 4 of 8 / 第07章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of marking missing values with nan values**.

本脚本演示 **example of marking missing values with nan values**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of marking missing values with nan values

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — replace '0' values with 'nan'

```python
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
```

---
## Step 4 — count the number of nan values in each column

```python
# 打印输出 / Print output
print(dataset.isnull().sum())
```

---
## Learning Notes / 学习笔记

- **概念**: example of marking missing values with nan values 是机器学习中的常用技术。  
  *example of marking missing values with nan values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mark Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# example of marking missing values with nan values
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# count the number of nan values in each column
# 打印输出 / Print output
print(dataset.isnull().sum())
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Review Marked Missing

# 05 — Review Marked Missing / 缺失值处理

**Chapter 07 — File 5 of 8 / 第07章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of review data with missing values marked with a nan**.

本脚本演示 **example of review data with missing values marked with a nan**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — example of review data with missing values marked with a nan

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — replace '0' values with 'nan'

```python
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
```

---
## Step 4 — summarize the first 20 rows of data

```python
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---
## Learning Notes / 学习笔记

- **概念**: example of review data with missing values marked with a nan 是机器学习中的常用技术。  
  *example of review data with missing values marked with a nan is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Review Marked Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# example of review data with missing values marked with a nan
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# summarize the first 20 rows of data
# 查看前几行数据（快速预览） / View first rows (quick preview)
print(dataset.head(20))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Missing Values Cause Errors

# 06 — Missing Values Cause Errors / 缺失值处理

**Chapter 07 — File 6 of 8 / 第07章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example where missing values cause errors**.

本脚本演示 **example where missing values cause errors**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — example where missing values cause errors

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — replace '0' values with 'nan'

```python
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
```

---
## Step 4 — split dataset into inputs and outputs

```python
# 转换为NumPy数组 / Convert to NumPy array
values = dataset.values
X = values[:,0:8]
y = values[:,8]
```

---
## Step 5 — define the model

```python
model = LinearDiscriminantAnalysis()
```

---
## Step 6 — define the model evaluation procedure

```python
cv = KFold(n_splits=3, shuffle=True, random_state=1)
```

---
## Step 7 — evaluate the model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

---
## Step 8 — report the mean performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f' % result.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: example where missing values cause errors 是机器学习中的常用技术。  
  *example where missing values cause errors is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Missing Values Cause Errors / 缺失值处理
# Complete Code / 完整代码
# ===============================

# example where missing values cause errors
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# split dataset into inputs and outputs
# 转换为NumPy数组 / Convert to NumPy array
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# define the model
model = LinearDiscriminantAnalysis()
# define the model evaluation procedure
cv = KFold(n_splits=3, shuffle=True, random_state=1)
# evaluate the model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
# report the mean performance
# 打印输出 / Print output
print('Accuracy: %.3f' % result.mean())
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Remove Missing

# 07 — Remove Missing / 缺失值处理

**Chapter 07 — File 7 of 8 / 第07章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of removing rows that contain missing values**.

本脚本演示 **example of removing rows that contain missing values**。

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
## Step 1 — example of removing rows that contain missing values

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — summarize the shape of the raw data

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Step 4 — replace '0' values with 'nan'

```python
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
```

---
## Step 5 — drop rows with missing values

```python
# 删除含缺失值的行 / Drop rows with missing values
dataset.dropna(inplace=True)
```

---
## Step 6 — summarize the shape of the data with missing rows removed

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: example of removing rows that contain missing values 是机器学习中的常用技术。  
  *example of removing rows that contain missing values is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `dropna` | 删除缺失值 | Drop missing values |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Remove Missing / 缺失值处理
# Complete Code / 完整代码
# ===============================

# example of removing rows that contain missing values
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# summarize the shape of the raw data
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# drop rows with missing values
# 删除含缺失值的行 / Drop rows with missing values
dataset.dropna(inplace=True)
# summarize the shape of the data with missing rows removed
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(dataset.shape)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Remove Missing Evaluate Model

# 08 — Remove Missing Evaluate Model / 模型评估

**Chapter 07 — File 8 of 8 / 第07章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate model on data after rows with missing data are removed**.

本脚本演示 **evaluate model on data after rows with missing data are removed**。

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
## Step 1 — evaluate model on data after rows with missing data are removed

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
```

---
## Step 2 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
```

---
## Step 3 — replace '0' values with 'nan'

```python
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
```

---
## Step 4 — drop rows with missing values

```python
# 删除含缺失值的行 / Drop rows with missing values
dataset.dropna(inplace=True)
```

---
## Step 5 — split dataset into inputs and outputs

```python
# 转换为NumPy数组 / Convert to NumPy array
values = dataset.values
X = values[:,0:8]
y = values[:,8]
```

---
## Step 6 — define the model

```python
model = LinearDiscriminantAnalysis()
```

---
## Step 7 — define the model evaluation procedure

```python
cv = KFold(n_splits=3, shuffle=True, random_state=1)
```

---
## Step 8 — evaluate the model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

---
## Step 9 — report the mean performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f' % result.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate model on data after rows with missing data are removed 是机器学习中的常用技术。  
  *evaluate model on data after rows with missing data are removed is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `dropna` | 删除缺失值 | Drop missing values |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Remove Missing Evaluate Model / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate model on data after rows with missing data are removed
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import nan
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv('pima-indians-diabetes.csv', header=None)
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# drop rows with missing values
# 删除含缺失值的行 / Drop rows with missing values
dataset.dropna(inplace=True)
# split dataset into inputs and outputs
# 转换为NumPy数组 / Convert to NumPy array
values = dataset.values
X = values[:,0:8]
y = values[:,8]
# define the model
model = LinearDiscriminantAnalysis()
# define the model evaluation procedure
cv = KFold(n_splits=3, shuffle=True, random_state=1)
# evaluate the model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
# report the mean performance
# 打印输出 / Print output
print('Accuracy: %.3f' % result.mean())
```

---

### Chapter Summary / 章节总结

# Chapter 07 Summary / 第07章总结

## Theme / 主题: Chapter 07 / Chapter 07

This chapter contains **8 code files** demonstrating chapter 07.

本章包含 **8 个代码文件**，演示Chapter 07。

---
## Evolution / 演化路线

  1. `01_load_dataset.ipynb` — Load Dataset
  2. `02_load_show_rows.ipynb` — Load Show Rows
  3. `03_num_missing.ipynb` — Num Missing
  4. `04_mark_missing.ipynb` — Mark Missing
  5. `05_review_marked_missing.ipynb` — Review Marked Missing
  6. `06_missing_values_cause_errors.ipynb` — Missing Values Cause Errors
  7. `07_remove_missing.ipynb` — Remove Missing
  8. `08_remove_missing_evaluate_model.ipynb` — Remove Missing Evaluate Model

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 07) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 07）是机器学习流水线中的基础构建块。

---
