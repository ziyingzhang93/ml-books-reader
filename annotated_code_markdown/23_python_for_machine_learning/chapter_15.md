# Python 机器学习 / Python for Machine Learning
## Chapter 15

---

### Iris



---

### Normal

# 02 — Normal / 02 Normal

**Chapter 15 — File 2 of 6 / 第15章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **Load dataset**.

本脚本演示 **Load dataset**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — Load dataset

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, names=names)
```

---
## Step 3 — Split-out validation dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
y = array[:,4]
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)
```

---
## Step 4 — Train

```python
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
clf.fit(X_train, y_train)
```

---
## Step 5 — Test

```python
score = clf.score(X_val, y_val)
# 打印输出 / Print output
print("Validation accuracy", score)
```

---
## Learning Notes / 学习笔记

- **概念**: Load dataset 是机器学习中的常用技术。  
  *Load dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Normal / 02 Normal
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, names=names)

# Split-out validation dataset
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
y = array[:,4]
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)

# Train
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
clf = Pipeline([('scaler',StandardScaler()), ('classifier',SVC())])
clf.fit(X_train, y_train)

# Test
score = clf.score(X_val, y_val)
# 打印输出 / Print output
print("Validation accuracy", score)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Toggle

# 06 — Toggle / 06 Toggle

**Chapter 15 — File 3 of 6 / 第15章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **toggle between options**.

本脚本演示 **toggle between options**。

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
```

---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

---
## Step 2 — toggle between options

```python
SCALER = "maxmin"    # "standard", "maxmin", or None
CLASSIFIER = "cart"  # "svc" or "cart"
```

---
## Step 3 — Load dataset

```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, names=names)
```

---
## Step 4 — Split-out validation dataset

```python
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
y = array[:,4]
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)
```

---
## Step 5 — Create model

```python
if CLASSIFIER == "svc":
    # 支持向量机 / Support Vector Machine
    model = SVC()
elif CLASSIFIER == "cart":
    # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
    model = DecisionTreeClassifier()
else:
    raise NotImplementedError

if SCALER == "standard":
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',model)])
elif SCALER == "maxmin":
    # 归一化到[0,1]范围 / Normalize to [0,1] range
    clf = Pipeline([('scaler',MinMaxScaler()), ('classifier',model)])
elif SCALER == None:
    clf = model
else:
    raise NotImplementedError
```

---
## Step 6 — Train and test

```python
clf.fit(X_train, y_train)
score = clf.score(X_val, y_val)
# 打印输出 / Print output
print("Validation accuracy", score)
```

---
## Learning Notes / 学习笔记

- **概念**: toggle between options 是机器学习中的常用技术。  
  *toggle between options is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `SVM` | 支持向量机 | Support Vector Machine |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Toggle / 06 Toggle
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# toggle between options
SCALER = "maxmin"    # "standard", "maxmin", or None
CLASSIFIER = "cart"  # "svc" or "cart"

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = read_csv(url, names=names)

# Split-out validation dataset
# 转换为NumPy数组 / Convert to NumPy array
array = dataset.values
X = array[:,0:4]
y = array[:,4]
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)

# Create model
if CLASSIFIER == "svc":
    # 支持向量机 / Support Vector Machine
    model = SVC()
elif CLASSIFIER == "cart":
    # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
    model = DecisionTreeClassifier()
else:
    raise NotImplementedError

if SCALER == "standard":
    # 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
    clf = Pipeline([('scaler',StandardScaler()), ('classifier',model)])
elif SCALER == "maxmin":
    # 归一化到[0,1]范围 / Normalize to [0,1] range
    clf = Pipeline([('scaler',MinMaxScaler()), ('classifier',model)])
elif SCALER == None:
    clf = model
else:
    raise NotImplementedError

# Train and test
clf.fit(X_train, y_train)
score = clf.score(X_val, y_val)
# 打印输出 / Print output
print("Validation accuracy", score)
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### String Hack



---

### Functions



---

### Wrapper



---

### Chapter Summary / 章节总结



---
