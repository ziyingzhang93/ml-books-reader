# Python 机器学习实战 / ML Mastery with Python
## Chapter 14

---

### Chapter Summary / 章节总结



---

### Feature Union Model Pipeline

# 01 — Feature Union Model Pipeline / 特征工程

**Chapter 14 — File 1 of 2 / 第14章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Create a pipeline that extracts features from the data then creates a model**.

本脚本演示 **Create a pipeline that extracts features from the data then creates a model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
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
## Step 1 — Create a pipeline that extracts features from the data then creates a model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import FeatureUnion
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — create feature union

```python
features = []
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
features.append(('pca', PCA(n_components=3)))
# 添加元素到列表末尾 / Append element to list end
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
```

---
## Step 4 — create pipeline

```python
estimators = []
# 添加元素到列表末尾 / Append element to list end
estimators.append(('feature_union', feature_union))
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
estimators.append(('logistic', LogisticRegression(solver='liblinear')))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(estimators)
```

---
## Step 5 — evaluate pipeline

```python
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Create a pipeline that extracts features from the data then creates a model 是机器学习中的常用技术。  
  *Create a pipeline that extracts features from the data then creates a model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Feature Union Model Pipeline / 特征工程
# Complete Code / 完整代码
# ===============================

# Create a pipeline that extracts features from the data then creates a model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import FeatureUnion
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.feature_selection import SelectKBest
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create feature union
features = []
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
features.append(('pca', PCA(n_components=3)))
# 添加元素到列表末尾 / Append element to list end
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
# 添加元素到列表末尾 / Append element to list end
estimators.append(('feature_union', feature_union))
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
estimators.append(('logistic', LogisticRegression(solver='liblinear')))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Standardize Model Pipeline

# 01 — Standardize Model Pipeline / 管道

**Chapter 14 — File 2 of 2 / 第14章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Create a pipeline that standardizes the data then creates a model**.

本脚本演示 **Create a pipeline that standardizes the data then creates a model**。

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
## Step 1 — Create a pipeline that standardizes the data then creates a model

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

---
## Step 2 — load data

```python
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
```

---
## Step 3 — create pipeline

```python
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('lda', LinearDiscriminantAnalysis()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(estimators)
```

---
## Step 4 — evaluate pipeline

```python
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
## Learning Notes / 学习笔记

- **概念**: Create a pipeline that standardizes the data then creates a model 是机器学习中的常用技术。  
  *Create a pipeline that standardizes the data then creates a model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Standardize Model Pipeline / 管道
# Complete Code / 完整代码
# ===============================

# Create a pipeline that standardizes the data then creates a model
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv(filename, names=names)
# 转换为NumPy数组 / Convert to NumPy array
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
# 标准化：均值=0，标准差=1 / Standardize: mean=0, std=1
estimators.append(('standardize', StandardScaler()))
# 添加元素到列表末尾 / Append element to list end
estimators.append(('lda', LinearDiscriminantAnalysis()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, Y, cv=kfold)
# 打印输出 / Print output
print(results.mean())
```

---
