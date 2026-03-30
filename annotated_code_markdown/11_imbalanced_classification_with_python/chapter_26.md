# 不平衡分类
## Chapter 26

---

### Load Summarize

# 01 — Load Summarize / 01 Load Summarize

**Chapter 26 — File 1 of 8 / 第26章 — 第1个文件（共8个）**

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
from pandas import read_csv
```

---
## Step 2 — define the dataset location

```python
filename = 'haberman.csv'
```

---
## Step 3 — define the dataset column names

```python
columns = ['age', 'year', 'nodes', 'class']
```

---
## Step 4 — load the csv file as a data frame

```python
dataframe = read_csv(filename, header=None, names=columns)
```

---
## Step 5 — summarize each column

```python
report = dataframe.describe()
print(report)
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
| `describe()` | 统计摘要信息 | Statistical summary |
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
from pandas import read_csv
# define the dataset location
filename = 'haberman.csv'
# define the dataset column names
columns = ['age', 'year', 'nodes', 'class']
# load the csv file as a data frame
dataframe = read_csv(filename, header=None, names=columns)
# summarize each column
report = dataframe.describe()
print(report)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Histograms

# 02 — Histograms / 02 Histograms

**Chapter 26 — File 2 of 8 / 第26章 — 第2个文件（共8个）**

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
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — define the dataset location

```python
filename = 'haberman.csv'
```

---
## Step 3 — define the dataset column names

```python
columns = ['age', 'year', 'nodes', 'class']
```

---
## Step 4 — load the csv file as a data frame

```python
dataframe = read_csv(filename, header=None, names=columns)
```

---
## Step 5 — create a histogram plot of each variable

```python
dataframe.hist()
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
from pandas import read_csv
from matplotlib import pyplot
# define the dataset location
filename = 'haberman.csv'
# define the dataset column names
columns = ['age', 'year', 'nodes', 'class']
# load the csv file as a data frame
dataframe = read_csv(filename, header=None, names=columns)
# create a histogram plot of each variable
dataframe.hist()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Class Distribution

# 03 — Class Distribution / 03 Class Distribution

**Chapter 26 — File 3 of 8 / 第26章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **summarize the class ratio**.

本脚本演示 **summarize the class ratio**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — summarize the class ratio

```python
from pandas import read_csv
from collections import Counter
```

---
## Step 2 — define the dataset location

```python
filename = 'haberman.csv'
```

---
## Step 3 — define the dataset column names

```python
columns = ['age', 'year', 'nodes', 'class']
```

---
## Step 4 — load the csv file as a data frame

```python
dataframe = read_csv(filename, header=None, names=columns)
```

---
## Step 5 — summarize the class distribution

```python
target = dataframe['class'].values
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the class ratio 是机器学习中的常用技术。  
  *summarize the class ratio is a common technique in machine learning.*

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
# Class Distribution / 03 Class Distribution
# Complete Code / 完整代码
# ===============================

# summarize the class ratio
from pandas import read_csv
from collections import Counter
# define the dataset location
filename = 'haberman.csv'
# define the dataset column names
columns = ['age', 'year', 'nodes', 'class']
# load the csv file as a data frame
dataframe = read_csv(filename, header=None, names=columns)
# summarize the class distribution
target = dataframe['class'].values
counter = Counter(target)
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Baseline

# 04 — Baseline / 04 Baseline

**Chapter 26 — File 4 of 8 / 第26章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **baseline model and test harness for the haberman dataset**.

本脚本演示 **baseline model and test harness for the haberman dataset**。

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
## Step 1 — baseline model and test harness for the haberman dataset

```python
from collections import Counter
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
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
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
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
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — calculate brier skill score (BSS)

```python
def brier_skill_score(y_true, y_prob):
```

---
## Step 8 — calculate reference brier score

```python
ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
```

---
## Step 9 — calculate model brier score

```python
bs_model = brier_score_loss(y_true, y_prob)
```

---
## Step 10 — calculate skill score

```python
return 1.0 - (bs_model / bs_ref)
```

---
## Step 11 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 12 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — define the model evaluation metric

```python
metric = make_scorer(brier_skill_score, needs_proba=True)
```

---
## Step 14 — evaluate model

```python
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 15 — define the location of the dataset

```python
full_path = 'haberman.csv'
```

---
## Step 16 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 17 — summarize the loaded dataset

```python
print(X.shape, y.shape, Counter(y))
```

---
## Step 18 — define the reference model

```python
model = DummyClassifier(strategy='prior')
```

---
## Step 19 — evaluate the model

```python
scores = evaluate_model(X, y, model)
```

---
## Step 20 — summarize performance

```python
print('Mean BSS: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: baseline model and test harness for the haberman dataset 是机器学习中的常用技术。  
  *baseline model and test harness for the haberman dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
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

# baseline model and test harness for the haberman dataset
from collections import Counter
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y

# calculate brier skill score (BSS)
def brier_skill_score(y_true, y_prob):
	# calculate reference brier score
	ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
	# calculate model brier score
	bs_model = brier_score_loss(y_true, y_prob)
	# calculate skill score
	return 1.0 - (bs_model / bs_ref)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(brier_skill_score, needs_proba=True)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define the location of the dataset
full_path = 'haberman.csv'
# load the dataset
X, y = load_dataset(full_path)
# summarize the loaded dataset
print(X.shape, y.shape, Counter(y))
# define the reference model
model = DummyClassifier(strategy='prior')
# evaluate the model
scores = evaluate_model(X, y, model)
# summarize performance
print('Mean BSS: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Evaluate Scaled Models

# 06 — Evaluate Scaled Models / 模型评估

**Chapter 26 — File 6 of 8 / 第26章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare probabilistic models with standardized input on the haberman dataset**.

本脚本演示 **compare probabilistic models with standardized input on the haberman dataset**。

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
## Step 1 — compare probabilistic models with standardized input on the haberman dataset

```python
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
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
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — calculate brier skill score (BSS)

```python
def brier_skill_score(y_true, y_prob):
```

---
## Step 8 — calculate reference brier score

```python
ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
```

---
## Step 9 — calculate model brier score

```python
bs_model = brier_score_loss(y_true, y_prob)
```

---
## Step 10 — calculate skill score

```python
return 1.0 - (bs_model / bs_ref)
```

---
## Step 11 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 12 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — define the model evaluation metric

```python
metric = make_scorer(brier_skill_score, needs_proba=True)
```

---
## Step 14 — evaluate model

```python
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 15 — define models to test

```python
def get_models():
	models, names = list(), list()
```

---
## Step 16 — LR

```python
models.append(LogisticRegression(solver='lbfgs'))
	names.append('LR')
```

---
## Step 17 — LDA

```python
models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
```

---
## Step 18 — QDA

```python
models.append(QuadraticDiscriminantAnalysis())
	names.append('QDA')
```

---
## Step 19 — GNB

```python
models.append(GaussianNB())
	names.append('GNB')
```

---
## Step 20 — GPC

```python
models.append(GaussianProcessClassifier())
	names.append('GPC')
	return models, names
```

---
## Step 21 — define the location of the dataset

```python
full_path = 'haberman.csv'
```

---
## Step 22 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 23 — define models

```python
models, names = get_models()
results = list()
```

---
## Step 24 — evaluate each model

```python
for i in range(len(models)):
```

---
## Step 25 — create a pipeline

```python
pipeline = Pipeline(steps=[('t', StandardScaler()),('m',models[i])])
```

---
## Step 26 — evaluate the model and store results

```python
scores = evaluate_model(X, y, pipeline)
	results.append(scores)
```

---
## Step 27 — summarize and store

```python
print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
```

---
## Step 28 — plot the results

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare probabilistic models with standardized input on the haberman dataset 是机器学习中的常用技术。  
  *compare probabilistic models with standardized input on the haberman dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Scaled Models / 模型评估
# Complete Code / 完整代码
# ===============================

# compare probabilistic models with standardized input on the haberman dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y

# calculate brier skill score (BSS)
def brier_skill_score(y_true, y_prob):
	# calculate reference brier score
	ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
	# calculate model brier score
	bs_model = brier_score_loss(y_true, y_prob)
	# calculate skill score
	return 1.0 - (bs_model / bs_ref)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(brier_skill_score, needs_proba=True)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# LR
	models.append(LogisticRegression(solver='lbfgs'))
	names.append('LR')
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# QDA
	models.append(QuadraticDiscriminantAnalysis())
	names.append('QDA')
	# GNB
	models.append(GaussianNB())
	names.append('GNB')
	# GPC
	models.append(GaussianProcessClassifier())
	names.append('GPC')
	return models, names

# define the location of the dataset
full_path = 'haberman.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
	# create a pipeline
	pipeline = Pipeline(steps=[('t', StandardScaler()),('m',models[i])])
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
	results.append(scores)
	# summarize and store
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Evaluate Power Models

# 07 — Evaluate Power Models / 模型评估

**Chapter 26 — File 7 of 8 / 第26章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare probabilistic models with power transforms on the haberman dataset**.

本脚本演示 **compare probabilistic models with power transforms on the haberman dataset**。

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
## Step 1 — compare probabilistic models with power transforms on the haberman dataset

```python
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
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
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — calculate brier skill score (BSS)

```python
def brier_skill_score(y_true, y_prob):
```

---
## Step 8 — calculate reference brier score

```python
ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
```

---
## Step 9 — calculate model brier score

```python
bs_model = brier_score_loss(y_true, y_prob)
```

---
## Step 10 — calculate skill score

```python
return 1.0 - (bs_model / bs_ref)
```

---
## Step 11 — evaluate a model

```python
def evaluate_model(X, y, model):
```

---
## Step 12 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — define the model evaluation metric

```python
metric = make_scorer(brier_skill_score, needs_proba=True)
```

---
## Step 14 — evaluate model

```python
scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
```

---
## Step 15 — define models to test

```python
def get_models():
	models, names = list(), list()
```

---
## Step 16 — LR

```python
models.append(LogisticRegression(solver='lbfgs'))
	names.append('LR')
```

---
## Step 17 — LDA

```python
models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
```

---
## Step 18 — GPC

```python
models.append(GaussianProcessClassifier())
	names.append('GPC')
	return models, names
```

---
## Step 19 — define the location of the dataset

```python
full_path = 'haberman.csv'
```

---
## Step 20 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 21 — define models

```python
models, names = get_models()
results = list()
```

---
## Step 22 — evaluate each model

```python
for i in range(len(models)):
```

---
## Step 23 — create a pipeline

```python
steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()),('m',models[i])]
	pipeline = Pipeline(steps=steps)
```

---
## Step 24 — evaluate the model and store results

```python
scores = evaluate_model(X, y, pipeline)
	results.append(scores)
```

---
## Step 25 — summarize and store

```python
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

- **概念**: compare probabilistic models with power transforms on the haberman dataset 是机器学习中的常用技术。  
  *compare probabilistic models with power transforms on the haberman dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
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
# Evaluate Power Models / 模型评估
# Complete Code / 完整代码
# ===============================

# compare probabilistic models with power transforms on the haberman dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y

# calculate brier skill score (BSS)
def brier_skill_score(y_true, y_prob):
	# calculate reference brier score
	ref_probs = [0.26471 for _ in range(len(y_true))]
	bs_ref = brier_score_loss(y_true, ref_probs)
	# calculate model brier score
	bs_model = brier_score_loss(y_true, y_prob)
	# calculate skill score
	return 1.0 - (bs_model / bs_ref)

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(brier_skill_score, needs_proba=True)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# LR
	models.append(LogisticRegression(solver='lbfgs'))
	names.append('LR')
	# LDA
	models.append(LinearDiscriminantAnalysis())
	names.append('LDA')
	# GPC
	models.append(GaussianProcessClassifier())
	names.append('GPC')
	return models, names

# define the location of the dataset
full_path = 'haberman.csv'
# load the dataset
X, y = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
	# create a pipeline
	steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()),('m',models[i])]
	pipeline = Pipeline(steps=steps)
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
	results.append(scores)
	# summarize and store
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Prediction

# 08 — Prediction / 08 Prediction

**Chapter 26 — File 8 of 8 / 第26章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **fit a model and make predictions for the haberman dataset**.

本脚本演示 **fit a model and make predictions for the haberman dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 训练模型 / Train the model

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
## Step 1 — fit a model and make predictions for the haberman dataset

```python
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
```

---
## Step 2 — load the dataset

```python
def load_dataset(full_path):
```

---
## Step 3 — load the dataset as a numpy array

```python
data = read_csv(full_path, header=None)
```

---
## Step 4 — retrieve numpy array

```python
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
y = LabelEncoder().fit_transform(y)
	return X, y
```

---
## Step 7 — define the location of the dataset

```python
full_path = 'haberman.csv'
```

---
## Step 8 — load the dataset

```python
X, y = load_dataset(full_path)
```

---
## Step 9 — fit the model

```python
steps = [('t1', MinMaxScaler()),('t2', PowerTransformer()),('m',LogisticRegression(solver='lbfgs'))]
model = Pipeline(steps=steps)
model.fit(X, y)
```

---
## Step 10 — some survival cases

```python
print('Survival Cases:')
data = [[31,59,2], [31,65,4], [34,60,1]]
for row in data:
```

---
## Step 11 — make prediction

```python
yhat = model.predict_proba([row])
```

---
## Step 12 — get percentage of survival

```python
p_survive = yhat[0, 0] * 100
```

---
## Step 13 — summarize

```python
print('>data=%s, Survival=%.3f%%' % (row, p_survive))
```

---
## Step 14 — some non-survival cases

```python
print('Non-Survival Cases:')
data = [[44,64,6], [34,66,9], [38,69,21]]
for row in data:
```

---
## Step 15 — make prediction

```python
yhat = model.predict_proba([row])
```

---
## Step 16 — get percentage of survival

```python
p_survive = yhat[0, 0] * 100
```

---
## Step 17 — summarize

```python
print('>data=%s, Survival=%.3f%%' % (row, p_survive))
```

---
## Learning Notes / 学习笔记

- **概念**: fit a model and make predictions for the haberman dataset 是机器学习中的常用技术。  
  *fit a model and make predictions for the haberman dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
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
# Prediction / 08 Prediction
# Complete Code / 完整代码
# ===============================

# fit a model and make predictions for the haberman dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	data = read_csv(full_path, header=None)
	# retrieve numpy array
	data = data.values
	# split into input and output elements
	X, y = data[:, :-1], data[:, -1]
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X, y

# define the location of the dataset
full_path = 'haberman.csv'
# load the dataset
X, y = load_dataset(full_path)
# fit the model
steps = [('t1', MinMaxScaler()),('t2', PowerTransformer()),('m',LogisticRegression(solver='lbfgs'))]
model = Pipeline(steps=steps)
model.fit(X, y)
# some survival cases
print('Survival Cases:')
data = [[31,59,2], [31,65,4], [34,60,1]]
for row in data:
	# make prediction
	yhat = model.predict_proba([row])
	# get percentage of survival
	p_survive = yhat[0, 0] * 100
	# summarize
	print('>data=%s, Survival=%.3f%%' % (row, p_survive))
# some non-survival cases
print('Non-Survival Cases:')
data = [[44,64,6], [34,66,9], [38,69,21]]
for row in data:
	# make prediction
	yhat = model.predict_proba([row])
	# get percentage of survival
	p_survive = yhat[0, 0] * 100
	# summarize
	print('>data=%s, Survival=%.3f%%' % (row, p_survive))
```

---
