# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 25

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 25 — File 1 of 8 / 第25章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **synthetic binary classification dataset**.

本脚本演示 **synthetic binary classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — synthetic binary classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
```

---
## Step 3 — summarize the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: synthetic binary classification dataset 是机器学习中的常用技术。  
  *synthetic binary classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Dataset / 分类
# Complete Code / 完整代码
# ===============================

# synthetic binary classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Hard Voting Evaluate

# 02 — Hard Voting Evaluate / 模型评估

**Chapter 25 — File 2 of 8 / 第25章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare hard voting to standalone classifiers**.

本脚本演示 **compare hard voting to standalone classifiers**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
## Step 1 — compare hard voting to standalone classifiers

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
	return X, y
```

---
## Step 3 — get a list of standalone models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the number of neighbors to consider

```python
neighbors = [1, 3, 5, 7, 9]
	for n in neighbors:
		key = 'knn' + str(n)
  # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
		models[key] = KNeighborsClassifier(n_neighbors=n)
```

---
## Step 5 — define the voting ensemble

```python
# 获取字典的键值对 / Get dict key-value pairs
members = [(n,m) for n,m in models.items()]
	models['hard_voting'] = VotingClassifier(estimators=members, voting='hard')
	return models
```

---
## Step 6 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 9 — define dataset

```python
X, y = get_dataset()
```

---
## Step 10 — get the models to evaluate

```python
models = get_models()
```

---
## Step 11 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 12 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 13 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 14 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 15 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare hard voting to standalone classifiers 是机器学习中的常用技术。  
  *compare hard voting to standalone classifiers is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hard Voting Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# compare hard voting to standalone classifiers
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
	return X, y

# get a list of standalone models to evaluate
def get_models():
	models = dict()
	# define the number of neighbors to consider
	neighbors = [1, 3, 5, 7, 9]
	for n in neighbors:
		key = 'knn' + str(n)
  # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
		models[key] = KNeighborsClassifier(n_neighbors=n)
	# define the voting ensemble
 # 获取字典的键值对 / Get dict key-value pairs
	members = [(n,m) for n,m in models.items()]
	models['hard_voting'] = VotingClassifier(estimators=members, voting='hard')
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	# summarize the performance along the way
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Hard Voting Predict



---

### Soft Voting Evaluate

# 04 — Soft Voting Evaluate / 模型评估

**Chapter 25 — File 4 of 8 / 第25章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare soft voting ensemble to standalone classifiers**.

本脚本演示 **compare soft voting ensemble to standalone classifiers**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
## Step 1 — compare soft voting ensemble to standalone classifiers

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
	return X, y
```

---
## Step 3 — get a list of standalone models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the degrees to consider

```python
# 生成整数序列 / Generate integer sequence
for n in range(1,6):
		key = 'svm' + str(n)
  # 支持向量机 / Support Vector Machine
		models[key] = SVC(probability=True, kernel='poly', degree=n)
```

---
## Step 5 — define the voting ensemble

```python
# 获取字典的键值对 / Get dict key-value pairs
members = [(n,m) for n,m in models.items()]
	models['soft_voting'] = VotingClassifier(estimators=members, voting='soft')
	return models
```

---
## Step 6 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 9 — define dataset

```python
X, y = get_dataset()
```

---
## Step 10 — get the models to evaluate

```python
models = get_models()
```

---
## Step 11 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 12 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 13 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 14 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 15 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare soft voting ensemble to standalone classifiers 是机器学习中的常用技术。  
  *compare soft voting ensemble to standalone classifiers is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Soft Voting Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# compare soft voting ensemble to standalone classifiers
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
	return X, y

# get a list of standalone models to evaluate
def get_models():
	models = dict()
	# define the degrees to consider
 # 生成整数序列 / Generate integer sequence
	for n in range(1,6):
		key = 'svm' + str(n)
  # 支持向量机 / Support Vector Machine
		models[key] = SVC(probability=True, kernel='poly', degree=n)
	# define the voting ensemble
 # 获取字典的键值对 / Get dict key-value pairs
	members = [(n,m) for n,m in models.items()]
	models['soft_voting'] = VotingClassifier(estimators=members, voting='soft')
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	# summarize the performance along the way
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Soft Voting Predict

# 05 — Soft Voting Predict / 05 Soft Voting Predict

**Chapter 25 — File 5 of 8 / 第25章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **make a prediction with a soft voting ensemble**.

本脚本演示 **make a prediction with a soft voting ensemble**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — make a prediction with a soft voting ensemble

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
```

---
## Step 3 — define the base models

```python
models = list()
# 生成整数序列 / Generate integer sequence
for n in range(1,6):
 # 支持向量机 / Support Vector Machine
	models.append(('svm'+str(n), SVC(probability=True, kernel='poly', degree=n)))
```

---
## Step 4 — define the soft voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft')
```

---
## Step 5 — fit the model on all available data

```python
ensemble.fit(X, y)
```

---
## Step 6 — make a prediction for one example

```python
row = [5.88891819, 2.64867662, -0.42728226, -1.24988856, -0.00822, -3.57895574, 2.87938412, -1.55614691, -0.38168784, 7.50285659, -1.16710354, -5.02492712, -0.46196105, -0.64539455, -1.71297469, 0.25987852, -0.193401, -5.52022952, 0.0364453, -1.960039]
yhat = ensemble.predict([row])
```

---
## Step 7 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with a soft voting ensemble 是机器学习中的常用技术。  
  *make a prediction with a soft voting ensemble is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Soft Voting Predict / 05 Soft Voting Predict
# Complete Code / 完整代码
# ===============================

# make a prediction with a soft voting ensemble
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
# define the base models
models = list()
# 生成整数序列 / Generate integer sequence
for n in range(1,6):
 # 支持向量机 / Support Vector Machine
	models.append(('svm'+str(n), SVC(probability=True, kernel='poly', degree=n)))
# define the soft voting ensemble
ensemble = VotingClassifier(estimators=models, voting='soft')
# fit the model on all available data
ensemble.fit(X, y)
# make a prediction for one example
row = [5.88891819, 2.64867662, -0.42728226, -1.24988856, -0.00822, -3.57895574, 2.87938412, -1.55614691, -0.38168784, 7.50285659, -1.16710354, -5.02492712, -0.46196105, -0.64539455, -1.71297469, 0.25987852, -0.193401, -5.52022952, 0.0364453, -1.960039]
yhat = ensemble.predict([row])
# summarize prediction
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Regression Dataset

# 06 — Regression Dataset / 回归

**Chapter 25 — File 6 of 8 / 第25章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **synthetic regression dataset**.

本脚本演示 **synthetic regression dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — synthetic regression dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
```

---
## Step 3 — summarize the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: synthetic regression dataset 是机器学习中的常用技术。  
  *synthetic regression dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Dataset / 回归
# Complete Code / 完整代码
# ===============================

# synthetic regression dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Regression Voting Evaluate

# 07 — Regression Voting Evaluate / 回归

**Chapter 25 — File 7 of 8 / 第25章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare voting ensemble to each standalone models for regression**.

本脚本演示 **compare voting ensemble to each standalone models for regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
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
## Step 1 — compare voting ensemble to each standalone models for regression

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingRegressor
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
	return X, y
```

---
## Step 3 — get a list of standalone models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the tree depths to consider

```python
# 生成整数序列 / Generate integer sequence
for n in range(1,6):
		key = 'cart' + str(n)
  # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
		models[key] = DecisionTreeRegressor(max_depth=n)
```

---
## Step 5 — define the voting ensemble

```python
# 获取字典的键值对 / Get dict key-value pairs
members = [(n,m) for n,m in models.items()]
	models['voting'] = VotingRegressor(estimators=members)
	return models
```

---
## Step 6 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 9 — define dataset

```python
X, y = get_dataset()
```

---
## Step 10 — get the models to evaluate

```python
models = get_models()
```

---
## Step 11 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 12 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 13 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 14 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 15 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare voting ensemble to each standalone models for regression 是机器学习中的常用技术。  
  *compare voting ensemble to each standalone models for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Voting Evaluate / 回归
# Complete Code / 完整代码
# ===============================

# compare voting ensemble to each standalone models for regression
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import VotingRegressor
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
	return X, y

# get a list of standalone models to evaluate
def get_models():
	models = dict()
	# define the tree depths to consider
 # 生成整数序列 / Generate integer sequence
	for n in range(1,6):
		key = 'cart' + str(n)
  # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
		models[key] = DecisionTreeRegressor(max_depth=n)
	# define the voting ensemble
 # 获取字典的键值对 / Get dict key-value pairs
	members = [(n,m) for n,m in models.items()]
	models['voting'] = VotingRegressor(estimators=members)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	# summarize the performance along the way
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Regression Voting Predict



---

### Chapter Summary / 章节总结

# Chapter 25 Summary / 第25章总结

## Theme / 主题: Chapter 25 / Chapter 25

This chapter contains **8 code files** demonstrating chapter 25.

本章包含 **8 个代码文件**，演示Chapter 25。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_hard_voting_evaluate.ipynb` — Hard Voting Evaluate
  3. `03_hard_voting_predict.ipynb` — Hard Voting Predict
  4. `04_soft_voting_evaluate.ipynb` — Soft Voting Evaluate
  5. `05_soft_voting_predict.ipynb` — Soft Voting Predict
  6. `06_regression_dataset.ipynb` — Regression Dataset
  7. `07_regression_voting_evaluate.ipynb` — Regression Voting Evaluate
  8. `08_regression_voting_predict.ipynb` — Regression Voting Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 25) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 25）是机器学习流水线中的基础构建块。

---
