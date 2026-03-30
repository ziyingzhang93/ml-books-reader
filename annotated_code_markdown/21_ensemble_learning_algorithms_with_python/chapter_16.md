# 集成学习
## Chapter 16

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 16 — File 1 of 8 / 第16章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **synthetic classification dataset**.

本脚本演示 **synthetic classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — synthetic classification dataset

```python
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
```

---
## Step 3 — summarize the dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: synthetic classification dataset 是机器学习中的常用技术。  
  *synthetic classification dataset is a common technique in machine learning.*

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
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# synthetic classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Anova Ensemble

# 03 — Anova Ensemble / 集成方法

**Chapter 16 — File 3 of 8 / 第16章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of an ensemble created from features selected with the anova f-statistic**.

本脚本演示 **example of an ensemble created from features selected with the anova f-statistic**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of an ensemble created from features selected with the anova f-statistic

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features):
```

---
## Step 3 — define the base models

```python
models = list()
```

---
## Step 4 — enumerate the features in the training dataset

```python
for i in range(1, n_features+1):
```

---
## Step 5 — create the feature selection transform

```python
fs = SelectKBest(score_func=f_classif, k=i)
```

---
## Step 6 — create the model

```python
model = DecisionTreeClassifier()
```

---
## Step 7 — create the pipeline

```python
pipe = Pipeline([('fs', fs), ('m', model)])
```

---
## Step 8 — add as a tuple to the list of models for voting

```python
models.append((str(i),pipe))
```

---
## Step 9 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

---
## Step 10 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
```

---
## Step 11 — get the ensemble model

```python
ensemble = get_ensemble(X.shape[1])
```

---
## Step 12 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — evaluate the model on the dataset

```python
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 14 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of an ensemble created from features selected with the anova f-statistic 是机器学习中的常用技术。  
  *example of an ensemble created from features selected with the anova f-statistic is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Anova Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# example of an ensemble created from features selected with the anova f-statistic
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=f_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs', fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Mut Info Ensemble

# 04 — Mut Info Ensemble / 集成方法

**Chapter 16 — File 4 of 8 / 第16章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of an ensemble created from features selected with mutual information**.

本脚本演示 **example of an ensemble created from features selected with mutual information**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of an ensemble created from features selected with mutual information

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features):
```

---
## Step 3 — define the base models

```python
models = list()
```

---
## Step 4 — enumerate the features in the training dataset

```python
for i in range(1, n_features+1):
```

---
## Step 5 — create the feature selection transform

```python
fs = SelectKBest(score_func=mutual_info_classif, k=i)
```

---
## Step 6 — create the model

```python
model = DecisionTreeClassifier()
```

---
## Step 7 — create the pipeline

```python
pipe = Pipeline([('fs', fs), ('m', model)])
```

---
## Step 8 — add as a tuple to the list of models for voting

```python
models.append((str(i),pipe))
```

---
## Step 9 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

---
## Step 10 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
```

---
## Step 11 — get the ensemble model

```python
ensemble = get_ensemble(X.shape[1])
```

---
## Step 12 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — evaluate the model on the dataset

```python
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 14 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of an ensemble created from features selected with mutual information 是机器学习中的常用技术。  
  *example of an ensemble created from features selected with mutual information is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mut Info Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# example of an ensemble created from features selected with mutual information
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs', fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Rfe Ensemble

# 05 — Rfe Ensemble / 集成方法

**Chapter 16 — File 5 of 8 / 第16章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of an ensemble created from features selected with RFE**.

本脚本演示 **example of an ensemble created from features selected with RFE**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of an ensemble created from features selected with RFE

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features):
```

---
## Step 3 — define the base models

```python
models = list()
```

---
## Step 4 — enumerate the features in the training dataset

```python
for i in range(1, n_features+1):
```

---
## Step 5 — create the feature selection transform

```python
fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
```

---
## Step 6 — create the model

```python
model = DecisionTreeClassifier()
```

---
## Step 7 — create the pipeline

```python
pipe = Pipeline([('fs', fs), ('m', model)])
```

---
## Step 8 — add as a tuple to the list of models for voting

```python
models.append((str(i),pipe))
```

---
## Step 9 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

---
## Step 10 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
```

---
## Step 11 — get the ensemble model

```python
ensemble = get_ensemble(X.shape[1])
```

---
## Step 12 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 13 — evaluate the model on the dataset

```python
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 14 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of an ensemble created from features selected with RFE 是机器学习中的常用技术。  
  *example of an ensemble created from features selected with RFE is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Rfe Ensemble / 集成方法
# Complete Code / 完整代码
# ===============================

# example of an ensemble created from features selected with RFE
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# enumerate the features in the training dataset
	for i in range(1, n_features+1):
		# create the feature selection transform
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		# create the model
		model = DecisionTreeClassifier()
		# create the pipeline
		pipe = Pipeline([('fs', fs), ('m', model)])
		# add as a tuple to the list of models for voting
		models.append((str(i),pipe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# get the ensemble model
ensemble = get_ensemble(X.shape[1])
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Combined Fixed

# 06 — Combined Fixed / 06 Combined Fixed

**Chapter 16 — File 6 of 8 / 第16章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **ensemble of a fixed number of features selected by different feature selection methods**.

本脚本演示 **ensemble of a fixed number of features selected by different feature selection methods**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — ensemble of a fixed number of features selected by different feature selection methods

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features):
```

---
## Step 3 — define the base models

```python
models = list()
```

---
## Step 4 — anova member

```python
fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
```

---
## Step 5 — mutual information member

```python
fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
```

---
## Step 6 — rfe member

```python
fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
```

---
## Step 7 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

---
## Step 8 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
```

---
## Step 9 — get the ensemble model

```python
ensemble = get_ensemble(15)
```

---
## Step 10 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 11 — evaluate the model on the dataset

```python
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 12 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: ensemble of a fixed number of features selected by different feature selection methods 是机器学习中的常用技术。  
  *ensemble of a fixed number of features selected by different feature selection methods is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Combined Fixed / 06 Combined Fixed
# Complete Code / 完整代码
# ===============================

# ensemble of a fixed number of features selected by different feature selection methods
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# anova member
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	# mutual information member
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	# rfe member
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
ensemble = get_ensemble(15)
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Combined Fixed Compare

# 07 — Combined Fixed Compare / 07 Combined Fixed Compare

**Chapter 16 — File 7 of 8 / 第16章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **comparison of ensemble of a fixed number of features to standalone models**.

本脚本演示 **comparison of ensemble of a fixed number of features to standalone models**。

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
## Step 1 — comparison of ensemble of a fixed number of features to standalone models

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features):
```

---
## Step 3 — define the base models

```python
models, names = list(), list()
```

---
## Step 4 — anova member

```python
fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	names.append('anova')
```

---
## Step 5 — mutual information member

```python
fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	names.append('mutinfo')
```

---
## Step 6 — rfe member

```python
fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	names.append('rfe')
```

---
## Step 7 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	names.append('ensemble')
	return names, [anova, mutinfo, rfe, ensemble]
```

---
## Step 8 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
```

---
## Step 9 — get the ensemble model

```python
names, models = get_ensemble(15)
```

---
## Step 10 — evaluate each model

```python
results = list()
for model,name in zip(models,names):
```

---
## Step 11 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 12 — evaluate the model on the dataset

```python
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 13 — store the results

```python
results.append(n_scores)
```

---
## Step 14 — report performance

```python
print('>%s: %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
```

---
## Step 15 — plot the results for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: comparison of ensemble of a fixed number of features to standalone models 是机器学习中的常用技术。  
  *comparison of ensemble of a fixed number of features to standalone models is a common technique in machine learning.*

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
# Combined Fixed Compare / 07 Combined Fixed Compare
# Complete Code / 完整代码
# ===============================

# comparison of ensemble of a fixed number of features to standalone models
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models, names = list(), list()
	# anova member
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	names.append('anova')
	# mutual information member
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	names.append('mutinfo')
	# rfe member
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	names.append('rfe')
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	names.append('ensemble')
	return names, [anova, mutinfo, rfe, ensemble]

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
names, models = get_ensemble(15)
# evaluate each model
results = list()
for model,name in zip(models,names):
	# define the evaluation method
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model on the dataset
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# store the results
	results.append(n_scores)
	# report performance
	print('>%s: %.3f (%.3f)' % (name, mean(n_scores), std(n_scores)))
# plot the results for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Combined Contiguous

# 08 — Combined Contiguous / 08 Combined Contiguous

**Chapter 16 — File 8 of 8 / 第16章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **ensemble of many subsets of features selected by multiple feature selection methods**.

本脚本演示 **ensemble of many subsets of features selected by multiple feature selection methods**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — ensemble of many subsets of features selected by multiple feature selection methods

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
```

---
## Step 2 — get a voting ensemble of models

```python
def get_ensemble(n_features_start, n_features_end):
```

---
## Step 3 — define the base models

```python
models = list()
	for i in range(n_features_start, n_features_end+1):
```

---
## Step 4 — anova member

```python
fs = SelectKBest(score_func=f_classif, k=i)
		anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('anova'+str(i), anova))
```

---
## Step 5 — mutual information member

```python
fs = SelectKBest(score_func=mutual_info_classif, k=i)
		mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('mutinfo'+str(i), mutinfo))
```

---
## Step 6 — rfe member

```python
fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('rfe'+str(i), rfe))
```

---
## Step 7 — define the voting ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble
```

---
## Step 8 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
```

---
## Step 9 — get the ensemble model

```python
ensemble = get_ensemble(1, 20)
```

---
## Step 10 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 11 — evaluate the model on the dataset

```python
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 12 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: ensemble of many subsets of features selected by multiple feature selection methods 是机器学习中的常用技术。  
  *ensemble of many subsets of features selected by multiple feature selection methods is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Combined Contiguous / 08 Combined Contiguous
# Complete Code / 完整代码
# ===============================

# ensemble of many subsets of features selected by multiple feature selection methods
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features_start, n_features_end):
	# define the base models
	models = list()
	for i in range(n_features_start, n_features_end+1):
		# anova member
		fs = SelectKBest(score_func=f_classif, k=i)
		anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('anova'+str(i), anova))
		# mutual information member
		fs = SelectKBest(score_func=mutual_info_classif, k=i)
		mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('mutinfo'+str(i), mutinfo))
		# rfe member
		fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
		rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
		models.append(('rfe'+str(i), rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
ensemble = get_ensemble(1, 20)
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

### Chapter Summary

# Chapter 16 Summary / 第16章总结

## Theme / 主题: Chapter 16 / Chapter 16

This chapter contains **8 code files** demonstrating chapter 16.

本章包含 **8 个代码文件**，演示Chapter 16。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_baseline_model.ipynb` — Baseline Model
  3. `03_anova_ensemble.ipynb` — Anova Ensemble
  4. `04_mut_info_ensemble.ipynb` — Mut Info Ensemble
  5. `05_rfe_ensemble.ipynb` — Rfe Ensemble
  6. `06_combined_fixed.ipynb` — Combined Fixed
  7. `07_combined_fixed_compare.ipynb` — Combined Fixed Compare
  8. `08_combined_contiguous.ipynb` — Combined Contiguous

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 16) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 16）是机器学习流水线中的基础构建块。

---
