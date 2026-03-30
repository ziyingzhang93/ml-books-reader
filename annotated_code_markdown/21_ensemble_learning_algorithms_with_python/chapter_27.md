# 集成学习
## Chapter 27

---

### Evaluate Standalone

# 02 — Evaluate Standalone / 模型评估

**Chapter 27 — File 2 of 5 / 第27章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **evaluate standard models on the synthetic dataset**.

本脚本演示 **evaluate standard models on the synthetic dataset**。

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
## Step 1 — evaluate standard models on the synthetic dataset

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models
```

---
## Step 4 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 5 — define the model evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate the model

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 7 — define dataset

```python
X, y = get_dataset()
```

---
## Step 8 — get the models to evaluate

```python
models = get_models()
```

---
## Step 9 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models:
```

---
## Step 10 — evaluate model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 11 — store results

```python
results.append(scores)
	names.append(name)
```

---
## Step 12 — summarize result

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 13 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate standard models on the synthetic dataset 是机器学习中的常用技术。  
  *evaluate standard models on the synthetic dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Standalone / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate standard models on the synthetic dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y

# get a list of models to evaluate
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the model evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models:
	# evaluate model
	scores = evaluate_model(model, X, y)
	# store results
	results.append(scores)
	names.append(name)
	# summarize result
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Evaluate Voting

# 03 — Evaluate Voting / 模型评估

**Chapter 27 — File 3 of 5 / 第27章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of a voting ensemble with soft voting of ensemble members**.

本脚本演示 **example of a voting ensemble with soft voting of ensemble members**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of a voting ensemble with soft voting of ensemble members

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models
```

---
## Step 4 — define dataset

```python
X, y = get_dataset()
```

---
## Step 5 — get the models to evaluate

```python
models = get_models()
```

---
## Step 6 — create the ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft')
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the ensemble

```python
scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 9 — summarize the result

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: example of a voting ensemble with soft voting of ensemble members 是机器学习中的常用技术。  
  *example of a voting ensemble with soft voting of ensemble members is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Voting / 模型评估
# Complete Code / 完整代码
# ===============================

# example of a voting ensemble with soft voting of ensemble members
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y

# get a list of models to evaluate
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft')
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the ensemble
scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Evaluate Pruning

# 04 — Evaluate Pruning / 模型评估

**Chapter 27 — File 4 of 5 / 第27章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of ensemble pruning for classification**.

本脚本演示 **example of ensemble pruning for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of ensemble pruning for classification

```python
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models
```

---
## Step 4 — evaluate a list of models

```python
def evaluate_ensemble(models, X, y):
```

---
## Step 5 — check for no models

```python
if len(models) == 0:
		return 0.0
```

---
## Step 6 — create the ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft')
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the ensemble

```python
scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 9 — return mean score

```python
return mean(scores)
```

---
## Step 10 — perform a single round of pruning the ensemble

```python
def prune_round(models_in, X, y):
```

---
## Step 11 — establish a baseline

```python
baseline = evaluate_ensemble(models_in, X, y)
	best_score, removed = baseline, None
```

---
## Step 12 — enumerate removing each candidate and see if we can improve performance

```python
for m in models_in:
```

---
## Step 13 — copy the list of chosen models

```python
dup = models_in.copy()
```

---
## Step 14 — remove this model

```python
dup.remove(m)
```

---
## Step 15 — evaluate new ensemble

```python
result = evaluate_ensemble(dup, X, y)
```

---
## Step 16 — check for new best

```python
if result > best_score:
```

---
## Step 17 — store the new best

```python
best_score, removed = result, m
	return best_score, removed
```

---
## Step 18 — prune an ensemble from scratch

```python
def prune_ensemble(models, X, y):
	best_score = 0.0
```

---
## Step 19 — prune ensemble until no further improvement

```python
while True:
```

---
## Step 20 — remove one model to the ensemble

```python
score, removed = prune_round(models, X, y)
```

---
## Step 21 — check for no improvement

```python
if removed is None:
			print('>no further improvement')
			break
```

---
## Step 22 — keep track of best score

```python
best_score = score
```

---
## Step 23 — remove model from the list

```python
models.remove(removed)
```

---
## Step 24 — report results along the way

```python
print('>%.3f (removed: %s)' % (score, removed[0]))
	return best_score, models
```

---
## Step 25 — define dataset

```python
X, y = get_dataset()
```

---
## Step 26 — get the models to evaluate

```python
models = get_models()
```

---
## Step 27 — prune the ensemble

```python
score, model_list = prune_ensemble(models, X, y)
names = ','.join([n for n,_ in model_list])
print('Models: %s' % names)
print('Final Mean Accuracy: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of ensemble pruning for classification 是机器学习中的常用技术。  
  *example of ensemble pruning for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Pruning / 模型评估
# Complete Code / 完整代码
# ===============================

# example of ensemble pruning for classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y

# get a list of models to evaluate
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models

# evaluate a list of models
def evaluate_ensemble(models, X, y):
	# check for no models
	if len(models) == 0:
		return 0.0
	# create the ensemble
	ensemble = VotingClassifier(estimators=models, voting='soft')
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the ensemble
	scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# return mean score
	return mean(scores)

# perform a single round of pruning the ensemble
def prune_round(models_in, X, y):
	# establish a baseline
	baseline = evaluate_ensemble(models_in, X, y)
	best_score, removed = baseline, None
	# enumerate removing each candidate and see if we can improve performance
	for m in models_in:
		# copy the list of chosen models
		dup = models_in.copy()
		# remove this model
		dup.remove(m)
		# evaluate new ensemble
		result = evaluate_ensemble(dup, X, y)
		# check for new best
		if result > best_score:
			# store the new best
			best_score, removed = result, m
	return best_score, removed

# prune an ensemble from scratch
def prune_ensemble(models, X, y):
	best_score = 0.0
	# prune ensemble until no further improvement
	while True:
		# remove one model to the ensemble
		score, removed = prune_round(models, X, y)
		# check for no improvement
		if removed is None:
			print('>no further improvement')
			break
		# keep track of best score
		best_score = score
		# remove model from the list
		models.remove(removed)
		# report results along the way
		print('>%.3f (removed: %s)' % (score, removed[0]))
	return best_score, models

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# prune the ensemble
score, model_list = prune_ensemble(models, X, y)
names = ','.join([n for n,_ in model_list])
print('Models: %s' % names)
print('Final Mean Accuracy: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Evaluate Growing

# 05 — Evaluate Growing / 模型评估

**Chapter 27 — File 5 of 5 / 第27章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **example of ensemble growing for classification**.

本脚本演示 **example of ensemble growing for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of ensemble growing for classification

```python
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models
```

---
## Step 4 — evaluate a list of models

```python
def evaluate_ensemble(models, X, y):
```

---
## Step 5 — check for no models

```python
if len(models) == 0:
		return 0.0
```

---
## Step 6 — create the ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft')
```

---
## Step 7 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 8 — evaluate the ensemble

```python
scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 9 — return mean score

```python
return mean(scores)
```

---
## Step 10 — perform a single round of growing the ensemble

```python
def grow_round(models_in, models_candidate, X, y):
```

---
## Step 11 — establish a baseline

```python
baseline = evaluate_ensemble(models_in, X, y)
	best_score, addition = baseline, None
```

---
## Step 12 — enumerate adding each candidate and see if we can improve performance

```python
for m in models_candidate:
```

---
## Step 13 — copy the list of chosen models

```python
dup = models_in.copy()
```

---
## Step 14 — add the candidate

```python
dup.append(m)
```

---
## Step 15 — evaluate new ensemble

```python
result = evaluate_ensemble(dup, X, y)
```

---
## Step 16 — check for new best

```python
if result > best_score:
```

---
## Step 17 — store the new best

```python
best_score, addition = result, m
	return best_score, addition
```

---
## Step 18 — grow an ensemble from scratch

```python
def grow_ensemble(models, X, y):
	best_score, best_list = 0.0, list()
```

---
## Step 19 — grow ensemble until no further improvement

```python
while True:
```

---
## Step 20 — add one model to the ensemble

```python
score, addition = grow_round(best_list, models, X, y)
```

---
## Step 21 — check for no improvement

```python
if addition is None:
			print('>no further improvement')
			break
```

---
## Step 22 — keep track of best score

```python
best_score = score
```

---
## Step 23 — remove new model from the list of candidates

```python
models.remove(addition)
```

---
## Step 24 — add new model to the list of models in the ensemble

```python
best_list.append(addition)
```

---
## Step 25 — report results along the way

```python
names = ','.join([n for n,_ in best_list])
		print('>%.3f (%s)' % (score, names))
	return best_score, best_list
```

---
## Step 26 — define dataset

```python
X, y = get_dataset()
```

---
## Step 27 — get the models to evaluate

```python
models = get_models()
```

---
## Step 28 — grow the ensemble

```python
score, model_list = grow_ensemble(models, X, y)
names = ','.join([n for n,_ in model_list])
print('Models: %s' % names)
print('Final Mean Accuracy: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of ensemble growing for classification 是机器学习中的常用技术。  
  *example of ensemble growing for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Growing / 模型评估
# Complete Code / 完整代码
# ===============================

# example of ensemble growing for classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
	return X, y

# get a list of models to evaluate
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('knn', KNeighborsClassifier()))
	models.append(('tree', DecisionTreeClassifier()))
	models.append(('nb', GaussianNB()))
	models.append(('svm', SVC(probability=True)))
	return models

# evaluate a list of models
def evaluate_ensemble(models, X, y):
	# check for no models
	if len(models) == 0:
		return 0.0
	# create the ensemble
	ensemble = VotingClassifier(estimators=models, voting='soft')
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the ensemble
	scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# return mean score
	return mean(scores)

# perform a single round of growing the ensemble
def grow_round(models_in, models_candidate, X, y):
	# establish a baseline
	baseline = evaluate_ensemble(models_in, X, y)
	best_score, addition = baseline, None
	# enumerate adding each candidate and see if we can improve performance
	for m in models_candidate:
		# copy the list of chosen models
		dup = models_in.copy()
		# add the candidate
		dup.append(m)
		# evaluate new ensemble
		result = evaluate_ensemble(dup, X, y)
		# check for new best
		if result > best_score:
			# store the new best
			best_score, addition = result, m
	return best_score, addition

# grow an ensemble from scratch
def grow_ensemble(models, X, y):
	best_score, best_list = 0.0, list()
	# grow ensemble until no further improvement
	while True:
		# add one model to the ensemble
		score, addition = grow_round(best_list, models, X, y)
		# check for no improvement
		if addition is None:
			print('>no further improvement')
			break
		# keep track of best score
		best_score = score
		# remove new model from the list of candidates
		models.remove(addition)
		# add new model to the list of models in the ensemble
		best_list.append(addition)
		# report results along the way
		names = ','.join([n for n,_ in best_list])
		print('>%.3f (%s)' % (score, names))
	return best_score, best_list

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# grow the ensemble
score, model_list = grow_ensemble(models, X, y)
names = ','.join([n for n,_ in model_list])
print('Models: %s' % names)
print('Final Mean Accuracy: %.3f' % score)
```

---

### Chapter Summary

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **5 code files** demonstrating chapter 27.

本章包含 **5 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_evaluate_standalone.ipynb` — Evaluate Standalone
  3. `03_evaluate_voting.ipynb` — Evaluate Voting
  4. `04_evaluate_pruning.ipynb` — Evaluate Pruning
  5. `05_evaluate_growing.ipynb` — Evaluate Growing

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---
