# 集成学习
## Chapter 26

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 26 — File 1 of 8 / 第26章 — 第1个文件（共8个）**

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
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — summarize the dataset

```python
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
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Classification Evaluate

# 02 — Classification Evaluate / 分类

**Chapter 26 — File 2 of 8 / 第26章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate a weighted average ensemble for classification**.

本脚本演示 **evaluate a weighted average ensemble for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — evaluate a weighted average ensemble for classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
```

---
## Step 2 — get a list of base models

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models
```

---
## Step 3 — evaluate each base model

```python
def evaluate_models(models, X_train, X_val, y_train, y_val):
```

---
## Step 4 — fit and evaluate the models

```python
scores = list()
	for _, model in models:
```

---
## Step 5 — fit the model

```python
model.fit(X_train, y_train)
```

---
## Step 6 — evaluate the model

```python
yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
```

---
## Step 7 — store the performance

```python
scores.append(acc)
```

---
## Step 8 — report model performance

```python
return scores
```

---
## Step 9 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 10 — split dataset into train and test sets

```python
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
```

---
## Step 11 — split the full train set into train and validation sets

```python
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 12 — create the base models

```python
models = get_models()
```

---
## Step 13 — fit and evaluate each model

```python
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
```

---
## Step 14 — create the ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
```

---
## Step 15 — fit the ensemble on the training dataset

```python
ensemble.fit(X_train_full, y_train_full)
```

---
## Step 16 — make predictions on test set

```python
yhat = ensemble.predict(X_test)
```

---
## Step 17 — evaluate predictions

```python
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a weighted average ensemble for classification 是机器学习中的常用技术。  
  *evaluate a weighted average ensemble for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Evaluate / 分类
# Complete Code / 完整代码
# ===============================

# evaluate a weighted average ensemble for classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for _, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
		# store the performance
		scores.append(acc)
		# report model performance
	return scores

# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Classification Compare

# 03 — Classification Compare / 分类

**Chapter 26 — File 3 of 8 / 第26章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate a weighted average ensemble for classification compared to base model**.

本脚本演示 **evaluate a weighted average ensemble for classification compared to base model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — evaluate a weighted average ensemble for classification compared to base model

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
```

---
## Step 2 — get a list of base models

```python
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models
```

---
## Step 3 — evaluate each base model

```python
def evaluate_models(models, X_train, X_val, y_train, y_val):
```

---
## Step 4 — fit and evaluate the models

```python
scores = list()
	for _, model in models:
```

---
## Step 5 — fit the model

```python
model.fit(X_train, y_train)
```

---
## Step 6 — evaluate the model

```python
yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
```

---
## Step 7 — store the performance

```python
scores.append(acc)
```

---
## Step 8 — report model performance

```python
return scores
```

---
## Step 9 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 10 — split dataset into train and test sets

```python
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
```

---
## Step 11 — split the full train set into train and validation sets

```python
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 12 — create the base models

```python
models = get_models()
```

---
## Step 13 — fit and evaluate each model

```python
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
```

---
## Step 14 — create the ensemble

```python
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
```

---
## Step 15 — fit the ensemble on the training dataset

```python
ensemble.fit(X_train_full, y_train_full)
```

---
## Step 16 — make predictions on test set

```python
yhat = ensemble.predict(X_test)
```

---
## Step 17 — evaluate predictions

```python
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
```

---
## Step 18 — evaluate each standalone model

```python
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]*100))
```

---
## Step 19 — evaluate equal weighting

```python
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Voting Accuracy: %.3f' % (score*100))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a weighted average ensemble for classification compared to base model 是机器学习中的常用技术。  
  *evaluate a weighted average ensemble for classification compared to base model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Compare / 分类
# Complete Code / 完整代码
# ===============================

# evaluate a weighted average ensemble for classification compared to base model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# get a list of base models
def get_models():
	models = list()
	models.append(('lr', LogisticRegression()))
	models.append(('cart', DecisionTreeClassifier()))
	models.append(('bayes', GaussianNB()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for _, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		acc = accuracy_score(y_val, yhat)
		# store the performance
		scores.append(acc)
		# report model performance
	return scores

# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingClassifier(estimators=models, voting='soft', weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = accuracy_score(y_test, yhat)
print('Weighted Avg Accuracy: %.3f' % (score*100))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]*100))
# evaluate equal weighting
ensemble = VotingClassifier(estimators=models, voting='soft')
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = accuracy_score(y_test, yhat)
print('Voting Accuracy: %.3f' % (score*100))
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Regression Dataset

# 04 — Regression Dataset / 回归

**Chapter 26 — File 4 of 8 / 第26章 — 第4个文件（共8个）**

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
from sklearn.datasets import make_regression
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
```

---
## Step 3 — summarize the dataset

```python
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
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Regression Compare

# 05 — Regression Compare / 回归

**Chapter 26 — File 5 of 8 / 第26章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate a weighted average ensemble for regression**.

本脚本演示 **evaluate a weighted average ensemble for regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — evaluate a weighted average ensemble for regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
```

---
## Step 2 — get a list of base models

```python
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models
```

---
## Step 3 — evaluate each base model

```python
def evaluate_models(models, X_train, X_val, y_train, y_val):
```

---
## Step 4 — fit and evaluate the models

```python
scores = list()
	for _, model in models:
```

---
## Step 5 — fit the model

```python
model.fit(X_train, y_train)
```

---
## Step 6 — evaluate the model

```python
yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
```

---
## Step 7 — store the performance

```python
scores.append(-mae)
```

---
## Step 8 — report model performance

```python
return scores
```

---
## Step 9 — define dataset

```python
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
```

---
## Step 10 — split dataset into train and test sets

```python
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
```

---
## Step 11 — split the full train set into train and validation sets

```python
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 12 — create the base models

```python
models = get_models()
```

---
## Step 13 — fit and evaluate each model

```python
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
```

---
## Step 14 — create the ensemble

```python
ensemble = VotingRegressor(estimators=models, weights=scores)
```

---
## Step 15 — fit the ensemble on the training dataset

```python
ensemble.fit(X_train_full, y_train_full)
```

---
## Step 16 — make predictions on test set

```python
yhat = ensemble.predict(X_test)
```

---
## Step 17 — evaluate predictions

```python
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
```

---
## Step 18 — evaluate each standalone model

```python
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
```

---
## Step 19 — evaluate equal weighting

```python
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a weighted average ensemble for regression 是机器学习中的常用技术。  
  *evaluate a weighted average ensemble for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `SVM` | 支持向量机 | Support Vector Machine |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Compare / 回归
# Complete Code / 完整代码
# ===============================

# evaluate a weighted average ensemble for regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

# get a list of base models
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for _, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
		# store the performance
		scores.append(-mae)
		# report model performance
	return scores

# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Argsort Example

# 06 — Argsort Example / 06 Argsort Example

**Chapter 26 — File 6 of 8 / 第26章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **demonstrate argsort**.

本脚本演示 **demonstrate argsort**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — demonstrate argsort

```python
from numpy import argsort
```

---
## Step 2 — data

```python
x = [300, 100, 200]
print(x)
```

---
## Step 3 — argsort of data

```python
print(argsort(x))
```

---
## Step 4 — arg sort of argsort of data

```python
print(argsort(argsort(x)))
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate argsort 是机器学习中的常用技术。  
  *demonstrate argsort is a common technique in machine learning.*

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
# Argsort Example / 06 Argsort Example
# Complete Code / 完整代码
# ===============================

# demonstrate argsort
from numpy import argsort
# data
x = [300, 100, 200]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Argsort Negative Example

# 07 — Argsort Negative Example / 07 Argsort Negative Example

**Chapter 26 — File 7 of 8 / 第26章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **demonstrate argsort with negative scores**.

本脚本演示 **demonstrate argsort with negative scores**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — demonstrate argsort with negative scores

```python
from numpy import argsort
```

---
## Step 2 — data

```python
x = [-10, -100, -80]
print(x)
```

---
## Step 3 — argsort of data

```python
print(argsort(x))
```

---
## Step 4 — arg sort of argsort of data

```python
print(argsort(argsort(x)))
```

---
## Learning Notes / 学习笔记

- **概念**: demonstrate argsort with negative scores 是机器学习中的常用技术。  
  *demonstrate argsort with negative scores is a common technique in machine learning.*

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
# Argsort Negative Example / 07 Argsort Negative Example
# Complete Code / 完整代码
# ===============================

# demonstrate argsort with negative scores
from numpy import argsort
# data
x = [-10, -100, -80]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Regression Rank Compare

# 08 — Regression Rank Compare / 回归

**Chapter 26 — File 8 of 8 / 第26章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **evaluate a weighted average ensemble for regression with rankings for model weights**.

本脚本演示 **evaluate a weighted average ensemble for regression with rankings for model weights**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
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
## Step 1 — evaluate a weighted average ensemble for regression with rankings for model weights

```python
from numpy import argsort
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
```

---
## Step 2 — get a list of base models

```python
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models
```

---
## Step 3 — evaluate each base model

```python
def evaluate_models(models, X_train, X_val, y_train, y_val):
```

---
## Step 4 — fit and evaluate the models

```python
scores = list()
	for _, model in models:
```

---
## Step 5 — fit the model

```python
model.fit(X_train, y_train)
```

---
## Step 6 — evaluate the model

```python
yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
```

---
## Step 7 — store the performance

```python
scores.append(-mae)
```

---
## Step 8 — report model performance

```python
return scores
```

---
## Step 9 — define dataset

```python
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
```

---
## Step 10 — split dataset into train and test sets

```python
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
```

---
## Step 11 — split the full train set into train and validation sets

```python
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 12 — create the base models

```python
models = get_models()
```

---
## Step 13 — fit and evaluate each model

```python
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
ranking = 1 + argsort(argsort(scores))
print(ranking)
```

---
## Step 14 — create the ensemble

```python
ensemble = VotingRegressor(estimators=models, weights=ranking)
```

---
## Step 15 — fit the ensemble on the training dataset

```python
ensemble.fit(X_train_full, y_train_full)
```

---
## Step 16 — make predictions on test set

```python
yhat = ensemble.predict(X_test)
```

---
## Step 17 — evaluate predictions

```python
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
```

---
## Step 18 — evaluate each standalone model

```python
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
```

---
## Step 19 — evaluate equal weighting

```python
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate a weighted average ensemble for regression with rankings for model weights 是机器学习中的常用技术。  
  *evaluate a weighted average ensemble for regression with rankings for model weights is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `SVM` | 支持向量机 | Support Vector Machine |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Rank Compare / 回归
# Complete Code / 完整代码
# ===============================

# evaluate a weighted average ensemble for regression with rankings for model weights
from numpy import argsort
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor

# get a list of base models
def get_models():
	models = list()
	models.append(('knn', KNeighborsRegressor()))
	models.append(('cart', DecisionTreeRegressor()))
	models.append(('svm', SVR()))
	return models

# evaluate each base model
def evaluate_models(models, X_train, X_val, y_train, y_val):
	# fit and evaluate the models
	scores = list()
	for _, model in models:
		# fit the model
		model.fit(X_train, y_train)
		# evaluate the model
		yhat = model.predict(X_val)
		mae = mean_absolute_error(y_val, yhat)
		# store the performance
		scores.append(-mae)
		# report model performance
	return scores

# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# split dataset into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1)
# split the full train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# create the base models
models = get_models()
# fit and evaluate each model
scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
ranking = 1 + argsort(argsort(scores))
print(ranking)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=ranking)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
	print('>%s: %.3f' % (models[i][0], scores[i]))
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))
```

---

### Chapter Summary

# Chapter 26 Summary / 第26章总结

## Theme / 主题: Chapter 26 / Chapter 26

This chapter contains **8 code files** demonstrating chapter 26.

本章包含 **8 个代码文件**，演示Chapter 26。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_classification_evaluate.ipynb` — Classification Evaluate
  3. `03_classification_compare.ipynb` — Classification Compare
  4. `04_regression_dataset.ipynb` — Regression Dataset
  5. `05_regression_compare.ipynb` — Regression Compare
  6. `06_argsort_example.ipynb` — Argsort Example
  7. `07_argsort_negative_example.ipynb` — Argsort Negative Example
  8. `08_regression_rank_compare.ipynb` — Regression Rank Compare

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 26) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 26）是机器学习流水线中的基础构建块。

---
