# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 30

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 30 — File 1 of 7 / 第30章 — 第1个文件（共7个）**

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

➡️ **Next / 下一步**: File 2 of 7

---

### Classification Manual Evaluate

# 02 — Classification Manual Evaluate / 分类

**Chapter 30 — File 2 of 7 / 第30章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of a super learner model for binary classification**.

本脚本演示 **example of a super learner model for binary classification**。

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
## Step 1 — example of a super learner model for binary classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
```

---
## Step 2 — create a list of base-models

```python
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='liblinear'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeClassifier())
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale', probability=True))
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianNB())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingClassifier(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestClassifier(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesClassifier(n_estimators=10))
	return models
```

---
## Step 3 — collect out of fold predictions from cross validation

```python
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
```

---
## Step 4 — define split of data

```python
kfold = KFold(n_splits=10, shuffle=True)
```

---
## Step 5 — enumerate splits

```python
for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
```

---
## Step 6 — get data

```python
train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
```

---
## Step 7 — fit and make predictions with each sub-model

```python
for model in models:
   # 训练模型 / Train the model
			model.fit(train_X, train_y)
			yhat = model.predict_proba(test_X)
```

---
## Step 8 — store columns

```python
# 添加元素到列表末尾 / Append element to list end
fold_yhats.append(yhat)
```

---
## Step 9 — store fold yhats as columns

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)
```

---
## Step 10 — fit all base models on the training dataset

```python
def fit_base_models(X, y, models):
	for model in models:
  # 训练模型 / Train the model
		model.fit(X, y)
```

---
## Step 11 — fit a meta model

```python
def fit_meta_model(X, y):
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 训练模型 / Train the model
	model.fit(X, y)
	return model
```

---
## Step 12 — evaluate a list of models on a dataset

```python
def evaluate_models(X, y, models):
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
  # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
		score = accuracy_score(y, yhat)
  # 打印输出 / Print output
		print('%s: %.3f' % (model.__class__.__name__, score))
```

---
## Step 13 — make predictions with stacked model

```python
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict_proba(X)
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	meta_X = hstack(meta_X)
```

---
## Step 14 — predict

```python
# 用模型做预测 / Make predictions with model
return meta_model.predict(meta_X)
```

---
## Step 15 — create the inputs and outputs

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
```

---
## Step 16 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
```

---
## Step 17 — get models

```python
models = get_models()
```

---
## Step 18 — get out of fold predictions

```python
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Meta ', meta_X.shape, meta_y.shape)
```

---
## Step 19 — fit base models

```python
fit_base_models(X, y, models)
```

---
## Step 20 — fit the meta model

```python
meta_model = fit_meta_model(meta_X, meta_y)
```

---
## Step 21 — evaluate base models

```python
evaluate_models(X_val, y_val, models)
```

---
## Step 22 — evaluate meta model

```python
yhat = super_learner_predictions(X_val, models, meta_model)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_val, yhat)
# 打印输出 / Print output
print('Super Learner: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a super learner model for binary classification 是机器学习中的常用技术。  
  *example of a super learner model for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
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
# Classification Manual Evaluate / 分类
# Complete Code / 完整代码
# ===============================

# example of a super learner model for binary classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier

# create a list of base-models
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='liblinear'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeClassifier())
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale', probability=True))
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianNB())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingClassifier(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestClassifier(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesClassifier(n_estimators=10))
	return models

# collect out of fold predictions from cross validation
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = KFold(n_splits=10, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
		# get data
		train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for model in models:
   # 训练模型 / Train the model
			model.fit(train_X, train_y)
			yhat = model.predict_proba(test_X)
			# store columns
   # 添加元素到列表末尾 / Append element to list end
			fold_yhats.append(yhat)
		# store fold yhats as columns
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)

# fit all base models on the training dataset
def fit_base_models(X, y, models):
	for model in models:
  # 训练模型 / Train the model
		model.fit(X, y)

# fit a meta model
def fit_meta_model(X, y):
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	model = LogisticRegression(solver='liblinear')
 # 训练模型 / Train the model
	model.fit(X, y)
	return model

# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
  # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
		score = accuracy_score(y, yhat)
  # 打印输出 / Print output
		print('%s: %.3f' % (model.__class__.__name__, score))

# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict_proba(X)
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	meta_X = hstack(meta_X)
	# predict
 # 用模型做预测 / Make predictions with model
	return meta_model.predict(meta_X)

# create the inputs and outputs
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
# split
# 划分训练集和测试集 / Split into train and test sets
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
# get models
models = get_models()
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
fit_base_models(X, y, models)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(X_val, y_val, models)
# evaluate meta model
yhat = super_learner_predictions(X_val, models, meta_model)
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_val, yhat)
# 打印输出 / Print output
print('Super Learner: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Regression Dataset

# 03 — Regression Dataset / 回归

**Chapter 30 — File 3 of 7 / 第30章 — 第3个文件（共7个）**

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
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
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
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Regression Manual Evaluate

# 04 — Regression Manual Evaluate / 回归

**Chapter 30 — File 4 of 7 / 第30章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of a super learner model for regression**.

本脚本演示 **example of a super learner model for regression**。

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
## Step 1 — example of a super learner model for regression

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesRegressor
```

---
## Step 2 — create a list of base-models

```python
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(ElasticNet())
 # 支持向量机 / Support Vector Machine
	models.append(SVR(gamma='scale'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeRegressor())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsRegressor())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostRegressor())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingRegressor(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestRegressor(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesRegressor(n_estimators=10))
	return models
```

---
## Step 3 — collect out of fold predictions from cross validation

```python
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
```

---
## Step 4 — define split of data

```python
kfold = KFold(n_splits=10, shuffle=True)
```

---
## Step 5 — enumerate splits

```python
for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
```

---
## Step 6 — get data

```python
train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
```

---
## Step 7 — fit and make predictions with each sub-model

```python
for model in models:
   # 训练模型 / Train the model
			model.fit(train_X, train_y)
   # 用模型做预测 / Make predictions with model
			yhat = model.predict(test_X)
```

---
## Step 8 — store columns

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
fold_yhats.append(yhat.reshape(len(yhat),1))
```

---
## Step 9 — store fold yhats as columns

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)
```

---
## Step 10 — fit all base models on the training dataset

```python
def fit_base_models(X, y, models):
	for model in models:
  # 训练模型 / Train the model
		model.fit(X, y)
```

---
## Step 11 — fit a meta model

```python
def fit_meta_model(X, y):
	model = LinearRegression()
 # 训练模型 / Train the model
	model.fit(X, y)
	return model
```

---
## Step 12 — evaluate a list of models on a dataset

```python
def evaluate_models(X, y, models):
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
		mae = mean_absolute_error(y, yhat)
  # 打印输出 / Print output
		print('%s: MAE %.3f' % (model.__class__.__name__, mae))
```

---
## Step 13 — make predictions with stacked model

```python
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		meta_X.append(yhat.reshape(len(yhat),1))
	meta_X = hstack(meta_X)
```

---
## Step 14 — predict

```python
# 用模型做预测 / Make predictions with model
return meta_model.predict(meta_X)
```

---
## Step 15 — create the inputs and outputs

```python
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
```

---
## Step 16 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
```

---
## Step 17 — get models

```python
models = get_models()
```

---
## Step 18 — get out of fold predictions

```python
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Meta ', meta_X.shape, meta_y.shape)
```

---
## Step 19 — fit base models

```python
fit_base_models(X, y, models)
```

---
## Step 20 — fit the meta model

```python
meta_model = fit_meta_model(meta_X, meta_y)
```

---
## Step 21 — evaluate base models

```python
evaluate_models(X_val, y_val, models)
```

---
## Step 22 — evaluate meta model

```python
yhat = super_learner_predictions(X_val, models, meta_model)
score = mean_absolute_error(y_val, yhat)
# 打印输出 / Print output
print('Super Learner: MAE %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a super learner model for regression 是机器学习中的常用技术。  
  *example of a super learner model for regression is a common technique in machine learning.*

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
# Regression Manual Evaluate / 回归
# Complete Code / 完整代码
# ===============================

# example of a super learner model for regression
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import KFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import ElasticNet
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesRegressor

# create a list of base-models
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(ElasticNet())
 # 支持向量机 / Support Vector Machine
	models.append(SVR(gamma='scale'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeRegressor())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsRegressor())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostRegressor())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingRegressor(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestRegressor(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesRegressor(n_estimators=10))
	return models

# collect out of fold predictions from cross validation
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = KFold(n_splits=10, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
		# get data
		train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for model in models:
   # 训练模型 / Train the model
			model.fit(train_X, train_y)
   # 用模型做预测 / Make predictions with model
			yhat = model.predict(test_X)
			# store columns
   # 改变数组形状（不改变数据） / Reshape array (data unchanged)
			fold_yhats.append(yhat.reshape(len(yhat),1))
		# store fold yhats as columns
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)

# fit all base models on the training dataset
def fit_base_models(X, y, models):
	for model in models:
  # 训练模型 / Train the model
		model.fit(X, y)

# fit a meta model
def fit_meta_model(X, y):
	model = LinearRegression()
 # 训练模型 / Train the model
	model.fit(X, y)
	return model

# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
		mae = mean_absolute_error(y, yhat)
  # 打印输出 / Print output
		print('%s: MAE %.3f' % (model.__class__.__name__, mae))

# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X)
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		meta_X.append(yhat.reshape(len(yhat),1))
	meta_X = hstack(meta_X)
	# predict
 # 用模型做预测 / Make predictions with model
	return meta_model.predict(meta_X)

# create the inputs and outputs
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
# split
# 划分训练集和测试集 / Split into train and test sets
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
# get models
models = get_models()
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
fit_base_models(X, y, models)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(X_val, y_val, models)
# evaluate meta model
yhat = super_learner_predictions(X_val, models, meta_model)
score = mean_absolute_error(y_val, yhat)
# 打印输出 / Print output
print('Super Learner: MAE %.3f' % score)
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Version

# 05 — Version / 库版本信息

**Chapter 30 — File 5 of 7 / 第30章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **check the version of the mlens library**.

本脚本演示 **check the version of the mlens library**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — check the version of the mlens library

```python
import mlens
```

---
## Step 2 — report library version

```python
# 打印输出 / Print output
print(mlens.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: check the version of the mlens library 是机器学习中的常用技术。  
  *check the version of the mlens library is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Version / 库版本信息
# Complete Code / 完整代码
# ===============================

# check the version of the mlens library
import mlens
# report library version
# 打印输出 / Print output
print(mlens.__version__)
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Classification Mlens

# 06 — Classification Mlens / 分类

**Chapter 30 — File 6 of 7 / 第30章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **example of evaluating a super learner ensemble for classification with the mlens library**.

本脚本演示 **example of evaluating a super learner ensemble for classification with the mlens library**。

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
## Step 1 — example of evaluating a super learner ensemble for classification with the mlens library

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner
```

---
## Step 2 — create a list of base-models

```python
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='liblinear'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeClassifier())
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale', probability=True))
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianNB())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingClassifier(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestClassifier(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesClassifier(n_estimators=10))
	return models
```

---
## Step 3 — create the inputs and outputs

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
```

---
## Step 4 — split

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

---
## Step 5 — create the super learner

```python
# 获取长度 / Get length
ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X_train))
```

---
## Step 6 — add the base models

```python
ensemble.add(get_models())
```

---
## Step 7 — add the meta model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
ensemble.add_meta(LogisticRegression(solver='lbfgs'))
```

---
## Step 8 — fit the super learner

```python
ensemble.fit(X_train, y_train)
```

---
## Step 9 — summarize base learners

```python
# 打印输出 / Print output
print(ensemble.data)
```

---
## Step 10 — make predictions on hold out set

```python
yhat = ensemble.predict(X_test)
```

---
## Step 11 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Super Learner Accuracy: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of evaluating a super learner ensemble for classification with the mlens library 是机器学习中的常用技术。  
  *example of evaluating a super learner ensemble for classification with the mlens library is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Mlens / 分类
# Complete Code / 完整代码
# ===============================

# example of evaluating a super learner ensemble for classification with the mlens library
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import BaggingClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import RandomForestClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner

# create a list of base-models
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(LogisticRegression(solver='liblinear'))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(DecisionTreeClassifier())
 # 支持向量机 / Support Vector Machine
	models.append(SVC(gamma='scale', probability=True))
 # 添加元素到列表末尾 / Append element to list end
	models.append(GaussianNB())
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(KNeighborsClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(AdaBoostClassifier())
 # 添加元素到列表末尾 / Append element to list end
	models.append(BaggingClassifier(n_estimators=10))
 # 随机森林：多棵决策树投票 / Random Forest: multiple trees vote
	models.append(RandomForestClassifier(n_estimators=10))
 # 添加元素到列表末尾 / Append element to list end
	models.append(ExtraTreesClassifier(n_estimators=10))
	return models

# create the inputs and outputs
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
# split
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# create the super learner
# 获取长度 / Get length
ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X_train))
# add the base models
ensemble.add(get_models())
# add the meta model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
ensemble.add_meta(LogisticRegression(solver='lbfgs'))
# fit the super learner
ensemble.fit(X_train, y_train)
# summarize base learners
# 打印输出 / Print output
print(ensemble.data)
# make predictions on hold out set
yhat = ensemble.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Super Learner Accuracy: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Regression Mlens



---

### Chapter Summary / 章节总结

# Chapter 30 Summary / 第30章总结

## Theme / 主题: Chapter 30 / Chapter 30

This chapter contains **7 code files** demonstrating chapter 30.

本章包含 **7 个代码文件**，演示Chapter 30。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_classification_manual_evaluate.ipynb` — Classification Manual Evaluate
  3. `03_regression_dataset.ipynb` — Regression Dataset
  4. `04_regression_manual_evaluate.ipynb` — Regression Manual Evaluate
  5. `05_version.ipynb` — Version
  6. `06_classification_mlens.ipynb` — Classification Mlens
  7. `07_regression_mlens.ipynb` — Regression Mlens

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 30) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 30）是机器学习流水线中的基础构建块。

---
