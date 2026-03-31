# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 29

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 29 — File 1 of 9 / 第29章 — 第1个文件（共9个）**

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
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
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
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 9

---

### Classification Evaluate

# 02 — Classification Evaluate / 分类

**Chapter 29 — File 2 of 9 / 第29章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **blending ensemble for classification using hard voting**.

本脚本演示 **blending ensemble for classification using hard voting**。

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
## Step 1 — blending ensemble for classification using hard voting

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of base models

```python
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC()))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models
```

---
## Step 4 — fit the blending ensemble

```python
def fit_ensemble(models, X_train, X_val, y_train, y_val):
```

---
## Step 5 — fit all models on the training set and predict on hold out set

```python
meta_X = list()
	for _, model in models:
```

---
## Step 6 — fit in training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — predict on hold out set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_val)
```

---
## Step 8 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 9 — store predictions as input for blending

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 10 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 11 — define blending model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
blender = LogisticRegression()
```

---
## Step 12 — fit on predictions from base models

```python
blender.fit(meta_X, y_val)
	return blender
```

---
## Step 13 — make a prediction with the blending ensemble

```python
def predict_ensemble(models, blender, X_test):
```

---
## Step 14 — make predictions with base models

```python
meta_X = list()
	for _, model in models:
```

---
## Step 15 — predict with base model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 16 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 17 — store prediction

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 18 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 19 — predict

```python
return blender.predict(meta_X)
```

---
## Step 20 — define dataset

```python
X, y = get_dataset()
```

---
## Step 21 — split dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

---
## Step 22 — split training set into train and validation sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 23 — summarize data split

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
```

---
## Step 24 — create the base models

```python
models = get_models()
```

---
## Step 25 — train the blending ensemble

```python
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
```

---
## Step 26 — make predictions on test set

```python
yhat = predict_ensemble(models, blender, X_test)
```

---
## Step 27 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Blending Accuracy: %.3f' % (score*100))
```

---
## Learning Notes / 学习笔记

- **概念**: blending ensemble for classification using hard voting 是机器学习中的常用技术。  
  *blending ensemble for classification using hard voting is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
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
# Classification Evaluate / 分类
# Complete Code / 完整代码
# ===============================

# blending ensemble for classification using hard voting
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC()))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
  # 训练模型 / Train the model
		model.fit(X_train, y_train)
		# predict on hold out set
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Blending Accuracy: %.3f' % (score*100))
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Classification Evaluate Probabilities

# 03 — Classification Evaluate Probabilities / 分类

**Chapter 29 — File 3 of 9 / 第29章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **blending ensemble for classification using soft voting**.

本脚本演示 **blending ensemble for classification using soft voting**。

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
## Step 1 — blending ensemble for classification using soft voting

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of base models

```python
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC(probability=True)))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models
```

---
## Step 4 — fit the blending ensemble

```python
def fit_ensemble(models, X_train, X_val, y_train, y_val):
```

---
## Step 5 — fit all models on the training set and predict on hold out set

```python
meta_X = list()
	for _, model in models:
```

---
## Step 6 — fit in training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — predict on hold out set

```python
yhat = model.predict_proba(X_val)
```

---
## Step 8 — store predictions as input for blending

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 9 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 10 — define blending model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
blender = LogisticRegression()
```

---
## Step 11 — fit on predictions from base models

```python
blender.fit(meta_X, y_val)
	return blender
```

---
## Step 12 — make a prediction with the blending ensemble

```python
def predict_ensemble(models, blender, X_test):
```

---
## Step 13 — make predictions with base models

```python
meta_X = list()
	for _, model in models:
```

---
## Step 14 — predict with base model

```python
yhat = model.predict_proba(X_test)
```

---
## Step 15 — store prediction

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 16 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 17 — predict

```python
return blender.predict(meta_X)
```

---
## Step 18 — define dataset

```python
X, y = get_dataset()
```

---
## Step 19 — split dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

---
## Step 20 — split training set into train and validation sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 21 — summarize data split

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
```

---
## Step 22 — create the base models

```python
models = get_models()
```

---
## Step 23 — train the blending ensemble

```python
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
```

---
## Step 24 — make predictions on test set

```python
yhat = predict_ensemble(models, blender, X_test)
```

---
## Step 25 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Blending Accuracy: %.3f' % (score*100))
```

---
## Learning Notes / 学习笔记

- **概念**: blending ensemble for classification using soft voting 是机器学习中的常用技术。  
  *blending ensemble for classification using soft voting is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
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
# Classification Evaluate Probabilities / 分类
# Complete Code / 完整代码
# ===============================

# blending ensemble for classification using soft voting
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC(probability=True)))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
  # 训练模型 / Train the model
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict_proba(X_val)
		# store predictions as input for blending
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
		yhat = model.predict_proba(X_test)
		# store prediction
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Blending Accuracy: %.3f' % (score*100))
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Classification Standalone



---

### Classification Predict

# 05 — Classification Predict / 分类

**Chapter 29 — File 5 of 9 / 第29章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **example of making a prediction with a blending ensemble for classification**.

本脚本演示 **example of making a prediction with a blending ensemble for classification**。

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
## Step 1 — example of making a prediction with a blending ensemble for classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of base models

```python
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC(probability=True)))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models
```

---
## Step 4 — fit the blending ensemble

```python
def fit_ensemble(models, X_train, X_val, y_train, y_val):
```

---
## Step 5 — fit all models on the training set and predict on hold out set

```python
meta_X = list()
	for _, model in models:
```

---
## Step 6 — fit in training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — predict on hold out set

```python
yhat = model.predict_proba(X_val)
```

---
## Step 8 — store predictions as input for blending

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 9 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 10 — define blending model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
blender = LogisticRegression()
```

---
## Step 11 — fit on predictions from base models

```python
blender.fit(meta_X, y_val)
	return blender
```

---
## Step 12 — make a prediction with the blending ensemble

```python
def predict_ensemble(models, blender, X_test):
```

---
## Step 13 — make predictions with base models

```python
meta_X = list()
	for _, model in models:
```

---
## Step 14 — predict with base model

```python
yhat = model.predict_proba(X_test)
```

---
## Step 15 — store prediction

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 16 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 17 — predict

```python
return blender.predict(meta_X)
```

---
## Step 18 — define dataset

```python
X, y = get_dataset()
```

---
## Step 19 — split dataset set into train and validation sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 20 — summarize data split

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
```

---
## Step 21 — create the base models

```python
models = get_models()
```

---
## Step 22 — train the blending ensemble

```python
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
```

---
## Step 23 — make a prediction on a new row of data

```python
row = [-0.30335011, 2.68066314, 2.07794281, 1.15253537, -2.0583897, -2.51936601, 0.67513028, -3.20651939, -1.60345385, 3.68820714, 0.05370913, 1.35804433, 0.42011397, 1.4732839, 2.89997622, 1.61119399, 7.72630965, -2.84089477, -1.83977415, 1.34381989]
yhat = predict_ensemble(models, blender, [row])
```

---
## Step 24 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---
## Learning Notes / 学习笔记

- **概念**: example of making a prediction with a blending ensemble for classification 是机器学习中的常用技术。  
  *example of making a prediction with a blending ensemble for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
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
# Classification Predict / 分类
# Complete Code / 完整代码
# ===============================

# example of making a prediction with a blending ensemble for classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVC(probability=True)))
 # 添加元素到列表末尾 / Append element to list end
	models.append(('bayes', GaussianNB()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
  # 训练模型 / Train the model
		model.fit(X_train, y_train)
		# predict on hold out set
		yhat = model.predict_proba(X_val)
		# store predictions as input for blending
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	blender = LogisticRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
		yhat = model.predict_proba(X_test)
		# store prediction
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset set into train and validation sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize data split
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make a prediction on a new row of data
row = [-0.30335011, 2.68066314, 2.07794281, 1.15253537, -2.0583897, -2.51936601, 0.67513028, -3.20651939, -1.60345385, 3.68820714, 0.05370913, 1.35804433, 0.42011397, 1.4732839, 2.89997622, 1.61119399, 7.72630965, -2.84089477, -1.83977415, 1.34381989]
yhat = predict_ensemble(models, blender, [row])
# summarize prediction
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Regression Dataset

# 06 — Regression Dataset / 回归

**Chapter 29 — File 6 of 9 / 第29章 — 第6个文件（共9个）**

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
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
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
X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Regression Evaluate

# 07 — Regression Evaluate / 回归

**Chapter 29 — File 7 of 9 / 第29章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluate blending ensemble for regression**.

本脚本演示 **evaluate blending ensemble for regression**。

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
## Step 1 — evaluate blending ensemble for regression

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y
```

---
## Step 3 — get a list of base models

```python
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(('lr', LinearRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVR()))
	return models
```

---
## Step 4 — fit the blending ensemble

```python
def fit_ensemble(models, X_train, X_val, y_train, y_val):
```

---
## Step 5 — fit all models on the training set and predict on hold out set

```python
meta_X = list()
	for _, model in models:
```

---
## Step 6 — fit in training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — predict on hold out set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_val)
```

---
## Step 8 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 9 — store predictions as input for blending

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 10 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 11 — define blending model

```python
blender = LinearRegression()
```

---
## Step 12 — fit on predictions from base models

```python
blender.fit(meta_X, y_val)
	return blender
```

---
## Step 13 — make a prediction with the blending ensemble

```python
def predict_ensemble(models, blender, X_test):
```

---
## Step 14 — make predictions with base models

```python
meta_X = list()
	for _, model in models:
```

---
## Step 15 — predict with base model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 16 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 17 — store prediction

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 18 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 19 — predict

```python
return blender.predict(meta_X)
```

---
## Step 20 — define dataset

```python
X, y = get_dataset()
```

---
## Step 21 — split dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

---
## Step 22 — split training set into train and validation sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
```

---
## Step 23 — summarize data split

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
```

---
## Step 24 — create the base models

```python
models = get_models()
```

---
## Step 25 — train the blending ensemble

```python
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
```

---
## Step 26 — make predictions on test set

```python
yhat = predict_ensemble(models, blender, X_test)
```

---
## Step 27 — evaluate predictions

```python
score = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('Blending MAE: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate blending ensemble for regression 是机器学习中的常用技术。  
  *evaluate blending ensemble for regression is a common technique in machine learning.*

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
# Regression Evaluate / 回归
# Complete Code / 完整代码
# ===============================

# evaluate blending ensemble for regression
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_absolute_error
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(('lr', LinearRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVR()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
  # 训练模型 / Train the model
		model.fit(X_train, y_train)
		# predict on hold out set
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LinearRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# split training set into train and validation sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)
# summarize data split
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
# 打印输出 / Print output
print('Blending MAE: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Regression Standalone



---

### Regression Predict

# 09 — Regression Predict / 回归

**Chapter 29 — File 9 of 9 / 第29章 — 第9个文件（共9个）**

---

## Summary / 总结

This script demonstrates **example of making a prediction with a blending ensemble for regression**.

本脚本演示 **example of making a prediction with a blending ensemble for regression**。

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
## Step 1 — example of making a prediction with a blending ensemble for regression

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y
```

---
## Step 3 — get a list of base models

```python
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(('lr', LinearRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVR()))
	return models
```

---
## Step 4 — fit the blending ensemble

```python
def fit_ensemble(models, X_train, X_val, y_train, y_val):
```

---
## Step 5 — fit all models on the training set and predict on hold out set

```python
meta_X = list()
	for _, model in models:
```

---
## Step 6 — fit in training set

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — predict on hold out set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_val)
```

---
## Step 8 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 9 — store predictions as input for blending

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 10 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 11 — define blending model

```python
blender = LinearRegression()
```

---
## Step 12 — fit on predictions from base models

```python
blender.fit(meta_X, y_val)
	return blender
```

---
## Step 13 — make a prediction with the blending ensemble

```python
def predict_ensemble(models, blender, X_test):
```

---
## Step 14 — make predictions with base models

```python
meta_X = list()
	for _, model in models:
```

---
## Step 15 — predict with base model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 16 — reshape predictions into a matrix with one column

```python
# 改变数组形状（不改变数据） / Reshape array (data unchanged)
yhat = yhat.reshape(len(yhat), 1)
```

---
## Step 17 — store prediction

```python
# 添加元素到列表末尾 / Append element to list end
meta_X.append(yhat)
```

---
## Step 18 — create 2d array from predictions, each set is an input feature

```python
meta_X = hstack(meta_X)
```

---
## Step 19 — predict

```python
return blender.predict(meta_X)
```

---
## Step 20 — define dataset

```python
X, y = get_dataset()
```

---
## Step 21 — split dataset set into train and validation sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 22 — summarize data split

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
```

---
## Step 23 — create the base models

```python
models = get_models()
```

---
## Step 24 — train the blending ensemble

```python
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
```

---
## Step 25 — make a prediction on a new row of data

```python
row = [-0.24038754, 0.55423865, -0.48979221, 1.56074459, -1.16007611, 1.10049103, 1.18385406, -1.57344162, 0.97862519, -0.03166643, 1.77099821, 1.98645499, 0.86780193, 2.01534177, 2.51509494, -1.04609004, -0.19428148, -0.05967386, -2.67168985, 1.07182911]
yhat = predict_ensemble(models, blender, [row])
```

---
## Step 26 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted: %.3f' % (yhat[0]))
```

---
## Learning Notes / 学习笔记

- **概念**: example of making a prediction with a blending ensemble for regression 是机器学习中的常用技术。  
  *example of making a prediction with a blending ensemble for regression is a common technique in machine learning.*

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
# Regression Predict / 回归
# Complete Code / 完整代码
# ===============================

# example of making a prediction with a blending ensemble for regression
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import hstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=10000, n_features=20, n_informative=10, noise=0.3, random_state=7)
	return X, y

# get a list of base models
def get_models():
	models = list()
 # 添加元素到列表末尾 / Append element to list end
	models.append(('lr', LinearRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	models.append(('svm', SVR()))
	return models

# fit the blending ensemble
def fit_ensemble(models, X_train, X_val, y_train, y_val):
	# fit all models on the training set and predict on hold out set
	meta_X = list()
	for _, model in models:
		# fit in training set
  # 训练模型 / Train the model
		model.fit(X_train, y_train)
		# predict on hold out set
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_val)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store predictions as input for blending
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# define blending model
	blender = LinearRegression()
	# fit on predictions from base models
	blender.fit(meta_X, y_val)
	return blender

# make a prediction with the blending ensemble
def predict_ensemble(models, blender, X_test):
	# make predictions with base models
	meta_X = list()
	for _, model in models:
		# predict with base model
  # 用模型做预测 / Make predictions with model
		yhat = model.predict(X_test)
		# reshape predictions into a matrix with one column
  # 改变数组形状（不改变数据） / Reshape array (data unchanged)
		yhat = yhat.reshape(len(yhat), 1)
		# store prediction
  # 添加元素到列表末尾 / Append element to list end
		meta_X.append(yhat)
	# create 2d array from predictions, each set is an input feature
	meta_X = hstack(meta_X)
	# predict
	return blender.predict(meta_X)

# define dataset
X, y = get_dataset()
# split dataset set into train and validation sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize data split
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print('Train: %s, Val: %s' % (X_train.shape, X_val.shape))
# create the base models
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make a prediction on a new row of data
row = [-0.24038754, 0.55423865, -0.48979221, 1.56074459, -1.16007611, 1.10049103, 1.18385406, -1.57344162, 0.97862519, -0.03166643, 1.77099821, 1.98645499, 0.86780193, 2.01534177, 2.51509494, -1.04609004, -0.19428148, -0.05967386, -2.67168985, 1.07182911]
yhat = predict_ensemble(models, blender, [row])
# summarize prediction
# 打印输出 / Print output
print('Predicted: %.3f' % (yhat[0]))
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **9 code files** demonstrating chapter 29.

本章包含 **9 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_classification_evaluate.ipynb` — Classification Evaluate
  3. `03_classification_evaluate_probabilities.ipynb` — Classification Evaluate Probabilities
  4. `04_classification_standalone.ipynb` — Classification Standalone
  5. `05_classification_predict.ipynb` — Classification Predict
  6. `06_regression_dataset.ipynb` — Regression Dataset
  7. `07_regression_evaluate.ipynb` — Regression Evaluate
  8. `08_regression_standalone.ipynb` — Regression Standalone
  9. `09_regression_predict.ipynb` — Regression Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
