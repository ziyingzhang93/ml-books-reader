# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 28

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 28 — File 1 of 8 / 第28章 — 第1个文件（共8个）**

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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Classification Compare Standalone



---

### Classification Stacking Compare

# 03 — Classification Stacking Compare / 分类

**Chapter 28 — File 3 of 8 / 第28章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare ensemble to each baseline classifier**.

本脚本演示 **compare ensemble to each baseline classifier**。

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
## Step 1 — compare ensemble to each baseline classifier

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
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
	return X, y
```

---
## Step 3 — get a stacking ensemble of models

```python
def get_stacking():
```

---
## Step 4 — define the base models

```python
level0 = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	level0.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	level0.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	level0.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	level0.append(('svm', SVC()))
 # 添加元素到列表末尾 / Append element to list end
	level0.append(('bayes', GaussianNB()))
```

---
## Step 5 — define meta learner model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
level1 = LogisticRegression()
```

---
## Step 6 — define the stacking ensemble

```python
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
```

---
## Step 7 — get a list of models to evaluate

```python
def get_models():
	models = dict()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models['lr'] = LogisticRegression()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models['knn'] = KNeighborsClassifier()
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models['cart'] = DecisionTreeClassifier()
 # 支持向量机 / Support Vector Machine
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models
```

---
## Step 8 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 9 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 10 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 11 — define dataset

```python
X, y = get_dataset()
```

---
## Step 12 — get the models to evaluate

```python
models = get_models()
```

---
## Step 13 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 14 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 15 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 16 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 17 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare ensemble to each baseline classifier 是机器学习中的常用技术。  
  *compare ensemble to each baseline classifier is a common technique in machine learning.*

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
# Classification Stacking Compare / 分类
# Complete Code / 完整代码
# ===============================

# compare ensemble to each baseline classifier
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
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVC
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
	return X, y

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	level0.append(('lr', LogisticRegression()))
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	level0.append(('knn', KNeighborsClassifier()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	level0.append(('cart', DecisionTreeClassifier()))
 # 支持向量机 / Support Vector Machine
	level0.append(('svm', SVC()))
 # 添加元素到列表末尾 / Append element to list end
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	models['lr'] = LogisticRegression()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models['knn'] = KNeighborsClassifier()
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models['cart'] = DecisionTreeClassifier()
 # 支持向量机 / Support Vector Machine
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
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

➡️ **Next / 下一步**: File 4 of 8

---

### Classification Predict

# 04 — Classification Predict / 分类

**Chapter 28 — File 4 of 8 / 第28章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **make a prediction with a stacking ensemble**.

本脚本演示 **make a prediction with a stacking ensemble**。

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
## Step 1 — make a prediction with a stacking ensemble

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingClassifier
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
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
```

---
## Step 3 — define the base models

```python
level0 = list()
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
level0.append(('lr', LogisticRegression()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
level0.append(('knn', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
level0.append(('cart', DecisionTreeClassifier()))
# 支持向量机 / Support Vector Machine
level0.append(('svm', SVC()))
# 添加元素到列表末尾 / Append element to list end
level0.append(('bayes', GaussianNB()))
```

---
## Step 4 — define meta learner model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
level1 = LogisticRegression()
```

---
## Step 5 — define the stacking ensemble

```python
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
```

---
## Step 6 — fit the model on all available data

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 7 — make a prediction for one example

```python
row = [2.47475454, 0.40165523, 1.68081787, 2.88940715, 0.91704519, -3.07950644, 4.39961206, 0.72464273, -4.86563631, -6.06338084, -1.22209949, -0.4699618, 1.01222748, -0.6899355, -0.53000581, 6.86966784, -3.27211075, -6.59044146, -2.21290585, -3.139579]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 8 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with a stacking ensemble 是机器学习中的常用技术。  
  *make a prediction with a stacking ensemble is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Predict / 分类
# Complete Code / 完整代码
# ===============================

# make a prediction with a stacking ensemble
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingClassifier
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
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# define the base models
level0 = list()
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
level0.append(('lr', LogisticRegression()))
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
level0.append(('knn', KNeighborsClassifier()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
level0.append(('cart', DecisionTreeClassifier()))
# 支持向量机 / Support Vector Machine
level0.append(('svm', SVC()))
# 添加元素到列表末尾 / Append element to list end
level0.append(('bayes', GaussianNB()))
# define meta learner model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
level1 = LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
# 训练模型 / Train the model
model.fit(X, y)
# make a prediction for one example
row = [2.47475454, 0.40165523, 1.68081787, 2.88940715, 0.91704519, -3.07950644, 4.39961206, 0.72464273, -4.86563631, -6.06338084, -1.22209949, -0.4699618, 1.01222748, -0.6899355, -0.53000581, 6.86966784, -3.27211075, -6.59044146, -2.21290585, -3.139579]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize prediction
# 打印输出 / Print output
print('Predicted Class: %d' % (yhat))
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Regression Dataset

# 05 — Regression Dataset / 回归

**Chapter 28 — File 5 of 8 / 第28章 — 第5个文件（共8个）**

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

➡️ **Next / 下一步**: File 6 of 8

---

### Regression Compare Standalone



---

### Regression Stacking Compare

# 07 — Regression Stacking Compare / 回归

**Chapter 28 — File 7 of 8 / 第28章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **compare ensemble to each standalone models for regression**.

本脚本演示 **compare ensemble to each standalone models for regression**。

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
## Step 1 — compare ensemble to each standalone models for regression

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
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingRegressor
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
## Step 3 — get a stacking ensemble of models

```python
def get_stacking():
```

---
## Step 4 — define the base models

```python
level0 = list()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	level0.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	level0.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	level0.append(('svm', SVR()))
```

---
## Step 5 — define meta learner model

```python
level1 = LinearRegression()
```

---
## Step 6 — define the stacking ensemble

```python
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model
```

---
## Step 7 — get a list of models to evaluate

```python
def get_models():
	models = dict()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models['knn'] = KNeighborsRegressor()
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models['cart'] = DecisionTreeRegressor()
 # 支持向量机 / Support Vector Machine
	models['svm'] = SVR()
	models['stacking'] = get_stacking()
	return models
```

---
## Step 8 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 9 — define the evaluation procedure

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 10 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 11 — define dataset

```python
X, y = get_dataset()
```

---
## Step 12 — get the models to evaluate

```python
models = get_models()
```

---
## Step 13 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 14 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 15 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 16 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 17 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare ensemble to each standalone models for regression 是机器学习中的常用技术。  
  *compare ensemble to each standalone models for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Stacking Compare / 回归
# Complete Code / 完整代码
# ===============================

# compare ensemble to each standalone models for regression
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
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingRegressor
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
	return X, y

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	level0.append(('knn', KNeighborsRegressor()))
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	level0.append(('cart', DecisionTreeRegressor()))
 # 支持向量机 / Support Vector Machine
	level0.append(('svm', SVR()))
	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
 # K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
	models['knn'] = KNeighborsRegressor()
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	models['cart'] = DecisionTreeRegressor()
 # 支持向量机 / Support Vector Machine
	models['svm'] = SVR()
	models['stacking'] = get_stacking()
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

### Regression Predict

# 08 — Regression Predict / 回归

**Chapter 28 — File 8 of 8 / 第28章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **make a prediction with a stacking ensemble**.

本脚本演示 **make a prediction with a stacking ensemble**。

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
## Step 1 — make a prediction with a stacking ensemble

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingRegressor
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
```

---
## Step 3 — define the base models

```python
level0 = list()
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
level0.append(('knn', KNeighborsRegressor()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
level0.append(('cart', DecisionTreeRegressor()))
# 支持向量机 / Support Vector Machine
level0.append(('svm', SVR()))
```

---
## Step 4 — define meta learner model

```python
level1 = LinearRegression()
```

---
## Step 5 — define the stacking ensemble

```python
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
```

---
## Step 6 — fit the model on all available data

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 7 — make a prediction for one example

```python
row = [0.59332206, -0.56637507, 1.34808718, -0.57054047, -0.72480487, 1.05648449, 0.77744852, 0.07361796, 0.88398267, 2.02843157, 1.01902732, 0.11227799, 0.94218853, 0.26741783, 0.91458143, -0.72759572, 1.08842814, -0.61450942, -0.69387293, 1.69169009]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 8 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted Value: %.3f' % (yhat))
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with a stacking ensemble 是机器学习中的常用技术。  
  *make a prediction with a stacking ensemble is a common technique in machine learning.*

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

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Predict / 回归
# Complete Code / 完整代码
# ===============================

# make a prediction with a stacking ensemble
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LinearRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import KNeighborsRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeRegressor
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import SVR
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import StackingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
# define the base models
level0 = list()
# K近邻：看周围K个最近的点投票 / KNN: vote from K nearest neighbors
level0.append(('knn', KNeighborsRegressor()))
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
level0.append(('cart', DecisionTreeRegressor()))
# 支持向量机 / Support Vector Machine
level0.append(('svm', SVR()))
# define meta learner model
level1 = LinearRegression()
# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
# 训练模型 / Train the model
model.fit(X, y)
# make a prediction for one example
row = [0.59332206, -0.56637507, 1.34808718, -0.57054047, -0.72480487, 1.05648449, 0.77744852, 0.07361796, 0.88398267, 2.02843157, 1.01902732, 0.11227799, 0.94218853, 0.26741783, 0.91458143, -0.72759572, 1.08842814, -0.61450942, -0.69387293, 1.69169009]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize prediction
# 打印输出 / Print output
print('Predicted Value: %.3f' % (yhat))
```

---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **8 code files** demonstrating chapter 28.

本章包含 **8 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_classification_compare_standalone.ipynb` — Classification Compare Standalone
  3. `03_classification_stacking_compare.ipynb` — Classification Stacking Compare
  4. `04_classification_predict.ipynb` — Classification Predict
  5. `05_regression_dataset.ipynb` — Regression Dataset
  6. `06_regression_compare_standalone.ipynb` — Regression Compare Standalone
  7. `07_regression_stacking_compare.ipynb` — Regression Stacking Compare
  8. `08_regression_predict.ipynb` — Regression Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
