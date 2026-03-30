# 集成学习
## Chapter 24

---

### Version

# 01 — Version / 库版本信息

**Chapter 24 — File 1 of 11 / 第24章 — 第1个文件（共11个）**

---

## Summary / 总结

This script demonstrates **check lightgbm library version**.

本脚本演示 **check lightgbm library version**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — check lightgbm library version

```python
import lightgbm
print(lightgbm.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: check lightgbm library version 是机器学习中的常用技术。  
  *check lightgbm library version is a common technique in machine learning.*

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

# check lightgbm library version
import lightgbm
print(lightgbm.__version__)
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Classification Dataset

# 02 — Classification Dataset / 分类

**Chapter 24 — File 2 of 11 / 第24章 — 第2个文件（共11个）**

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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Classification Evaluate

# 03 — Classification Evaluate / 分类

**Chapter 24 — File 3 of 11 / 第24章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **evaluate lightgbm algorithm for classification**.

本脚本演示 **evaluate lightgbm algorithm for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — evaluate lightgbm algorithm for classification

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = LGBMClassifier()
```

---
## Step 4 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model on the dataset

```python
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate lightgbm algorithm for classification 是机器学习中的常用技术。  
  *evaluate lightgbm algorithm for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Classification Evaluate / 分类
# Complete Code / 完整代码
# ===============================

# evaluate lightgbm algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = LGBMClassifier()
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Classification Predict

# 04 — Classification Predict / 分类

**Chapter 24 — File 4 of 11 / 第24章 — 第4个文件（共11个）**

---

## Summary / 总结

This script demonstrates **make predictions using lightgbm for classification**.

本脚本演示 **make predictions using lightgbm for classification**。

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
## Step 1 — make predictions using lightgbm for classification

```python
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = LGBMClassifier()
```

---
## Step 4 — fit the model on the whole dataset

```python
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
yhat = model.predict([row])
```

---
## Step 6 — summarize prediction

```python
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make predictions using lightgbm for classification 是机器学习中的常用技术。  
  *make predictions using lightgbm for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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

# make predictions using lightgbm for classification
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = LGBMClassifier()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
yhat = model.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Regression Evaluate

# 06 — Regression Evaluate / 回归

**Chapter 24 — File 6 of 11 / 第24章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **evaluate lightgbm ensemble for regression**.

本脚本演示 **evaluate lightgbm ensemble for regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — evaluate lightgbm ensemble for regression

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
```

---
## Step 3 — define the model

```python
model = LGBMRegressor()
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model

```python
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate lightgbm ensemble for regression 是机器学习中的常用技术。  
  *evaluate lightgbm ensemble for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Regression Evaluate / 回归
# Complete Code / 完整代码
# ===============================

# evaluate lightgbm ensemble for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from lightgbm import LGBMRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = LGBMRegressor()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Regression Predict

# 07 — Regression Predict / 回归

**Chapter 24 — File 7 of 11 / 第24章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **make predictions using lightgbm for regression**.

本脚本演示 **make predictions using lightgbm for regression**。

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
## Step 1 — make predictions using lightgbm for regression

```python
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
```

---
## Step 3 — define the model

```python
model = LGBMRegressor()
```

---
## Step 4 — fit the model on the whole dataset

```python
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [0.20543991, -0.97049844, -0.81403429, -0.23842689, -0.60704084, -0.48541492, 0.53113006, 2.01834338, -0.90745243, -1.85859731, -1.02334791, -0.6877744, 0.60984819, -0.70630121, -1.29161497, 1.32385441, 1.42150747, 1.26567231, 2.56569098, -0.11154792]
yhat = model.predict([row])
```

---
## Step 6 — summarize prediction

```python
print('Prediction: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make predictions using lightgbm for regression 是机器学习中的常用技术。  
  *make predictions using lightgbm for regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
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

# make predictions using lightgbm for regression
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=7)
# define the model
model = LGBMRegressor()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.20543991, -0.97049844, -0.81403429, -0.23842689, -0.60704084, -0.48541492, 0.53113006, 2.01834338, -0.90745243, -1.85859731, -1.02334791, -0.6877744, 0.60984819, -0.70630121, -1.29161497, 1.32385441, 1.42150747, 1.26567231, 2.56569098, -0.11154792]
yhat = model.predict([row])
# summarize prediction
print('Prediction: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 8 of 11

---

### Tune Num Trees

# 08 — Tune Num Trees / 决策树

**Chapter 24 — File 8 of 11 / 第24章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore lightgbm number of trees effect on performance**.

本脚本演示 **explore lightgbm number of trees effect on performance**。

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
## Step 1 — explore lightgbm number of trees effect on performance

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the number of trees to explore

```python
trees = [10, 50, 100, 500, 1000, 5000]
	for n in trees:
		models[str(n)] = LGBMClassifier(n_estimators=n)
	return models
```

---
## Step 5 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 6 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — evaluate the model and collect the results

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 8 — define dataset

```python
X, y = get_dataset()
```

---
## Step 9 — get the models to evaluate

```python
models = get_models()
```

---
## Step 10 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models.items():
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 12 — store the results

```python
results.append(scores)
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore lightgbm number of trees effect on performance 是机器学习中的常用技术。  
  *explore lightgbm number of trees effect on performance is a common technique in machine learning.*

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
# Tune Num Trees / 决策树
# Complete Code / 完整代码
# ===============================

# explore lightgbm number of trees effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# define the number of trees to explore
	trees = [10, 50, 100, 500, 1000, 5000]
	for n in trees:
		models[str(n)] = LGBMClassifier(n_estimators=n)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 9 of 11

---

### Tune Tree Depth

# 09 — Tune Tree Depth / 决策树

**Chapter 24 — File 9 of 11 / 第24章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore lightgbm tree depth effect on performance**.

本脚本演示 **explore lightgbm tree depth effect on performance**。

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
## Step 1 — explore lightgbm tree depth effect on performance

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — explore tree depths from 1 to 10

```python
for i in range(1,11):
		models[str(i)] = LGBMClassifier(max_depth=i, num_leaves=2**i)
	return models
```

---
## Step 5 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 6 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — evaluate the model and collect the results

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 8 — define dataset

```python
X, y = get_dataset()
```

---
## Step 9 — get the models to evaluate

```python
models = get_models()
```

---
## Step 10 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models.items():
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 12 — store the results

```python
results.append(scores)
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore lightgbm tree depth effect on performance 是机器学习中的常用技术。  
  *explore lightgbm tree depth effect on performance is a common technique in machine learning.*

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
# Tune Tree Depth / 决策树
# Complete Code / 完整代码
# ===============================

# explore lightgbm tree depth effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# explore tree depths from 1 to 10
	for i in range(1,11):
		models[str(i)] = LGBMClassifier(max_depth=i, num_leaves=2**i)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Tune Learning Rate

# 10 — Tune Learning Rate / 超参数调优

**Chapter 24 — File 10 of 11 / 第24章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore lightgbm learning rate effect on performance**.

本脚本演示 **explore lightgbm learning rate effect on performance**。

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
## Step 1 — explore lightgbm learning rate effect on performance

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the learning rates to explore

```python
rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
	for r in rates:
		key = '%.4f' % r
		models[key] = LGBMClassifier(learning_rate=r)
	return models
```

---
## Step 5 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 6 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — evaluate the model and collect the results

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 8 — define dataset

```python
X, y = get_dataset()
```

---
## Step 9 — get the models to evaluate

```python
models = get_models()
```

---
## Step 10 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models.items():
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 12 — store the results

```python
results.append(scores)
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore lightgbm learning rate effect on performance 是机器学习中的常用技术。  
  *explore lightgbm learning rate effect on performance is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tune Learning Rate / 超参数调优
# Complete Code / 完整代码
# ===============================

# explore lightgbm learning rate effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# define the learning rates to explore
	rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
	for r in rates:
		key = '%.4f' % r
		models[key] = LGBMClassifier(learning_rate=r)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Tune Boosting Type

# 11 — Tune Boosting Type / 超参数调优

**Chapter 24 — File 11 of 11 / 第24章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore lightgbm boosting type effect on performance**.

本脚本演示 **explore lightgbm boosting type effect on performance**。

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
## Step 1 — explore lightgbm boosting type effect on performance

```python
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define the boosting types to explore

```python
types = ['gbdt', 'dart', 'goss']
	for t in types:
		models[t] = LGBMClassifier(boosting_type=t)
	return models
```

---
## Step 5 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 6 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — evaluate the model and collect the results

```python
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 8 — define dataset

```python
X, y = get_dataset()
```

---
## Step 9 — get the models to evaluate

```python
models = get_models()
```

---
## Step 10 — evaluate the models and store results

```python
results, names = list(), list()
for name, model in models.items():
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 12 — store the results

```python
results.append(scores)
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore lightgbm boosting type effect on performance 是机器学习中的常用技术。  
  *explore lightgbm boosting type effect on performance is a common technique in machine learning.*

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
# Tune Boosting Type / 超参数调优
# Complete Code / 完整代码
# ===============================

# explore lightgbm boosting type effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# define the boosting types to explore
	types = ['gbdt', 'dart', 'goss']
	for t in types:
		models[t] = LGBMClassifier(boosting_type=t)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

### Chapter Summary

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **11 code files** demonstrating chapter 24.

本章包含 **11 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `01_version.ipynb` — Version
  2. `02_classification_dataset.ipynb` — Classification Dataset
  3. `03_classification_evaluate.ipynb` — Classification Evaluate
  4. `04_classification_predict.ipynb` — Classification Predict
  5. `05_regression_dataset.ipynb` — Regression Dataset
  6. `06_regression_evaluate.ipynb` — Regression Evaluate
  7. `07_regression_predict.ipynb` — Regression Predict
  8. `08_tune_num_trees.ipynb` — Tune Num Trees
  9. `09_tune_tree_depth.ipynb` — Tune Tree Depth
  10. `10_tune_learning_rate.ipynb` — Tune Learning Rate
  11. `11_tune_boosting_type.ipynb` — Tune Boosting Type

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
