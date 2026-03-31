# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 21

---

### Classification Dataset

# 01 — Classification Dataset / 分类

**Chapter 21 — File 1 of 11 / 第21章 — 第1个文件（共11个）**

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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
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
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 11

---

### Classification Evaluate

# 02 — Classification Evaluate / 分类

**Chapter 21 — File 2 of 11 / 第21章 — 第2个文件（共11个）**

---

## Summary / 总结

This script demonstrates **evaluate adaboost algorithm for classification**.

本脚本演示 **evaluate adaboost algorithm for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate adaboost algorithm for classification

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
from sklearn.ensemble import AdaBoostClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
```

---
## Step 3 — define the model

```python
model = AdaBoostClassifier()
```

---
## Step 4 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model on the dataset

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate adaboost algorithm for classification 是机器学习中的常用技术。  
  *evaluate adaboost algorithm for classification is a common technique in machine learning.*

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

# evaluate adaboost algorithm for classification
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
from sklearn.ensemble import AdaBoostClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# define the model
model = AdaBoostClassifier()
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 3 of 11

---

### Classification Predict

# 03 — Classification Predict / 分类

**Chapter 21 — File 3 of 11 / 第21章 — 第3个文件（共11个）**

---

## Summary / 总结

This script demonstrates **make predictions using adaboost for classification**.

本脚本演示 **make predictions using adaboost for classification**。

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
## Step 1 — make predictions using adaboost for classification

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
```

---
## Step 3 — define the model

```python
model = AdaBoostClassifier()
```

---
## Step 4 — fit the model on the whole dataset

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [-3.47224758, 1.95378146, 0.04875169, -0.91592588, -3.54022468, 1.96405547, -7.72564954, -2.64787168, -1.81726906, -1.67104974, 2.33762043, -4.30273117, 0.4839841, -1.28253034, -10.6704077, -0.7641103, -3.58493721, 2.07283886, 0.08385173, 0.91461126]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 6 — summarize prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make predictions using adaboost for classification 是机器学习中的常用技术。  
  *make predictions using adaboost for classification is a common technique in machine learning.*

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

# make predictions using adaboost for classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# define the model
model = AdaBoostClassifier()
# fit the model on the whole dataset
# 训练模型 / Train the model
model.fit(X, y)
# make a single prediction
row = [-3.47224758, 1.95378146, 0.04875169, -0.91592588, -3.54022468, 1.96405547, -7.72564954, -2.64787168, -1.81726906, -1.67104974, 2.33762043, -4.30273117, 0.4839841, -1.28253034, -10.6704077, -0.7641103, -3.58493721, 2.07283886, 0.08385173, 0.91461126]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize prediction
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 4 of 11

---

### Regression Dataset

# 04 — Regression Dataset / 回归

**Chapter 21 — File 4 of 11 / 第21章 — 第4个文件（共11个）**

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
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
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
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 5 of 11

---

### Regression Evaluate

# 05 — Regression Evaluate / 回归

**Chapter 21 — File 5 of 11 / 第21章 — 第5个文件（共11个）**

---

## Summary / 总结

This script demonstrates **evaluate adaboost ensemble for regression**.

本脚本演示 **evaluate adaboost ensemble for regression**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate adaboost ensemble for regression

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
from sklearn.ensemble import AdaBoostRegressor
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
```

---
## Step 3 — define the model

```python
model = AdaBoostRegressor()
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate adaboost ensemble for regression 是机器学习中的常用技术。  
  *evaluate adaboost ensemble for regression is a common technique in machine learning.*

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

# evaluate adaboost ensemble for regression
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
from sklearn.ensemble import AdaBoostRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
# define the model
model = AdaBoostRegressor()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 6 of 11

---

### Regression Predict

# 06 — Regression Predict / 回归

**Chapter 21 — File 6 of 11 / 第21章 — 第6个文件（共11个）**

---

## Summary / 总结

This script demonstrates **make predictions using adaboost for regression**.

本脚本演示 **make predictions using adaboost for regression**。

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
## Step 1 — make predictions using adaboost for regression

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
```

---
## Step 3 — define the model

```python
model = AdaBoostRegressor()
```

---
## Step 4 — fit the model on the whole dataset

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [1.20871625, 0.88440466, -0.9030013, -0.22687731, -0.82940077, -1.14410988, 1.26554256, -0.2842871, 1.43929072, 0.74250241, 0.34035501, 0.45363034, 0.1778756, -1.75252881, -1.33337384, -1.50337215, -0.45099008, 0.46160133, 0.58385557, -1.79936198]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 6 — summarize prediction

```python
# 打印输出 / Print output
print('Prediction: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make predictions using adaboost for regression 是机器学习中的常用技术。  
  *make predictions using adaboost for regression is a common technique in machine learning.*

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

# make predictions using adaboost for regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=6)
# define the model
model = AdaBoostRegressor()
# fit the model on the whole dataset
# 训练模型 / Train the model
model.fit(X, y)
# make a single prediction
row = [1.20871625, 0.88440466, -0.9030013, -0.22687731, -0.82940077, -1.14410988, 1.26554256, -0.2842871, 1.43929072, 0.74250241, 0.34035501, 0.45363034, 0.1778756, -1.75252881, -1.33337384, -1.50337215, -0.45099008, 0.46160133, 0.58385557, -1.79936198]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize prediction
# 打印输出 / Print output
print('Prediction: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 7 of 11

---

### Tune Num Trees

# 07 — Tune Num Trees / 决策树

**Chapter 21 — File 7 of 11 / 第21章 — 第7个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore adaboost ensemble number of trees effect on performance**.

本脚本演示 **explore adaboost ensemble number of trees effect on performance**。

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
## Step 1 — explore adaboost ensemble number of trees effect on performance

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
from sklearn.ensemble import AdaBoostClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — define number of trees to consider

```python
n_trees = [10, 50, 100, 500, 1000, 5000]
	for n in n_trees:
		models[str(n)] = AdaBoostClassifier(n_estimators=n)
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
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
# 获取字典的键值对 / Get dict key-value pairs
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
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
# 打印输出 / Print output
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

- **概念**: explore adaboost ensemble number of trees effect on performance 是机器学习中的常用技术。  
  *explore adaboost ensemble number of trees effect on performance is a common technique in machine learning.*

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

# explore adaboost ensemble number of trees effect on performance
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
from sklearn.ensemble import AdaBoostClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# define number of trees to consider
	n_trees = [10, 50, 100, 500, 1000, 5000]
	for n in n_trees:
		models[str(n)] = AdaBoostClassifier(n_estimators=n)
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

➡️ **Next / 下一步**: File 8 of 11

---

### Tune Max Depth

# 08 — Tune Max Depth / 超参数调优

**Chapter 21 — File 8 of 11 / 第21章 — 第8个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore adaboost ensemble tree depth effect on performance**.

本脚本演示 **explore adaboost ensemble tree depth effect on performance**。

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
## Step 1 — explore adaboost ensemble tree depth effect on performance

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
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — explore depths from 1 to 10

```python
# 生成整数序列 / Generate integer sequence
for i in range(1,11):
```

---
## Step 5 — define base model

```python
# 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
base = DecisionTreeClassifier(max_depth=i)
```

---
## Step 6 — define ensemble model

```python
models[str(i)] = AdaBoostClassifier(base_estimator=base)
	return models
```

---
## Step 7 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 8 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 9 — evaluate the model and collect the results

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 10 — define dataset

```python
X, y = get_dataset()
```

---
## Step 11 — get the models to evaluate

```python
models = get_models()
```

---
## Step 12 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 13 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 14 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 15 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 16 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore adaboost ensemble tree depth effect on performance 是机器学习中的常用技术。  
  *explore adaboost ensemble tree depth effect on performance is a common technique in machine learning.*

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
# Tune Max Depth / 超参数调优
# Complete Code / 完整代码
# ===============================

# explore adaboost ensemble tree depth effect on performance
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
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# explore depths from 1 to 10
 # 生成整数序列 / Generate integer sequence
	for i in range(1,11):
		# define base model
  # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
		base = DecisionTreeClassifier(max_depth=i)
		# define ensemble model
		models[str(i)] = AdaBoostClassifier(base_estimator=base)
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

➡️ **Next / 下一步**: File 9 of 11

---

### Tune Learning Rate

# 09 — Tune Learning Rate / 超参数调优

**Chapter 21 — File 9 of 11 / 第21章 — 第9个文件（共11个）**

---

## Summary / 总结

This script demonstrates **explore adaboost ensemble learning rate effect on performance**.

本脚本演示 **explore adaboost ensemble learning rate effect on performance**。

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
## Step 1 — explore adaboost ensemble learning rate effect on performance

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — explore learning rates from 0.1 to 2 in 0.1 increments

```python
# 生成整数序列 / Generate integer sequence
for i in arange(0.1, 2.1, 0.1):
		key = '%.1f' % i
		models[key] = AdaBoostClassifier(learning_rate=i)
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
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
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
# 获取字典的键值对 / Get dict key-value pairs
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
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 13 — summarize the performance along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore adaboost ensemble learning rate effect on performance 是机器学习中的常用技术。  
  *explore adaboost ensemble learning rate effect on performance is a common technique in machine learning.*

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

# explore adaboost ensemble learning rate effect on performance
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import std
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import arange
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# explore learning rates from 0.1 to 2 in 0.1 increments
 # 生成整数序列 / Generate integer sequence
	for i in arange(0.1, 2.1, 0.1):
		key = '%.1f' % i
		models[key] = AdaBoostClassifier(learning_rate=i)
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
pyplot.xticks(rotation=45)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 11

---

### Change Base Estimator

# 10 — Change Base Estimator / 10 Change Base Estimator

**Chapter 21 — File 10 of 11 / 第21章 — 第10个文件（共11个）**

---

## Summary / 总结

This script demonstrates **evaluate adaboost algorithm with logistic regression weak learner for classification**.

本脚本演示 **evaluate adaboost algorithm with logistic regression weak learner for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate adaboost algorithm with logistic regression weak learner for classification

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
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
```

---
## Step 3 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = AdaBoostClassifier(base_estimator=LogisticRegression())
```

---
## Step 4 — define the evaluation method

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model on the dataset

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate adaboost algorithm with logistic regression weak learner for classification 是机器学习中的常用技术。  
  *evaluate adaboost algorithm with logistic regression weak learner for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Change Base Estimator / 10 Change Base Estimator
# Complete Code / 完整代码
# ===============================

# evaluate adaboost algorithm with logistic regression weak learner for classification
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
from sklearn.ensemble import AdaBoostClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = AdaBoostClassifier(base_estimator=LogisticRegression())
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 11 of 11

---

### Grid Search

# 11 — Grid Search / 11 Grid Search

**Chapter 21 — File 11 of 11 / 第21章 — 第11个文件（共11个）**

---

## Summary / 总结

This script demonstrates **example of grid searching key hyperparameters for adaboost on a classification dataset**.

本脚本演示 **example of grid searching key hyperparameters for adaboost on a classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏋️ 训练模型 / Train Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — example of grid searching key hyperparameters for adaboost on a classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
```

---
## Step 3 — define the model with default hyperparameters

```python
model = AdaBoostClassifier()
```

---
## Step 4 — define the grid of values to search

```python
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
```

---
## Step 5 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — define the grid search procedure

```python
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
```

---
## Step 7 — execute the grid search

```python
grid_result = grid_search.fit(X, y)
```

---
## Step 8 — summarize the best score and configuration

```python
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

---
## Step 9 — summarize all scores that were evaluated

```python
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---
## Learning Notes / 学习笔记

- **概念**: example of grid searching key hyperparameters for adaboost on a classification dataset 是机器学习中的常用技术。  
  *example of grid searching key hyperparameters for adaboost on a classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `GridSearchCV` | 网格搜索超参数调优 | Grid search for hyperparameter tuning |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search / 11 Grid Search
# Complete Code / 完整代码
# ===============================

# example of grid searching key hyperparameters for adaboost on a classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import GridSearchCV
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import AdaBoostClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# define the model with default hyperparameters
model = AdaBoostClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
# 网格搜索：自动尝试所有参数组合找最优 / GridSearch: try all parameter combos to find best
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X, y)
# summarize the best score and configuration
# 打印输出 / Print output
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# 将多个序列配对 / Pair multiple sequences
for mean, stdev, param in zip(means, stds, params):
    # 打印输出 / Print output
    print("%f (%f) with: %r" % (mean, stdev, param))
```

---

### Chapter Summary / 章节总结

# Chapter 21 Summary / 第21章总结

## Theme / 主题: Chapter 21 / Chapter 21

This chapter contains **11 code files** demonstrating chapter 21.

本章包含 **11 个代码文件**，演示Chapter 21。

---
## Evolution / 演化路线

  1. `01_classification_dataset.ipynb` — Classification Dataset
  2. `02_classification_evaluate.ipynb` — Classification Evaluate
  3. `03_classification_predict.ipynb` — Classification Predict
  4. `04_regression_dataset.ipynb` — Regression Dataset
  5. `05_regression_evaluate.ipynb` — Regression Evaluate
  6. `06_regression_predict.ipynb` — Regression Predict
  7. `07_tune_num_trees.ipynb` — Tune Num Trees
  8. `08_tune_max_depth.ipynb` — Tune Max Depth
  9. `09_tune_learning_rate.ipynb` — Tune Learning Rate
  10. `10_change_base_estimator.ipynb` — Change Base Estimator
  11. `11_grid_search.ipynb` — Grid Search

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 21) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 21）是机器学习流水线中的基础构建块。

---
