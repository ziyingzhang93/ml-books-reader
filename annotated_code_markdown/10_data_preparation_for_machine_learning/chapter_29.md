# 机器学习数据准备 / Data Preparation for ML
## Chapter 29

---

### Define Dataset

# 01 — Define Dataset / 01 Define Dataset

**Chapter 29 — File 1 of 4 / 第29章 — 第1个文件（共4个）**

---

## Summary / 总结

This script demonstrates **test classification dataset**.

本脚本演示 **test classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — test classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
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
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: test classification dataset 是机器学习中的常用技术。  
  *test classification dataset is a common technique in machine learning.*

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
# Define Dataset / 01 Define Dataset
# Complete Code / 完整代码
# ===============================

# test classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# summarize the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 4

---

### Model With Pca Transform

# 02 — Model With Pca Transform / 数据变换

**Chapter 29 — File 2 of 4 / 第29章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **evaluate pca with logistic regression algorithm for classification**.

本脚本演示 **evaluate pca with logistic regression algorithm for classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — evaluate pca with logistic regression algorithm for classification

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
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the pipeline

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps = [('pca', PCA(n_components=10)), ('m', LogisticRegression())]
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(steps=steps)
```

---
## Step 4 — evaluate model

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 5 — report performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate pca with logistic regression algorithm for classification 是机器学习中的常用技术。  
  *evaluate pca with logistic regression algorithm for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Model With Pca Transform / 数据变换
# Complete Code / 完整代码
# ===============================

# evaluate pca with logistic regression algorithm for classification
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
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the pipeline
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps = [('pca', PCA(n_components=10)), ('m', LogisticRegression())]
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Compare Num Components

# 03 — Compare Num Components / 03 Compare Num Components

**Chapter 29 — File 3 of 4 / 第29章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **compare pca number of components with logistic regression algorithm for classification**.

本脚本演示 **compare pca number of components with logistic regression algorithm for classification**。

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
## Step 1 — compare pca number of components with logistic regression algorithm for classification

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
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
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
 # 生成整数序列 / Generate integer sequence
	for i in range(1,21):
  # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
		steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
  # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
		models[str(i)] = Pipeline(steps=steps)
	return models
```

---
## Step 4 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 5 — define dataset

```python
X, y = get_dataset()
```

---
## Step 6 — get the models to evaluate

```python
models = get_models()
```

---
## Step 7 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 8 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: compare pca number of components with logistic regression algorithm for classification 是机器学习中的常用技术。  
  *compare pca number of components with logistic regression algorithm for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Compare Num Components / 03 Compare Num Components
# Complete Code / 完整代码
# ===============================

# compare pca number of components with logistic regression algorithm for classification
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
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
 # 生成整数序列 / Generate integer sequence
	for i in range(1,21):
  # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
		steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
  # 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
		models[str(i)] = Pipeline(steps=steps)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
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
	scores = evaluate_model(model, X, y)
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Make Prediction

# 04 — Make Prediction / 04 Make Prediction

**Chapter 29 — File 4 of 4 / 第29章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **make predictions using pca with logistic regression**.

本脚本演示 **make predictions using pca with logistic regression**。

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
## Step 1 — make predictions using pca with logistic regression

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())]
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(steps=steps)
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
row = [[0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]]
# 用模型做预测 / Make predictions with model
yhat = model.predict(row)
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make predictions using pca with logistic regression 是机器学习中的常用技术。  
  *make predictions using pca with logistic regression is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Prediction / 04 Make Prediction
# Complete Code / 完整代码
# ===============================

# make predictions using pca with logistic regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())]
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
model = Pipeline(steps=steps)
# fit the model on the whole dataset
# 训练模型 / Train the model
model.fit(X, y)
# make a single prediction
row = [[0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]]
# 用模型做预测 / Make predictions with model
yhat = model.predict(row)
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---

### Chapter Summary / 章节总结

# Chapter 29 Summary / 第29章总结

## Theme / 主题: Chapter 29 / Chapter 29

This chapter contains **4 code files** demonstrating chapter 29.

本章包含 **4 个代码文件**，演示Chapter 29。

---
## Evolution / 演化路线

  1. `01_define_dataset.ipynb` — Define Dataset
  2. `02_model_with_pca_transform.ipynb` — Model With Pca Transform
  3. `03_compare_num_components.ipynb` — Compare Num Components
  4. `04_make_prediction.ipynb` — Make Prediction

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 29) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 29）是机器学习流水线中的基础构建块。

---
