# 机器学习数据准备 / Data Preparation for ML
## Chapter 04

---

### Define Dataset

# 01 — Define Dataset / 01 Define Dataset

**Chapter 04 — File 1 of 5 / 第04章 — 第1个文件（共5个）**

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

➡️ **Next / 下一步**: File 2 of 5

---

### Train Test Eval Leakage

# 02 — Train Test Eval Leakage / 模型评估

**Chapter 04 — File 2 of 5 / 第04章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **naive approach to normalizing the data before splitting the data and evaluating the model**.

本脚本演示 **naive approach to normalizing the data before splitting the data and evaluating the model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — naive approach to normalizing the data before splitting the data and evaluating the model

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — standardize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = scaler.fit_transform(X)
```

---
## Step 4 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 5 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 6 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 7 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: naive approach to normalizing the data before splitting the data and evaluating the model 是机器学习中的常用技术。  
  *naive approach to normalizing the data before splitting the data and evaluating the model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Train Test Eval Leakage / 模型评估
# Complete Code / 完整代码
# ===============================

# naive approach to normalizing the data before splitting the data and evaluating the model
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# standardize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = scaler.fit_transform(X)
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Train Test Eval Correct

# 03 — Train Test Eval Correct / 模型评估

**Chapter 04 — File 3 of 5 / 第04章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **correct approach for normalizing the data after the data is split before the model is evaluated**.

本脚本演示 **correct approach for normalizing the data after the data is split before the model is evaluated**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — correct approach for normalizing the data after the data is split before the model is evaluated

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — split into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

---
## Step 4 — define the scaler

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
```

---
## Step 5 — fit on the training dataset

```python
scaler.fit(X_train)
```

---
## Step 6 — scale the training dataset

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train)
```

---
## Step 7 — scale the test dataset

```python
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test)
```

---
## Step 8 — fit the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 9 — evaluate the model

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 10 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (accuracy*100))
```

---
## Learning Notes / 学习笔记

- **概念**: correct approach for normalizing the data after the data is split before the model is evaluated 是机器学习中的常用技术。  
  *correct approach for normalizing the data after the data is split before the model is evaluated is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
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
# Train Test Eval Correct / 模型评估
# Complete Code / 完整代码
# ===============================

# correct approach for normalizing the data after the data is split before the model is evaluated
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define the scaler
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# fit on the training dataset
scaler.fit(X_train)
# scale the training dataset
# 用已拟合的模型转换数据 / Transform data with fitted model
X_train = scaler.transform(X_train)
# scale the test dataset
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test = scaler.transform(X_test)
# fit the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# 训练模型 / Train the model
model.fit(X_train, y_train)
# evaluate the model
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (accuracy*100))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Cv Eval Leakage

# 04 — Cv Eval Leakage / 模型评估

**Chapter 04 — File 4 of 5 / 第04章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **naive data preparation for model evaluation with k-fold cross-validation**.

本脚本演示 **naive data preparation for model evaluation with k-fold cross-validation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — naive data preparation for model evaluation with k-fold cross-validation

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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — standardize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = scaler.fit_transform(X)
```

---
## Step 4 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
```

---
## Step 5 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate the model using cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — report performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

---
## Learning Notes / 学习笔记

- **概念**: naive data preparation for model evaluation with k-fold cross-validation 是机器学习中的常用技术。  
  *naive data preparation for model evaluation with k-fold cross-validation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv Eval Leakage / 模型评估
# Complete Code / 完整代码
# ===============================

# naive data preparation for model evaluation with k-fold cross-validation
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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# standardize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler()
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X = scaler.fit_transform(X)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Cv Eval Correct

# 05 — Cv Eval Correct / 模型评估

**Chapter 04 — File 5 of 5 / 第04章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **correct data preparation for model evaluation with k-fold cross-validation**.

本脚本演示 **correct data preparation for model evaluation with k-fold cross-validation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
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
## Step 1 — correct data preparation for model evaluation with k-fold cross-validation

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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the pipeline

```python
steps = list()
# 归一化到[0,1]范围 / Normalize to [0,1] range
steps.append(('scaler', MinMaxScaler()))
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps.append(('model', LogisticRegression()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=steps)
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model using cross-validation

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

---
## Learning Notes / 学习笔记

- **概念**: correct data preparation for model evaluation with k-fold cross-validation 是机器学习中的常用技术。  
  *correct data preparation for model evaluation with k-fold cross-validation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cv Eval Correct / 模型评估
# Complete Code / 完整代码
# ===============================

# correct data preparation for model evaluation with k-fold cross-validation
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
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.pipeline import Pipeline
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the pipeline
steps = list()
# 归一化到[0,1]范围 / Normalize to [0,1] range
steps.append(('scaler', MinMaxScaler()))
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
steps.append(('model', LogisticRegression()))
# 管道：将多个步骤串成流水线 / Pipeline: chain multiple steps into workflow
pipeline = Pipeline(steps=steps)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Accuracy: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))
```

---

### Chapter Summary / 章节总结

# Chapter 04 Summary / 第04章总结

## Theme / 主题: Chapter 04 / Chapter 04

This chapter contains **5 code files** demonstrating chapter 04.

本章包含 **5 个代码文件**，演示Chapter 04。

---
## Evolution / 演化路线

  1. `01_define_dataset.ipynb` — Define Dataset
  2. `02_train_test_eval_leakage.ipynb` — Train Test Eval Leakage
  3. `03_train_test_eval_correct.ipynb` — Train Test Eval Correct
  4. `04_cv_eval_leakage.ipynb` — Cv Eval Leakage
  5. `05_cv_eval_correct.ipynb` — Cv Eval Correct

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 04) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 04）是机器学习流水线中的基础构建块。

---
