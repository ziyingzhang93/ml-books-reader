# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 08

---

### Dataset



---

### Ovr Builtin Evaluate

# 02 — Ovr Builtin Evaluate / 模型评估

**Chapter 08 — File 2 of 7 / 第08章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluate logistic regression for multi-class classification using built-in one-vs-rest**.

本脚本演示 **evaluate logistic regression for multi-class classification using built-in one-vs-rest**。

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
## Step 1 — evaluate logistic regression for multi-class classification using built-in one-vs-rest

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
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
```

---
## Step 3 — define the model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(multi_class='ovr')
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model and collect the scores

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — summarize the performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate logistic regression for multi-class classification using built-in one-vs-rest 是机器学习中的常用技术。  
  *evaluate logistic regression for multi-class classification using built-in one-vs-rest is a common technique in machine learning.*

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
# Ovr Builtin Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate logistic regression for multi-class classification using built-in one-vs-rest
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
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define the model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(multi_class='ovr')
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Ovr Evaluate

# 03 — Ovr Evaluate / 模型评估

**Chapter 08 — File 3 of 7 / 第08章 — 第3个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluate logistic regression for multi-class classification using one-vs-rest**.

本脚本演示 **evaluate logistic regression for multi-class classification using one-vs-rest**。

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
## Step 1 — evaluate logistic regression for multi-class classification using one-vs-rest

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
from sklearn.multiclass import OneVsRestClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
```

---
## Step 3 — define model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
```

---
## Step 4 — define the one-vs-rest strategy

```python
ovr = OneVsRestClassifier(model)
```

---
## Step 5 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate the model and collect the scores

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(ovr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — summarize the performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate logistic regression for multi-class classification using one-vs-rest 是机器学习中的常用技术。  
  *evaluate logistic regression for multi-class classification using one-vs-rest is a common technique in machine learning.*

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
# Ovr Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate logistic regression for multi-class classification using one-vs-rest
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
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# define the one-vs-rest strategy
ovr = OneVsRestClassifier(model)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(ovr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 4 of 7

---

### Ovr Predict

# 04 — Ovr Predict / 04 Ovr Predict

**Chapter 08 — File 4 of 7 / 第08章 — 第4个文件（共7个）**

---

## Summary / 总结

This script demonstrates **make a prediction with logistic regression using one-vs-rest**.

本脚本演示 **make a prediction with logistic regression using one-vs-rest**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — make a prediction with logistic regression using one-vs-rest

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.multiclass import OneVsRestClassifier
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
```

---
## Step 3 — define model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
```

---
## Step 4 — define the one-vs-rest strategy

```python
ovr = OneVsRestClassifier(model)
```

---
## Step 5 — fit the model on the whole dataset

```python
ovr.fit(X, y)
```

---
## Step 6 — make a single prediction

```python
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223, 1.80998823, 0.58810926, -0.02542177, -0.52835426]
yhat = ovr.predict([row])
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with logistic regression using one-vs-rest 是机器学习中的常用技术。  
  *make a prediction with logistic regression using one-vs-rest is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ovr Predict / 04 Ovr Predict
# Complete Code / 完整代码
# ===============================

# make a prediction with logistic regression using one-vs-rest
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression()
# define the one-vs-rest strategy
ovr = OneVsRestClassifier(model)
# fit the model on the whole dataset
ovr.fit(X, y)
# make a single prediction
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223, 1.80998823, 0.58810926, -0.02542177, -0.52835426]
yhat = ovr.predict([row])
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 5 of 7

---

### Ovo Builtin Evaluate

# 05 — Ovo Builtin Evaluate / 模型评估

**Chapter 08 — File 5 of 7 / 第08章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **evaluate SVM for multi-class classification using built-in one-vs-one**.

本脚本演示 **evaluate SVM for multi-class classification using built-in one-vs-one**。

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
## Step 1 — evaluate SVM for multi-class classification using built-in one-vs-one

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
from sklearn.svm import SVC
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
```

---
## Step 3 — define the model

```python
# 支持向量机 / Support Vector Machine
model = SVC(decision_function_shape='ovo')
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model and collect the scores

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — summarize the performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate SVM for multi-class classification using built-in one-vs-one 是机器学习中的常用技术。  
  *evaluate SVM for multi-class classification using built-in one-vs-one is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ovo Builtin Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate SVM for multi-class classification using built-in one-vs-one
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
from sklearn.svm import SVC
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define the model
# 支持向量机 / Support Vector Machine
model = SVC(decision_function_shape='ovo')
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Ovo Evaluate



---

### Ovo Predict



---

### Chapter Summary / 章节总结

# Chapter 08 Summary / 第08章总结

## Theme / 主题: Chapter 08 / Chapter 08

This chapter contains **7 code files** demonstrating chapter 08.

本章包含 **7 个代码文件**，演示Chapter 08。

---
## Evolution / 演化路线

  1. `01_dataset.ipynb` — Dataset
  2. `02_ovr_builtin_evaluate.ipynb` — Ovr Builtin Evaluate
  3. `03_ovr_evaluate.ipynb` — Ovr Evaluate
  4. `04_ovr_predict.ipynb` — Ovr Predict
  5. `05_ovo_builtin_evaluate.ipynb` — Ovo Builtin Evaluate
  6. `06_ovo_evaluate.ipynb` — Ovo Evaluate
  7. `07_ovo_predict.ipynb` — Ovo Predict

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 08) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 08）是机器学习流水线中的基础构建块。

---
