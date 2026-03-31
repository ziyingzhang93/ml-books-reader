# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 21

---

### Roc Curve



---

### Roc Optimal Threshold

# 02 — Roc Optimal Threshold / 优化

**Chapter 21 — File 2 of 7 / 第21章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **roc curve for logistic regression model with optimal threshold**.

本脚本演示 **roc curve for logistic regression model with optimal threshold**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — roc curve for logistic regression model with optimal threshold

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_curve
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — fit a model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
```

---
## Step 5 — predict probabilities

```python
yhat = model.predict_proba(testX)
```

---
## Step 6 — keep probabilities for the positive outcome only

```python
yhat = yhat[:, 1]
```

---
## Step 7 — calculate roc curves

```python
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, thresholds = roc_curve(testy, yhat)
```

---
## Step 8 — calculate the g-mean for each threshold

```python
gmeans = sqrt(tpr * (1-fpr))
```

---
## Step 9 — locate the index of the largest g-mean

```python
ix = argmax(gmeans)
# 打印输出 / Print output
print('Best Threshold=%f, G-mean=%.3f' % (thresholds[ix], gmeans[ix]))
```

---
## Step 10 — plot the roc curve for the model

```python
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
```

---
## Step 11 — axis labels

```python
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: roc curve for logistic regression model with optimal threshold 是机器学习中的常用技术。  
  *roc curve for logistic regression model with optimal threshold is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `matplotlib` | 绑图库 | Plotting library |
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
# Roc Optimal Threshold / 优化
# Complete Code / 完整代码
# ===============================

# roc curve for logistic regression model with optimal threshold
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_curve
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, thresholds = roc_curve(testy, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
# 打印输出 / Print output
print('Best Threshold=%f, G-mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Roc Optimal Threshold J Stat



---

### Pr Curve



---

### Pr Optimal Threshold

# 05 — Pr Optimal Threshold / 优化

**Chapter 21 — File 5 of 7 / 第21章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **optimal threshold for precision-recall curve with logistic regression model**.

本脚本演示 **optimal threshold for precision-recall curve with logistic regression model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — optimal threshold for precision-recall curve with logistic regression model

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — fit a model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
```

---
## Step 5 — predict probabilities

```python
yhat = model.predict_proba(testX)
```

---
## Step 6 — keep probabilities for the positive outcome only

```python
yhat = yhat[:, 1]
```

---
## Step 7 — calculate roc curves

```python
precision, recall, thresholds = precision_recall_curve(testy, yhat)
```

---
## Step 8 — convert to f-measure

```python
fscore = (2 * precision * recall) / (precision + recall)
```

---
## Step 9 — locate the index of the largest f-measure

```python
ix = argmax(fscore)
# 打印输出 / Print output
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix], fscore[ix]))
```

---
## Step 10 — plot the roc curve for the model

```python
# 获取长度 / Get length
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
```

---
## Step 11 — axis labels

```python
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: optimal threshold for precision-recall curve with logistic regression model 是机器学习中的常用技术。  
  *optimal threshold for precision-recall curve with logistic regression model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `matplotlib` | 绑图库 | Plotting library |
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
# Pr Optimal Threshold / 优化
# Complete Code / 完整代码
# ===============================

# optimal threshold for precision-recall curve with logistic regression model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmax
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(testy, yhat)
# convert to f-measure
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f-measure
ix = argmax(fscore)
# 打印输出 / Print output
print('Best Threshold=%f, F-measure=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
# 获取长度 / Get length
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Logistic F1



---

### Grid Search



---

### Chapter Summary / 章节总结



---
