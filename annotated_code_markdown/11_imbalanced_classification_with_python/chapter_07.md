# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 07

---

### Roc Curve Balanced

# 01 — Roc Curve Balanced / 01 Roc Curve Balanced

**Chapter 07 — File 1 of 8 / 第07章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a roc curve for a predictive model**.

本脚本演示 **example of a roc curve for a predictive model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — example of a roc curve for a predictive model

```python
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
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
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
## Step 6 — retrieve just the probabilities for the positive class

```python
pos_probs = yhat[:, 1]
```

---
## Step 7 — plot no skill roc curve

```python
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
```

---
## Step 8 — calculate roc curve for model

```python
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, _ = roc_curve(testy, pos_probs)
```

---
## Step 9 — plot model roc curve

```python
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
```

---
## Step 10 — axis labels

```python
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
```

---
## Step 11 — show the legend

```python
pyplot.legend()
```

---
## Step 12 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a roc curve for a predictive model 是机器学习中的常用技术。  
  *example of a roc curve for a predictive model is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Roc Curve Balanced / 01 Roc Curve Balanced
# Complete Code / 完整代码
# ===============================

# example of a roc curve for a predictive model
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
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, _ = roc_curve(testy, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Roc Auc Balanced



---

### Pr Curve Balanced

# 03 — Pr Curve Balanced / 03 Pr Curve Balanced

**Chapter 07 — File 3 of 8 / 第07章 — 第3个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a precision-recall curve for a predictive model**.

本脚本演示 **example of a precision-recall curve for a predictive model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — example of a precision-recall curve for a predictive model

```python
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
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
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
## Step 6 — retrieve just the probabilities for the positive class

```python
pos_probs = yhat[:, 1]
```

---
## Step 7 — calculate the no skill line as the proportion of the positive class

```python
# 获取长度 / Get length
no_skill = len(y[y==1]) / len(y)
```

---
## Step 8 — plot the no skill precision-recall curve

```python
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
```

---
## Step 9 — calculate model precision-recall curve

```python
precision, recall, _ = precision_recall_curve(testy, pos_probs)
```

---
## Step 10 — plot the model precision-recall curve

```python
pyplot.plot(recall, precision, marker='.', label='Logistic')
```

---
## Step 11 — axis labels

```python
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
```

---
## Step 12 — show the legend

```python
pyplot.legend()
```

---
## Step 13 — show the plot

```python
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: example of a precision-recall curve for a predictive model 是机器学习中的常用技术。  
  *example of a precision-recall curve for a predictive model is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pr Curve Balanced / 03 Pr Curve Balanced
# Complete Code / 完整代码
# ===============================

# example of a precision-recall curve for a predictive model
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
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no skill line as the proportion of the positive class
# 获取长度 / Get length
no_skill = len(y[y==1]) / len(y)
# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(testy, pos_probs)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```

---

➡️ **Next / 下一步**: File 4 of 8

---

### Pr Auc Balanced

# 04 — Pr Auc Balanced / 04 Pr Auc Balanced

**Chapter 07 — File 4 of 8 / 第07章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **example of a precision-recall auc for a predictive model**.

本脚本演示 **example of a precision-recall auc for a predictive model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
```

---
## Step 1 — example of a precision-recall auc for a predictive model

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import auc
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
```

---
## Step 4 — no skill model, stratified random class predictions

```python
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
```

---
## Step 5 — calculate the precision-recall auc

```python
precision, recall, _ = precision_recall_curve(testy, pos_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('No Skill PR AUC: %.3f' % auc_score)
```

---
## Step 6 — fit a model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
```

---
## Step 7 — calculate the precision-recall auc

```python
precision, recall, _ = precision_recall_curve(testy, pos_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('Logistic PR AUC: %.3f' % auc_score)
```

---
## Learning Notes / 学习笔记

- **概念**: example of a precision-recall auc for a predictive model 是机器学习中的常用技术。  
  *example of a precision-recall auc for a predictive model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pr Auc Balanced / 04 Pr Auc Balanced
# Complete Code / 完整代码
# ===============================

# example of a precision-recall auc for a predictive model
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import auc
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, pos_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('No Skill PR AUC: %.3f' % auc_score)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
pos_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, pos_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('Logistic PR AUC: %.3f' % auc_score)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Imbalanced Dataset

# 05 — Imbalanced Dataset / 不平衡数据

**Chapter 07 — File 5 of 8 / 第07章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **create an imbalanced dataset**.

本脚本演示 **create an imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


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
```

---
## Step 1 — create an imbalanced dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
```

---
## Step 3 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — summarize dataset

```python
# 打印输出 / Print output
print('Dataset: Class0=%d, Class1=%d' % (len(y[y==0]), len(y[y==1])))
# 打印输出 / Print output
print('Train: Class0=%d, Class1=%d' % (len(trainy[trainy==0]), len(trainy[trainy==1])))
# 打印输出 / Print output
print('Test: Class0=%d, Class1=%d' % (len(testy[testy==0]), len(testy[testy==1])))
```

---
## Learning Notes / 学习笔记

- **概念**: create an imbalanced dataset 是机器学习中的常用技术。  
  *create an imbalanced dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Imbalanced Dataset / 不平衡数据
# Complete Code / 完整代码
# ===============================

# create an imbalanced dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# summarize dataset
# 打印输出 / Print output
print('Dataset: Class0=%d, Class1=%d' % (len(y[y==0]), len(y[y==1])))
# 打印输出 / Print output
print('Train: Class0=%d, Class1=%d' % (len(trainy[trainy==0]), len(trainy[trainy==1])))
# 打印输出 / Print output
print('Test: Class0=%d, Class1=%d' % (len(testy[testy==0]), len(testy[testy==1])))
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Roc Imbalanced

# 06 — Roc Imbalanced / 不平衡数据

**Chapter 07 — File 6 of 8 / 第07章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **roc curve and roc auc on an imbalanced dataset**.

本脚本演示 **roc curve and roc auc on an imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — roc curve and roc auc on an imbalanced dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — plot no skill and model roc curves

```python
# 生成ROC曲线数据 / Generate ROC curve data
def plot_roc_curve(test_y, naive_probs, model_probs):
```

---
## Step 3 — plot naive skill roc curve

```python
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, _ = roc_curve(test_y, naive_probs)
	pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')
```

---
## Step 4 — plot model roc curve

```python
# 生成ROC曲线数据 / Generate ROC curve data
fpr, tpr, _ = roc_curve(test_y, model_probs)
	pyplot.plot(fpr, tpr, marker='.', label='Logistic')
```

---
## Step 5 — axis labels

```python
pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
```

---
## Step 6 — show the legend

```python
pyplot.legend()
```

---
## Step 7 — show the plot

```python
pyplot.show()
```

---
## Step 8 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
```

---
## Step 9 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 10 — no skill model, stratified random class predictions

```python
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
```

---
## Step 11 — calculate roc auc

```python
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
roc_auc = roc_auc_score(testy, naive_probs)
# 打印输出 / Print output
print('No Skill ROC AUC %.3f' % roc_auc)
```

---
## Step 12 — skilled model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
```

---
## Step 13 — calculate roc auc

```python
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
roc_auc = roc_auc_score(testy, model_probs)
# 打印输出 / Print output
print('Logistic ROC AUC %.3f' % roc_auc)
```

---
## Step 14 — plot roc curves

```python
# 生成ROC曲线数据 / Generate ROC curve data
plot_roc_curve(testy, naive_probs, model_probs)
```

---
## Learning Notes / 学习笔记

- **概念**: roc curve and roc auc on an imbalanced dataset 是机器学习中的常用技术。  
  *roc curve and roc auc on an imbalanced dataset is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Roc Imbalanced / 不平衡数据
# Complete Code / 完整代码
# ===============================

# roc curve and roc auc on an imbalanced dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import roc_auc_score
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# plot no skill and model roc curves
# 生成ROC曲线数据 / Generate ROC curve data
def plot_roc_curve(test_y, naive_probs, model_probs):
	# plot naive skill roc curve
 # 生成ROC曲线数据 / Generate ROC curve data
	fpr, tpr, _ = roc_curve(test_y, naive_probs)
	pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')
	# plot model roc curve
 # 生成ROC曲线数据 / Generate ROC curve data
	fpr, tpr, _ = roc_curve(test_y, model_probs)
	pyplot.plot(fpr, tpr, marker='.', label='Logistic')
	# axis labels
	pyplot.xlabel('False Positive Rate')
	pyplot.ylabel('True Positive Rate')
	# show the legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
# calculate roc auc
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
roc_auc = roc_auc_score(testy, naive_probs)
# 打印输出 / Print output
print('No Skill ROC AUC %.3f' % roc_auc)
# skilled model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
# calculate roc auc
# 计算ROC-AUC分数（分类器好坏） / ROC-AUC score (classifier quality)
roc_auc = roc_auc_score(testy, model_probs)
# 打印输出 / Print output
print('Logistic ROC AUC %.3f' % roc_auc)
# plot roc curves
# 生成ROC曲线数据 / Generate ROC curve data
plot_roc_curve(testy, naive_probs, model_probs)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Pr Imbalanced

# 07 — Pr Imbalanced / 不平衡数据

**Chapter 07 — File 7 of 8 / 第07章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **pr curve and pr auc on an imbalanced dataset**.

本脚本演示 **pr curve and pr auc on an imbalanced dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
## Step 1 — pr curve and pr auc on an imbalanced dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import auc
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — plot no skill and model precision-recall curves

```python
def plot_pr_curve(test_y, model_probs):
```

---
## Step 3 — calculate the no skill line as the proportion of the positive class

```python
# 获取长度 / Get length
no_skill = len(test_y[test_y==1]) / len(test_y)
```

---
## Step 4 — plot the no skill precision-recall curve

```python
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
```

---
## Step 5 — plot model precision-recall curve

```python
precision, recall, _ = precision_recall_curve(testy, model_probs)
	pyplot.plot(recall, precision, marker='.', label='Logistic')
```

---
## Step 6 — axis labels

```python
pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
```

---
## Step 7 — show the legend

```python
pyplot.legend()
```

---
## Step 8 — show the plot

```python
pyplot.show()
```

---
## Step 9 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
```

---
## Step 10 — split into train/test sets with same class ratio

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 11 — no skill model, stratified random class predictions

```python
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
```

---
## Step 12 — calculate the precision-recall auc

```python
precision, recall, _ = precision_recall_curve(testy, naive_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('No Skill PR AUC: %.3f' % auc_score)
```

---
## Step 13 — fit a model

```python
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
```

---
## Step 14 — calculate the precision-recall auc

```python
precision, recall, _ = precision_recall_curve(testy, model_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('Logistic PR AUC: %.3f' % auc_score)
```

---
## Step 15 — plot precision-recall curves

```python
plot_pr_curve(testy, model_probs)
```

---
## Learning Notes / 学习笔记

- **概念**: pr curve and pr auc on an imbalanced dataset 是机器学习中的常用技术。  
  *pr curve and pr auc on an imbalanced dataset is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pr Imbalanced / 不平衡数据
# Complete Code / 完整代码
# ===============================

# pr curve and pr auc on an imbalanced dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.dummy import DummyClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import precision_recall_curve
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import auc
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# plot no skill and model precision-recall curves
def plot_pr_curve(test_y, model_probs):
	# calculate the no skill line as the proportion of the positive class
 # 获取长度 / Get length
	no_skill = len(test_y[test_y==1]) / len(test_y)
	# plot the no skill precision-recall curve
	pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	# plot model precision-recall curve
	precision, recall, _ = precision_recall_curve(testy, model_probs)
	pyplot.plot(recall, precision, marker='.', label='Logistic')
	# axis labels
	pyplot.xlabel('Recall')
	pyplot.ylabel('Precision')
	# show the legend
	pyplot.legend()
	# show the plot
	pyplot.show()

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, naive_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('No Skill PR AUC: %.3f' % auc_score)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, model_probs)
auc_score = auc(recall, precision)
# 打印输出 / Print output
print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(testy, model_probs)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Summarize Probs

# 08 — Summarize Probs / 08 Summarize Probs

**Chapter 07 — File 8 of 8 / 第07章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **summarize the distribution of predicted probabilities**.

本脚本演示 **summarize the distribution of predicted probabilities**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — summarize the distribution of predicted probabilities

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
```

---
## Step 2 — generate 2 class dataset

```python
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
```

---
## Step 3 — split into train/test sets with same class ratio

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
## Step 6 — retrieve just the probabilities for the positive class

```python
pos_probs = yhat[:, 1]
```

---
## Step 7 — predict class labels

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
```

---
## Step 8 — summarize the distribution of class labels

```python
# 打印输出 / Print output
print(Counter(yhat))
```

---
## Step 9 — create a histogram of the predicted probabilities

```python
pyplot.hist(pos_probs, bins=100)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: summarize the distribution of predicted probabilities 是机器学习中的常用技术。  
  *summarize the distribution of predicted probabilities is a common technique in machine learning.*

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
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Summarize Probs / 08 Summarize Probs
# Complete Code / 完整代码
# ===============================

# summarize the distribution of predicted probabilities
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
# split into train/test sets with same class ratio
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
# 逻辑回归：线性分类器 / Logistic Regression: linear classifier
model = LogisticRegression(solver='lbfgs')
# 训练模型 / Train the model
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# predict class labels
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# summarize the distribution of class labels
# 打印输出 / Print output
print(Counter(yhat))
# create a histogram of the predicted probabilities
pyplot.hist(pos_probs, bins=100)
pyplot.show()
```

---

### Chapter Summary / 章节总结



---
