# 不平衡分类问题 / Imbalanced Classification with Python
## Chapter 24

---

### Dataset

# 01 — Dataset / 01 Dataset

**Chapter 24 — File 1 of 5 / 第24章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Generate and plot a synthetic imbalanced classification dataset**.

本脚本演示 **Generate and plot a synthetic imbalanced classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🏗️ 定义模型 / Define Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

---
## Step 1 — Generate and plot a synthetic imbalanced classification dataset

```python
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
```

---
## Step 3 — summarize class distribution

```python
counter = Counter(y)
# 打印输出 / Print output
print(counter)
```

---
## Step 4 — scatter plot of examples by class label

```python
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Generate and plot a synthetic imbalanced classification dataset 是机器学习中的常用技术。  
  *Generate and plot a synthetic imbalanced classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Dataset / 01 Dataset
# Complete Code / 完整代码
# ===============================

# Generate and plot a synthetic imbalanced classification dataset
# 导入高级数据结构 / Import advanced data structures
from collections import Counter
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# summarize class distribution
counter = Counter(y)
# 打印输出 / Print output
print(counter)
# scatter plot of examples by class label
# 获取字典的键值对 / Get dict key-value pairs
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### One Class Svm

# 02 — One Class Svm / 支持向量机

**Chapter 24 — File 2 of 5 / 第24章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **one-class svm for imbalanced binary classification**.

本脚本演示 **one-class svm for imbalanced binary classification**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — one-class svm for imbalanced binary classification

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import OneClassSVM
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — define outlier detection model

```python
model = OneClassSVM(gamma='scale', nu=0.01)
```

---
## Step 5 — fit on majority class

```python
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
```

---
## Step 6 — detect outliers in the test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
```

---
## Step 7 — mark inliers 1, outliers -1

```python
testy[testy == 1] = -1
testy[testy == 0] = 1
```

---
## Step 8 — calculate score

```python
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: one-class svm for imbalanced binary classification 是机器学习中的常用技术。  
  *one-class svm for imbalanced binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `SVM` | 支持向量机 | Support Vector Machine |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# One Class Svm / 支持向量机
# Complete Code / 完整代码
# ===============================

# one-class svm for imbalanced binary classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.svm import OneClassSVM
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)
# fit on majority class
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
# detect outliers in the test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# mark inliers 1, outliers -1
testy[testy == 1] = -1
testy[testy == 0] = 1
# calculate score
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Isolation Forest

# 03 — Isolation Forest / 随机森林

**Chapter 24 — File 3 of 5 / 第24章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **isolation forest for imbalanced classification**.

本脚本演示 **isolation forest for imbalanced classification**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — isolation forest for imbalanced classification

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import IsolationForest
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — define outlier detection model

```python
model = IsolationForest(contamination=0.01)
```

---
## Step 5 — fit on majority class

```python
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
```

---
## Step 6 — detect outliers in the test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
```

---
## Step 7 — mark inliers 1, outliers -1

```python
testy[testy == 1] = -1
testy[testy == 0] = 1
```

---
## Step 8 — calculate score

```python
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: isolation forest for imbalanced classification 是机器学习中的常用技术。  
  *isolation forest for imbalanced classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Isolation Forest / 随机森林
# Complete Code / 完整代码
# ===============================

# isolation forest for imbalanced classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.ensemble import IsolationForest
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# define outlier detection model
model = IsolationForest(contamination=0.01)
# fit on majority class
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
# detect outliers in the test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# mark inliers 1, outliers -1
testy[testy == 1] = -1
testy[testy == 0] = 1
# calculate score
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Elliptic Envelope

# 04 — Elliptic Envelope / 04 Elliptic Envelope

**Chapter 24 — File 4 of 5 / 第24章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **elliptic envelope for imbalanced classification**.

本脚本演示 **elliptic envelope for imbalanced classification**。

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
  │
  ▼
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
```

---
## Step 1 — elliptic envelope for imbalanced classification

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.covariance import EllipticEnvelope
```

---
## Step 2 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
```

---
## Step 3 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 4 — define outlier detection model

```python
model = EllipticEnvelope(contamination=0.01)
```

---
## Step 5 — fit on majority class

```python
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
```

---
## Step 6 — detect outliers in the test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
```

---
## Step 7 — mark inliers 1, outliers -1

```python
testy[testy == 1] = -1
testy[testy == 0] = 1
```

---
## Step 8 — calculate score

```python
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: elliptic envelope for imbalanced classification 是机器学习中的常用技术。  
  *elliptic envelope for imbalanced classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Elliptic Envelope / 04 Elliptic Envelope
# Complete Code / 完整代码
# ===============================

# elliptic envelope for imbalanced classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.covariance import EllipticEnvelope
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# define outlier detection model
model = EllipticEnvelope(contamination=0.01)
# fit on majority class
trainX = trainX[trainy==0]
# 训练模型 / Train the model
model.fit(trainX)
# detect outliers in the test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(testX)
# mark inliers 1, outliers -1
testy[testy == 1] = -1
testy[testy == 0] = 1
# calculate score
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Lof

# 05 — Lof / 05 Lof

**Chapter 24 — File 5 of 5 / 第24章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **local outlier factor for imbalanced classification**.

本脚本演示 **local outlier factor for imbalanced classification**。

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
## Step 1 — local outlier factor for imbalanced classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import LocalOutlierFactor
```

---
## Step 2 — make a prediction with a lof model

```python
def lof_predict(model, trainX, testX):
```

---
## Step 3 — create one large dataset

```python
composite = vstack((trainX, testX))
```

---
## Step 4 — make prediction on composite dataset

```python
yhat = model.fit_predict(composite)
```

---
## Step 5 — return just the predictions on the test set

```python
# 获取长度 / Get length
return yhat[len(trainX):]
```

---
## Step 6 — generate dataset

```python
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
```

---
## Step 7 — split into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
```

---
## Step 8 — define outlier detection model

```python
model = LocalOutlierFactor(contamination=0.01)
```

---
## Step 9 — get examples for just the majority class

```python
trainX = trainX[trainy==0]
```

---
## Step 10 — detect outliers in the test set

```python
yhat = lof_predict(model, trainX, testX)
```

---
## Step 11 — mark inliers 1, outliers -1

```python
testy[testy == 1] = -1
testy[testy == 0] = 1
```

---
## Step 12 — calculate score

```python
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: local outlier factor for imbalanced classification 是机器学习中的常用技术。  
  *local outlier factor for imbalanced classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lof / 05 Lof
# Complete Code / 完整代码
# ===============================

# local outlier factor for imbalanced classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import f1_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.neighbors import LocalOutlierFactor

# make a prediction with a lof model
def lof_predict(model, trainX, testX):
	# create one large dataset
	composite = vstack((trainX, testX))
	# make prediction on composite dataset
	yhat = model.fit_predict(composite)
	# return just the predictions on the test set
 # 获取长度 / Get length
	return yhat[len(trainX):]

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# split into train/test sets
# 划分训练集和测试集 / Split into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# define outlier detection model
model = LocalOutlierFactor(contamination=0.01)
# get examples for just the majority class
trainX = trainX[trainy==0]
# detect outliers in the test set
yhat = lof_predict(model, trainX, testX)
# mark inliers 1, outliers -1
testy[testy == 1] = -1
testy[testy == 0] = 1
# calculate score
# 计算F1分数 = 精确率和召回率的调和均值 / F1 = harmonic mean of precision and recall
score = f1_score(testy, yhat, pos_label=-1)
# 打印输出 / Print output
print('F-measure: %.3f' % score)
```

---

### Chapter Summary / 章节总结



---
