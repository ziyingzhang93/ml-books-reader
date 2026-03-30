# DL时间序列
## Chapter 23

---

### Evaluate Ml Models

# 01 — Evaluate Ml Models / 模型评估

**Chapter 23 — File 1 of 2 / 第23章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **spot check ml algorithms on engineered-features from the har dataset**.

本脚本演示 **spot check ml algorithms on engineered-features from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
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
## Step 1 — spot check ml algorithms on engineered-features from the har dataset

```python
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a dataset group, such as train or test

```python
def load_dataset_group(group, prefix=''):
```

---
## Step 4 — load input data

```python
X = load_file(prefix + group + '/X_'+group+'.txt')
```

---
## Step 5 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 6 — load the dataset, returns train and test X and y elements

```python
def load_dataset(prefix=''):
```

---
## Step 7 — load all train

```python
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
```

---
## Step 8 — load all test

```python
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
```

---
## Step 9 — flatten y

```python
trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy
```

---
## Step 10 — create a dict of standard models to evaluate {name:object}

```python
def define_models(models=dict()):
```

---
## Step 11 — nonlinear models

```python
models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
```

---
## Step 12 — ensemble models

```python
models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models
```

---
## Step 13 — evaluate a single model

```python
def evaluate_model(trainX, trainy, testX, testy, model):
```

---
## Step 14 — fit the model

```python
model.fit(trainX, trainy)
```

---
## Step 15 — make predictions

```python
yhat = model.predict(testX)
```

---
## Step 16 — evaluate predictions

```python
accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0
```

---
## Step 17 — evaluate a dict of models {name:object}, returns {name:score}

```python
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
```

---
## Step 18 — evaluate the model

```python
results[name] = evaluate_model(trainX, trainy, testX, testy, model)
```

---
## Step 19 — show process

```python
print('>%s: %.3f' % (name, results[name]))
	return results
```

---
## Step 20 — print and plot the results

```python
def summarize_results(results, maximize=True):
```

---
## Step 21 — create a list of (name, mean(scores)) tuples

```python
mean_scores = [(k,v) for k,v in results.items()]
```

---
## Step 22 — sort tuples by mean score

```python
mean_scores = sorted(mean_scores, key=lambda x: x[1])
```

---
## Step 23 — reverse for descending order (e.g. for accuracy)

```python
if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))
```

---
## Step 24 — load dataset

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 25 — get model list

```python
models = define_models()
```

---
## Step 26 — evaluate models

```python
results = evaluate_models(trainX, trainy, testX, testy, models)
```

---
## Step 27 — summarize results

```python
summarize_results(results)
```

---
## Learning Notes / 学习笔记

- **概念**: spot check ml algorithms on engineered-features from the har dataset 是机器学习中的常用技术。  
  *spot check ml algorithms on engineered-features from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Ml Models / 模型评估
# Complete Code / 完整代码
# ===============================

# spot check ml algorithms on engineered-features from the har dataset
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	# load input data
	X = load_file(prefix + group + '/X_'+group+'.txt')
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Evaluate Ml Models Raw Data

# 02 — Evaluate Ml Models Raw Data / 模型评估

**Chapter 23 — File 2 of 2 / 第23章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **spot check on raw data from the har dataset**.

本脚本演示 **spot check on raw data from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
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
## Step 1 — spot check on raw data from the har dataset

```python
from numpy import dstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files into a 3D array of [samples, timesteps, features]

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — load the dataset, returns train and test X and y elements

```python
def load_dataset(prefix=''):
```

---
## Step 13 — load all train

```python
trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
```

---
## Step 14 — load all test

```python
testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
```

---
## Step 15 — flatten X

```python
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
	testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
```

---
## Step 16 — flatten y

```python
trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy
```

---
## Step 17 — create a dict of standard models to evaluate {name:object}

```python
def define_models(models=dict()):
```

---
## Step 18 — nonlinear models

```python
models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
```

---
## Step 19 — ensemble models

```python
models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models
```

---
## Step 20 — evaluate a single model

```python
def evaluate_model(trainX, trainy, testX, testy, model):
```

---
## Step 21 — fit the model

```python
model.fit(trainX, trainy)
```

---
## Step 22 — make predictions

```python
yhat = model.predict(testX)
```

---
## Step 23 — evaluate predictions

```python
accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0
```

---
## Step 24 — evaluate a dict of models {name:object}, returns {name:score}

```python
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
```

---
## Step 25 — evaluate the model

```python
results[name] = evaluate_model(trainX, trainy, testX, testy, model)
```

---
## Step 26 — show process

```python
print('>%s: %.3f' % (name, results[name]))
	return results
```

---
## Step 27 — print and plot the results

```python
def summarize_results(results, maximize=True):
```

---
## Step 28 — create a list of (name, mean(scores)) tuples

```python
mean_scores = [(k,v) for k,v in results.items()]
```

---
## Step 29 — sort tuples by mean score

```python
mean_scores = sorted(mean_scores, key=lambda x: x[1])
```

---
## Step 30 — reverse for descending order (e.g. for accuracy)

```python
if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))
```

---
## Step 31 — load dataset

```python
trainX, trainy, testX, testy = load_dataset()
```

---
## Step 32 — get model list

```python
models = define_models()
```

---
## Step 33 — evaluate models

```python
results = evaluate_models(trainX, trainy, testX, testy, models)
```

---
## Step 34 — summarize results

```python
summarize_results(results)
```

---
## Learning Notes / 学习笔记

- **概念**: spot check on raw data from the har dataset 是机器学习中的常用技术。  
  *spot check on raw data from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `GradientBoosting` | 梯度提升算法 | Gradient Boosting algorithm |
| `RandomForestClassifier` | 随机森林分类器 | Random Forest classifier |
| `SVM` | 支持向量机 | Support Vector Machine |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `gradient` | 梯度：指示参数调整方向 | Gradient: direction to adjust parameters |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Evaluate Ml Models Raw Data / 模型评估
# Complete Code / 完整代码
# ===============================

# spot check on raw data from the har dataset
from numpy import dstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	# flatten X
	trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
	testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)
```

---

### Chapter Summary

# Chapter 23 Summary / 第23章总结

## Theme / 主题: Chapter 23 / Chapter 23

This chapter contains **2 code files** demonstrating chapter 23.

本章包含 **2 个代码文件**，演示Chapter 23。

---
## Evolution / 演化路线

  1. `01_evaluate_ml_models.ipynb` — Evaluate Ml Models
  2. `02_evaluate_ml_models_raw_data.ipynb` — Evaluate Ml Models Raw Data

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 23) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 23）是机器学习流水线中的基础构建块。

---
