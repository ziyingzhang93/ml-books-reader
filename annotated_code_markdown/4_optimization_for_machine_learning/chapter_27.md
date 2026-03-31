# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 27

---

### Make Regression Dataset



---

### Random Regression

# 07 — Random Regression / 回归

**Chapter 27 — File 2 of 6 / 第27章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **linear regression model**.

本脚本演示 **linear regression model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — linear regression model

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — linear regression

```python
def predict_row(row, coefficients):
```

---
## Step 3 — add the bias, the last coefficient

```python
result = coefficients[-1]
```

---
## Step 4 — add the weighted input

```python
# 获取长度 / Get length
for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result
```

---
## Step 5 — use model coefficients to generate predictions for a dataset of rows

```python
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
```

---
## Step 6 — make a prediction

```python
yhat = predict_row(row, coefficients)
```

---
## Step 7 — store the prediction

```python
# 添加元素到列表末尾 / Append element to list end
yhats.append(yhat)
	return yhats
```

---
## Step 8 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
```

---
## Step 9 — determine the number of coefficients

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coeff = X.shape[1] + 1
```

---
## Step 10 — generate random coefficients

```python
coefficients = rand(n_coeff)
```

---
## Step 11 — generate predictions for dataset

```python
yhat = predict_dataset(X, coefficients)
```

---
## Step 12 — calculate model prediction error

```python
# 计算均方误差 / Calculate Mean Squared Error
score = mean_squared_error(y, yhat)
# 打印输出 / Print output
print('MSE: %f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: linear regression model 是机器学习中的常用技术。  
  *linear regression model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Regression / 回归
# Complete Code / 完整代码
# ===============================

# linear regression model
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
 # 获取长度 / Get length
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
  # 添加元素到列表末尾 / Append element to list end
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# determine the number of coefficients
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# calculate model prediction error
# 计算均方误差 / Calculate Mean Squared Error
score = mean_squared_error(y, yhat)
# 打印输出 / Print output
print('MSE: %f' % score)
```

---

➡️ **Next / 下一步**: File 3 of 6

---

### Hillclimbing Regression

# 13 — Hillclimbing Regression / 回归

**Chapter 27 — File 3 of 6 / 第27章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **optimize linear regression coefficients for regression dataset**.

本脚本演示 **optimize linear regression coefficients for regression dataset**。

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
## Step 1 — optimize linear regression coefficients for regression dataset

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — linear regression

```python
def predict_row(row, coefficients):
```

---
## Step 3 — add the bias, the last coefficient

```python
result = coefficients[-1]
```

---
## Step 4 — add the weighted input

```python
# 获取长度 / Get length
for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result
```

---
## Step 5 — use model coefficients to generate predictions for a dataset of rows

```python
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
```

---
## Step 6 — make a prediction

```python
yhat = predict_row(row, coefficients)
```

---
## Step 7 — store the prediction

```python
# 添加元素到列表末尾 / Append element to list end
yhats.append(yhat)
	return yhats
```

---
## Step 8 — objective function

```python
def objective(X, y, coefficients):
```

---
## Step 9 — generate predictions for dataset

```python
yhat = predict_dataset(X, coefficients)
```

---
## Step 10 — calculate accuracy

```python
# 计算均方误差 / Calculate Mean Squared Error
score = mean_squared_error(y, yhat)
	return score
```

---
## Step 11 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, solution, n_iter, step_size):
```

---
## Step 12 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 13 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 14 — take a step

```python
# 获取长度 / Get length
candidate = solution + randn(len(solution)) * step_size
```

---
## Step 15 — evaluate candidate point

```python
candidte_eval = objective(X, y, candidate)
```

---
## Step 16 — check if we should keep the new point

```python
if candidte_eval <= solution_eval:
```

---
## Step 17 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 18 — report progress

```python
# 打印输出 / Print output
print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]
```

---
## Step 19 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
```

---
## Step 20 — split into train test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

---
## Step 21 — define the total iterations

```python
n_iter = 2000
```

---
## Step 22 — define the maximum step size

```python
step_size = 0.15
```

---
## Step 23 — determine the number of coefficients

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coef = X.shape[1] + 1
```

---
## Step 24 — define the initial solution

```python
solution = rand(n_coef)
```

---
## Step 25 — perform the hill climbing search

```python
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Coefficients: %s' % coefficients)
# 打印输出 / Print output
print('Train MSE: %f' % (score))
```

---
## Step 26 — generate predictions for the test dataset

```python
yhat = predict_dataset(X_test, coefficients)
```

---
## Step 27 — calculate accuracy

```python
# 计算均方误差 / Calculate Mean Squared Error
score = mean_squared_error(y_test, yhat)
# 打印输出 / Print output
print('Test MSE: %f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: optimize linear regression coefficients for regression dataset 是机器学习中的常用技术。  
  *optimize linear regression coefficients for regression dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Regression / 回归
# Complete Code / 完整代码
# ===============================

# optimize linear regression coefficients for regression dataset
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_regression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
 # 获取长度 / Get length
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	return result

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
  # 添加元素到列表末尾 / Append element to list end
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# calculate accuracy
 # 计算均方误差 / Calculate Mean Squared Error
	score = mean_squared_error(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
  # 获取长度 / Get length
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
   # 打印输出 / Print output
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# split into train test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.15
# determine the number of coefficients
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Coefficients: %s' % coefficients)
# 打印输出 / Print output
print('Train MSE: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# calculate accuracy
# 计算均方误差 / Calculate Mean Squared Error
score = mean_squared_error(y_test, yhat)
# 打印输出 / Print output
print('Test MSE: %f' % (score))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Make Classification Dataset

# 14 — Make Classification Dataset / 分类

**Chapter 27 — File 4 of 6 / 第27章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **define a binary classification dataset**.

本脚本演示 **define a binary classification dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — define a binary classification dataset

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 3 — summarize the shape of the dataset

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: define a binary classification dataset 是机器学习中的常用技术。  
  *define a binary classification dataset is a common technique in machine learning.*

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
# Make Classification Dataset / 分类
# Complete Code / 完整代码
# ===============================

# define a binary classification dataset
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# summarize the shape of the dataset
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Random Logistic

# 19 — Random Logistic / 19 Random Logistic

**Chapter 27 — File 5 of 6 / 第27章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **logistic regression function for binary classification**.

本脚本演示 **logistic regression function for binary classification**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — logistic regression function for binary classification

```python
from math import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — logistic regression

```python
def predict_row(row, coefficients):
```

---
## Step 3 — add the bias, the last coefficient

```python
result = coefficients[-1]
```

---
## Step 4 — add the weighted input

```python
# 获取长度 / Get length
for i in range(len(row)):
		result += coefficients[i] * row[i]
```

---
## Step 5 — logistic function

```python
logistic = 1.0 / (1.0 + exp(-result))
	return logistic
```

---
## Step 6 — use model coefficients to generate predictions for a dataset of rows

```python
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
```

---
## Step 7 — make a prediction

```python
yhat = predict_row(row, coefficients)
```

---
## Step 8 — store the prediction

```python
# 添加元素到列表末尾 / Append element to list end
yhats.append(yhat)
	return yhats
```

---
## Step 9 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 10 — determine the number of coefficients

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coeff = X.shape[1] + 1
```

---
## Step 11 — generate random coefficients

```python
coefficients = rand(n_coeff)
```

---
## Step 12 — generate predictions for dataset

```python
yhat = predict_dataset(X, coefficients)
```

---
## Step 13 — round predictions to labels

```python
yhat = [round(y) for y in yhat]
```

---
## Step 14 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y, yhat)
# 打印输出 / Print output
print('Accuracy: %f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: logistic regression function for binary classification 是机器学习中的常用技术。  
  *logistic regression function for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Logistic / 19 Random Logistic
# Complete Code / 完整代码
# ===============================

# logistic regression function for binary classification
from math import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
 # 获取长度 / Get length
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
  # 添加元素到列表末尾 / Append element to list end
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of coefficients
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y, yhat)
# 打印输出 / Print output
print('Accuracy: %f' % score)
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Hillclimbing Logistic

# 23 — Hillclimbing Logistic / 23 Hillclimbing Logistic

**Chapter 27 — File 6 of 6 / 第27章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **optimize logistic regression model with a stochastic hill climber**.

本脚本演示 **optimize logistic regression model with a stochastic hill climber**。

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
## Step 1 — optimize logistic regression model with a stochastic hill climber

```python
from math import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
```

---
## Step 2 — logistic regression

```python
def predict_row(row, coefficients):
```

---
## Step 3 — add the bias, the last coefficient

```python
result = coefficients[-1]
```

---
## Step 4 — add the weighted input

```python
# 获取长度 / Get length
for i in range(len(row)):
		result += coefficients[i] * row[i]
```

---
## Step 5 — logistic function

```python
logistic = 1.0 / (1.0 + exp(-result))
	return logistic
```

---
## Step 6 — use model coefficients to generate predictions for a dataset of rows

```python
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
```

---
## Step 7 — make a prediction

```python
yhat = predict_row(row, coefficients)
```

---
## Step 8 — store the prediction

```python
# 添加元素到列表末尾 / Append element to list end
yhats.append(yhat)
	return yhats
```

---
## Step 9 — objective function

```python
def objective(X, y, coefficients):
```

---
## Step 10 — generate predictions for dataset

```python
yhat = predict_dataset(X, coefficients)
```

---
## Step 11 — round predictions to labels

```python
yhat = [round(y) for y in yhat]
```

---
## Step 12 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y, yhat)
	return score
```

---
## Step 13 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, solution, n_iter, step_size):
```

---
## Step 14 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 15 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 16 — take a step

```python
# 获取长度 / Get length
candidate = solution + randn(len(solution)) * step_size
```

---
## Step 17 — evaluate candidate point

```python
candidte_eval = objective(X, y, candidate)
```

---
## Step 18 — check if we should keep the new point

```python
if candidte_eval >= solution_eval:
```

---
## Step 19 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 20 — report progress

```python
# 打印输出 / Print output
print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]
```

---
## Step 21 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 22 — split into train test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

---
## Step 23 — define the total iterations

```python
n_iter = 2000
```

---
## Step 24 — define the maximum step size

```python
step_size = 0.1
```

---
## Step 25 — determine the number of coefficients

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coef = X.shape[1] + 1
```

---
## Step 26 — define the initial solution

```python
solution = rand(n_coef)
```

---
## Step 27 — perform the hill climbing search

```python
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Coefficients: %s' % coefficients)
# 打印输出 / Print output
print('Train Accuracy: %f' % (score))
```

---
## Step 28 — generate predictions for the test dataset

```python
yhat = predict_dataset(X_test, coefficients)
```

---
## Step 29 — round predictions to labels

```python
yhat = [round(y) for y in yhat]
```

---
## Step 30 — calculate accuracy

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Test Accuracy: %f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: optimize logistic regression model with a stochastic hill climber 是机器学习中的常用技术。  
  *optimize logistic regression model with a stochastic hill climber is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Logistic / 23 Hillclimbing Logistic
# Complete Code / 完整代码
# ===============================

# optimize logistic regression model with a stochastic hill climber
from math import exp
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
 # 获取长度 / Get length
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
  # 添加元素到列表末尾 / Append element to list end
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# round predictions to labels
	yhat = [round(y) for y in yhat]
	# calculate accuracy
 # 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
	score = accuracy_score(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
  # 获取长度 / Get length
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
   # 打印输出 / Print output
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.1
# determine the number of coefficients
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('Coefficients: %s' % coefficients)
# 打印输出 / Print output
print('Train Accuracy: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Test Accuracy: %f' % (score))
```

---

### Chapter Summary / 章节总结



---
