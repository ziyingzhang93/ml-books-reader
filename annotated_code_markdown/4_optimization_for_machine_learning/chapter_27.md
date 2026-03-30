# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 27

---

### Make Regression Dataset

# 01 — Make Regression Dataset / 回归

**Chapter 27 — File 1 of 6 / 第27章 — 第1个文件（共6个）**

---

## Summary / 总结

This script demonstrates **define a regression dataset**.

本脚本演示 **define a regression dataset**。

---
## Step 1 — define a regression dataset

```python
from sklearn.datasets import make_regression
```

---
## Step 2 — define dataset

```python
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
```

---
## Step 3 — summarize the shape of the dataset

```python
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: define a regression dataset 是机器学习中的常用技术。  
  *define a regression dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Regression Dataset / 回归
# Complete Code / 完整代码
# ===============================

# define a regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# summarize the shape of the dataset
print(X.shape, y.shape)
```

---

➡️ **Next / 下一步**: File 2 of 6

---

### Random Regression

# 07 — Random Regression / 回归

**Chapter 27 — File 2 of 6 / 第27章 — 第2个文件（共6个）**

---

## Summary / 总结

This script demonstrates **linear regression model**.

本脚本演示 **linear regression model**。

---
## Step 1 — linear regression model

```python
from numpy.random import rand
from sklearn.datasets import make_regression
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
score = mean_squared_error(y, yhat)
print('MSE: %f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: linear regression model 是机器学习中的常用技术。  
  *linear regression model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Regression / 回归
# Complete Code / 完整代码
# ===============================

# linear regression model
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
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
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# calculate model prediction error
score = mean_squared_error(y, yhat)
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
## Step 1 — optimize linear regression coefficients for regression dataset

```python
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
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
for i in range(n_iter):
```

---
## Step 14 — take a step

```python
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
print('Done!')
print('Coefficients: %s' % coefficients)
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
score = mean_squared_error(y_test, yhat)
print('Test MSE: %f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: optimize linear regression coefficients for regression dataset 是机器学习中的常用技术。  
  *optimize linear regression coefficients for regression dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Regression / 回归
# Complete Code / 完整代码
# ===============================

# optimize linear regression coefficients for regression dataset
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# linear regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
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
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# calculate accuracy
	score = mean_squared_error(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, noise=0.2, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.15
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train MSE: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# calculate accuracy
score = mean_squared_error(y_test, yhat)
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
## Step 1 — define a binary classification dataset

```python
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
print(X.shape, y.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: define a binary classification dataset 是机器学习中的常用技术。  
  *define a binary classification dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Make Classification Dataset / 分类
# Complete Code / 完整代码
# ===============================

# define a binary classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# summarize the shape of the dataset
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
## Step 1 — logistic regression function for binary classification

```python
from math import exp
from numpy.random import rand
from sklearn.datasets import make_classification
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
score = accuracy_score(y, yhat)
print('Accuracy: %f' % score)
```

---
## Learning Notes / 学习笔记

- **概念**: logistic regression function for binary classification 是机器学习中的常用技术。  
  *logistic regression function for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
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
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of coefficients
n_coeff = X.shape[1] + 1
# generate random coefficients
coefficients = rand(n_coeff)
# generate predictions for dataset
yhat = predict_dataset(X, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y, yhat)
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
## Step 1 — optimize logistic regression model with a stochastic hill climber

```python
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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
for i in range(n_iter):
```

---
## Step 16 — take a step

```python
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
print('Done!')
print('Coefficients: %s' % coefficients)
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
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: optimize logistic regression model with a stochastic hill climber 是机器学习中的常用技术。  
  *optimize logistic regression model with a stochastic hill climber is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

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
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
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
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# round predictions to labels
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.1
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train Accuracy: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %f' % (score))
```

---

### Chapter Summary / 章节总结

# Chapter 27 Summary / 第27章总结

## Theme / 主题: Chapter 27 / Chapter 27

This chapter contains **6 code files** demonstrating chapter 27.

本章包含 **6 个代码文件**，演示Chapter 27。

---
## Evolution / 演化路线

  1. `01_make_regression_dataset.ipynb` — Make Regression Dataset
  2. `07_random_regression.ipynb` — Random Regression
  3. `13_hillclimbing_regression.ipynb` — Hillclimbing Regression
  4. `14_make_classification_dataset.ipynb` — Make Classification Dataset
  5. `19_random_logistic.ipynb` — Random Logistic
  6. `23_hillclimbing_logistic.ipynb` — Hillclimbing Logistic

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 27) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 27）是机器学习流水线中的基础构建块。

---
