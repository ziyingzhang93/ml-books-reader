# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 28

---

### Make Classification Dataset

# 01 — Make Classification Dataset / 分类

**Chapter 28 — File 1 of 5 / 第28章 — 第1个文件（共5个）**

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

➡️ **Next / 下一步**: File 2 of 5

---

### Random Perceptron

# 09 — Random Perceptron / 09 Random Perceptron

**Chapter 28 — File 2 of 5 / 第28章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **simple perceptron model for binary classification**.

本脚本演示 **simple perceptron model for binary classification**。

---
## Step 1 — simple perceptron model for binary classification

```python
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
```

---
## Step 2 — transfer function

```python
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0
```

---
## Step 3 — activation function

```python
def activate(row, weights):
```

---
## Step 4 — add the bias, the last weight

```python
activation = weights[-1]
```

---
## Step 5 — add the weighted input

```python
for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
```

---
## Step 6 — use model weights to predict 0 or 1 for a given row of data

```python
def predict_row(row, weights):
```

---
## Step 7 — activate for input

```python
activation = activate(row, weights)
```

---
## Step 8 — transfer for activation

```python
return transfer(activation)
```

---
## Step 9 — use model weights to generate predictions for a dataset of rows

```python
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats
```

---
## Step 10 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 11 — determine the number of weights

```python
n_weights = X.shape[1] + 1
```

---
## Step 12 — generate random weights

```python
weights = rand(n_weights)
```

---
## Step 13 — generate predictions for dataset

```python
yhat = predict_dataset(X, weights)
```

---
## Step 14 — calculate accuracy

```python
score = accuracy_score(y, yhat)
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: simple perceptron model for binary classification 是机器学习中的常用技术。  
  *simple perceptron model for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Perceptron / 09 Random Perceptron
# Complete Code / 完整代码
# ===============================

# simple perceptron model for binary classification
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# use model weights to predict 0 or 1 for a given row of data
def predict_row(row, weights):
	# activate for input
	activation = activate(row, weights)
	# transfer for activation
	return transfer(activation)

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of weights
n_weights = X.shape[1] + 1
# generate random weights
weights = rand(n_weights)
# generate predictions for dataset
yhat = predict_dataset(X, weights)
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Hillclimbing Perceptron

# 15 — Hillclimbing Perceptron / 15 Hillclimbing Perceptron

**Chapter 28 — File 3 of 5 / 第28章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **hill climbing to optimize weights of a perceptron model for classification**.

本脚本演示 **hill climbing to optimize weights of a perceptron model for classification**。

---
## Step 1 — hill climbing to optimize weights of a perceptron model for classification

```python
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---
## Step 2 — transfer function

```python
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0
```

---
## Step 3 — activation function

```python
def activate(row, weights):
```

---
## Step 4 — add the bias, the last weight

```python
activation = weights[-1]
```

---
## Step 5 — add the weighted input

```python
for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
```

---
## Step 6 — # use model weights to predict 0 or 1 for a given row of data

```python
def predict_row(row, weights):
```

---
## Step 7 — activate for input

```python
activation = activate(row, weights)
```

---
## Step 8 — transfer for activation

```python
return transfer(activation)
```

---
## Step 9 — use model weights to generate predictions for a dataset of rows

```python
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats
```

---
## Step 10 — objective function

```python
def objective(X, y, weights):
```

---
## Step 11 — generate predictions for dataset

```python
yhat = predict_dataset(X, weights)
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
n_iter = 1000
```

---
## Step 24 — define the maximum step size

```python
step_size = 0.05
```

---
## Step 25 — determine the number of weights

```python
n_weights = X.shape[1] + 1
```

---
## Step 26 — define the initial solution

```python
solution = rand(n_weights)
```

---
## Step 27 — perform the hill climbing search

```python
weights, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (weights, score))
```

---
## Step 28 — generate predictions for the test dataset

```python
yhat = predict_dataset(X_test, weights)
```

---
## Step 29 — calculate accuracy

```python
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

---
## Learning Notes / 学习笔记

- **概念**: hill climbing to optimize weights of a perceptron model for classification 是机器学习中的常用技术。  
  *hill climbing to optimize weights of a perceptron model for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Perceptron / 15 Hillclimbing Perceptron
# Complete Code / 完整代码
# ===============================

# hill climbing to optimize weights of a perceptron model for classification
from numpy import asarray
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	if activation >= 0.0:
		return 1
	return 0

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# # use model weights to predict 0 or 1 for a given row of data
def predict_row(row, weights):
	# activate for input
	activation = activate(row, weights)
	# transfer for activation
	return transfer(activation)

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, weights):
	yhats = list()
	for row in X:
		yhat = predict_row(row, weights)
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, weights):
	# generate predictions for dataset
	yhat = predict_dataset(X, weights)
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
n_iter = 1000
# define the maximum step size
step_size = 0.05
# determine the number of weights
n_weights = X.shape[1] + 1
# define the initial solution
solution = rand(n_weights)
# perform the hill climbing search
weights, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (weights, score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, weights)
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Random Mlp

# 22 — Random Mlp / 22 Random Mlp

**Chapter 28 — File 4 of 5 / 第28章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **develop an mlp model for classification**.

本脚本演示 **develop an mlp model for classification**。

---
## Step 1 — develop an mlp model for classification

```python
from math import exp
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
```

---
## Step 2 — transfer function

```python
def transfer(activation):
```

---
## Step 3 — sigmoid transfer function

```python
return 1.0 / (1.0 + exp(-activation))
```

---
## Step 4 — activation function

```python
def activate(row, weights):
```

---
## Step 5 — add the bias, the last weight

```python
activation = weights[-1]
```

---
## Step 6 — add the weighted input

```python
for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
```

---
## Step 7 — activation function for a network

```python
def predict_row(row, network):
	inputs = row
```

---
## Step 8 — enumerate the layers in the network from input to output

```python
for layer in network:
		new_inputs = list()
```

---
## Step 9 — enumerate nodes in the layer

```python
for node in layer:
```

---
## Step 10 — activate the node

```python
activation = activate(inputs, node)
```

---
## Step 11 — transfer activation

```python
output = transfer(activation)
```

---
## Step 12 — store output

```python
new_inputs.append(output)
```

---
## Step 13 — output from this layer is input to the next layer

```python
inputs = new_inputs
	return inputs[0]
```

---
## Step 14 — use model weights to generate predictions for a dataset of rows

```python
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats
```

---
## Step 15 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 16 — determine the number of inputs

```python
n_inputs = X.shape[1]
```

---
## Step 17 — one hidden layer and an output layer

```python
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
```

---
## Step 18 — generate predictions for dataset

```python
yhat = predict_dataset(X, network)
```

---
## Step 19 — round the predictions

```python
yhat = [round(y) for y in yhat]
```

---
## Step 20 — calculate accuracy

```python
score = accuracy_score(y, yhat)
print(score)
```

---
## Learning Notes / 学习笔记

- **概念**: develop an mlp model for classification 是机器学习中的常用技术。  
  *develop an mlp model for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Mlp / 22 Random Mlp
# Complete Code / 完整代码
# ===============================

# develop an mlp model for classification
from math import exp
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# determine the number of inputs
n_inputs = X.shape[1]
# one hidden layer and an output layer
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
# generate predictions for dataset
yhat = predict_dataset(X, network)
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y, yhat)
print(score)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Hillclimbing Mlp

# 25 — Hillclimbing Mlp / 25 Hillclimbing Mlp

**Chapter 28 — File 5 of 5 / 第28章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **stochastic hill climbing to optimize a multilayer perceptron for classification**.

本脚本演示 **stochastic hill climbing to optimize a multilayer perceptron for classification**。

---
## Step 1 — stochastic hill climbing to optimize a multilayer perceptron for classification

```python
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---
## Step 2 — transfer function

```python
def transfer(activation):
```

---
## Step 3 — sigmoid transfer function

```python
return 1.0 / (1.0 + exp(-activation))
```

---
## Step 4 — activation function

```python
def activate(row, weights):
```

---
## Step 5 — add the bias, the last weight

```python
activation = weights[-1]
```

---
## Step 6 — add the weighted input

```python
for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation
```

---
## Step 7 — activation function for a network

```python
def predict_row(row, network):
	inputs = row
```

---
## Step 8 — enumerate the layers in the network from input to output

```python
for layer in network:
		new_inputs = list()
```

---
## Step 9 — enumerate nodes in the layer

```python
for node in layer:
```

---
## Step 10 — activate the node

```python
activation = activate(inputs, node)
```

---
## Step 11 — transfer activation

```python
output = transfer(activation)
```

---
## Step 12 — store output

```python
new_inputs.append(output)
```

---
## Step 13 — output from this layer is input to the next layer

```python
inputs = new_inputs
	return inputs[0]
```

---
## Step 14 — use model weights to generate predictions for a dataset of rows

```python
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats
```

---
## Step 15 — objective function

```python
def objective(X, y, network):
```

---
## Step 16 — generate predictions for dataset

```python
yhat = predict_dataset(X, network)
```

---
## Step 17 — round the predictions

```python
yhat = [round(y) for y in yhat]
```

---
## Step 18 — calculate accuracy

```python
score = accuracy_score(y, yhat)
	return score
```

---
## Step 19 — take a step in the search space

```python
def step(network, step_size):
	new_net = list()
```

---
## Step 20 — enumerate layers in the network

```python
for layer in network:
		new_layer = list()
```

---
## Step 21 — enumerate nodes in this layer

```python
for node in layer:
```

---
## Step 22 — mutate the node

```python
new_node = node.copy() + randn(len(node)) * step_size
```

---
## Step 23 — store node in layer

```python
new_layer.append(new_node)
```

---
## Step 24 — store layer in network

```python
new_net.append(new_layer)
	return new_net
```

---
## Step 25 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, solution, n_iter, step_size):
```

---
## Step 26 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 27 — run the hill climb

```python
for i in range(n_iter):
```

---
## Step 28 — take a step

```python
candidate = step(solution, step_size)
```

---
## Step 29 — evaluate candidate point

```python
candidte_eval = objective(X, y, candidate)
```

---
## Step 30 — check if we should keep the new point

```python
if candidte_eval >= solution_eval:
```

---
## Step 31 — store the new point

```python
solution, solution_eval = candidate, candidte_eval
```

---
## Step 32 — report progress

```python
print('>%d %f' % (i, solution_eval))
	return [solution, solution_eval]
```

---
## Step 33 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 34 — split into train test sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

---
## Step 35 — define the total iterations

```python
n_iter = 1000
```

---
## Step 36 — define the maximum step size

```python
step_size = 0.1
```

---
## Step 37 — determine the number of inputs

```python
n_inputs = X.shape[1]
```

---
## Step 38 — one hidden layer and an output layer, each perceptron has a bias term

```python
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
```

---
## Step 39 — perform the hill climbing search

```python
network, score = hillclimbing(X_train, y_train, objective, network, n_iter, step_size)
print('Done!')
print('Best: %f' % (score))
```

---
## Step 40 — generate predictions for the test dataset

```python
yhat = predict_dataset(X_test, network)
```

---
## Step 41 — round the predictions

```python
yhat = [round(y) for y in yhat]
```

---
## Step 42 — calculate accuracy

```python
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

---
## Learning Notes / 学习笔记

- **概念**: stochastic hill climbing to optimize a multilayer perceptron for classification 是机器学习中的常用技术。  
  *stochastic hill climbing to optimize a multilayer perceptron for classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Mlp / 25 Hillclimbing Mlp
# Complete Code / 完整代码
# ===============================

# stochastic hill climbing to optimize a multilayer perceptron for classification
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, network):
	# generate predictions for dataset
	yhat = predict_dataset(X, network)
	# round the predictions
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# take a step in the search space
def step(network, step_size):
	new_net = list()
	# enumerate layers in the network
	for layer in network:
		new_layer = list()
		# enumerate nodes in this layer
		for node in layer:
			# mutate the node
			new_node = node.copy() + randn(len(node)) * step_size
			# store node in layer
			new_layer.append(new_node)
		# store layer in network
		new_net.append(new_layer)
	return new_net

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 1000
# define the maximum step size
step_size = 0.1
# determine the number of inputs
n_inputs = X.shape[1]
# one hidden layer and an output layer, each perceptron has a bias term
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
# perform the hill climbing search
network, score = hillclimbing(X_train, y_train, objective, network, n_iter, step_size)
print('Done!')
print('Best: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, network)
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))
```

---

### Chapter Summary / 章节总结

# Chapter 28 Summary / 第28章总结

## Theme / 主题: Chapter 28 / Chapter 28

This chapter contains **5 code files** demonstrating chapter 28.

本章包含 **5 个代码文件**，演示Chapter 28。

---
## Evolution / 演化路线

  1. `01_make_classification_dataset.ipynb` — Make Classification Dataset
  2. `09_random_perceptron.ipynb` — Random Perceptron
  3. `15_hillclimbing_perceptron.ipynb` — Hillclimbing Perceptron
  4. `22_random_mlp.ipynb` — Random Mlp
  5. `25_hillclimbing_mlp.ipynb` — Hillclimbing Mlp

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 28) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 28）是机器学习流水线中的基础构建块。

---
