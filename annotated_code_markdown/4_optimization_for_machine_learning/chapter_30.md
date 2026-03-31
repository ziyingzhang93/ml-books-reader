# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 30

---

### Make Classification Dataset

# 01 — Make Classification Dataset / 分类

**Chapter 30 — File 1 of 6 / 第30章 — 第1个文件（共6个）**

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

➡️ **Next / 下一步**: File 2 of 6

---

### Kfold Perceptron



---

### Hillclimbing Perceptron

# 10 — Hillclimbing Perceptron / 10 Hillclimbing Perceptron

**Chapter 30 — File 3 of 6 / 第30章 — 第3个文件（共6个）**

---

## Summary / 总结

This script demonstrates **manually search perceptron hyperparameters for binary classification**.

本脚本演示 **manually search perceptron hyperparameters for binary classification**。

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
## Step 1 — manually search perceptron hyperparameters for binary classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron
```

---
## Step 2 — objective function

```python
def objective(X, y, cfg):
```

---
## Step 3 — unpack config

```python
eta, alpha = cfg
```

---
## Step 4 — define model

```python
model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
```

---
## Step 5 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — calculate mean accuracy

```python
result = mean(scores)
	return result
```

---
## Step 8 — take a step in the search space

```python
def step(cfg, step_size):
```

---
## Step 9 — unpack the configuration

```python
eta, alpha = cfg
```

---
## Step 10 — step eta

```python
new_eta = eta + randn() * step_size
```

---
## Step 11 — check the bounds of eta

```python
if new_eta <= 0.0:
		new_eta = 1e-8
	if new_eta > 1.0:
		new_eta = 1.0
```

---
## Step 12 — step alpha

```python
new_alpha = alpha + randn() * step_size
```

---
## Step 13 — check the bounds of alpha

```python
if new_alpha < 0.0:
		new_alpha = 0.0
```

---
## Step 14 — return the new configuration

```python
return [new_eta, new_alpha]
```

---
## Step 15 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, n_iter, step_size):
```

---
## Step 16 — starting point for the search

```python
solution = [rand(), rand()]
```

---
## Step 17 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 18 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 19 — take a step

```python
candidate = step(solution, step_size)
```

---
## Step 20 — evaluate candidate point

```python
candidate_eval = objective(X, y, candidate)
```

---
## Step 21 — check if we should keep the new point

```python
if candidate_eval >= solution_eval:
```

---
## Step 22 — store the new point

```python
solution, solution_eval = candidate, candidate_eval
```

---
## Step 23 — report progress

```python
# 打印输出 / Print output
print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 24 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 25 — define the total iterations

```python
n_iter = 100
```

---
## Step 26 — step size in the search space

```python
step_size = 0.1
```

---
## Step 27 — perform the hill climbing search

```python
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

---
## Learning Notes / 学习笔记

- **概念**: manually search perceptron hyperparameters for binary classification 是机器学习中的常用技术。  
  *manually search perceptron hyperparameters for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Perceptron / 10 Hillclimbing Perceptron
# Complete Code / 完整代码
# ===============================

# manually search perceptron hyperparameters for binary classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import Perceptron

# objective function
def objective(X, y, cfg):
	# unpack config
	eta, alpha = cfg
	# define model
	model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result

# take a step in the search space
def step(cfg, step_size):
	# unpack the configuration
	eta, alpha = cfg
	# step eta
	new_eta = eta + randn() * step_size
	# check the bounds of eta
	if new_eta <= 0.0:
		new_eta = 1e-8
	if new_eta > 1.0:
		new_eta = 1.0
	# step alpha
	new_alpha = alpha + randn() * step_size
	# check the bounds of alpha
	if new_alpha < 0.0:
		new_alpha = 0.0
	# return the new configuration
	return [new_eta, new_alpha]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, step_size):
	# starting point for the search
	solution = [rand(), rand()]
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
   # 打印输出 / Print output
			print('>%d, cfg=%s %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 100
# step size in the search space
step_size = 0.1
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter, step_size)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=%s: Mean Accuracy: %f' % (cfg, score))
```

---

➡️ **Next / 下一步**: File 4 of 6

---

### Print Xgboost

# 11 — Print Xgboost / 提升方法

**Chapter 30 — File 4 of 6 / 第30章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **xgboost**.

本脚本演示 **xgboost**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — xgboost

```python
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
import xgboost
# 打印输出 / Print output
print("xgboost", xgboost.__version__)
```

---
## Learning Notes / 学习笔记

- **概念**: xgboost 是机器学习中的常用技术。  
  *xgboost is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Print Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# xgboost
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
import xgboost
# 打印输出 / Print output
print("xgboost", xgboost.__version__)
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Kfold Xgboost



---

### Hillclimbing Xgboost

# 17 — Hillclimbing Xgboost / 提升方法

**Chapter 30 — File 6 of 6 / 第30章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **xgboost manual hyperparameter optimization for binary classification**.

本脚本演示 **xgboost manual hyperparameter optimization for binary classification**。

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
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — xgboost manual hyperparameter optimization for binary classification

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
```

---
## Step 2 — objective function

```python
def objective(X, y, cfg):
```

---
## Step 3 — unpack config

```python
lrate, n_tree, subsam, depth = cfg
```

---
## Step 4 — define model

```python
model = XGBClassifier(learning_rate=lrate, n_estimators=n_tree, subsample=subsam, max_depth=depth, use_label_encoder=False, eval_metric="logloss")
```

---
## Step 5 — define evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 6 — evaluate model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 7 — calculate mean accuracy

```python
result = mean(scores)
	return result
```

---
## Step 8 — take a step in the search space

```python
def step(cfg):
```

---
## Step 9 — unpack config

```python
lrate, n_tree, subsam, depth = cfg
```

---
## Step 10 — learning rate

```python
lrate = lrate + randn() * 0.01
	if lrate <= 0.0:
		lrate = 1e-8
	if lrate > 1:
		lrate = 1.0
```

---
## Step 11 — number of trees

```python
n_tree = round(n_tree + randn() * 50)
	if n_tree <= 0.0:
		n_tree = 1
```

---
## Step 12 — subsample percentage

```python
subsam = subsam + randn() * 0.1
	if subsam <= 0.0:
		subsam = 1e-8
	if subsam > 1:
		subsam = 1.0
```

---
## Step 13 — max tree depth

```python
depth = round(depth + randn() * 7)
	if depth <= 1:
		depth = 1
```

---
## Step 14 — return new config

```python
return [lrate, n_tree, subsam, depth]
```

---
## Step 15 — hill climbing local search algorithm

```python
def hillclimbing(X, y, objective, n_iter):
```

---
## Step 16 — starting point for the search

```python
solution = step([0.1, 100, 1.0, 7])
```

---
## Step 17 — evaluate the initial point

```python
solution_eval = objective(X, y, solution)
```

---
## Step 18 — run the hill climb

```python
# 生成整数序列 / Generate integer sequence
for i in range(n_iter):
```

---
## Step 19 — take a step

```python
candidate = step(solution)
```

---
## Step 20 — evaluate candidate point

```python
candidate_eval = objective(X, y, candidate)
```

---
## Step 21 — check if we should keep the new point

```python
if candidate_eval >= solution_eval:
```

---
## Step 22 — store the new point

```python
solution, solution_eval = candidate, candidate_eval
```

---
## Step 23 — report progress

```python
# 打印输出 / Print output
print('>%d, cfg=[%s] %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]
```

---
## Step 24 — define dataset

```python
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
```

---
## Step 25 — define the total iterations

```python
n_iter = 200
```

---
## Step 26 — perform the hill climbing search

```python
cfg, score = hillclimbing(X, y, objective, n_iter)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=[%s]: Mean Accuracy: %f' % (cfg, score))
```

---
## Learning Notes / 学习笔记

- **概念**: xgboost manual hyperparameter optimization for binary classification 是机器学习中的常用技术。  
  *xgboost manual hyperparameter optimization for binary classification is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `learning_rate` | 学习率：参数更新步长 | Learning rate: step size for parameter updates |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `numpy` | 数值计算库 | Numerical computing library |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Hillclimbing Xgboost / 提升方法
# Complete Code / 完整代码
# ===============================

# xgboost manual hyperparameter optimization for binary classification
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randn
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import randint
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import RepeatedStratifiedKFold
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier

# objective function
def objective(X, y, cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# define model
	model = XGBClassifier(learning_rate=lrate, n_estimators=n_tree, subsample=subsam, max_depth=depth, use_label_encoder=False, eval_metric="logloss")
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# calculate mean accuracy
	result = mean(scores)
	return result

# take a step in the search space
def step(cfg):
	# unpack config
	lrate, n_tree, subsam, depth = cfg
	# learning rate
	lrate = lrate + randn() * 0.01
	if lrate <= 0.0:
		lrate = 1e-8
	if lrate > 1:
		lrate = 1.0
	# number of trees
	n_tree = round(n_tree + randn() * 50)
	if n_tree <= 0.0:
		n_tree = 1
	# subsample percentage
	subsam = subsam + randn() * 0.1
	if subsam <= 0.0:
		subsam = 1e-8
	if subsam > 1:
		subsam = 1.0
	# max tree depth
	depth = round(depth + randn() * 7)
	if depth <= 1:
		depth = 1
	# return new config
	return [lrate, n_tree, subsam, depth]

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter):
	# starting point for the search
	solution = step([0.1, 100, 1.0, 7])
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
 # 生成整数序列 / Generate integer sequence
	for i in range(n_iter):
		# take a step
		candidate = step(solution)
		# evaluate candidate point
		candidate_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidate_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# report progress
   # 打印输出 / Print output
			print('>%d, cfg=[%s] %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# define the total iterations
n_iter = 200
# perform the hill climbing search
cfg, score = hillclimbing(X, y, objective, n_iter)
# 打印输出 / Print output
print('Done!')
# 打印输出 / Print output
print('cfg=[%s]: Mean Accuracy: %f' % (cfg, score))
```

---

### Chapter Summary / 章节总结



---
