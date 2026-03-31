# 集成学习算法 / Ensemble Learning Algorithms
## Chapter 12

---

### Dataset



---

### Knora E Evaluate

# 02 — Knora E Evaluate / 模型评估

**Chapter 12 — File 2 of 9 / 第12章 — 第2个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluate dynamic KNORA-E dynamic ensemble selection for binary classification**.

本脚本演示 **evaluate dynamic KNORA-E dynamic ensemble selection for binary classification**。

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
## Step 1 — evaluate dynamic KNORA-E dynamic ensemble selection for binary classification

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
from deslib.des.knora_e import KNORAE
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = KNORAE()
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate dynamic KNORA-E dynamic ensemble selection for binary classification 是机器学习中的常用技术。  
  *evaluate dynamic KNORA-E dynamic ensemble selection for binary classification is a common technique in machine learning.*

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
# Knora E Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate dynamic KNORA-E dynamic ensemble selection for binary classification
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
from deslib.des.knora_e import KNORAE
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = KNORAE()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 3 of 9

---

### Knora E Predict

# 03 — Knora E Predict / 03 Knora E Predict

**Chapter 12 — File 3 of 9 / 第12章 — 第3个文件（共9个）**

---

## Summary / 总结

This script demonstrates **make a prediction with KNORA-E dynamic ensemble selection**.

本脚本演示 **make a prediction with KNORA-E dynamic ensemble selection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — make a prediction with KNORA-E dynamic ensemble selection

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
from deslib.des.knora_e import KNORAE
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = KNORAE()
```

---
## Step 4 — fit the model on the whole dataset

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 6 — summarize the prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with KNORA-E dynamic ensemble selection 是机器学习中的常用技术。  
  *make a prediction with KNORA-E dynamic ensemble selection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Knora E Predict / 03 Knora E Predict
# Complete Code / 完整代码
# ===============================

# make a prediction with KNORA-E dynamic ensemble selection
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
from deslib.des.knora_e import KNORAE
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = KNORAE()
# fit the model on the whole dataset
# 训练模型 / Train the model
model.fit(X, y)
# make a single prediction
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize the prediction
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 4 of 9

---

### Knora U Evaluate

# 04 — Knora U Evaluate / 模型评估

**Chapter 12 — File 4 of 9 / 第12章 — 第4个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluate dynamic KNORA-U dynamic ensemble selection for binary classification**.

本脚本演示 **evaluate dynamic KNORA-U dynamic ensemble selection for binary classification**。

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
## Step 1 — evaluate dynamic KNORA-U dynamic ensemble selection for binary classification

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
from deslib.des.knora_u import KNORAU
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = KNORAU()
```

---
## Step 4 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 5 — evaluate the model

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
```

---
## Step 6 — report performance

```python
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate dynamic KNORA-U dynamic ensemble selection for binary classification 是机器学习中的常用技术。  
  *evaluate dynamic KNORA-U dynamic ensemble selection for binary classification is a common technique in machine learning.*

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
# Knora U Evaluate / 模型评估
# Complete Code / 完整代码
# ===============================

# evaluate dynamic KNORA-U dynamic ensemble selection for binary classification
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
from deslib.des.knora_u import KNORAU
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = KNORAU()
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
# 打印输出 / Print output
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

---

➡️ **Next / 下一步**: File 5 of 9

---

### Knora U Predict

# 05 — Knora U Predict / 05 Knora U Predict

**Chapter 12 — File 5 of 9 / 第12章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **make a prediction with KNORA-U dynamic ensemble selection**.

本脚本演示 **make a prediction with KNORA-U dynamic ensemble selection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — make a prediction with KNORA-U dynamic ensemble selection

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
from deslib.des.knora_u import KNORAU
```

---
## Step 2 — define dataset

```python
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 3 — define the model

```python
model = KNORAU()
```

---
## Step 4 — fit the model on the whole dataset

```python
# 训练模型 / Train the model
model.fit(X, y)
```

---
## Step 5 — make a single prediction

```python
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
```

---
## Step 6 — summarize the prediction

```python
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---
## Learning Notes / 学习笔记

- **概念**: make a prediction with KNORA-U dynamic ensemble selection 是机器学习中的常用技术。  
  *make a prediction with KNORA-U dynamic ensemble selection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Knora U Predict / 05 Knora U Predict
# Complete Code / 完整代码
# ===============================

# make a prediction with KNORA-U dynamic ensemble selection
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
from deslib.des.knora_u import KNORAU
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# define the model
model = KNORAU()
# fit the model on the whole dataset
# 训练模型 / Train the model
model.fit(X, y)
# make a single prediction
row = [0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799, 3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]
# 用模型做预测 / Make predictions with model
yhat = model.predict([row])
# summarize the prediction
# 打印输出 / Print output
print('Predicted Class: %d' % yhat[0])
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Tune Knn

# 06 — Tune Knn / 超参数调优

**Chapter 12 — File 6 of 9 / 第12章 — 第6个文件（共9个）**

---

## Summary / 总结

This script demonstrates **explore k in knn for KNORA-U dynamic ensemble selection**.

本脚本演示 **explore k in knn for KNORA-U dynamic ensemble selection**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — explore k in knn for KNORA-U dynamic ensemble selection

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
from deslib.des.knora_u import KNORAU
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — get the dataset

```python
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y
```

---
## Step 3 — get a list of models to evaluate

```python
def get_models():
	models = dict()
```

---
## Step 4 — explore k values from 2 to 21

```python
# 生成整数序列 / Generate integer sequence
for n in range(2,22):
		models[str(n)] = KNORAU(k=n)
	return models
```

---
## Step 5 — evaluate a given model using cross-validation

```python
def evaluate_model(model, X, y):
```

---
## Step 6 — define the evaluation procedure

```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
```

---
## Step 7 — evaluate the model and collect the scores

```python
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
```

---
## Step 8 — define dataset

```python
X, y = get_dataset()
```

---
## Step 9 — get the models to evaluate

```python
models = get_models()
```

---
## Step 10 — evaluate the models and store results

```python
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
```

---
## Step 11 — evaluate the model

```python
scores = evaluate_model(model, X, y)
```

---
## Step 12 — store the results

```python
# 添加元素到列表末尾 / Append element to list end
results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
```

---
## Step 13 — summarize the results along the way

```python
# 打印输出 / Print output
print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
```

---
## Step 14 — plot model performance for comparison

```python
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: explore k in knn for KNORA-U dynamic ensemble selection 是机器学习中的常用技术。  
  *explore k in knn for KNORA-U dynamic ensemble selection is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Tune Knn / 超参数调优
# Complete Code / 完整代码
# ===============================

# explore k in knn for KNORA-U dynamic ensemble selection
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
from deslib.des.knora_u import KNORAU
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
	return X, y

# get a list of models to evaluate
def get_models():
	models = dict()
	# explore k values from 2 to 21
 # 生成整数序列 / Generate integer sequence
	for n in range(2,22):
		models[str(n)] = KNORAU(k=n)
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the scores
 # 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
# 获取字典的键值对 / Get dict key-value pairs
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
 # 添加元素到列表末尾 / Append element to list end
	results.append(scores)
 # 添加元素到列表末尾 / Append element to list end
	names.append(name)
	# summarize the results along the way
 # 打印输出 / Print output
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 7 of 9

---

### Pool Classifiers

# 07 — Pool Classifiers / 07 Pool Classifiers

**Chapter 12 — File 7 of 9 / 第12章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms**.

本脚本演示 **evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
## Step 1 — evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
```

---
## Step 2 — split the dataset into train and test sets

```python
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

---
## Step 3 — define classifiers to use in the pool

```python
classifiers = [
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	LogisticRegression(),
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	DecisionTreeClassifier(),
	GaussianNB()]
```

---
## Step 4 — fit each classifier on the training set

```python
for c in classifiers:
	c.fit(X_train, y_train)
```

---
## Step 5 — define the KNORA-U model

```python
model = KNORAU(pool_classifiers=classifiers)
```

---
## Step 6 — fit the model

```python
# 训练模型 / Train the model
model.fit(X_train, y_train)
```

---
## Step 7 — make predictions on the test set

```python
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
```

---
## Step 8 — evaluate predictions

```python
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (score))
```

---
## Learning Notes / 学习笔记

- **概念**: evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms 是机器学习中的常用技术。  
  *evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `DecisionTree` | 决策树 | Decision Tree |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
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
# Pool Classifiers / 07 Pool Classifiers
# Complete Code / 完整代码
# ===============================

# evaluate KNORA-U dynamic ensemble selection with a custom pool of algorithms
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classification
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.linear_model import LogisticRegression
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.tree import DecisionTreeClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
# split the dataset into train and test sets
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# define classifiers to use in the pool
classifiers = [
 # 逻辑回归：线性分类器 / Logistic Regression: linear classifier
	LogisticRegression(),
 # 决策树：if-else规则分类 / Decision Tree: if-else rules for classification
	DecisionTreeClassifier(),
	GaussianNB()]
# fit each classifier on the training set
for c in classifiers:
	c.fit(X_train, y_train)
# define the KNORA-U model
model = KNORAU(pool_classifiers=classifiers)
# fit the model
# 训练模型 / Train the model
model.fit(X_train, y_train)
# make predictions on the test set
# 用模型做预测 / Make predictions with model
yhat = model.predict(X_test)
# evaluate predictions
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
score = accuracy_score(y_test, yhat)
# 打印输出 / Print output
print('Accuracy: %.3f' % (score))
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Pool Classifiers Standalone



---

### Pool Classifiers Rf



---

### Chapter Summary / 章节总结



---
