# Python XGBoost 实战 / XGBoost with Python
## Chapter 11

---

### Chapter Summary / 章节总结



---

### Eval Num Threads

# 01 — Eval Num Threads / 模型评估

**Chapter 11 — File 1 of 2 / 第11章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Otto, tune number of threads**.

本脚本演示 **Otto, tune number of threads**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Otto, tune number of threads

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
from time import time
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:94]
y = dataset[:,94]
```

---
## Step 4 — encode string class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
```

---
## Step 5 — evaluate the effect of the number of threads

```python
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
	start = time()
	model = XGBClassifier(nthread=n)
 # 训练模型 / Train the model
	model.fit(X, label_encoded_y)
	elapsed = time() - start
 # 打印输出 / Print output
	print(n, elapsed)
 # 添加元素到列表末尾 / Append element to list end
	results.append(elapsed)
```

---
## Step 6 — plot results

```python
pyplot.plot(num_threads, results)
pyplot.ylabel('Speed (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Otto, tune number of threads 是机器学习中的常用技术。  
  *Otto, tune number of threads is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Eval Num Threads / 模型评估
# Complete Code / 完整代码
# ===============================

# Otto, tune number of threads
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
from time import time
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
# evaluate the effect of the number of threads
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
	start = time()
	model = XGBClassifier(nthread=n)
 # 训练模型 / Train the model
	model.fit(X, label_encoded_y)
	elapsed = time() - start
 # 打印输出 / Print output
	print(n, elapsed)
 # 添加元素到列表末尾 / Append element to list end
	results.append(elapsed)
# plot results
pyplot.plot(num_threads, results)
pyplot.ylabel('Speed (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show()
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Eval Parallel Cv And Xgboost

# 01 — Eval Parallel Cv And Xgboost / 模型评估

**Chapter 11 — File 2 of 2 / 第11章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **Otto, parallel cross validation**.

本脚本演示 **Otto, parallel cross validation**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
```

---
## Step 1 — Otto, parallel cross validation

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入时间处理模块 / Import time module
import time
```

---
## Step 2 — load data

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
```

---
## Step 3 — split data into X and y

```python
X = dataset[:,0:94]
y = dataset[:,94]
```

---
## Step 4 — encode string class values as integers

```python
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
```

---
## Step 5 — prepare cross validation

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
```

---
## Step 6 — Single Thread XGBoost, Parallel Thread CV

```python
start = time.time()
model = XGBClassifier(nthread=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Single Thread XGBoost, Parallel Thread CV: %f" % (elapsed))
```

---
## Step 7 — Parallel Thread XGBoost, Single Thread CV

```python
start = time.time()
model = XGBClassifier(nthread=-1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))
```

---
## Step 8 — Parallel Thread XGBoost and CV

```python
start = time.time()
model = XGBClassifier(nthread=-1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Parallel Thread XGBoost and CV: %f" % (elapsed))
```

---
## Learning Notes / 学习笔记

- **概念**: Otto, parallel cross validation 是机器学习中的常用技术。  
  *Otto, parallel cross validation is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `XGBClassifier` | XGBoost分类器 | XGBoost classifier |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `xgboost` | 梯度提升框架 | Gradient boosting framework |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Eval Parallel Cv And Xgboost / 模型评估
# Complete Code / 完整代码
# ===============================

# Otto, parallel cross validation
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入XGBoost梯度提升库 / Import XGBoost gradient boosting library
from xgboost import XGBClassifier
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import StratifiedKFold
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import cross_val_score
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import LabelEncoder
# 导入时间处理模块 / Import time module
import time
# load data
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
data = read_csv('train.csv')
# 转换为NumPy数组 / Convert to NumPy array
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
# 将类别标签编码为数字 / Encode categorical labels to numbers
label_encoded_y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# Single Thread XGBoost, Parallel Thread CV
start = time.time()
model = XGBClassifier(nthread=1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Single Thread XGBoost, Parallel Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost, Single Thread CV
start = time.time()
model = XGBClassifier(nthread=-1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost and CV
start = time.time()
model = XGBClassifier(nthread=-1)
# 交叉验证：多次划分数据评估模型稳定性 / Cross-validation: evaluate model stability
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
# 打印输出 / Print output
print("Parallel Thread XGBoost and CV: %f" % (elapsed))
```

---
