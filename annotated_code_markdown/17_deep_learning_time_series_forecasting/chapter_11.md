# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 11

---

### Persistence Forecast

# 01 — Persistence Forecast / 预测

**Chapter 11 — File 1 of 12 / 第11章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **example of a one-step naive forecast**.

本脚本演示 **example of a one-step naive forecast**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of a one-step naive forecast

```python
def naive_forecast(history, n):
	return history[-n]
```

---
## Step 2 — define dataset

```python
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
# 打印输出 / Print output
print(data)
```

---
## Step 3 — test naive forecast

```python
# 获取长度 / Get length
for i in range(1, len(data)+1):
 # 打印输出 / Print output
	print(naive_forecast(data, i))
```

---
## Learning Notes / 学习笔记

- **概念**: example of a one-step naive forecast 是机器学习中的常用技术。  
  *example of a one-step naive forecast is a common technique in machine learning.*

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
# Persistence Forecast / 预测
# Complete Code / 完整代码
# ===============================

# example of a one-step naive forecast
def naive_forecast(history, n):
	return history[-n]

# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
# 打印输出 / Print output
print(data)
# test naive forecast
# 获取长度 / Get length
for i in range(1, len(data)+1):
 # 打印输出 / Print output
	print(naive_forecast(data, i))
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Average Forecast



---

### Average Forecast Seasonality

# 03 — Average Forecast Seasonality / 预测

**Chapter 11 — File 3 of 12 / 第11章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **example of an average forecast for seasonal data**.

本脚本演示 **example of an average forecast for seasonal data**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — example of an average forecast for seasonal data

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median
```

---
## Step 2 — one-step average forecast

```python
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
```

---
## Step 3 — skip bad configs

```python
# 获取长度 / Get length
if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
```

---
## Step 4 — try and collect n values using offset

```python
# 生成整数序列 / Generate integer sequence
for i in range(1, n+1):
			ix = i * offset
   # 添加元素到列表末尾 / Append element to list end
			values.append(history[-ix])
```

---
## Step 5 — mean of last n values

```python
if avg_type is 'mean':
		return mean(values)
```

---
## Step 6 — median of last n values

```python
return median(values)
```

---
## Step 7 — define dataset

```python
data = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
# 打印输出 / Print output
print(data)
```

---
## Step 8 — test naive forecast

```python
for i in [1, 2, 3]:
 # 打印输出 / Print output
	print(average_forecast(data, (i, 3, 'mean')))
```

---
## Learning Notes / 学习笔记

- **概念**: example of an average forecast for seasonal data 是机器学习中的常用技术。  
  *example of an average forecast for seasonal data is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Average Forecast Seasonality / 预测
# Complete Code / 完整代码
# ===============================

# example of an average forecast for seasonal data
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median

# one-step average forecast
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
  # 获取长度 / Get length
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
  # 生成整数序列 / Generate integer sequence
		for i in range(1, n+1):
			ix = i * offset
   # 添加元素到列表末尾 / Append element to list end
			values.append(history[-ix])
	# mean of last n values
	if avg_type is 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# define dataset
data = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
# 打印输出 / Print output
print(data)
# test naive forecast
for i in [1, 2, 3]:
 # 打印输出 / Print output
	print(average_forecast(data, (i, 3, 'mean')))
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Grid Search

# 04 — Grid Search / 04 Grid Search

**Chapter 11 — File 4 of 12 / 第11章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **grid search simple forecasts**.

本脚本演示 **grid search simple forecasts**。

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
## Step 1 — grid search simple forecasts

```python
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — one-step simple forecast

```python
def simple_forecast(history, config):
	n, offset, avg_type = config
```

---
## Step 3 — persist value, ignore other config

```python
if avg_type == 'persist':
		return history[-n]
```

---
## Step 4 — collect values to average

```python
values = list()
	if offset == 1:
		values = history[-n:]
	else:
```

---
## Step 5 — skip bad configs

```python
# 获取长度 / Get length
if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
```

---
## Step 6 — try and collect n values using offset

```python
# 生成整数序列 / Generate integer sequence
for i in range(1, n+1):
			ix = i * offset
   # 添加元素到列表末尾 / Append element to list end
			values.append(history[-ix])
```

---
## Step 7 — check if we can average

```python
# 获取长度 / Get length
if len(values) < 2:
		raise Exception('Cannot calculate average')
```

---
## Step 8 — mean of last n values

```python
if avg_type == 'mean':
		return mean(values)
```

---
## Step 9 — median of last n values

```python
return median(values)
```

---
## Step 10 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 11 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 12 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 13 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 14 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 15 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 16 — fit model and make forecast for history

```python
yhat = simple_forecast(history, cfg)
```

---
## Step 17 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 18 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 19 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	return error
```

---
## Step 20 — score a model, return None on failure

```python
def score_model(data, n_test, cfg, debug=False):
	result = None
```

---
## Step 21 — convert config to a key

```python
key = str(cfg)
```

---
## Step 22 — show all warnings and fail on exception if debugging

```python
if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
```

---
## Step 23 — one failure during model validation suggests an unstable config

```python
try:
```

---
## Step 24 — never show warnings when grid searching, too noisy

```python
with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
```

---
## Step 25 — check for an interesting result

```python
if result is not None:
  # 打印输出 / Print output
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 26 — grid search configs

```python
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
```

---
## Step 27 — execute configs in parallel

```python
executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
```

---
## Step 28 — remove empty results

```python
scores = [r for r in scores if r[1] != None]
```

---
## Step 29 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 30 — create a set of simple configs to try

```python
def simple_configs(max_length, offsets=[1]):
	configs = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
    # 添加元素到列表末尾 / Append element to list end
				configs.append(cfg)
	return configs

if __name__ == '__main__':
```

---
## Step 31 — define dataset

```python
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
```

---
## Step 32 — data split

```python
n_test = 4
```

---
## Step 33 — model configs

```python
# 获取长度 / Get length
max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
```

---
## Step 34 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
 # 打印输出 / Print output
	print('done')
```

---
## Step 35 — list top 3 configs

```python
for cfg, error in scores[:3]:
  # 打印输出 / Print output
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search simple forecasts 是机器学习中的常用技术。  
  *grid search simple forecasts is a common technique in machine learning.*

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
# Grid Search / 04 Grid Search
# Complete Code / 完整代码
# ===============================

# grid search simple forecasts
from math import sqrt
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import mean
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import median
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error

# one-step simple forecast
def simple_forecast(history, config):
	n, offset, avg_type = config
	# persist value, ignore other config
	if avg_type == 'persist':
		return history[-n]
	# collect values to average
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# skip bad configs
  # 获取长度 / Get length
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n,offset))
		# try and collect n values using offset
  # 生成整数序列 / Generate integer sequence
		for i in range(1, n+1):
			ix = i * offset
   # 添加元素到列表末尾 / Append element to list end
			values.append(history[-ix])
	# check if we can average
 # 获取长度 / Get length
	if len(values) < 2:
		raise Exception('Cannot calculate average')
	# mean of last n values
	if avg_type == 'mean':
		return mean(values)
	# median of last n values
	return median(values)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
 # 划分训练集和测试集 / Split into train and test sets
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
 # 获取长度 / Get length
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = simple_forecast(history, cfg)
		# store forecast in list of predictions
  # 添加元素到列表末尾 / Append element to list end
		predictions.append(yhat)
		# add actual observation to history for the next loop
  # 添加元素到列表末尾 / Append element to list end
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
  # 打印输出 / Print output
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
	configs = list()
 # 生成整数序列 / Generate integer sequence
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
    # 添加元素到列表末尾 / Append element to list end
				configs.append(cfg)
	return configs

if __name__ == '__main__':
	# define dataset
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	# data split
	n_test = 4
	# model configs
 # 获取长度 / Get length
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
	# grid search
	scores = grid_search(data, cfg_list, n_test)
 # 打印输出 / Print output
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
  # 打印输出 / Print output
		print(cfg, error)
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Load Plot Daily Births

# 05 — Load Plot Daily Births / 05 Load Plot Daily Births

**Chapter 11 — File 5 of 12 / 第11章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load and plot daily births dataset**.

本脚本演示 **load and plot daily births dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — load and plot daily births dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
```

---
## Step 3 — summarize shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
```

---
## Step 4 — plot

```python
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot daily births dataset 是机器学习中的常用技术。  
  *load and plot daily births dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Plot Daily Births / 05 Load Plot Daily Births
# Complete Code / 完整代码
# ===============================

# load and plot daily births dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
# summarize shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Grid Search Daily Births



---

### Load Plot Monthly Shampoo

# 07 — Load Plot Monthly Shampoo / 07 Load Plot Monthly Shampoo

**Chapter 11 — File 7 of 12 / 第11章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load and plot monthly shampoo sales dataset**.

本脚本演示 **load and plot monthly shampoo sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — load and plot monthly shampoo sales dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)
```

---
## Step 3 — summarize shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
```

---
## Step 4 — plot

```python
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot monthly shampoo sales dataset 是机器学习中的常用技术。  
  *load and plot monthly shampoo sales dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Plot Monthly Shampoo / 07 Load Plot Monthly Shampoo
# Complete Code / 完整代码
# ===============================

# load and plot monthly shampoo sales dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-shampoo-sales.csv', header=0, index_col=0)
# summarize shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Grid Search Shampoo Sales



---

### Load Plot Monthly Mean Temp

# 09 — Load Plot Monthly Mean Temp / 09 Load Plot Monthly Mean Temp

**Chapter 11 — File 9 of 12 / 第11章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load and plot monthly mean temp dataset**.

本脚本演示 **load and plot monthly mean temp dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — load and plot monthly mean temp dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
```

---
## Step 3 — summarize shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
```

---
## Step 4 — plot

```python
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot monthly mean temp dataset 是机器学习中的常用技术。  
  *load and plot monthly mean temp dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Plot Monthly Mean Temp / 09 Load Plot Monthly Mean Temp
# Complete Code / 完整代码
# ===============================

# load and plot monthly mean temp dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
# summarize shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Grid Search Mean Temp



---

### Load Plot Monthly Car Sales

# 11 — Load Plot Monthly Car Sales / 11 Load Plot Monthly Car Sales

**Chapter 11 — File 11 of 12 / 第11章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **load and plot monthly car sales dataset**.

本脚本演示 **load and plot monthly car sales dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌────────────────────┐
│  加载数据 Load Data  │
└────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — load and plot monthly car sales dataset

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — load

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
```

---
## Step 3 — summarize shape

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
```

---
## Step 4 — plot

```python
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: load and plot monthly car sales dataset 是机器学习中的常用技术。  
  *load and plot monthly car sales dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Plot Monthly Car Sales / 11 Load Plot Monthly Car Sales
# Complete Code / 完整代码
# ===============================

# load and plot monthly car sales dataset
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# load
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
# summarize shape
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(series.shape)
# plot
pyplot.plot(series)
pyplot.xticks([])
pyplot.show()
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Grid Search Car Sales



---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **12 code files** demonstrating chapter 11.

本章包含 **12 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_persistence_forecast.ipynb` — Persistence Forecast
  2. `02_average_forecast.ipynb` — Average Forecast
  3. `03_average_forecast_seasonality.ipynb` — Average Forecast Seasonality
  4. `04_grid_search.ipynb` — Grid Search
  5. `05_load_plot_daily_births.ipynb` — Load Plot Daily Births
  6. `06_grid_search_daily_births.ipynb` — Grid Search Daily Births
  7. `07_load_plot_monthly_shampoo.ipynb` — Load Plot Monthly Shampoo
  8. `08_grid_search_shampoo_sales.ipynb` — Grid Search Shampoo Sales
  9. `09_load_plot_monthly_mean_temp.ipynb` — Load Plot Monthly Mean Temp
  10. `10_grid_search_mean_temp.ipynb` — Grid Search Mean Temp
  11. `11_load_plot_monthly_car_sales.ipynb` — Load Plot Monthly Car Sales
  12. `12_grid_search_car_sales.ipynb` — Grid Search Car Sales

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
