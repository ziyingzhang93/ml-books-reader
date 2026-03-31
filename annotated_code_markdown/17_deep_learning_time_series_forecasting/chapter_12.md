# 深度学习时间序列预测 / DL Time Series Forecasting
## Chapter 12

---

### Grid Search

# 01 — Grid Search / 01 Grid Search

**Chapter 12 — File 1 of 5 / 第12章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search holt winter's exponential smoothing**.

本脚本演示 **grid search holt winter's exponential smoothing**。

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
## Step 1 — grid search holt winter's exponential smoothing

```python
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

---
## Step 2 — one-step Holt Winter’s Exponential Smoothing forecast

```python
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
```

---
## Step 3 — define model

```python
history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
```

---
## Step 5 — make one step forecast

```python
# 获取长度 / Get length
yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
```

---
## Step 6 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 7 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 8 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 9 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 10 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 11 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = exp_smoothing_forecast(history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 15 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	return error
```

---
## Step 16 — score a model, return None on failure

```python
def score_model(data, n_test, cfg, debug=False):
	result = None
```

---
## Step 17 — convert config to a key

```python
key = str(cfg)
```

---
## Step 18 — show all warnings and fail on exception if debugging

```python
if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
```

---
## Step 19 — one failure during model validation suggests an unstable config

```python
try:
```

---
## Step 20 — never show warnings when grid searching, too noisy

```python
with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
```

---
## Step 21 — check for an interesting result

```python
if result is not None:
  # 打印输出 / Print output
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 22 — grid search configs

```python
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
```

---
## Step 23 — execute configs in parallel

```python
executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
```

---
## Step 24 — remove empty results

```python
scores = [r for r in scores if r[1] != None]
```

---
## Step 25 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 26 — create a set of exponential smoothing configs to try

```python
def exp_smoothing_configs(seasonal=[None]):
	models = list()
```

---
## Step 27 — define config lists

```python
t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
```

---
## Step 28 — create config instances

```python
for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
       # 添加元素到列表末尾 / Append element to list end
							models.append(cfg)
	return models

if __name__ == '__main__':
```

---
## Step 29 — define dataset

```python
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
 # 打印输出 / Print output
	print(data)
```

---
## Step 30 — data split

```python
n_test = 4
```

---
## Step 31 — model configs

```python
cfg_list = exp_smoothing_configs()
```

---
## Step 32 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
 # 打印输出 / Print output
	print('done')
```

---
## Step 33 — list top 3 configs

```python
for cfg, error in scores[:3]:
  # 打印输出 / Print output
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search holt winter's exponential smoothing 是机器学习中的常用技术。  
  *grid search holt winter's exponential smoothing is a common technique in machine learning.*

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
# Grid Search / 01 Grid Search
# Complete Code / 完整代码
# ===============================

# grid search holt winter's exponential smoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model
	history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
 # 训练模型 / Train the model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
 # 获取长度 / Get length
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = exp_smoothing_forecast(history, cfg)
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

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
       # 添加元素到列表末尾 / Append element to list end
							models.append(cfg)
	return models

if __name__ == '__main__':
	# define dataset
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
 # 打印输出 / Print output
	print(data)
	# data split
	n_test = 4
	# model configs
	cfg_list = exp_smoothing_configs()
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

➡️ **Next / 下一步**: File 2 of 5

---

### Grid Search Daily Births



---

### Grid Search Monthly Shampoo Sales



---

### Grid Search Monthly Mean Temp

# 04 — Grid Search Monthly Mean Temp / 04 Grid Search Monthly Mean Temp

**Chapter 12 — File 4 of 5 / 第12章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search ets hyperparameters for monthly mean temp dataset**.

本脚本演示 **grid search ets hyperparameters for monthly mean temp dataset**。

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
## Step 1 — grid search ets hyperparameters for monthly mean temp dataset

```python
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array
```

---
## Step 2 — one-step Holt Winter’s Exponential Smoothing forecast

```python
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
```

---
## Step 3 — define model

```python
history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
```

---
## Step 4 — fit model

```python
# 训练模型 / Train the model
model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
```

---
## Step 5 — make one step forecast

```python
# 获取长度 / Get length
yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
```

---
## Step 6 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
 # 计算均方误差 / Calculate Mean Squared Error
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 7 — split a univariate dataset into train/test sets

```python
# 划分训练集和测试集 / Split into train and test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
```

---
## Step 8 — walk-forward validation for univariate data

```python
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
```

---
## Step 9 — split dataset

```python
# 划分训练集和测试集 / Split into train and test sets
train, test = train_test_split(data, n_test)
```

---
## Step 10 — seed history with training dataset

```python
history = [x for x in train]
```

---
## Step 11 — step over each time-step in the test set

```python
# 获取长度 / Get length
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = exp_smoothing_forecast(history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
# 添加元素到列表末尾 / Append element to list end
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
# 添加元素到列表末尾 / Append element to list end
history.append(test[i])
```

---
## Step 15 — estimate prediction error

```python
error = measure_rmse(test, predictions)
	return error
```

---
## Step 16 — score a model, return None on failure

```python
def score_model(data, n_test, cfg, debug=False):
	result = None
```

---
## Step 17 — convert config to a key

```python
key = str(cfg)
```

---
## Step 18 — show all warnings and fail on exception if debugging

```python
if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
```

---
## Step 19 — one failure during model validation suggests an unstable config

```python
try:
```

---
## Step 20 — never show warnings when grid searching, too noisy

```python
with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
```

---
## Step 21 — check for an interesting result

```python
if result is not None:
  # 打印输出 / Print output
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
```

---
## Step 22 — grid search configs

```python
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
```

---
## Step 23 — execute configs in parallel

```python
executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
```

---
## Step 24 — remove empty results

```python
scores = [r for r in scores if r[1] != None]
```

---
## Step 25 — sort configs by error, asc

```python
scores.sort(key=lambda tup: tup[1])
	return scores
```

---
## Step 26 — create a set of exponential smoothing configs to try

```python
def exp_smoothing_configs(seasonal=[None]):
	models = list()
```

---
## Step 27 — define config lists

```python
t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
```

---
## Step 28 — create config instances

```python
for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
       # 添加元素到列表末尾 / Append element to list end
							models.append(cfg)
	return models

if __name__ == '__main__':
```

---
## Step 29 — load dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
 # 转换为NumPy数组 / Convert to NumPy array
	data = series.values
```

---
## Step 30 — trim dataset to 5 years

```python
data = data[-(5*12):]
```

---
## Step 31 — data split

```python
n_test = 12
```

---
## Step 32 — model configs

```python
cfg_list = exp_smoothing_configs(seasonal=[0,12])
```

---
## Step 33 — grid search

```python
scores = grid_search(data[:,0], cfg_list, n_test)
 # 打印输出 / Print output
	print('done')
```

---
## Step 34 — list top 3 configs

```python
for cfg, error in scores[:3]:
  # 打印输出 / Print output
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search ets hyperparameters for monthly mean temp dataset 是机器学习中的常用技术。  
  *grid search ets hyperparameters for monthly mean temp dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Monthly Mean Temp / 04 Grid Search Monthly Mean Temp
# Complete Code / 完整代码
# ===============================

# grid search ets hyperparameters for monthly mean temp dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import array

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model
	history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
 # 训练模型 / Train the model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
 # 获取长度 / Get length
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

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
		yhat = exp_smoothing_forecast(history, cfg)
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

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
       # 添加元素到列表末尾 / Append element to list end
							models.append(cfg)
	return models

if __name__ == '__main__':
	# load dataset
 # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
	series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
 # 转换为NumPy数组 / Convert to NumPy array
	data = series.values
	# trim dataset to 5 years
	data = data[-(5*12):]
	# data split
	n_test = 12
	# model configs
	cfg_list = exp_smoothing_configs(seasonal=[0,12])
	# grid search
	scores = grid_search(data[:,0], cfg_list, n_test)
 # 打印输出 / Print output
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
  # 打印输出 / Print output
		print(cfg, error)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Grid Search Monthly Car Sales



---

### Chapter Summary / 章节总结



---
