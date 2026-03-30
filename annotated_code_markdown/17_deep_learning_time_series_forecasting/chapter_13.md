# DL时间序列
## Chapter 13

---

### Grid Search

# 01 — Grid Search / 01 Grid Search

**Chapter 13 — File 1 of 5 / 第13章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search sarima hyperparameters**.

本脚本演示 **grid search sarima hyperparameters**。

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
## Step 1 — grid search sarima hyperparameters

```python
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```

---
## Step 2 — one-step sarima forecast

```python
def sarima_forecast(history, config):
	order, sorder, trend = config
```

---
## Step 3 — define model

```python
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
```

---
## Step 4 — fit model

```python
model_fit = model.fit(disp=False)
```

---
## Step 5 — make one step forecast

```python
yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
```

---
## Step 6 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 7 — split a univariate dataset into train/test sets

```python
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
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = sarima_forecast(history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
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
## Step 26 — create a set of sarima configs to try

```python
def sarima_configs(seasonal=[0]):
	models = list()
```

---
## Step 27 — define config lists

```python
p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
```

---
## Step 28 — create config instances

```python
for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
```

---
## Step 29 — define dataset

```python
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
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
cfg_list = sarima_configs()
```

---
## Step 32 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
	print('done')
```

---
## Step 33 — list top 3 configs

```python
for cfg, error in scores[:3]:
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search sarima hyperparameters 是机器学习中的常用技术。  
  *grid search sarima hyperparameters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
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

# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
	# define dataset
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	print(data)
	# data split
	n_test = 4
	# model configs
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Grid Search Monthly Mean Temp

# 04 — Grid Search Monthly Mean Temp / 04 Grid Search Monthly Mean Temp

**Chapter 13 — File 4 of 5 / 第13章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search sarima hyperparameters for monthly mean temp dataset**.

本脚本演示 **grid search sarima hyperparameters for monthly mean temp dataset**。

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
## Step 1 — grid search sarima hyperparameters for monthly mean temp dataset

```python
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
```

---
## Step 2 — one-step sarima forecast

```python
def sarima_forecast(history, config):
	order, sorder, trend = config
```

---
## Step 3 — define model

```python
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
```

---
## Step 4 — fit model

```python
model_fit = model.fit(disp=False)
```

---
## Step 5 — make one step forecast

```python
yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
```

---
## Step 6 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 7 — split a univariate dataset into train/test sets

```python
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
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = sarima_forecast(history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
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
## Step 26 — create a set of sarima configs to try

```python
def sarima_configs(seasonal=[0]):
	models = list()
```

---
## Step 27 — define config lists

```python
p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
```

---
## Step 28 — create config instances

```python
for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
```

---
## Step 29 — load dataset

```python
series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
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
cfg_list = sarima_configs(seasonal=[0, 12])
```

---
## Step 33 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
	print('done')
```

---
## Step 34 — list top 3 configs

```python
for cfg, error in scores[:3]:
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search sarima hyperparameters for monthly mean temp dataset 是机器学习中的常用技术。  
  *grid search sarima hyperparameters for monthly mean temp dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
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

# grid search sarima hyperparameters for monthly mean temp dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
	# load dataset
	series = read_csv('monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	# trim dataset to 5 years
	data = data[-(5*12):]
	# data split
	n_test = 12
	# model configs
	cfg_list = sarima_configs(seasonal=[0, 12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Grid Search Monthly Car Sales

# 05 — Grid Search Monthly Car Sales / 05 Grid Search Monthly Car Sales

**Chapter 13 — File 5 of 5 / 第13章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **grid search sarima hyperparameters for monthly car sales dataset**.

本脚本演示 **grid search sarima hyperparameters for monthly car sales dataset**。

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
## Step 1 — grid search sarima hyperparameters for monthly car sales dataset

```python
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
```

---
## Step 2 — one-step sarima forecast

```python
def sarima_forecast(history, config):
	order, sorder, trend = config
```

---
## Step 3 — define model

```python
model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
```

---
## Step 4 — fit model

```python
model_fit = model.fit(disp=False)
```

---
## Step 5 — make one step forecast

```python
yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
```

---
## Step 6 — root mean squared error or rmse

```python
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))
```

---
## Step 7 — split a univariate dataset into train/test sets

```python
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
for i in range(len(test)):
```

---
## Step 12 — fit model and make forecast for history

```python
yhat = sarima_forecast(history, cfg)
```

---
## Step 13 — store forecast in list of predictions

```python
predictions.append(yhat)
```

---
## Step 14 — add actual observation to history for the next loop

```python
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
## Step 26 — create a set of sarima configs to try

```python
def sarima_configs(seasonal=[0]):
	models = list()
```

---
## Step 27 — define config lists

```python
p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
```

---
## Step 28 — create config instances

```python
for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
```

---
## Step 29 — load dataset

```python
series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
```

---
## Step 30 — data split

```python
n_test = 12
```

---
## Step 31 — model configs

```python
cfg_list = sarima_configs(seasonal=[0,6,12])
```

---
## Step 32 — grid search

```python
scores = grid_search(data, cfg_list, n_test)
	print('done')
```

---
## Step 33 — list top 3 configs

```python
for cfg, error in scores[:3]:
		print(cfg, error)
```

---
## Learning Notes / 学习笔记

- **概念**: grid search sarima hyperparameters for monthly car sales dataset 是机器学习中的常用技术。  
  *grid search sarima hyperparameters for monthly car sales dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.fit` | 训练模型 | Train the model |
| `pandas` | 数据分析库 | Data analysis library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Grid Search Monthly Car Sales / 05 Grid Search Monthly Car Sales
# Complete Code / 完整代码
# ===============================

# grid search sarima hyperparameters for monthly car sales dataset
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
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

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models

if __name__ == '__main__':
	# load dataset
	series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	# data split
	n_test = 12
	# model configs
	cfg_list = sarima_configs(seasonal=[0,6,12])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
```

---
