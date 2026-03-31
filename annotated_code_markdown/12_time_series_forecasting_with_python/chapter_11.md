# 时间序列预测 / Time Series Forecasting with Python
## Chapter 11

---

### Chapter Summary / 章节总结



---

### Random Series



---

### Random Walk



---

### Random Walk Autocorrelation



---

### Random Walk Differenced

# 01 — Random Walk Differenced / Random Walk Differenced

**Chapter 11 — File 4 of 8 / 第11章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **calculate and plot a differenced random walk**.

本脚本演示 **calculate and plot a differenced random walk**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — calculate and plot a differenced random walk

```python
from random import seed
from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — create random walk

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
```

---
## Step 3 — take difference

```python
diff = list()
# 获取长度 / Get length
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
```

---
## Step 4 — line plot

```python
pyplot.plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: calculate and plot a differenced random walk 是机器学习中的常用技术。  
  *calculate and plot a differenced random walk is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Walk Differenced / Random Walk Differenced
# Complete Code / 完整代码
# ===============================

# calculate and plot a differenced random walk
from random import seed
from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# create random walk
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
# take difference
diff = list()
# 获取长度 / Get length
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
# line plot
pyplot.plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Random Walk Differenced Autocorrelation

# 01 — Random Walk Differenced Autocorrelation / Random Walk Differenced Autocorrelation

**Chapter 11 — File 5 of 8 / 第11章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **plot the autocorrelation of a differenced random walk**.

本脚本演示 **plot the autocorrelation of a differenced random walk**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — plot the autocorrelation of a differenced random walk

```python
from random import seed
from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
```

---
## Step 2 — create random walk

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
```

---
## Step 3 — take difference

```python
diff = list()
# 获取长度 / Get length
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
```

---
## Step 4 — line plot

```python
autocorrelation_plot(diff)
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: plot the autocorrelation of a differenced random walk 是机器学习中的常用技术。  
  *plot the autocorrelation of a differenced random walk is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Walk Differenced Autocorrelation / Random Walk Differenced Autocorrelation
# Complete Code / 完整代码
# ===============================

# plot the autocorrelation of a differenced random walk
from random import seed
from random import random
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas.plotting import autocorrelation_plot
# create random walk
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
# take difference
diff = list()
# 获取长度 / Get length
for i in range(1, len(random_walk)):
	value = random_walk[i] - random_walk[i - 1]
 # 添加元素到列表末尾 / Append element to list end
	diff.append(value)
# line plot
autocorrelation_plot(diff)
pyplot.show()
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Random Walk Persistence



---

### Random Walk Random

# 01 — Random Walk Random / Random Walk Random

**Chapter 11 — File 7 of 8 / 第11章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **random predictions for a random walk**.

本脚本演示 **random predictions for a random walk**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — random predictions for a random walk

```python
from random import seed
from random import random
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
```

---
## Step 2 — generate the random walk

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
```

---
## Step 3 — prepare dataset

```python
# 获取长度 / Get length
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
```

---
## Step 4 — random prediction

```python
predictions = list()
history = train[-1]
# 获取长度 / Get length
for i in range(len(test)):
	yhat = history + (-1 if random() < 0.5 else 1)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	history = test[i]
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Random RMSE: %.3f' % rmse)
```

---
## Learning Notes / 学习笔记

- **概念**: random predictions for a random walk 是机器学习中的常用技术。  
  *random predictions for a random walk is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Walk Random / Random Walk Random
# Complete Code / 完整代码
# ===============================

# random predictions for a random walk
from random import seed
from random import random
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
from math import sqrt
# generate the random walk
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
# prepare dataset
# 获取长度 / Get length
train_size = int(len(random_walk) * 0.66)
train, test = random_walk[0:train_size], random_walk[train_size:]
# random prediction
predictions = list()
history = train[-1]
# 获取长度 / Get length
for i in range(len(test)):
	yhat = history + (-1 if random() < 0.5 else 1)
 # 添加元素到列表末尾 / Append element to list end
	predictions.append(yhat)
	history = test[i]
# 计算均方误差 / Calculate Mean Squared Error
rmse = sqrt(mean_squared_error(test, predictions))
# 打印输出 / Print output
print('Random RMSE: %.3f' % rmse)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Random Walk Stationarity

# 01 — Random Walk Stationarity / Random Walk Stationarity

**Chapter 11 — File 8 of 8 / 第11章 — 第8个文件（共8个）**

---

## Summary / 总结

This script demonstrates **calculate the stationarity of a random walk**.

本脚本演示 **calculate the stationarity of a random walk**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — calculate the stationarity of a random walk

```python
from random import seed
from random import random
from statsmodels.tsa.stattools import adfuller
```

---
## Step 2 — generate random walk

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
```

---
## Step 3 — statistical test

```python
result = adfuller(random_walk)
# 打印输出 / Print output
print('ADF Statistic: %f' % result[0])
# 打印输出 / Print output
print('p-value: %f' % result[1])
# 打印输出 / Print output
print('Critical Values:')
# 获取字典的键值对 / Get dict key-value pairs
for key, value in result[4].items():
 # 打印输出 / Print output
	print('\t%s: %.3f' % (key, value))
```

---
## Learning Notes / 学习笔记

- **概念**: calculate the stationarity of a random walk 是机器学习中的常用技术。  
  *calculate the stationarity of a random walk is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Random Walk Stationarity / Random Walk Stationarity
# Complete Code / 完整代码
# ===============================

# calculate the stationarity of a random walk
from random import seed
from random import random
from statsmodels.tsa.stattools import adfuller
# generate random walk
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
seed(1)
random_walk = list()
# 添加元素到列表末尾 / Append element to list end
random_walk.append(-1 if random() < 0.5 else 1)
# 生成整数序列 / Generate integer sequence
for i in range(1, 1000):
	movement = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + movement
 # 添加元素到列表末尾 / Append element to list end
	random_walk.append(value)
# statistical test
result = adfuller(random_walk)
# 打印输出 / Print output
print('ADF Statistic: %f' % result[0])
# 打印输出 / Print output
print('p-value: %f' % result[1])
# 打印输出 / Print output
print('Critical Values:')
# 获取字典的键值对 / Get dict key-value pairs
for key, value in result[4].items():
 # 打印输出 / Print output
	print('\t%s: %.3f' % (key, value))
```

---
