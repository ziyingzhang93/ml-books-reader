# Python 深度学习 / Deep Learning with Python
## Chapter 31

---

### Plot

# 01 — Plot / 01 Plot

**Chapter 31 — File 1 of 3 / 第31章 — 第1个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **01 Plot**。

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
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 绘制折线图 / Draw line plot
plt.plot(dataset)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Plot 是机器学习中的常用技术。  
  *Plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 01 Plot
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataset = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 绘制折线图 / Draw line plot
plt.plot(dataset)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 3

---

### Mlp Simple

# 10 — Mlp Simple / 10 Mlp Simple

**Chapter 31 — File 2 of 3 / 第31章 — 第2个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)**.

本脚本演示 **Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — convert an array of values into a dataset matrix

```python
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        # 添加元素到列表末尾 / Append element to list end
        dataX.append(a)
        # 添加元素到列表末尾 / Append element to list end
        dataY.append(dataset[i + look_back, 0])
    # 创建NumPy数组 / Create NumPy array
    return np.array(dataX), np.array(dataY)
```

---
## Step 3 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 4 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 5 — reshape into X=t and Y=t+1

```python
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 6 — create and fit Multilayer Perceptron model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, input_shape=(look_back,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)
```

---
## Step 7 — Estimate model performance

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
trainScore = model.evaluate(trainX, trainY, verbose=0)
# 打印输出 / Print output
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# 评估模型在测试集上的表现 / Evaluate model on test set
testScore = model.evaluate(testX, testY, verbose=0)
# 打印输出 / Print output
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
```

---
## Step 8 — generate predictions for training

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
```

---
## Step 9 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 10 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 11 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(dataset, 'b')
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot, 'g')
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot, 'r')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Multilayer Perceptron to Predict International Airline Passengers (t+1, given t) 是机器学习中的常用技术。  
  *Multilayer Perceptron to Predict International Airline Passengers (t+1, given t) is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Simple / 10 Mlp Simple
# Complete Code / 完整代码
# ===============================

# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        # 添加元素到列表末尾 / Append element to list end
        dataX.append(a)
        # 添加元素到列表末尾 / Append element to list end
        dataY.append(dataset[i + look_back, 0])
    # 创建NumPy数组 / Create NumPy array
    return np.array(dataX), np.array(dataY)

# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, input_shape=(look_back,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)
# Estimate model performance
# 评估模型在测试集上的表现 / Evaluate model on test set
trainScore = model.evaluate(trainX, trainY, verbose=0)
# 打印输出 / Print output
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# 评估模型在测试集上的表现 / Evaluate model on test set
testScore = model.evaluate(testX, testY, verbose=0)
# 打印输出 / Print output
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
# 绘制折线图 / Draw line plot
plt.plot(dataset, 'b')
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot, 'g')
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot, 'r')
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 3

---

### Mlp Window

# 11 — Mlp Window / 11 Mlp Window

**Chapter 31 — File 3 of 3 / 第31章 — 第3个文件（共3个）**

---

## Summary / 总结

This script demonstrates **Multilayer Perceptron to Predict International Airline Passengers (t+1, given**.

本脚本演示 **Multilayer Perceptron to Predict International Airline Passengers (t+1, given**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
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
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Multilayer Perceptron to Predict International Airline Passengers (t+1, given
t, t-1, t-2)

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
```

---
## Step 2 — convert an array of values into a dataset matrix

```python
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        # 添加元素到列表末尾 / Append element to list end
        dataX.append(a)
        # 添加元素到列表末尾 / Append element to list end
        dataY.append(dataset[i + look_back, 0])
    # 创建NumPy数组 / Create NumPy array
    return np.array(dataX), np.array(dataY)
```

---
## Step 3 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 4 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 5 — reshape dataset

```python
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 6 — create and fit Multilayer Perceptron model

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(look_back,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
```

---
## Step 7 — Estimate model performance

```python
# 评估模型在测试集上的表现 / Evaluate model on test set
trainScore = model.evaluate(trainX, trainY, verbose=0)
# 打印输出 / Print output
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# 评估模型在测试集上的表现 / Evaluate model on test set
testScore = model.evaluate(testX, testY, verbose=0)
# 打印输出 / Print output
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
```

---
## Step 8 — generate predictions for training

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
```

---
## Step 9 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 10 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 11 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(dataset, 'b')
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot, 'g')
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot, 'r')
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Multilayer Perceptron to Predict International Airline Passengers (t+1, given 是机器学习中的常用技术。  
  *Multilayer Perceptron to Predict International Airline Passengers (t+1, given is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.evaluate` | 评估模型 | Evaluate the model |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `optimizer` | 优化器，更新模型参数 | Optimizer: updates model parameters |
| `pandas` | 数据分析库 | Data analysis library |
| `plt.plot` | 绘制折线图 | Draw line plot |
| `plt.show` | 显示图表 | Display plot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mlp Window / 11 Mlp Window
# Complete Code / 完整代码
# ===============================

# Multilayer Perceptron to Predict International Airline Passengers (t+1, given
# t, t-1, t-2)
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    # 获取长度 / Get length
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        # 添加元素到列表末尾 / Append element to list end
        dataX.append(a)
        # 添加元素到列表末尾 / Append element to list end
        dataY.append(dataset[i + look_back, 0])
    # 创建NumPy数组 / Create NumPy array
    return np.array(dataX), np.array(dataY)

# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape dataset
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(Dense(12, input_shape=(look_back,), activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(8, activation='relu'))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
# Estimate model performance
# 评估模型在测试集上的表现 / Evaluate model on test set
trainScore = model.evaluate(trainX, trainY, verbose=0)
# 打印输出 / Print output
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
# 评估模型在测试集上的表现 / Evaluate model on test set
testScore = model.evaluate(testX, testY, verbose=0)
# 打印输出 / Print output
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
# 绘制折线图 / Draw line plot
plt.plot(dataset, 'b')
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot, 'g')
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot, 'r')
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结



---
