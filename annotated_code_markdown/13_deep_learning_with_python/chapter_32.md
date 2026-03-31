# Python 深度学习 / Deep Learning with Python
## Chapter 32

---

### Lstm

# 12 — Lstm / LSTM 网络

**Chapter 32 — File 1 of 5 / 第32章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM for international airline passengers problem with regression framing**.

本脚本演示 **LSTM for international airline passengers problem with regression framing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — LSTM for international airline passengers problem with regression framing

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
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
## Step 3 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 4 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 5 — normalize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
```

---
## Step 6 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 7 — reshape into X=t and Y=t+1

```python
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 8 — reshape input to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```

---
## Step 9 — create and fit the LSTM network

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(1, look_back)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```

---
## Step 10 — make predictions

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
```

---
## Step 11 — invert predictions

```python
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```

---
## Step 12 — calculate root mean squared error

```python
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
```

---
## Step 13 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 14 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 15 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM for international airline passengers problem with regression framing 是机器学习中的常用技术。  
  *LSTM for international airline passengers problem with regression framing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
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
# Lstm / LSTM 网络
# Complete Code / 完整代码
# ===============================

# LSTM for international airline passengers problem with regression framing
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入数学函数库 / Import math functions library
import math
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# normalize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
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
# reshape input to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(1, look_back)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
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
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Window

# 13 — Window / 13 Window

**Chapter 32 — File 2 of 5 / 第32章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM for international airline passengers problem with window regression framing**.

本脚本演示 **LSTM for international airline passengers problem with window regression framing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — LSTM for international airline passengers problem with window regression framing

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
## Step 3 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 4 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 5 — normalize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
```

---
## Step 6 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 7 — reshape into X=t and Y=t+1

```python
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 8 — reshape input to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
```

---
## Step 9 — create and fit the LSTM network

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(1, look_back)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```

---
## Step 10 — make predictions

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
```

---
## Step 11 — invert predictions

```python
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```

---
## Step 12 — calculate root mean squared error

```python
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
```

---
## Step 13 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 14 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 15 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM for international airline passengers problem with window regression framing 是机器学习中的常用技术。  
  *LSTM for international airline passengers problem with window regression framing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
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
# Window / 13 Window
# Complete Code / 完整代码
# ===============================

# LSTM for international airline passengers problem with window regression framing
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# normalize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(1, look_back)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
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
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Timestep

# 15 — Timestep / 15 Timestep

**Chapter 32 — File 3 of 5 / 第32章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM for international airline passengers problem with time step regression framing**.

本脚本演示 **LSTM for international airline passengers problem with time step regression framing**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — LSTM for international airline passengers problem with time step regression framing

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
## Step 3 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 4 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 5 — normalize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
```

---
## Step 6 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 7 — reshape into X=t and Y=t+1

```python
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 8 — reshape input to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
```

---
## Step 9 — create and fit the LSTM network

```python
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(look_back, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
```

---
## Step 10 — make predictions

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
```

---
## Step 11 — invert predictions

```python
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```

---
## Step 12 — calculate root mean squared error

```python
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
```

---
## Step 13 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 14 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 15 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM for international airline passengers problem with time step regression framing 是机器学习中的常用技术。  
  *LSTM for international airline passengers problem with time step regression framing is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
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
# Timestep / 15 Timestep
# Complete Code / 完整代码
# ===============================

# LSTM for international airline passengers problem with time step regression framing
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.models import Sequential
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import Dense
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
from tensorflow.keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# normalize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, input_shape=(look_back, 1)))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型 / Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX)
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
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
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Manual

# 19 — Manual / 19 Manual

**Chapter 32 — File 4 of 5 / 第32章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **LSTM for international airline passengers problem with memory**.

本脚本演示 **LSTM for international airline passengers problem with memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — LSTM for international airline passengers problem with memory

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
## Step 3 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 4 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 5 — normalize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
```

---
## Step 6 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 7 — reshape into X=t and Y=t+1

```python
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 8 — reshape input to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
```

---
## Step 9 — create and fit the LSTM network

```python
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 生成整数序列 / Generate integer sequence
for i in range(100):
    # 训练模型 / Train the model
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
```

---
## Step 10 — make predictions

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX, batch_size=batch_size)
```

---
## Step 11 — invert predictions

```python
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```

---
## Step 12 — calculate root mean squared error

```python
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
```

---
## Step 13 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 14 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 15 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: LSTM for international airline passengers problem with memory 是机器学习中的常用技术。  
  *LSTM for international airline passengers problem with memory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
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
# Manual / 19 Manual
# Complete Code / 完整代码
# ===============================

# LSTM for international airline passengers problem with memory
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# normalize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 生成整数序列 / Generate integer sequence
for i in range(100):
    # 训练模型 / Train the model
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
# make predictions
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
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
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Stacked

# 21 — Stacked / 堆叠方法

**Chapter 32 — File 5 of 5 / 第32章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Stacked LSTM for international airline passengers problem with memory**.

本脚本演示 **Stacked LSTM for international airline passengers problem with memory**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
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
## Step 1 — Stacked LSTM for international airline passengers problem with memory

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
## Step 3 — fix random seed for reproducibility

```python
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
```

---
## Step 4 — load the dataset

```python
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
```

---
## Step 5 — normalize the dataset

```python
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
```

---
## Step 6 — split into train and test sets

```python
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
```

---
## Step 7 — reshape into X=t and Y=t+1

```python
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
```

---
## Step 8 — reshape input to be [samples, time steps, features]

```python
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
```

---
## Step 9 — create and fit the LSTM network

```python
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True,
               return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 生成整数序列 / Generate integer sequence
for i in range(100):
    # 训练模型 / Train the model
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
```

---
## Step 10 — make predictions

```python
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX, batch_size=batch_size)
```

---
## Step 11 — invert predictions

```python
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
```

---
## Step 12 — calculate root mean squared error

```python
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
```

---
## Step 13 — shift train predictions for plotting

```python
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# 获取长度 / Get length
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
```

---
## Step 14 — shift test predictions for plotting

```python
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
# 获取长度 / Get length
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
```

---
## Step 15 — plot baseline and predictions

```python
# 绘制折线图 / Draw line plot
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Stacked LSTM for international airline passengers problem with memory 是机器学习中的常用技术。  
  *Stacked LSTM for international airline passengers problem with memory is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Adam` | 自适应学习率优化器 | Adaptive learning rate optimizer |
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `MinMaxScaler` | 归一化到[0,1]范围 | Normalize to [0,1] range |
| `Sequential` | 顺序模型，层层堆叠 | Sequential model: stack layers one by one |
| `batch_size` | 每次送入模型的样本数 | Number of samples per training step |
| `epoch` | 一个epoch=遍历全部训练数据一次 | One epoch = one pass through all training data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `model.compile` | 编译模型：设置损失函数和优化器 | Compile: set loss and optimizer |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `np.reshape` | 改变数组形状 | Reshape array dimensions |
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
# Stacked / 堆叠方法
# Complete Code / 完整代码
# ===============================

# Stacked LSTM for international airline passengers problem with memory
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
# 导入Pandas数据分析库 / Import Pandas data analysis library
from pandas import read_csv
# 导入TensorFlow深度学习框架 / Import TensorFlow framework
import tensorflow as tf
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.models import Sequential
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import Dense
# 导入Keras高级神经网络API / Import Keras high-level neural network API
from keras.layers import LSTM
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import MinMaxScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import mean_squared_error
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
# fix random seed for reproducibility
# 设置随机种子（保证可重复） / Set random seed (ensure reproducibility)
tf.random.set_seed(7)
# load the dataset
# 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# 转换为NumPy数组 / Convert to NumPy array
dataset = dataframe.values
# 转换数据类型 / Convert data type
dataset = dataset.astype('float32')
# normalize the dataset
# 归一化到[0,1]范围 / Normalize to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
dataset = scaler.fit_transform(dataset)
# split into train and test sets
# 获取长度 / Get length
train_size = int(len(dataset) * 0.67)
# 获取长度 / Get length
test_size = len(dataset) - train_size
# 获取长度 / Get length
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
batch_size = 1
# 创建顺序模型：逐层堆叠 / Create Sequential model: stack layers
model = Sequential()
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True,
               return_sequences=True))
# 向模型添加一层 / Add a layer to the model
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# 向模型添加一层 / Add a layer to the model
model.add(Dense(1))
# 编译模型：设置优化器和损失函数 / Compile: set optimizer and loss function
model.compile(loss='mean_squared_error', optimizer='adam')
# 生成整数序列 / Generate integer sequence
for i in range(100):
    # 训练模型 / Train the model
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
# make predictions
# 用模型做预测 / Make predictions with model
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
# 用模型做预测 / Make predictions with model
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# 计算均方误差 / Calculate Mean Squared Error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# 打印输出 / Print output
print('Train Score: %.2f RMSE' % (trainScore))
# 计算均方误差 / Calculate Mean Squared Error
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# 打印输出 / Print output
print('Test Score: %.2f RMSE' % (testScore))
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
plt.plot(scaler.inverse_transform(dataset))
# 绘制折线图 / Draw line plot
plt.plot(trainPredictPlot)
# 绘制折线图 / Draw line plot
plt.plot(testPredictPlot)
# 显示图表 / Display the plot
plt.show()
```

---

### Chapter Summary / 章节总结



---
