# DL时间序列
## Chapter 22

---

### Load File

# 01 — Load File / 01 Load File

**Chapter 22 — File 1 of 8 / 第22章 — 第1个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load one file from the har dataset**.

本脚本演示 **load one file from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load one file from the har dataset

```python
from pandas import read_csv
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

data = load_file('HARDataset/train/Inertial Signals/total_acc_y_train.txt')
print(data.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load one file from the har dataset 是机器学习中的常用技术。  
  *load one file from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load File / 01 Load File
# Complete Code / 完整代码
# ===============================

# load one file from the har dataset
from pandas import read_csv

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

data = load_file('HARDataset/train/Inertial Signals/total_acc_y_train.txt')
print(data.shape)
```

---

➡️ **Next / 下一步**: File 2 of 8

---

### Load Group Of Files

# 02 — Load Group Of Files / 02 Load Group Of Files

**Chapter 22 — File 2 of 8 / 第22章 — 第2个文件（共8个）**

---

## Summary / 总结

This script demonstrates **load group of files from the har dataset**.

本脚本演示 **load group of files from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — load group of files from the har dataset

```python
from numpy import dstack
from pandas import read_csv
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files, such as x, y, z data for a given variable

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load the total acc data

```python
filenames = ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
total_acc = load_group(filenames, prefix='HARDataset/train/Inertial Signals/')
print(total_acc.shape)
```

---
## Learning Notes / 学习笔记

- **概念**: load group of files from the har dataset 是机器学习中的常用技术。  
  *load group of files from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Load Group Of Files / 02 Load Group Of Files
# Complete Code / 完整代码
# ===============================

# load group of files from the har dataset
from numpy import dstack
from pandas import read_csv

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load the total acc data
filenames = ['total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
total_acc = load_group(filenames, prefix='HARDataset/train/Inertial Signals/')
print(total_acc.shape)
```

---

➡️ **Next / 下一步**: File 3 of 8

---

### Class Breakdown

# 04 — Class Breakdown / 04 Class Breakdown

**Chapter 22 — File 4 of 8 / 第22章 — 第4个文件（共8个）**

---

## Summary / 总结

This script demonstrates **summarize class balance from the har dataset**.

本脚本演示 **summarize class balance from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture


---
## Step 1 — summarize class balance from the har dataset

```python
from numpy import vstack
from pandas import read_csv
from pandas import DataFrame
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — summarize the balance of classes in an output variable column

```python
def class_breakdown(data):
```

---
## Step 4 — convert the numpy array into a dataframe

```python
df = DataFrame(data)
```

---
## Step 5 — group data by the class value and calculate the number of rows

```python
counts = df.groupby(0).size()
```

---
## Step 6 — retrieve raw rows

```python
counts = counts.values
```

---
## Step 7 — summarize

```python
for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))
```

---
## Step 8 — load train file

```python
trainy = load_file('HARDataset/train/y_train.txt')
```

---
## Step 9 — summarize class breakdown

```python
print('Train Dataset')
class_breakdown(trainy)
```

---
## Step 10 — load test file

```python
testy = load_file('HARDataset/test/y_test.txt')
```

---
## Step 11 — summarize class breakdown

```python
print('Test Dataset')
class_breakdown(testy)
```

---
## Step 12 — summarize combined class breakdown

```python
print('Both')
combined = vstack((trainy, testy))
class_breakdown(combined)
```

---
## Learning Notes / 学习笔记

- **概念**: summarize class balance from the har dataset 是机器学习中的常用技术。  
  *summarize class balance from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `groupby` | 分组聚合 | Group and aggregate |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Class Breakdown / 04 Class Breakdown
# Complete Code / 完整代码
# ===============================

# summarize class balance from the har dataset
from numpy import vstack
from pandas import read_csv
from pandas import DataFrame

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# summarize the balance of classes in an output variable column
def class_breakdown(data):
	# convert the numpy array into a dataframe
	df = DataFrame(data)
	# group data by the class value and calculate the number of rows
	counts = df.groupby(0).size()
	# retrieve raw rows
	counts = counts.values
	# summarize
	for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent))

# load train file
trainy = load_file('HARDataset/train/y_train.txt')
# summarize class breakdown
print('Train Dataset')
class_breakdown(trainy)

# load test file
testy = load_file('HARDataset/test/y_test.txt')
# summarize class breakdown
print('Test Dataset')
class_breakdown(testy)

# summarize combined class breakdown
print('Both')
combined = vstack((trainy, testy))
class_breakdown(combined)
```

---

➡️ **Next / 下一步**: File 5 of 8

---

### Plot Data For Subject

# 05 — Plot Data For Subject / 05 Plot Data For Subject

**Chapter 22 — File 5 of 8 / 第22章 — 第5个文件（共8个）**

---

## Summary / 总结

This script demonstrates **plot all vars for one subject in the har dataset**.

本脚本演示 **plot all vars for one subject in the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — plot all vars for one subject in the har dataset

```python
from numpy import dstack
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files, such as x, y, z data for a given variable

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — get all data for one subject

```python
def data_for_subject(X, y, sub_map, sub_id):
```

---
## Step 13 — get row indexes for the subject id

```python
ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
```

---
## Step 14 — return the selected samples

```python
return X[ix, :, :], y[ix]
```

---
## Step 15 — convert a series of windows to a 1D list

```python
def to_series(windows):
	series = list()
	for window in windows:
```

---
## Step 16 — remove the overlap from the window

```python
half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series
```

---
## Step 17 — plot the data for one subject

```python
def plot_subject(X, y):
	pyplot.figure()
```

---
## Step 18 — determine the total number of plots

```python
n, off = X.shape[2] + 1, 0
```

---
## Step 19 — plot total acc

```python
for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('total acc '+str(i), y=0, loc='left', size=7)
```

---
## Step 20 — turn off ticks to remove clutter

```python
pyplot.yticks([])
		pyplot.xticks([])
		off += 1
```

---
## Step 21 — plot body acc

```python
for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('body acc '+str(i), y=0, loc='left', size=7)
```

---
## Step 22 — turn off ticks to remove clutter

```python
pyplot.yticks([])
		pyplot.xticks([])
		off += 1
```

---
## Step 23 — plot body gyro

```python
for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('body gyro '+str(i), y=0, loc='left', size=7)
```

---
## Step 24 — turn off ticks to remove clutter

```python
pyplot.yticks([])
		pyplot.xticks([])
		off += 1
```

---
## Step 25 — plot activities

```python
pyplot.subplot(n, 1, n)
	pyplot.plot(y)
	pyplot.title('activity', y=0, loc='left', size=7)
```

---
## Step 26 — turn off ticks to remove clutter

```python
pyplot.yticks([])
	pyplot.xticks([])
	pyplot.show()
```

---
## Step 27 — load data

```python
trainX, trainy = load_dataset('train', 'HARDataset/')
```

---
## Step 28 — load mapping of rows to subjects

```python
sub_map = load_file('HARDataset/train/subject_train.txt')
train_subjects = unique(sub_map)
print(train_subjects)
```

---
## Step 29 — get the data for one subject

```python
sub_id = train_subjects[0]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
print(subX.shape, suby.shape)
```

---
## Step 30 — plot data for subject

```python
plot_subject(subX, suby)
```

---
## Learning Notes / 学习笔记

- **概念**: plot all vars for one subject in the har dataset 是机器学习中的常用技术。  
  *plot all vars for one subject in the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Data For Subject / 05 Plot Data For Subject
# Complete Code / 完整代码
# ===============================

# plot all vars for one subject in the har dataset
from numpy import dstack
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# get all data for one subject
def data_for_subject(X, y, sub_map, sub_id):
	# get row indexes for the subject id
	ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
	# return the selected samples
	return X[ix, :, :], y[ix]

# convert a series of windows to a 1D list
def to_series(windows):
	series = list()
	for window in windows:
		# remove the overlap from the window
		half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series

# plot the data for one subject
def plot_subject(X, y):
	pyplot.figure()
	# determine the total number of plots
	n, off = X.shape[2] + 1, 0
	# plot total acc
	for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('total acc '+str(i), y=0, loc='left', size=7)
		# turn off ticks to remove clutter
		pyplot.yticks([])
		pyplot.xticks([])
		off += 1
	# plot body acc
	for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('body acc '+str(i), y=0, loc='left', size=7)
		# turn off ticks to remove clutter
		pyplot.yticks([])
		pyplot.xticks([])
		off += 1
	# plot body gyro
	for i in range(3):
		pyplot.subplot(n, 1, off+1)
		pyplot.plot(to_series(X[:, :, off]))
		pyplot.title('body gyro '+str(i), y=0, loc='left', size=7)
		# turn off ticks to remove clutter
		pyplot.yticks([])
		pyplot.xticks([])
		off += 1
	# plot activities
	pyplot.subplot(n, 1, n)
	pyplot.plot(y)
	pyplot.title('activity', y=0, loc='left', size=7)
	# turn off ticks to remove clutter
	pyplot.yticks([])
	pyplot.xticks([])
	pyplot.show()

# load data
trainX, trainy = load_dataset('train', 'HARDataset/')
# load mapping of rows to subjects
sub_map = load_file('HARDataset/train/subject_train.txt')
train_subjects = unique(sub_map)
print(train_subjects)
# get the data for one subject
sub_id = train_subjects[0]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
print(subX.shape, suby.shape)
# plot data for subject
plot_subject(subX, suby)
```

---

➡️ **Next / 下一步**: File 6 of 8

---

### Plot Histograms For Subjects

# 06 — Plot Histograms For Subjects / 06 Plot Histograms For Subjects

**Chapter 22 — File 6 of 8 / 第22章 — 第6个文件（共8个）**

---

## Summary / 总结

This script demonstrates **plot histograms for multiple subjects from the har dataset**.

本脚本演示 **plot histograms for multiple subjects from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — plot histograms for multiple subjects from the har dataset

```python
from numpy import unique
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files, such as x, y, z data for a given variable

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — get all data for one subject

```python
def data_for_subject(X, y, sub_map, sub_id):
```

---
## Step 13 — get row indexes for the subject id

```python
ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
```

---
## Step 14 — return the selected samples

```python
return X[ix, :, :], y[ix]
```

---
## Step 15 — convert a series of windows to a 1D list

```python
def to_series(windows):
	series = list()
	for window in windows:
```

---
## Step 16 — remove the overlap from the window

```python
half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series
```

---
## Step 17 — plot histograms for multiple subjects

```python
def plot_subject_histograms(X, y, sub_map, offset, n=10):
	pyplot.figure()
```

---
## Step 18 — get unique subjects

```python
subject_ids = unique(sub_map[:,0])
```

---
## Step 19 — enumerate subjects

```python
for k in range(n):
		sub_id = subject_ids[k]
```

---
## Step 20 — get data for one subject

```python
subX, _ = data_for_subject(X, y, sub_map, sub_id)
```

---
## Step 21 — total acc

```python
for i in range(3):
			ax = pyplot.subplot(n, 1, k+1)
			ax.set_xlim(-1,1)
			ax.hist(to_series(subX[:,:,offset+i]), bins=100)
			pyplot.yticks([])
			pyplot.xticks([-1,0,1])
	pyplot.show()
```

---
## Step 22 — load training dataset

```python
X, y = load_dataset('train', 'HARDataset/')
```

---
## Step 23 — load mapping of rows to subjects

```python
sub_map = load_file('HARDataset/train/subject_train.txt')
```

---
## Step 24 — plot total acceleration histograms for subjects

```python
plot_subject_histograms(X, y, sub_map, 0)
```

---
## Step 25 — plot body acceleration histograms for subjects

```python
plot_subject_histograms(X, y, sub_map, 3)
```

---
## Step 26 — plot gyroscopic histograms for subjects

```python
plot_subject_histograms(X, y, sub_map, 6)
```

---
## Learning Notes / 学习笔记

- **概念**: plot histograms for multiple subjects from the har dataset 是机器学习中的常用技术。  
  *plot histograms for multiple subjects from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Histograms For Subjects / 06 Plot Histograms For Subjects
# Complete Code / 完整代码
# ===============================

# plot histograms for multiple subjects from the har dataset
from numpy import unique
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# get all data for one subject
def data_for_subject(X, y, sub_map, sub_id):
	# get row indexes for the subject id
	ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
	# return the selected samples
	return X[ix, :, :], y[ix]

# convert a series of windows to a 1D list
def to_series(windows):
	series = list()
	for window in windows:
		# remove the overlap from the window
		half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series

# plot histograms for multiple subjects
def plot_subject_histograms(X, y, sub_map, offset, n=10):
	pyplot.figure()
	# get unique subjects
	subject_ids = unique(sub_map[:,0])
	# enumerate subjects
	for k in range(n):
		sub_id = subject_ids[k]
		# get data for one subject
		subX, _ = data_for_subject(X, y, sub_map, sub_id)
		# total acc
		for i in range(3):
			ax = pyplot.subplot(n, 1, k+1)
			ax.set_xlim(-1,1)
			ax.hist(to_series(subX[:,:,offset+i]), bins=100)
			pyplot.yticks([])
			pyplot.xticks([-1,0,1])
	pyplot.show()

# load training dataset
X, y = load_dataset('train', 'HARDataset/')
# load mapping of rows to subjects
sub_map = load_file('HARDataset/train/subject_train.txt')
# plot total acceleration histograms for subjects
plot_subject_histograms(X, y, sub_map, 0)
# plot body acceleration histograms for subjects
plot_subject_histograms(X, y, sub_map, 3)
# plot gyroscopic histograms for subjects
plot_subject_histograms(X, y, sub_map, 6)
```

---

➡️ **Next / 下一步**: File 7 of 8

---

### Plot Histograms By Activity

# 07 — Plot Histograms By Activity / 07 Plot Histograms By Activity

**Chapter 22 — File 7 of 8 / 第22章 — 第7个文件（共8个）**

---

## Summary / 总结

This script demonstrates **plot histograms per activity for a subject from the har dataset**.

本脚本演示 **plot histograms per activity for a subject from the har dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 定义模型结构 / Define model architecture
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
## Step 1 — plot histograms per activity for a subject from the har dataset

```python
from numpy import dstack
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot
```

---
## Step 2 — load a single file as a numpy array

```python
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
```

---
## Step 3 — load a list of files, such as x, y, z data for a given variable

```python
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
```

---
## Step 4 — stack group so that features are the 3rd dimension

```python
loaded = dstack(loaded)
	return loaded
```

---
## Step 5 — load a dataset group, such as train or test

```python
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
```

---
## Step 6 — load all 9 files as a single array

```python
filenames = list()
```

---
## Step 7 — total acceleration

```python
filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
```

---
## Step 8 — body acceleration

```python
filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
```

---
## Step 9 — body gyroscope

```python
filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
```

---
## Step 10 — load input data

```python
X = load_group(filenames, filepath)
```

---
## Step 11 — load class output

```python
y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
```

---
## Step 12 — get all data for one subject

```python
def data_for_subject(X, y, sub_map, sub_id):
```

---
## Step 13 — get row indexes for the subject id

```python
ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
```

---
## Step 14 — return the selected samples

```python
return X[ix, :, :], y[ix]
```

---
## Step 15 — convert a series of windows to a 1D list

```python
def to_series(windows):
	series = list()
	for window in windows:
```

---
## Step 16 — remove the overlap from the window

```python
half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series
```

---
## Step 17 — group data by activity

```python
def data_by_activity(X, y, activities):
```

---
## Step 18 — group windows by activity

```python
return {a:X[y[:,0]==a, :, :] for a in activities}
```

---
## Step 19 — plot histograms for each activity for a subject

```python
def plot_activity_histograms(X, y, offset):
```

---
## Step 20 — get a list of unique activities for the subject

```python
activity_ids = unique(y[:,0])
```

---
## Step 21 — group windows by activity

```python
grouped = data_by_activity(X, y, activity_ids)
```

---
## Step 22 — plot per activity, histograms for each axis

```python
pyplot.figure()
	for k in range(len(activity_ids)):
		act_id = activity_ids[k]
```

---
## Step 23 — total acceleration

```python
for i in range(3):
			ax = pyplot.subplot(len(activity_ids), 1, k+1)
			ax.set_xlim(-1,1)
```

---
## Step 24 — create histogra,

```python
pyplot.hist(to_series(grouped[act_id][:,:,offset+i]), bins=100)
```

---
## Step 25 — create title

```python
pyplot.title('activity '+str(act_id), y=0, loc='left', size=10)
```

---
## Step 26 — simplify axis

```python
pyplot.yticks([])
			pyplot.xticks([-1,0,1])
	pyplot.show()
```

---
## Step 27 — load data

```python
trainX, trainy = load_dataset('train', 'HARDataset/')
```

---
## Step 28 — load mapping of rows to subjects

```python
sub_map = load_file('HARDataset/train/subject_train.txt')
train_subjects = unique(sub_map)
```

---
## Step 29 — get the data for one subject

```python
sub_id = train_subjects[0]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
```

---
## Step 30 — plot total acceleration histograms per activity for a subject

```python
plot_activity_histograms(subX, suby, 0)
```

---
## Step 31 — plot body acceleration histograms per activity for a subject

```python
plot_activity_histograms(subX, suby, 3)
```

---
## Step 32 — plot gyroscopic histograms per activity for a subject

```python
plot_activity_histograms(subX, suby, 6)
```

---
## Learning Notes / 学习笔记

- **概念**: plot histograms per activity for a subject from the har dataset 是机器学习中的常用技术。  
  *plot histograms per activity for a subject from the har dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `DataFrame` | 二维表格数据结构 | 2D tabular data structure |
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Histograms By Activity / 07 Plot Histograms By Activity
# Complete Code / 完整代码
# ===============================

# plot histograms per activity for a subject from the har dataset
from numpy import dstack
from numpy import unique
from pandas import read_csv
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# get all data for one subject
def data_for_subject(X, y, sub_map, sub_id):
	# get row indexes for the subject id
	ix = [i for i in range(len(sub_map)) if sub_map[i]==sub_id]
	# return the selected samples
	return X[ix, :, :], y[ix]

# convert a series of windows to a 1D list
def to_series(windows):
	series = list()
	for window in windows:
		# remove the overlap from the window
		half = int(len(window) / 2) - 1
		for value in window[-half:]:
			series.append(value)
	return series

# group data by activity
def data_by_activity(X, y, activities):
	# group windows by activity
	return {a:X[y[:,0]==a, :, :] for a in activities}

# plot histograms for each activity for a subject
def plot_activity_histograms(X, y, offset):
	# get a list of unique activities for the subject
	activity_ids = unique(y[:,0])
	# group windows by activity
	grouped = data_by_activity(X, y, activity_ids)
	# plot per activity, histograms for each axis
	pyplot.figure()
	for k in range(len(activity_ids)):
		act_id = activity_ids[k]
		# total acceleration
		for i in range(3):
			ax = pyplot.subplot(len(activity_ids), 1, k+1)
			ax.set_xlim(-1,1)
			# create histogra,
			pyplot.hist(to_series(grouped[act_id][:,:,offset+i]), bins=100)
			# create title
			pyplot.title('activity '+str(act_id), y=0, loc='left', size=10)
			# simplify axis
			pyplot.yticks([])
			pyplot.xticks([-1,0,1])
	pyplot.show()

# load data
trainX, trainy = load_dataset('train', 'HARDataset/')
# load mapping of rows to subjects
sub_map = load_file('HARDataset/train/subject_train.txt')
train_subjects = unique(sub_map)
# get the data for one subject
sub_id = train_subjects[0]
subX, suby = data_for_subject(trainX, trainy, sub_map, sub_id)
# plot total acceleration histograms per activity for a subject
plot_activity_histograms(subX, suby, 0)
# plot body acceleration histograms per activity for a subject
plot_activity_histograms(subX, suby, 3)
# plot gyroscopic histograms per activity for a subject
plot_activity_histograms(subX, suby, 6)
```

---

➡️ **Next / 下一步**: File 8 of 8

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **8 code files** demonstrating chapter 22.

本章包含 **8 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_load_file.ipynb` — Load File
  2. `02_load_group_of_files.ipynb` — Load Group Of Files
  3. `03_load_all_files.ipynb` — Load All Files
  4. `04_class_breakdown.ipynb` — Class Breakdown
  5. `05_plot_data_for_subject.ipynb` — Plot Data For Subject
  6. `06_plot_histograms_for_subjects.ipynb` — Plot Histograms For Subjects
  7. `07_plot_histograms_by_activity.ipynb` — Plot Histograms By Activity
  8. `08_plot_activity_durations.ipynb` — Plot Activity Durations

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
