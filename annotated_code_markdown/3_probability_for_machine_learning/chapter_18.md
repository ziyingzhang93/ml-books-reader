# 概率论与机器学习
## Chapter 18

---

### Test Problem

# 18 — Naive Bayes Classification / 根据贝叶斯定理的分类

**Chapter 18 — File 1 of 4**

## Summary / 汇总

This notebook demonstrates generating a 2D classification dataset using make_blobs. In Naive Bayes classification, we first create a synthetic dataset with two classes to understand the feature distribution that will be modeled probabilistically.

此笔记本演示了使用make_blobs生成二维分类数据集。在根据贝叶斯定理的分类中，我们需要事先创建一个人为的两类数据集来理解将被概率建模的特征分布。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


## Step 1 — Generate Classification Dataset / 生成分类数据集

```python
# example of generating a small classification dataset
from sklearn.datasets import make_blobs

# generate 2d classification dataset
# Generate a synthetic 2D dataset with 100 samples and 2 classes (centers=2)
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# summarize the dataset shape and sample data
print(X.shape, y.shape)
print('First 5 feature samples:')
print(X[:5])
print('First 5 class labels:')
print(y[:5])
```

## Learning Notes / 学习笔记

- **Concept**: make_blobs() creates synthetic datasets with spherical blobs, useful for testing classification algorithms. The centers parameter determines the number of class clusters.
  
  **概念**: make_blobs()创建具有球面块的合成数据集，有效于测试分类算法。centers参数确定类癶集的数量。

- **ML Application**: Naive Bayes assumes feature independence within each class. This dataset will be used to fit independent Gaussian distributions for each feature per class.
  
  **机器学习应用**: 根据贝叶斯定理假设每个类内特征一事符。此数据集将用于操纺每个类每个特征的辄助高斯分布。

➡️ **Next**: `02_fit_distributions.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

## Complete Code / 完整代码一览

```python
# example of generating a small classification dataset
from sklearn.datasets import make_blobs

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# summarize
print(X.shape, y.shape)
print(X[:5])
print(y[:5])
```

---

### Fit Distributions

# 18 — Naive Bayes Classification / 根据贝叶斯定理的分类

**Chapter 18 — File 2 of 4**

## Summary / 汇总

This notebook demonstrates fitting Normal distributions to features for each class. We calculate class priors (the probability of each class) and estimate Gaussian parameters (mean and standard deviation) for each feature within each class.

此笔记本演示了为不同类分别突訊特征齐中高斯分布。我们计算类先验（每个类或然率）以及每个类内每个特征的高斯参数（均值不标准差）。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


## Step 1 — Define fit_distribution Helper Function / 定义幐合分布辅助函数

```python
from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean, std

# fit a probability distribution to a univariate data sample
def fit_distribution(data):
    # estimate parameters: calculate mean and standard deviation
    mu = mean(data)          # 求平均值 (mean)
    sigma = std(data)        # 求标准差 (std)
    print(mu, sigma)
    # fit distribution: create a Normal distribution with these parameters
    dist = norm(mu, sigma)   # 使用scipy.stats创建正态分布对象
    return dist
```

## Step 2 — Generate and Separate Dataset / 生成并分离数据集

```python
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# sort data into classes: separate features by class label
Xy0 = X[y == 0]          # 模类0的特征
Xy1 = X[y == 1]          # 模类1的特征
print(Xy0.shape, Xy1.shape)
```

## Step 3 — Calculate Class Priors / 计算类先验

```python
# calculate priors: P(y=0) and P(y=1)
priory0 = len(Xy0) / len(X)   # class 0 模类0的先验
priory1 = len(Xy1) / len(X)   # class 1 模类1的先验
print('P(y=0):', priory0, 'P(y=1):', priory1)
```

## Step 4 — Fit Distributions for Each Feature and Class / 为不同特征和类墅合置分布

```python
# create PDFs for y==0: fit Gaussian distributions for both features when y=0
X1y0 = fit_distribution(Xy0[:, 0])  # Feature 1 for class 0 特征1的模类0分布
X2y0 = fit_distribution(Xy0[:, 1])  # Feature 2 for class 0 特征2的模类0分布

# create PDFs for y==1: fit Gaussian distributions for both features when y=1
X1y1 = fit_distribution(Xy1[:, 0])  # Feature 1 for class 1 特征1的模类1分布
X2y1 = fit_distribution(Xy1[:, 1])  # Feature 2 for class 1 特征2的模类1分布
```

## Learning Notes / 学习笔记

- **Concept**: Naive Bayes assumes each feature follows a Normal distribution within each class. We estimate the mean and variance separately for each feature-class pair, resulting in independent Gaussian PDFs.
  
  **概念**: 根据贝叶斯定理假设每个特征在每个类内都应归正态分布。我们为每个特征-类栶分別估计均值不算数方差，从而得到独立的高斯数密函数。

- **ML Application**: Prior probabilities P(y) represent the class distribution in the training data. These will be multiplied with conditional likelihoods to compute posterior probabilities P(y|X) during classification.
  
  **机器学习应用**: 先验概率P(y)代表训练数据中的类分布。这些将与条件似然率相乘，以计算分类期间的后验概率P(y|X)。

➡️ **Next**: `03_naive_bayes_scratch.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |

## Complete Code / 完整代码一览

```python
# summarize probability distributions of the dataset
from sklearn.datasets import make_blobs
from scipy.stats import norm
from numpy import mean
from numpy import std

# fit a probability distribution to a univariate data sample
def fit_distribution(data):
    # estimate parameters
    mu = mean(data)
    sigma = std(data)
    print(mu, sigma)
    # fit distribution
    dist = norm(mu, sigma)
    return dist

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print(Xy0.shape, Xy1.shape)
# calculate priors
priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)
print(priory0, priory1)
# create PDFs for y==0
X1y0 = fit_distribution(Xy0[:, 0])
X2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
X1y1 = fit_distribution(Xy1[:, 0])
X2y1 = fit_distribution(Xy1[:, 1])
```

---

### Naive Bayes Scratch

# 18 — Naive Bayes Classification / 根据贝叶斯定理的分类

**Chapter 18 — File 3 of 4**

## Summary / 汇总

This notebook implements Naive Bayes classifier from scratch. We calculate P(y|X) = prior * product of feature PDFs for both classes and make predictions.

本笔记本从头实现朴素贝叶斯分类器。我们为两个类计算P(y|X) = 先验 * 特征PDF的乘积，并进行预测。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Define probability Calculation Helper / 定义概率计算辅助函数

```python
from sklearn.datasets import make_blobsfrom scipy.stats import normfrom numpy import mean, std# fit a probability distribution to a univariate data sampledef fit_distribution(data):    # estimate parameters    mu = mean(data)    sigma = std(data)    # fit distribution    dist = norm(mu, sigma)    return dist# calculate the independent conditional probabilitydef probability(X, prior, dist1, dist2):    # P(y|X) = prior * PDF(X1|y) * PDF(X2|y)    # Calculate probability using prior and conditional feature PDFs    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])
```

## Step 2 — Generate Dataset and Fit Distributions / 生成数据集并拟合分布

```python
# generate 2d classification datasetX, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)# sort data into classesXy0 = X[y == 0]Xy1 = X[y == 1]# calculate priorspriory0 = len(Xy0) / len(X)priory1 = len(Xy1) / len(X)# create PDFs for y==0distX1y0 = fit_distribution(Xy0[:, 0])distX2y0 = fit_distribution(Xy0[:, 1])# create PDFs for y==1distX1y1 = fit_distribution(Xy1[:, 0])distX2y1 = fit_distribution(Xy1[:, 1])
```

## Step 3 — Classify Test Sample / 分类测试样本

```python
# classify one example: select first sampleXsample, ysample = X[0], y[0]# calculate P(y=0|X) for the samplepy0 = probability(Xsample, priory0, distX1y0, distX2y0)# calculate P(y=1|X) for the samplepy1 = probability(Xsample, priory1, distX1y1, distX2y1)print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))print('Truth: y=%d' % ysample)
```

## Learning Notes / 学习笔记

- **Concept**: Naive Bayes assumes feature independence. The posterior probability P(y|X) is computed by multiplying the prior P(y) with the product of conditional PDFs for each feature: P(y|X) ∝ P(y) * ∏ P(x_i|y). **概念**: 朴素贝叶斯假设特征独立。后验概率P(y|X)通过将先验P(y)与每个特征的条件PDF的乘积相乘来计算: P(y|X) ∝ P(y) * ∏ P(x_i|y)。

- **ML Application**: Naive Bayes is computationally efficient and works well for many classification tasks. The 'naive' assumption of independence, though often violated, allows using individual feature probabilities rather than computing joint distributions. **机器学习应用**: 朴素贝叶斯在计算上是有效的，适用于许多分类任务。'朴素'的独立性假设虽然经常被违反，但允许使用单个特征概率而不是计算联合分布。

➡️ **Next**: `04_naive_bayes_sklearn.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# example of preparing and making a prediction with a naive bayes modelfrom sklearn.datasets import make_blobsfrom scipy.stats import normfrom numpy import meanfrom numpy import std# fit a probability distribution to a univariate data sampledef fit_distribution(data):    # estimate parameters    mu = mean(data)    sigma = std(data)    # fit distribution    dist = norm(mu, sigma)    return dist# calculate the independent conditional probabilitydef probability(X, prior, dist1, dist2):    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])# generate 2d classification datasetX, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)# sort data into classesXy0 = X[y == 0]Xy1 = X[y == 1]# calculate priorspriory0 = len(Xy0) / len(X)priory1 = len(Xy1) / len(X)# create PDFs for y==0distX1y0 = fit_distribution(Xy0[:, 0])distX2y0 = fit_distribution(Xy0[:, 1])# create PDFs for y==1distX1y1 = fit_distribution(Xy1[:, 0])distX2y1 = fit_distribution(Xy1[:, 1])# classify one exampleXsample, ysample = X[0], y[0]py0 = probability(Xsample, priory0, distX1y0, distX2y0)py1 = probability(Xsample, priory1, distX1y1, distX2y1)print('P(y=0 | %s) = %.3f' % (Xsample, py0*100))print('P(y=1 | %s) = %.3f' % (Xsample, py1*100))print('Truth: y=%d' % ysample)
```

---
