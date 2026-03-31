# 概率论与机器学习 / Probability for Machine Learning
## Chapter 19

---

### Plot Domain



---

### Surrogate Function

# 19 — Bayesian Optimization / 贝叶斯优化

**Chapter 19 — File 2 of 4**

## Summary / 汇总

This notebook fits a Gaussian Process as a surrogate model to approximate the objective function. The surrogate learns from sparse observations and provides uncertainty estimates.

本笔记本将高斯过程作为代理模型拟合以近似目标函数。代理从稀疏观测学习并提供不确定性估计。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Import and Define Helper Functions / 导入并定义辅助函数

```python
# 用模型做预测 / Make predictions with model
from math import sin, pifrom numpy import arange, asarrayfrom numpy.random import normal, randomfrom matplotlib import pyplotfrom warnings import catch_warnings, simplefilterfrom sklearn.gaussian_process import GaussianProcessRegressor# objective functiondef objective(x, noise=0.1):    noise = normal(loc=0, scale=noise)    return (x**2 * sin(5 * pi * x)**6.0) + noise# surrogate or approximation for the objective functiondef surrogate(model, X):    # catch any warning generated when making a prediction    with catch_warnings():        # ignore generated warnings        simplefilter("ignore")        # Return mean predictions and standard deviation (uncertainty)        return model.predict(X, return_std=True)
```

## Step 2 — Define Plot Function / 定义绘图函数

```python
# plot real observations vs surrogate functiondef plot(X, y, model):    # scatter plot of inputs and real objective function    pyplot.scatter(X, y)    # line plot of surrogate function across domain    Xsamples = asarray(arange(0, 1, 0.001))    Xsamples = Xsamples.reshape(len(Xsamples), 1)    ysamples, _ = surrogate(model, Xsamples)    pyplot.plot(Xsamples, ysamples)    # show the plot    pyplot.show()
```

## Step 3 — Sample Domain and Fit Surrogate / 采样域并拟合代理

```python
# sample the domain sparsely with noiseX = random(100)y = asarray([objective(x) for x in X])# reshape into rows and colsX = X.reshape(len(X), 1)y = y.reshape(len(y), 1)# define the modelmodel = GaussianProcessRegressor()# fit the model to sparse observationsmodel.fit(X, y)# plot the surrogate functionplot(X, y, model)
```

## Learning Notes / 学习笔记

- **Concept**: Gaussian Process is a non-parametric Bayesian method that learns a distribution over functions. It provides both point predictions (mean) and uncertainty estimates (variance), which are crucial for acquisition functions. **概念**: 高斯过程是一种非参数贝叶斯方法，学习函数上的分布。它提供点预测（均值）和不确定性估计（方差），这对采集函数至关重要。

- **ML Application**: The surrogate model enables fast function evaluations without calling the expensive objective function. Its uncertainty estimates guide exploration vs exploitation trade-offs in optimization. **机器学习应用**: 代理模型能够快速评估函数而无需调用昂贵的目标函数。其不确定性估计指导优化中的探索与开发权衡。

➡️ **Next**: `03_bayes_opt_scratch.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# example of a gaussian process surrogate functionfrom math import sinfrom math import pifrom numpy import arangefrom numpy import asarrayfrom numpy.random import normalfrom numpy.random import randomfrom matplotlib import pyplotfrom warnings import catch_warningsfrom warnings import simplefilterfrom sklearn.gaussian_process import GaussianProcessRegressor# objective functiondef objective(x, noise=0.1):    noise = normal(loc=0, scale=noise)    return (x**2 * sin(5 * pi * x)**6.0) + noise# surrogate or approximation for the objective functiondef surrogate(model, X):    # catch any warning generated when making a prediction    with catch_warnings():        # ignore generated warnings        simplefilter("ignore")        return model.predict(X, return_std=True)# plot real observations vs surrogate functiondef plot(X, y, model):    # scatter plot of inputs and real objective function    pyplot.scatter(X, y)    # line plot of surrogate function across domain    Xsamples = asarray(arange(0, 1, 0.001))    Xsamples = Xsamples.reshape(len(Xsamples), 1)    ysamples, _ = surrogate(model, Xsamples)    pyplot.plot(Xsamples, ysamples)    # show the plot    pyplot.show()# sample the domain sparsely with noiseX = random(100)y = asarray([objective(x) for x in X])# reshape into rows and colsX = X.reshape(len(X), 1)y = y.reshape(len(y), 1)# define the modelmodel = GaussianProcessRegressor()# fit the modelmodel.fit(X, y)# plot the surrogate functionplot(X, y, model)
```

---

### Bayes Opt Scratch

# 19 — Bayesian Optimization / 贝叶斯优化

**Chapter 19 — File 3 of 4**

## Summary / 汇总

This notebook implements the complete Bayesian optimization loop from scratch. It combines the surrogate model with a Probability of Improvement (PoI) acquisition function to iteratively select points.

本笔记本从头实现完整的贝叶斯优化循环。它将代理模型与概率改进(PoI)采集函数结合起来以迭代地选择点。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 训练模型 / Train the model
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

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
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Define Acquisition Function / 定义采集函数

```python
from scipy.stats import norm# probability of improvement acquisition functiondef acquisition(X, Xsamples, model):    # calculate the best surrogate score found so far    yhat, _ = surrogate(model, X)    best = max(yhat)    # calculate mean and stdev via surrogate function    mu, std = surrogate(model, Xsamples)    mu = mu[:, 0]    # calculate the probability of improvement    # This measures how likely each point is to improve upon the best found    probs = norm.cdf((mu - best) / (std+1E-9))    return probs
```

## Step 2 — Define Acquisition Optimization / 定义采集优化

```python
# optimize the acquisition functiondef opt_acquisition(X, y, model):    # random search: generate random samples in the domain    Xsamples = random(100)    Xsamples = Xsamples.reshape(len(Xsamples), 1)    # calculate the acquisition function for each sample    scores = acquisition(X, Xsamples, model)    # locate the index of the largest scores (best acquisition value)    ix = argmax(scores)    # Return the point with highest acquisition value    return Xsamples[ix, 0]
```

## Step 3 — Main Bayesian Optimization Loop / 主贝叶斯优化循环

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import vstack# sample the domain sparsely with noiseX = random(100)y = asarray([objective(x) for x in X])# reshape into rows and colsX = X.reshape(len(X), 1)y = y.reshape(len(y), 1)# define the modelmodel = GaussianProcessRegressor()# fit the modelmodel.fit(X, y)# perform the optimization processfor i in range(100):    # select the next point to sample using acquisition function    x = opt_acquisition(X, y, model)    # sample the point (expensive objective function call)    actual = objective(x)    # summarize the finding    est, _ = surrogate(model, [[x]])    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))    # add the data to the dataset    X = vstack((X, [[x]]))    y = vstack((y, [[actual]]))    # update the model with new data    model.fit(X, y)# plot all samples and the final surrogate functionplot(X, y, model)# best resultix = argmax(y)print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```

## Learning Notes / 学习笔记

- **Concept**: Probability of Improvement balances exploration (high uncertainty) and exploitation (high predicted value). It quantifies the likelihood that a point will improve upon the current best observation. **概念**: 改进概率平衡探索（高不确定性）和利用（高预测值）。它量化了一个点改进当前最佳观测的可能性。

- **ML Application**: This loop iteratively refines the surrogate model while efficiently exploring the function space. By 100 iterations, the algorithm typically finds near-optimal solutions with far fewer function evaluations than grid search. **机器学习应用**: 此循环在有效探索函数空间的同时迭代地优化代理模型。经过100次迭代后，该算法通常比网格搜索少调用许多次函数即可找到接近最优的解。

➡️ **Next**: `04_bayes_opt_hyperparam.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# example of bayesian optimization for a 1d function from scratchfrom math import sinfrom math import pifrom numpy import arangefrom numpy import vstackfrom numpy import argmaxfrom numpy import asarrayfrom numpy.random import normalfrom numpy.random import randomfrom scipy.stats import normfrom sklearn.gaussian_process import GaussianProcessRegressorfrom warnings import catch_warningsfrom warnings import simplefilterfrom matplotlib import pyplot# objective functiondef objective(x, noise=0.1):    noise = normal(loc=0, scale=noise)    return (x**2 * sin(5 * pi * x)**6.0) + noise# surrogate or approximation for the objective functiondef surrogate(model, X):    # catch any warning generated when making a prediction    with catch_warnings():        # ignore generated warnings        simplefilter("ignore")        return model.predict(X, return_std=True)# probability of improvement acquisition functiondef acquisition(X, Xsamples, model):    # calculate the best surrogate score found so far    yhat, _ = surrogate(model, X)    best = max(yhat)    # calculate mean and stdev via surrogate function    mu, std = surrogate(model, Xsamples)    mu = mu[:, 0]    # calculate the probability of improvement    probs = norm.cdf((mu - best) / (std+1E-9))    return probs# optimize the acquisition functiondef opt_acquisition(X, y, model):    # random search, generate random samples    Xsamples = random(100)    Xsamples = Xsamples.reshape(len(Xsamples), 1)    # calculate the acquisition function for each sample    scores = acquisition(X, Xsamples, model)    # locate the index of the largest scores    ix = argmax(scores)    return Xsamples[ix, 0]# plot real observations vs surrogate functiondef plot(X, y, model):    # scatter plot of inputs and real objective function    pyplot.scatter(X, y)    # line plot of surrogate function across domain    Xsamples = asarray(arange(0, 1, 0.001))    Xsamples = Xsamples.reshape(len(Xsamples), 1)    ysamples, _ = surrogate(model, Xsamples)    pyplot.plot(Xsamples, ysamples)    # show the plot    pyplot.show()# sample the domain sparsely with noiseX = random(100)y = asarray([objective(x) for x in X])# reshape into rows and colsX = X.reshape(len(X), 1)y = y.reshape(len(y), 1)# define the modelmodel = GaussianProcessRegressor()# fit the modelmodel.fit(X, y)# plot before handplot(X, y, model)# perform the optimization processfor i in range(100):    # select the next point to sample    x = opt_acquisition(X, y, model)    # sample the point    actual = objective(x)    # summarize the finding    est, _ = surrogate(model, [[x]])    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))    # add the data to the dataset    X = vstack((X, [[x]]))    y = vstack((y, [[actual]]))    # update the model    model.fit(X, y)# plot all samples and the final surrogate functionplot(X, y, model)# best resultix = argmax(y)print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```

---

### Bayes Opt Hyperparam

# 19 — Bayesian Optimization / 贝叶斯优化

**Chapter 19 — File 4 of 4**

## Summary / 汇总

This notebook uses scikit-optimize's gp_minimize for Bayesian hyperparameter tuning of a KNN classifier. It demonstrates practical application to machine learning model optimization.

本笔记本使用scikit-optimize的gp_minimize进行KNN分类器的贝叶斯超参数调优。它演示了机器学习模型优化的实际应用。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  ✂️ 划分数据集 / Split Dataset
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
```

## Step 1 — Setup Dataset and Model / 设置数据集和模型

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meanfrom sklearn.datasets import make_blobsfrom sklearn.model_selection import cross_val_scorefrom sklearn.neighbors import KNeighborsClassifierfrom skopt.space import Integerfrom skopt.utils import use_named_argsfrom skopt import gp_minimize# generate 2d classification datasetX, y = make_blobs(n_samples=500, centers=3, n_features=2)# define the modelmodel = KNeighborsClassifier()# define the space of hyperparameters to searchsearch_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]
```

## Step 2 — Define Evaluation Function / 定义评估函数

```python
# define the function used to evaluate a given configuration@use_named_args(search_space)def evaluate_model(**params):    # something    model.set_params(**params)    # calculate 5-fold cross validation score    result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')    # calculate the mean of the scores    estimate = mean(result)    # return loss (1 - accuracy) for minimization    return 1.0 - estimate
```

## Step 3 — Perform Bayesian Optimization / 执行贝叶斯优化

```python
# perform optimization: gp_minimize uses Gaussian Process surrogateresult = gp_minimize(evaluate_model, search_space)# summarizing finding:print('Best Accuracy: %.3f' % (1.0 - result.fun))print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
```

## Learning Notes / 学习笔记

- **Concept**: gp_minimize automates the Bayesian optimization loop for hyperparameter tuning. It uses a Gaussian Process surrogate and Probability of Improvement acquisition to efficiently explore the hyperparameter space. **概念**: gp_minimize自动化超参数调优的贝叶斯优化循环。它使用高斯过程代理和改进概率采集以有效地探索超参数空间。

- **ML Application**: Bayesian hyperparameter optimization is more efficient than random or grid search, especially for expensive objectives like neural networks. scikit-optimize integrates seamlessly with scikit-learn pipelines. **机器学习应用**: 贝叶斯超参数优化比随机或网格搜索更有效，特别是对于神经网络等昂贵目标。scikit-optimize与scikit-learn管道无缝集成。

➡️ **Next**: `../chapter_21/01_coin_flip.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `cross_val_score` | 交叉验证评估模型 | Cross-validation model evaluation |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `numpy` | 数值计算库 | Numerical computing library |

## Complete Code / 完整代码一览

```python
# example of bayesian optimization with scikit-optimizefrom numpy import meanfrom sklearn.datasets import make_blobsfrom sklearn.model_selection import cross_val_scorefrom sklearn.neighbors import KNeighborsClassifierfrom skopt.space import Integerfrom skopt.utils import use_named_argsfrom skopt import gp_minimize# generate 2d classification datasetX, y = make_blobs(n_samples=500, centers=3, n_features=2)# define the modelmodel = KNeighborsClassifier()# define the space of hyperparameters to searchsearch_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]# define the function used to evaluate a given configuration@use_named_args(search_space)def evaluate_model(**params):    # something    model.set_params(**params)    # calculate 5-fold cross validation    result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')    # calculate the mean of the scores    estimate = mean(result)    return 1.0 - estimate# perform optimizationresult = gp_minimize(evaluate_model, search_space)# summarizing finding:print('Best Accuracy: %.3f' % (1.0 - result.fun))print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
```

---

### Chapter Summary / 章节总结



---
