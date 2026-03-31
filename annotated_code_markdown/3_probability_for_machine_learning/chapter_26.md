# 概率论与机器学习 / Probability for Machine Learning
## Chapter 26

---

### Log Loss Each Class

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 1 of 8**

## Summary / 汇总

This notebook plots log loss curves for correct and incorrect class predictions. It visualizes how log loss penalizes poor probability estimates.

本笔记本绘制正确和不正确类预测的对数损失曲线。它可视化对数损失如何惩罚较差的概率估计。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Plot Log Loss for Each Class / 为每个类绘制对数损失

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplot# predictions as 0 to 1 in 0.01 incrementsyhat = [x*0.01 for x in range(0, 101)]# evaluate predictions for a 0 true value (low probability = high loss)losses_0 = [log_loss([0], [x], labels=[0,1]) for x in yhat]# evaluate predictions for a 1 true value (high probability = high loss)losses_1 = [log_loss([1], [x], labels=[0,1]) for x in yhat]# plot input to losspyplot.plot(yhat, losses_0, label='true=0')pyplot.plot(yhat, losses_1, label='true=1')pyplot.legend()pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Log loss has symmetric curves. When true=0, loss increases as prediction→1. When true=1, loss increases as prediction→0. Confident wrong predictions carry heavy penalties. **概念**: 对数损失具有对称曲线。当真=0时，随着预测→1，损失增加。当真=1时，随着预测→0，损失增加。确信错误的预测会受到严重处罚。

- **ML Application**: Log loss heavily penalizes overconfident incorrect predictions. This makes it ideal for tasks where calibration and uncertainty quantification matter, like medical diagnosis or risk assessment. **机器学习应用**: 对数损失重罚过度自信的不正确预测。这使其非常适合需要校准和不确定性量化的任务，如医学诊断或风险评估。

➡️ **Next**: `02_log_loss_balanced.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplotyhat = [x*0.01 for x in range(0, 101)]losses_0 = [log_loss([0], [x], labels=[0,1]) for x in yhat]losses_1 = [log_loss([1], [x], labels=[0,1]) for x in yhat]pyplot.plot(yhat, losses_0, label='true=0')pyplot.plot(yhat, losses_1, label='true=1')pyplot.legend()pyplot.show()
```

---

### Log Loss Balanced

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 2 of 8**

## Summary / 汇总

This notebook plots log loss against fixed probability predictions on balanced data (50% each class). It shows how to evaluate constant prediction strategies.

本笔记本针对平衡数据上的固定概率预测绘制对数损失。它显示了如何评估恒定预测策略。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Log Loss with Balanced Dataset / 平衡数据集的对数损失

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplot# define a balanced dataset (50-50 split)testy = [0 for x in range(50)] + [1 for x in range(50)]# loss for predicting different fixed probability valuespredictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [log_loss(testy, [y for x in range(len(testy))]) for y in predictions]# plot predictions vs losspyplot.plot(predictions, losses)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: On balanced data, the optimal constant prediction is 0.5, achieving loss = 1 bit. Predictions away from 0.5 incur higher loss due to misalignment with 50-50 distribution. **概念**: 在平衡数据上，最优恒定预测是0.5，达到损失= 1比特。远离0.5的预测会因为与50-50分布不对齐而导致更高的损失。

- **ML Application**: This demonstrates why well-calibrated predictions matter. A model predicting [0.9, 0.9, 0.1, 0.1, ...] on balanced data will have high loss despite potential accuracy. **机器学习应用**: 这演示了为什么良好校准的预测很重要。在平衡数据上预测[0.9, 0.9, 0.1, 0.1, ...]的模型将具有高损失，尽管可能准确。

➡️ **Next**: `03_log_loss_imbalanced.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplottesty = [0 for x in range(50)] + [1 for x in range(50)]predictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [log_loss(testy, [y for x in range(len(testy))]) for y in predictions]pyplot.plot(predictions, losses)pyplot.show()
```

---

### Log Loss Imbalanced

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 3 of 8**

## Summary / 汇总

This notebook plots log loss on imbalanced data (100 class 0 vs 10 class 1). Imbalanced data shifts the optimal constant prediction toward the majority class.

本笔记本在不平衡数据(100个第0类对10个第1类)上绘制对数损失。不平衡数据将最优恒定预测转向多数类。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Log Loss with Imbalanced Dataset / 不平衡数据集的对数损失

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplot# define an imbalanced datasettesty = [0 for x in range(100)] + [1 for x in range(10)]# loss for predicting different fixed probability valuespredictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [log_loss(testy, [y for x in range(len(testy))]) for y in predictions]# plot predictions vs losspyplot.plot(predictions, losses)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: On imbalanced data (90% class 0), optimal constant prediction shifts toward the majority class (~0.1). The loss function naturally balances extremes. **概念**: 在不平衡数据(90%第0类)上，最优恒定预测向多数类转移(~0.1)。损失函数自然平衡极端。

- **ML Application**: Log loss automatically accommodates class imbalance without special weighting. This is why it's preferred over accuracy for imbalanced classification problems. **机器学习应用**: 对数损失自动适应类不平衡，无需特殊权衡。这是为什么对于不平衡分类问题首选它而不是准确率。

➡️ **Next**: `04_brier_errors.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import log_lossfrom matplotlib import pyplottesty = [0 for x in range(100)] + [1 for x in range(10)]predictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [log_loss(testy, [y for x in range(len(testy))]) for y in predictions]pyplot.plot(predictions, losses)pyplot.show()
```

---

### Brier Errors

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 4 of 8**

## Summary / 汇总

This notebook plots Brier score (mean squared error) for binary predictions. Unlike log loss, Brier score has a quadratic penalty.

本笔记本绘制二元预测的Brier评分(均方误差)。与对数损失不同，Brier评分具有二次惩罚。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Brier Score Curve / Brier评分曲线

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplot# predictions as 0 to 1 in 0.01 incrementsyhat = [x*0.01 for x in range(0, 101)]# evaluate predictions for a 1 true value (MSE penalty)losses = [brier_score_loss([1], [x], pos_label=[1]) for x in yhat]# plot input to losspyplot.plot(yhat, losses)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Brier score = mean((y - ŷ)²) is the quadratic penalty. It's less sensitive to extreme miscalibration than log loss, favoring well-calibrated moderate predictions. **概念**: Brier评分= mean((y - ŷ)²)是二次惩罚。它对极端错误校准的敏感度低于对数损失，偏好经过良好校准的中等预测。

- **ML Application**: Brier score is appropriate when you care equally about all prediction ranges. It's used in weather forecasting and other calibration-critical applications. **机器学习应用**: 当你对所有预测范围同样关心时，Brier评分是合适的。它用于天气预报和其他需要校准的应用中。

➡️ **Next**: `05_brier_balanced.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplotyhat = [x*0.01 for x in range(0, 101)]losses = [brier_score_loss([1], [x], pos_label=[1]) for x in yhat]pyplot.plot(yhat, losses)pyplot.show()
```

---

### Brier Balanced

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 5 of 8**

## Summary / 汇总

This notebook plots Brier score against fixed predictions on balanced data. It shows the quadratic nature of the penalty.

本笔记本针对平衡数据上的固定预测绘制Brier评分。它显示了惩罚的二次性质。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Brier Score with Balanced Dataset / 平衡数据集的Brier评分

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplot# define a balanced datasettesty = [0 for x in range(50)] + [1 for x in range(50)]# brier score for predicting different fixed probability valuespredictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions]# plot predictions vs losspyplot.plot(predictions, losses)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Brier score has minimum at prediction=0.5 for balanced data, just like log loss. However, the shape is parabolic (smooth) rather than exponential. **概念**: 对于平衡数据，Brier评分在预测=0.5处有最小值，就像对数损失一样。但是，形状是抛物线(光滑)而不是指数。

- **ML Application**: Brier score is easier to optimize numerically and has better theoretical properties for some algorithms. It's less prone to numerical instability than log loss. **机器学习应用**: Brier评分更容易数值优化，对某些算法具有更好的理论特性。它比对数损失更不容易数值不稳定。

➡️ **Next**: `06_brier_imbalanced.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplottesty = [0 for x in range(50)] + [1 for x in range(50)]predictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions]pyplot.plot(predictions, losses)pyplot.show()
```

---

### Brier Imbalanced

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 6 of 8**

## Summary / 汇总

This notebook plots Brier score on imbalanced data. Like log loss, Brier score naturally weights predictions by class distribution.

本笔记本在不平衡数据上绘制Brier评分。与对数损失一样，Brier评分自然按类分布加权预测。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results


---
## Code Flow / 代码流程

```
  ⚙️ 配置训练 / Configure Training
       │
       ▼
  📊 评估模型 / Evaluate Model
       │
       ▼
  📈 可视化结果 / Visualize Results
```

## Step 1 — Brier Score with Imbalanced Dataset / 不平衡数据集的Brier评分

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplot# define an imbalanced datasettesty = [0 for x in range(100)] + [1 for x in range(10)]# brier score for predicting different fixed probability valuespredictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions]# plot predictions vs losspyplot.plot(predictions, losses)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: On imbalanced data, Brier score has minimum near 0.09 (≈10% class 1 proportion). The parabola shifts left following data distribution. **概念**: 在不平衡数据上，Brier评分的最小值接近0.09(≈10%的第1类比例)。抛物线向左移动，跟随数据分布。

- **ML Application**: This shows Brier score naturally handles imbalance without modifications. It's an alternative to log loss with different regularization properties. **机器学习应用**: 这显示Brier评分自然处理不平衡，无需修改。这是对数损失的替代品，具有不同的正则化特性。

➡️ **Next**: `07_roc_curve.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `loss` | 损失函数：衡量预测与真实值的差距 | Loss: measures gap between prediction and truth |
| `matplotlib` | 绑图库 | Plotting library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import brier_score_lossfrom matplotlib import pyplottesty = [0 for x in range(100)] + [1 for x in range(10)]predictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]losses = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions]pyplot.plot(predictions, losses)pyplot.show()
```

---

### Roc Curve

# 26 — Probability Scoring Metrics / 概率评分指标

**Chapter 26 — File 7 of 8**

## Summary / 汇总

This notebook plots ROC curves. The ROC curve plots True Positive Rate vs False Positive Rate at different classification thresholds.

本笔记本绘制ROC曲线。ROC曲线在不同分类阈值处绘制真阳性率vs虚假正阳性率。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 训练模型 / Train the model
- 可视化结果 / Visualize results

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
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Plot ROC Curve / 绘制ROC曲线

```python
from sklearn.datasets import make_classificationfrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import roc_curvefrom matplotlib import pyplot# generate 2 class datasetX, y = make_classification(n_samples=1000, n_classes=2, random_state=1)# split into train/test setstrainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)# fit a modelmodel = LogisticRegression(solver='lbfgs')model.fit(trainX, trainy)# predict probabilitiesprobs = model.predict_proba(testX)# keep probabilities for the positive outcome onlyprobs = probs[:, 1]# calculate roc curvefpr, tpr, thresholds = roc_curve(testy, probs)# plot no skill (random classifier)pyplot.plot([0, 1], [0, 1], linestyle='--')# plot the roc curve for the modelpyplot.plot(fpr, tpr)# show the plotpyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: ROC curves plot TPR vs FPR at all thresholds. A random classifier is a diagonal line; higher curves indicate better discrimination. Area Under Curve (AUC) summarizes performance. **概念**: ROC曲线在所有阈值处绘制TPR vs FPR。随机分类器是对角线；更高的曲线表示更好的判别。曲线下的面积(AUC)总结性能。

- **ML Application**: ROC curves are threshold-independent and ideal for imbalanced data. They ignore class proportions, focusing on relative ranking of predictions. **机器学习应用**: ROC曲线与阈值无关，非常适合不平衡数据。它们忽略类比例，专注于预测的相对排名。

➡️ **Next**: `08_roc_auc.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `LogisticRegression` | 逻辑回归（分类算法） | Logistic Regression (classification) |
| `matplotlib` | 绑图库 | Plotting library |
| `model.fit` | 训练模型 | Train the model |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

## Complete Code / 完整代码一览

```python
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.datasets import make_classificationfrom sklearn.linear_model import LogisticRegressionfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import roc_curvefrom matplotlib import pyplotX, y = make_classification(n_samples=1000, n_classes=2, random_state=1)trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)model = LogisticRegression(solver='lbfgs')model.fit(trainX, trainy)probs = model.predict_proba(testX)probs = probs[:, 1]fpr, tpr, thresholds = roc_curve(testy, probs)pyplot.plot([0, 1], [0, 1], linestyle='--')pyplot.plot(fpr, tpr)pyplot.show()
```

---

### Roc Auc



---

### Chapter Summary / 章节总结

# Chapter 26: Probability Scoring

## Overview
This chapter explores **probability-based scoring metrics** for evaluating classification models. These metrics score how well a model assigns probabilities to classes, distinct from accuracy which only looks at hard predictions.

## Key Concepts
- **Log Loss (Cross-Entropy)**: Penalizes confidence in wrong predictions
- **Brier Score**: Mean squared error of probabilities
- **ROC Curve**: Sensitivity vs specificity at different thresholds
- **AUC**: Area under ROC curve, probability of correct ranking
- **Balanced vs Imbalanced**: How metrics behave with class imbalance

## Evolution of Examples

### Exploring Both Metrics × Both Data Scenarios
1. **01_log_loss_curves.py**: Plot log loss as function of predicted probability
2. **02_log_loss_balanced.py**: Evaluate on balanced dataset
3. **03_log_loss_imbalanced.py**: Evaluate on imbalanced dataset
4. **04_brier_curves.py**: Plot Brier score function
5. **05_brier_balanced.py**: Evaluate on balanced dataset
6. **06_brier_imbalanced.py**: Evaluate on imbalanced dataset
7. **07_roc_curve.py**: Plot ROC curve
8. **08_roc_auc_score.py**: Calculate AUC under ROC

## Log Loss (Cross-Entropy)

### Definition
```
LogLoss = -(1/n) × Σᵢ [yᵢ × log(pᵢ) + (1-yᵢ) × log(1-pᵢ)]
```
- yᵢ: True label (0 or 1)
- pᵢ: Predicted probability of class 1

### Properties
- **Range**: [0, ∞)
- **Lower is better**: 0 is perfect
- **Penalties**:
  - y=1, p→0: log(0) → ∞ (confident wrong)
  - y=0, p→1: log(0) → ∞ (confident wrong)
  - y=1, p=1: 0 (correct)

### Curve Properties
```
y=1: Loss = -log(p)
- p=1: Loss = 0
- p=0.5: Loss = 0.693
- p=0: Loss = ∞

y=0: Loss = -log(1-p)
- p=0: Loss = 0
- p=0.5: Loss = 0.693
- p=1: Loss = ∞
```
Symmetric around p=0.5, heavy penalties for extreme wrong predictions.

### Balanced Data (50-50)
- Naive classifier: Log Loss ≈ 0.693
- Good model: Log Loss < 0.3
- Great model: Log Loss < 0.1

### Imbalanced Data (90-10)
- Naive classifier (always 0.9): Log Loss ≈ 0.325
- Good model: Log Loss < 0.2
- Imbalance reduces baseline, making improvements harder to achieve

## Brier Score

### Definition
```
BrierScore = (1/n) × Σᵢ (yᵢ - pᵢ)²
```
Mean squared error between true label and predicted probability.

### Properties
- **Range**: [0, 1]
- **Lower is better**: 0 is perfect
- **Interpretation**: Average squared deviation
- **Softer penalties**: Quadratic vs logarithmic

### Curve Properties
```
y=1: BS = (1-p)²
- p=1: BS = 0
- p=0.5: BS = 0.25
- p=0: BS = 1

y=0: BS = p²
- p=0: BS = 0
- p=0.5: BS = 0.25
- p=1: BS = 1
```
Smooth quadratic penalty, less extreme than log loss.

### Balanced Data
- Naive classifier: BS ≈ 0.25
- Good model: BS < 0.10
- Great model: BS < 0.05

### Imbalanced Data (90-10)
- Naive classifier (always 0.9): BS ≈ 0.09
- Harder to improve: baseline already low

## Log Loss vs Brier Score

| Aspect | Log Loss | Brier Score |
|--------|----------|-------------|
| Penalty type | Logarithmic | Quadratic |
| Extreme errors | Heavily penalized | Moderately penalized |
| Range | [0, ∞) | [0, 1] |
| Interpretability | Bits of information | Squared error |
| When to use | Probability quality matters | Calibration matters |
| Sensitivity | High to outliers | Moderate |
| Use in practice | Deep learning losses | Probability calibration |

## ROC Curve

### Definition
Plot of **True Positive Rate (Sensitivity)** vs **False Positive Rate (1-Specificity)**

As we vary classification threshold from 0 to 1:
- Threshold=0: Classify all as positive → TPR=1, FPR=1 (top-right)
- Threshold=1: Classify all as negative → TPR=0, FPR=0 (bottom-left)
- Threshold=0.5: Standard decision boundary → somewhere in middle

### Interpretation
- **Perfect classifier**: Goes through (0,0) → (0,1) → (1,1)
- **Random classifier**: Diagonal from (0,0) to (1,1), AUC=0.5
- **Real classifier**: Curve above diagonal (hopefully)

### Why Threshold Matters
Default threshold p=0.5 may be suboptimal:
- Imbalanced data: Lowering threshold catches more minority class
- Cost-sensitive: Different cost for FP vs FN
- Applications: Fraud (want high recall) vs spam (want high precision)

## AUC (Area Under Curve)

### Definition
Area under ROC curve = probability that classifier ranks random positive higher than random negative.

### Interpretation
- AUC = 1.0: Perfect ranking (no threshold choice needed)
- AUC = 0.5: Random ranking
- AUC = 0.7: Good ranking ability
- AUC = 0.9: Excellent ranking ability

### Advantages
1. **Threshold-independent**: Doesn't depend on classification threshold
2. **Imbalance-robust**: Same scale regardless of class proportion
3. **Meaningful**: Probabilistic interpretation
4. **Ranking**: Measures relative likelihood ordering

### Balanced Data
- Naive classifier: AUC = 0.5
- Good model: AUC > 0.7
- Excellent: AUC > 0.85

### Imbalanced Data
- AUC = 0.5 baseline unchanged (robust property)
- Real model AUC still > 0.7 if good
- Much better metric than accuracy for imbalanced data

## Comparison: All Metrics

| Metric | Type | Scale | Robust to Imbalance | Threshold-Free | Use Case |
|--------|------|-------|---------------------|-----------------|----------|
| Log Loss | Probability | [0, ∞) | Moderate | Yes | Loss function |
| Brier | Probability | [0, 1] | Moderate | Yes | Calibration |
| ROC Curve | Ranking | [0, 1] | Yes | No | Visualization |
| AUC | Ranking | [0, 1] | Yes | Yes | Default metric |

## Practical Workflow

1. **Check Log Loss**: Overall probability quality
2. **Check AUC**: Ranking ability (threshold-independent)
3. **Plot ROC**: Visualize threshold tradeoffs
4. **Adjust threshold**: Based on cost/requirements
5. **Check Brier**: Calibration quality

## Key Takeaways
1. Log Loss and Brier Score measure probability quality
2. Log Loss penalizes extreme errors, Brier is smoother
3. Both have balanced/imbalanced variants
4. ROC curve shows threshold tradeoffs
5. AUC is robust metric: same scale on balanced and imbalanced data
6. For imbalanced data, use AUC not accuracy
7. Different metrics capture different aspects of model quality

---
