# 概率论与机器学习 / Probability for Machine Learning
## Chapter 25

---

### Dataset

# 25 — Baseline Classifiers / 基线分类器

**Chapter 25 — File 1 of 5**

## Summary / 汇总

This notebook creates an imbalanced dataset with 25% class 0 and 75% class 1. Imbalanced data provides a challenging baseline for classifier evaluation.

本笔记本创建了具有25%的0类和75%的1类的不平衡数据集。不平衡数据为分类器评估提供了具有挑战性的基准。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Create Imbalanced Dataset / 创建不平衡数据集

```python
# summarize a test dataset# define datasetclass0 = [0 for _ in range(25)]class1 = [1 for _ in range(75)]y = class0 + class1# summarize distributionprint('Class 0: %.3f' % (len(class0) / len(y) * 100))print('Class 1: %.3f' % (len(class1) / len(y) * 100))
```

## Learning Notes / 学习笔记

- **Concept**: Imbalanced datasets are common in practice (fraud detection, disease diagnosis, rare events). A naive classifier that always predicts the majority class achieves 75% accuracy without learning patterns. **概念**: 不平衡数据集在实践中很常见(欺诈检测、疾病诊断、稀有事件)。朴素分类器总是预测多数类，无需学习模式即可达到75%的准确率。

- **ML Application**: Baseline classifiers serve as sanity checks. Sophisticated models must outperform baselines; otherwise, they fail to learn meaningful patterns from imbalanced data. **机器学习应用**: 基线分类器作为健全性检查。复杂模型必须超越基线；否则，它们无法从不平衡数据中学习有意义的模式。

➡️ **Next**: `02_random_guess.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
class0 = [0 for _ in range(25)]class1 = [1 for _ in range(75)]y = class0 + class1print('Class 0: %.3f' % (len(class0) / len(y) * 100))print('Class 1: %.3f' % (len(class1) / len(y) * 100))
```

---

### Random Guess

# 25 — Baseline Classifiers / 基线分类器

**Chapter 25 — File 2 of 5**

## Summary / 汇总

This notebook implements a random 50/50 guesser classifier. Despite not learning the class distribution, it provides a baseline for comparison.

本笔记本实现了随机50/50猜测分类器。尽管没有学习类分布，但它提供了比较的基线。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


## Step 1 — Random 50/50 Guess Classifier / 随机50/50猜测分类器

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meanfrom numpy.random import randomfrom sklearn.metrics import accuracy_score# guess random class: 50% chance of 0 or 1def random_guess():    if random() < 0.5:        return 0    return 1# define datasetclass0 = [0 for _ in range(25)]class1 = [1 for _ in range(75)]y = class0 + class1# average performance over many repeatsresults = list()for _ in range(1000):    yhat = [random_guess() for _ in range(len(y))]    acc = accuracy_score(y, yhat)    results.append(acc)print('Mean: %.3f' % mean(results))
```

## Learning Notes / 学习笔记

- **Concept**: A random 50/50 guesser achieves ~50% accuracy regardless of true class distribution, because it ignores prior probabilities. This is worse than always predicting the majority class. **概念**: 随机50/50猜测器无论真实类分布如何都能达到约50%的准确率，因为它忽略了先验概率。这比总是预测多数类更差。

- **ML Application**: Random baselines are weak but useful for detecting obviously broken models. Any reasonable model should significantly outperform random chance on real datasets. **机器学习应用**: 随机基线很弱但对于检测明显破坏的模型很有用。任何合理的模型都应该在真实数据集上明显优于随机机会。

➡️ **Next**: `03_randomly_selected_class.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `numpy` | 数值计算库 | Numerical computing library |

## Complete Code / 完整代码一览

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import meanfrom numpy.random import randomfrom sklearn.metrics import accuracy_scoredef random_guess():    if random() < 0.5:        return 0    return 1class0 = [0 for _ in range(25)]class1 = [1 for _ in range(75)]y = class0 + class1results = list()for _ in range(1000):    yhat = [random_guess() for _ in range(len(y))]    acc = accuracy_score(y, yhat)    results.append(acc)print('Mean: %.3f' % mean(results))
```

---

### Randomly Selected Class



---

### Majority Class



---

### Majority Class Sklearn



---

### Chapter Summary / 章节总结



---
