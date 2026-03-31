# 概率论与机器学习 / Probability for Machine Learning
## Chapter 23

---

### Distributions



---

### Cross Entropy



---

### Cross Distribution Itself



---

### Cross Entropy Alternate

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 4 of 8**

## Summary / 汇总

This notebook demonstrates the relationship H(P,Q) = H(P) + KL(P||Q). Cross-entropy decomposes into entropy plus KL divergence, showing prediction error comes from both data uncertainty and misalignment.

本笔记本演示了关系H(P,Q) = H(P) + KL(P||Q)。交叉熵分解为熵加KL散度，显示预测误差来自数据不确定性和错误对齐。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Cross-Entropy as Entropy Plus KL / 交叉熵作为熵加KL

```python
from math import log2# calculate the kl divergence KL(P || Q)def kl_divergence(p, q):    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))# calculate entropy H(P)def entropy(p):    return -sum([p[i] * log2(p[i]) for i in range(len(p))])# calculate cross-entropy H(P, Q)def cross_entropy(p, q):    return entropy(p) + kl_divergence(p, q)# define datap = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]# calculate H(P)en_p = entropy(p)print('H(P): %.3f bits' % en_p)# calculate kl divergence KL(P || Q)kl_pq = kl_divergence(p, q)print('KL(P || Q): %.3f bits' % kl_pq)# calculate cross-entropy H(P, Q)ce_pq = cross_entropy(p, q)print('H(P, Q): %.3f bits' % ce_pq)
```

## Learning Notes / 学习笔记

- **Concept**: H(P,Q) = H(P) + KL(P||Q) decompose loss into two components: intrinsic entropy (irreducible uncertainty) and KL divergence (model misalignment). Only KL divergence can be minimized by the model. **概念**: H(P,Q) = H(P) + KL(P||Q)将损失分解为两个分量：固有熵(不可约不确定性)和KL散度(模型错误对齐)。只有KL散度可以通过模型最小化。

- **ML Application**: This decomposition explains why perfectly calibrated models on imbalanced datasets still have high cross-entropy - the data's inherent entropy is high. Reducing KL requires improving predictions. **机器学习应用**: 这种分解解释了为什么在不平衡数据集上完美校准的模型仍具有高交叉熵 - 数据的固有熵很高。减少KL需要改进预测。

➡️ **Next**: `05_entropy_labels.ipynb`

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from math import log2def kl_divergence(p, q):    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))def entropy(p):    return -sum([p[i] * log2(p[i]) for i in range(len(p))])def cross_entropy(p, q):    return entropy(p) + kl_divergence(p, q)p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]en_p = entropy(p)print('H(P): %.3f bits' % en_p)kl_pq = kl_divergence(p, q)print('KL(P || Q): %.3f bits' % kl_pq)ce_pq = cross_entropy(p, q)print('H(P, Q): %.3f bits' % ce_pq)
```

---

### Entropy Labels



---

### Average Cross Entropy

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 6 of 8**

## Summary / 汇总

This notebook calculates average cross-entropy for binary classification predictions. It demonstrates the loss for individual examples and the batch average.

本笔记本计算二元分类预测的平均交叉熵。它演示单个示例的损失和批次平均值。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Average Cross-Entropy / 计算平均交叉熵

```python
# 打印输出 / Print output
from math import logfrom numpy import mean# calculate cross entropy: cost for binary classificationdef cross_entropy(p, q):    return -sum([p[i]*log(q[i]) for i in range(len(p))])# define classification datap = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # True labelsq = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]  # Predicted probabilities# calculate cross entropy for each exampleresults = list()for i in range(len(p)):    # create the distribution for each event {0, 1}    expected = [1.0 - p[i], p[i]]    predicted = [1.0 - q[i], q[i]]    # calculate cross entropy for the two events    ce = cross_entropy(expected, predicted)    print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))    results.append(ce)# calculate the average cross entropymean_ce = mean(results)print('Average Cross Entropy: %.3f nats' % mean_ce)
```

## Learning Notes / 学习笔记

- **Concept**: Cross-entropy for binary classification H(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]. When y=1, only -log(ŷ) matters; when y=0, only -log(1-ŷ) matters. **概念**: 二元分类的交叉熵H(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]。当y=1时，只有-log(ŷ)重要；当y=0时，只有-log(1-ŷ)重要。

- **ML Application**: Averaging cross-entropy across the batch gives the final loss. This is the standard loss function in logistic regression and deep learning classification models. **机器学习应用**: 在批次间平均交叉熵得到最终损失。这是逻辑回归和深度学习分类模型中的标准损失函数。

➡️ **Next**: `07_average_cross_entropy_keras.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from math import logfrom numpy import meandef cross_entropy(p, q):    return -sum([p[i]*log(q[i]) for i in range(len(p))])p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]results = list()for i in range(len(p)):    expected = [1.0 - p[i], p[i]]    predicted = [1.0 - q[i], q[i]]    ce = cross_entropy(expected, predicted)    print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))    results.append(ce)mean_ce = mean(results)print('Average Cross Entropy: %.3f nats' % mean_ce)
```

---

### Average Cross Entropy Keras



---

### Log Loss



---

### Chapter Summary / 章节总结

# Chapter 23: Cross-Entropy

## Overview
This chapter explores **Cross-Entropy**, a fundamental loss function for classification and machine learning. The journey progresses from theory through properties to production APIs.

## Key Concepts
- **Cross-Entropy**: Expected bits to encode true labels using model probabilities
- **Relationship to KL**: H(P, Q) = H(P) + D_KL(P || Q)
- **Self Information vs Mutual Information**: Special cases
- **Log Loss**: Equivalent to cross-entropy for classification

## Evolution of Examples

### From Theory to Practice
1. **01_distributions.py**: Visualize true and model probability distributions
2. **02_cross_entropy.py**: Calculate cross-entropy from scratch
3. **03_cross_distribution_itself.py**: Cross-entropy of distribution with itself (self-information)
4. **04_cross_entropy_alternate.py**: Verify relationship to KL divergence
5. **05_entropy_labels.py**: Calculate entropy of true label distribution
6. **06_average_cross_entropy.py**: Calculate expected cross-entropy across dataset
7. **07_average_cross_entropy_keras.py**: Use Keras loss function
8. **08_log_loss.py**: Implement log loss (equivalent to cross-entropy)

## Cross-Entropy Formula

### Definition
```
H(P, Q) = -Σₓ P(x) × log(Q(x))
```
- P(x): True probability (ground truth labels)
- Q(x): Model probability (predicted probabilities)
- H ≥ H(P): Always greater than or equal to true entropy

### Interpretation
**Expected bits to encode samples from P using Q's code**

If we build an optimal encoder for Q's distribution but samples come from P:
- Cross-entropy: average bits wasted
- Minimizing CE ≡ minimizing KL divergence (when H(P) constant)

## Relationship to Entropy and KL Divergence

### The Decomposition
```
H(P, Q) = H(P) + D_KL(P || Q)
```

**Cross-entropy = True Entropy + KL Divergence**

Breaking down:
- H(P): Irreducible uncertainty in P (fixed)
- D_KL(P || Q): How badly Q approximates P (variable)

### Minimizing Cross-Entropy
Since H(P) is constant across models:
```
argmin_Q H(P, Q) = argmin_Q D_KL(P || Q)
```
Minimizing CE is equivalent to minimizing KL divergence!

## Special Cases

### Self Cross-Entropy
```
H(P, P) = H(P)
```
Distribution's cross-entropy with itself is its entropy.

### Binary Classification
True labels: y ∈ {0, 1}
Model outputs: p = P(y=1 | x)

```
CE = -[y × log(p) + (1-y) × log(1-p)]
```

- If y=1: CE = -log(p) (penalizes low probability for positive class)
- If y=0: CE = -log(1-p) (penalizes high probability for positive class)

### Multi-class Classification
True labels: one-hot y ∈ {0,1}^k (k classes)
Model outputs: p ∈ [0,1]^k, Σp = 1 (softmax)

```
CE = -Σᵢ yᵢ × log(pᵢ)
```
Sums over all classes; typically only one yᵢ=1.

## From Information to Classification Loss

### Average Cross-Entropy on Dataset
For n samples:
```
Loss = (1/n) × Σᵢ H(Pᵢ, Qᵢ)
     = (1/n) × Σᵢ -log(p_{yᵢ})
```
Average log loss across all samples.

### Log Loss Equivalence
```
Log Loss = -Σᵢ log(p_{yᵢ})
```
Negated log probability of true class.
- Perfect prediction (p=1): Loss = 0
- Poor prediction (p→0): Loss → ∞
- Penalizes confidence in wrong class heavily

## Production APIs

### Keras/TensorFlow
```python
loss = keras.losses.BinaryCrossentropy()        # Binary classification
loss = keras.losses.CategoricalCrossentropy()   # Multi-class (one-hot)
loss = keras.losses.SparseCategoricalCrossentropy()  # Multi-class (integer labels)
```

### PyTorch
```python
loss_fn = torch.nn.BCELoss()         # Binary with sigmoid
loss_fn = torch.nn.CrossEntropyLoss()  # Multi-class with softmax
```

### Scikit-learn
```python
from sklearn.metrics import log_loss
score = log_loss(y_true, y_pred_proba)
```

## Why Cross-Entropy for Classification?

### Natural Connection
Classification is fundamentally about assigning correct probabilities.
Cross-entropy directly measures this objective.

### Information-Theoretic Foundation
Grounded in information theory, not arbitrary choice.

### Computational Properties
- Gradient well-behaved: gradient ∝ (p - y)
- Well-conditioned: no saturation like squared error
- Avoids vanishing gradient problem

### Probabilistic Interpretation
Can interpret loss as negative log-likelihood (MLE framework).

## Properties

### Lower Bound
```
H(P, Q) ≥ H(P)
```
Cross-entropy at least true entropy (achieved when Q = P).

### Convexity
CE loss is convex in model parameters (for logistic/softmax models).
Guarantees unique global minimum.

### Connection to Likelihood
For Bernoulli/multinomial:
```
CE = -log(L)
```
Cross-entropy equals negative log-likelihood!

## Key Takeaways
1. Cross-entropy measures coding efficiency using model probabilities
2. Decomposes into true entropy + KL divergence
3. Minimizing CE = minimizing KL (when true entropy fixed)
4. Binary/multi-class forms are special cases of general formula
5. Equivalent to log loss and negative log-likelihood
6. Standard loss function for classification in deep learning
7. Information-theoretic justification makes it principled choice

---
