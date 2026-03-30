# 概率论与机器学习 / Probability for Machine Learning
## Chapter 23

---

### Distributions

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 1 of 8**

## Summary / 汇总

This notebook plots two probability distributions P and Q. Cross-entropy measures the average number of bits needed using a suboptimal code Q designed for P.

本笔记本绘制两个概率分布P和Q。交叉熵衡量使用为P设计的次优代码Q所需的平均比特数。

## Step 1 — Define and Plot Distributions / 定义并绘制分布

```python
from matplotlib import pyplotevents = ['red', 'green', 'blue']p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]print('P=%.3f Q=%.3f' % (sum(p), sum(q)))pyplot.subplot(2,1,1)pyplot.bar(events, p)pyplot.subplot(2,1,2)pyplot.bar(events, q)pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Cross-entropy H(P,Q) measures the average information needed to specify outcomes from P using a code optimized for Q. It's always ≥ entropy H(P). **概念**: 交叉熵H(P,Q)衡量使用为Q优化的代码来指定来自P的结果所需的平均信息。它总是≥熵H(P)。

- **ML Application**: Cross-entropy is the standard loss function for classification. minimizing H(P,Q) where P is true labels and Q is predicted probabilities trains the model to match target distribution. **机器学习应用**: 交叉熵是分类的标准损失函数。最小化H(P,Q)，其中P是真实标签，Q是预测概率，训练模型以匹配目标分布。

➡️ **Next**: `02_cross_entropy.ipynb`

## Complete Code / 完整代码一览

```python
from matplotlib import pyplotevents = ['red', 'green', 'blue']p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]print('P=%.3f Q=%.3f' % (sum(p), sum(q)))pyplot.subplot(2,1,1)pyplot.bar(events, p)pyplot.subplot(2,1,2)pyplot.bar(events, q)pyplot.show()
```

---

### Cross Entropy

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 2 of 8**

## Summary / 汇总

This notebook calculates cross-entropy from scratch. H(P,Q) = -Σ p*log₂(q) measures the cost of using distribution Q when P is the truth.

本笔记本从头计算交叉熵。H(P,Q) = -Σ p*log₂(q)衡量当P是真实时使用分布Q的成本。

## Step 1 — Define and Calculate Cross-Entropy / 定义并计算交叉熵

```python
from math import log2# calculate cross-entropy: cost of using code Q for events from Pdef cross_entropy(p, q):    # H(P,Q) = -sum(p[i] * log2(q[i]))    # Lower q values (incorrect predictions) incur higher cost    return -sum([p[i]*log2(q[i]) for i in range(len(p))])# define datap = [0.10, 0.40, 0.50]  # True distributionq = [0.80, 0.15, 0.05]  # Predicted distribution# calculate cross-entropy H(P, Q)ce_pq = cross_entropy(p, q)print('H(P, Q): %.3f bits' % ce_pq)# calculate cross-entropy H(Q, P) - different orderce_qp = cross_entropy(q, p)print('H(Q, P): %.3f bits' % ce_qp)
```

## Learning Notes / 学习笔记

- **Concept**: Cross-entropy is asymmetric. H(P,Q) weights errors by the probability in P, so misclassifying high-probability events is more costly. This makes it ideal for classification loss. **概念**: 交叉熵是非对称的。H(P,Q)按P中的概率权衡错误，因此误分类高概率事件的成本更高。这使其非常适合分类损失。

- **ML Application**: For a single binary classification example, cross-entropy reduces to -log(y_pred) when true label y=1. Averaging over the dataset gives the batch cross-entropy loss. **机器学习应用**: 对于单个二元分类示例，当真实标签y=1时，交叉熵减少到-log(y_pred)。在数据集上平均得到批次交叉熵损失。

➡️ **Next**: `03_cross_distribution_itself.ipynb`

## Complete Code / 完整代码一览

```python
from math import log2def cross_entropy(p, q):    return -sum([p[i]*log2(q[i]) for i in range(len(p))])p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]ce_pq = cross_entropy(p, q)print('H(P, Q): %.3f bits' % ce_pq)ce_qp = cross_entropy(q, p)print('H(Q, P): %.3f bits' % ce_qp)
```

---

### Cross Distribution Itself

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 3 of 8**

## Summary / 汇总

This notebook calculates cross-entropy when distributions are identical. H(P,P) = H(P) shows that cross-entropy with itself equals entropy.

本笔记本计算分布相同时的交叉熵。H(P,P) = H(P)显示交叉熵与自身相等于熵。

## Step 1 — Cross-Entropy of Distribution with Itself / 分布与自身的交叉熵

```python
from math import log2# calculate cross entropydef cross_entropy(p, q):    return -sum([p[i]*log2(q[i]) for i in range(len(p))])# define datap = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]# calculate cross entropy H(P, P)# When P predicts itself perfectly, this equals entropy H(P)ce_pp = cross_entropy(p, p)print('H(P, P): %.3f bits' % ce_pp)# calculate cross entropy H(Q, Q)ce_qq = cross_entropy(q, q)print('H(Q, Q): %.3f bits' % ce_qq)
```

## Learning Notes / 学习笔记

- **Concept**: When P=Q, cross-entropy reduces to entropy. This is the minimum possible cross-entropy. Any mismatch between true and predicted distributions increases the value. **概念**: 当P=Q时，交叉熵减少到熵。这是可能的最小交叉熵。真实分布与预测分布之间的任何不匹配都会增加该值。

- **ML Application**: Perfect predictions (Q=P) achieve the minimum possible loss, which equals entropy of labels. This is the theoretical lower bound of what a perfect model can achieve. **机器学习应用**: 完美预测(Q=P)达到最小可能的损失，等于标签的熵。这是完美模型可以达到的理论下界。

➡️ **Next**: `04_cross_entropy_alternate.ipynb`

## Complete Code / 完整代码一览

```python
from math import log2def cross_entropy(p, q):    return -sum([p[i]*log2(q[i]) for i in range(len(p))])p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]ce_pp = cross_entropy(p, p)print('H(P, P): %.3f bits' % ce_pp)ce_qq = cross_entropy(q, q)print('H(Q, Q): %.3f bits' % ce_qq)
```

---

### Cross Entropy Alternate

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 4 of 8**

## Summary / 汇总

This notebook demonstrates the relationship H(P,Q) = H(P) + KL(P||Q). Cross-entropy decomposes into entropy plus KL divergence, showing prediction error comes from both data uncertainty and misalignment.

本笔记本演示了关系H(P,Q) = H(P) + KL(P||Q)。交叉熵分解为熵加KL散度，显示预测误差来自数据不确定性和错误对齐。

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
from math import log2def kl_divergence(p, q):    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))def entropy(p):    return -sum([p[i] * log2(p[i]) for i in range(len(p))])def cross_entropy(p, q):    return entropy(p) + kl_divergence(p, q)p = [0.10, 0.40, 0.50]q = [0.80, 0.15, 0.05]en_p = entropy(p)print('H(P): %.3f bits' % en_p)kl_pq = kl_divergence(p, q)print('KL(P || Q): %.3f bits' % kl_pq)ce_pq = cross_entropy(p, q)print('H(P, Q): %.3f bits' % ce_pq)
```

---

### Entropy Labels

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 5 of 8**

## Summary / 汇总

This notebook calculates entropy of one-hot encoded class labels. A one-hot vector like [1,0,0] has zero entropy, representing perfect certainty.

本笔记本计算独热编码类标签的熵。独热向量如[1,0,0]的熵为零，代表完全确定性。

## Step 1 — Entropy of One-Hot Class Labels / 独热类标签的熵

```python
from math import log2from numpy import asarray# calculate entropydef entropy(p):    return -sum([p[i] * log2(p[i]) for i in range(len(p))])# class 1: [1, 0, 0] - complete certaintyp = asarray([1, 0, 0]) + 1e-15  # Add small epsilon to avoid log(0)print('Entropy of [1,0,0]: %.6f' % entropy(p))# class 2: [0, 1, 0] - complete certaintyp = asarray([0, 1, 0]) + 1e-15print('Entropy of [0,1,0]: %.6f' % entropy(p))# class 3: [0, 0, 1] - complete certaintyp = asarray([0, 0, 1]) + 1e-15print('Entropy of [0,0,1]: %.6f' % entropy(p))
```

## Learning Notes / 学习笔记

- **Concept**: One-hot encoded labels have zero entropy because each is completely determined. The epsilon prevents log(0) numerical issues. Pure one-hot vectors have no uncertainty. **概念**: 独热编码标签的熵为零，因为每个都是完全确定的。epsilon防止log(0)数值问题。纯独热向量没有不确定性。

- **ML Application**: When training with one-hot labels, the minimum achievable loss is the entropy of the label distribution (which is 0 for one-hot). Any > 0 indicates prediction error. **机器学习应用**: 使用独热标签训练时，可实现的最小损失是标签分布的熵(对于独热为0)。任何> 0表示预测误差。

➡️ **Next**: `06_average_cross_entropy.ipynb`

## Complete Code / 完整代码一览

```python
from math import log2from numpy import asarraydef entropy(p):    return -sum([p[i] * log2(p[i]) for i in range(len(p))])p = asarray([1, 0, 0]) + 1e-15print(entropy(p))p = asarray([0, 1, 0]) + 1e-15print(entropy(p))p = asarray([0, 0, 1]) + 1e-15print(entropy(p))
```

---

### Average Cross Entropy

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 6 of 8**

## Summary / 汇总

This notebook calculates average cross-entropy for binary classification predictions. It demonstrates the loss for individual examples and the batch average.

本笔记本计算二元分类预测的平均交叉熵。它演示单个示例的损失和批次平均值。

## Step 1 — Calculate Average Cross-Entropy / 计算平均交叉熵

```python
from math import logfrom numpy import mean# calculate cross entropy: cost for binary classificationdef cross_entropy(p, q):    return -sum([p[i]*log(q[i]) for i in range(len(p))])# define classification datap = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # True labelsq = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]  # Predicted probabilities# calculate cross entropy for each exampleresults = list()for i in range(len(p)):    # create the distribution for each event {0, 1}    expected = [1.0 - p[i], p[i]]    predicted = [1.0 - q[i], q[i]]    # calculate cross entropy for the two events    ce = cross_entropy(expected, predicted)    print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))    results.append(ce)# calculate the average cross entropymean_ce = mean(results)print('Average Cross Entropy: %.3f nats' % mean_ce)
```

## Learning Notes / 学习笔记

- **Concept**: Cross-entropy for binary classification H(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]. When y=1, only -log(ŷ) matters; when y=0, only -log(1-ŷ) matters. **概念**: 二元分类的交叉熵H(y, ŷ) = -[y*log(ŷ) + (1-y)*log(1-ŷ)]。当y=1时，只有-log(ŷ)重要；当y=0时，只有-log(1-ŷ)重要。

- **ML Application**: Averaging cross-entropy across the batch gives the final loss. This is the standard loss function in logistic regression and deep learning classification models. **机器学习应用**: 在批次间平均交叉熵得到最终损失。这是逻辑回归和深度学习分类模型中的标准损失函数。

➡️ **Next**: `07_average_cross_entropy_keras.ipynb`

## Complete Code / 完整代码一览

```python
from math import logfrom numpy import meandef cross_entropy(p, q):    return -sum([p[i]*log(q[i]) for i in range(len(p))])p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]results = list()for i in range(len(p)):    expected = [1.0 - p[i], p[i]]    predicted = [1.0 - q[i], q[i]]    ce = cross_entropy(expected, predicted)    print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))    results.append(ce)mean_ce = mean(results)print('Average Cross Entropy: %.3f nats' % mean_ce)
```

---

### Average Cross Entropy Keras

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 7 of 8**

## Summary / 汇总

This notebook calculates cross-entropy using Keras binary_crossentropy function. It demonstrates the integration with deep learning frameworks.

本笔记本使用Keras binary_crossentropy函数计算交叉熵。它演示了与深度学习框架的集成。

## Step 1 — Cross-Entropy with Keras / 使用Keras计算交叉熵

```python
from numpy import asarrayfrom keras import backendfrom keras.losses import binary_crossentropy# prepare classification datap = asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])q = asarray([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])# convert to keras variablesy_true = backend.variable(p)y_pred = backend.variable(q)# calculate the average cross-entropymean_ce = backend.eval(binary_crossentropy(y_true, y_pred))print('Average Cross Entropy: %.3f nats' % mean_ce)
```

## Learning Notes / 学习笔记

- **Concept**: Deep learning frameworks like Keras provide optimized, numerically stable implementations of cross-entropy loss. They handle gradient computation automatically for backpropagation. **概念**: 深度学习框架如Keras提供交叉熵损失的优化、数值稳定的实现。它们自动处理反向传播的梯度计算。

- **ML Application**: Using framework implementations rather than manual computation avoids numerical stability issues and integrates seamlessly with training loops and optimizers. **机器学习应用**: 使用框架实现而不是手动计算避免了数值稳定性问题，并与训练循环和优化器无缝集成。

➡️ **Next**: `08_log_loss.ipynb`

## Complete Code / 完整代码一览

```python
from numpy import asarrayfrom keras import backendfrom keras.losses import binary_crossentropyp = asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])q = asarray([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])y_true = backend.variable(p)y_pred = backend.variable(q)mean_ce = backend.eval(binary_crossentropy(y_true, y_pred))print('Average Cross Entropy: %.3f nats' % mean_ce)
```

---

### Log Loss

# 23 — Cross-Entropy Loss / 交叉熵损失

**Chapter 23 — File 8 of 8**

## Summary / 汇总

This notebook calculates log loss (cross-entropy) using scikit-learn's log_loss function. It shows the direct calculation with sklearn metrics.

本笔记本使用scikit-learn的log_loss函数计算对数损失(交叉熵)。它显示了与sklearn指标的直接计算。

## Step 1 — Log Loss with Scikit-Learn / 使用Scikit-Learn的对数损失

```python
from sklearn.metrics import log_lossfrom numpy import asarray# define classification datap = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]# define data as expected, e.g. probability for each event {0, 1}y_true = asarray([[1-v, v] for v in p])y_pred = asarray([[1-v, v] for v in q])# calculate the average log lossll = log_loss(y_true, y_pred)print('Average Log Loss: %.3f' % ll)
```

## Learning Notes / 学习笔记

- **Concept**: Log loss is another name for cross-entropy loss in classification. sklearn's log_loss expects class probabilities and handles multi-class classification automatically. **概念**: 对数损失是分类中交叉熵损失的另一个名称。sklearn的log_loss期望类概率并自动处理多类分类。

- **ML Application**: sklearn.metrics provides a convenient interface for computing standard losses. It's useful for evaluating trained models and comparing prediction quality across different algorithms. **机器学习应用**: sklearn.metrics为计算标准损失提供了便利的接口。它对于评估训练模型和比较不同算法的预测质量很有用。

➡️ **Next**: `../chapter_24/01_entropy.ipynb`

## Complete Code / 完整代码一览

```python
from sklearn.metrics import log_lossfrom numpy import asarrayp = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]y_true = asarray([[1-v, v] for v in p])y_pred = asarray([[1-v, v] for v in q])ll = log_loss(y_true, y_pred)print('Average Log Loss: %.3f' % ll)
```

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
