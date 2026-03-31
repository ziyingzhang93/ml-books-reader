# 概率论与机器学习 / Probability for Machine Learning
## Chapter 21

---

### Coin Flip



---

### Dice Roll

# 21 — Information and Entropy / 信息与熵

**Chapter 21 — File 2 of 6**

## Summary / 汇总

This notebook calculates the information content of a single outcome from a fair die roll. A rare outcome (p=1/6) carries more information than a common outcome.

本笔记本计算公平骰子单次结果的信息内容。罕见结果(p=1/6)比常见结果携带更多信息。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Information for Die Roll / 计算骰子滚动的信息

```python
from math import log2# probability of one event (fair die has 6 equally likely outcomes)p = 1.0 / 6.0# calculate information for eventh = -log2(p)# print the resultprint('p(x)=%.3f, information: %.3f bits' % (p, h))
```

## Learning Notes / 学习笔记

- **Concept**: For a fair die, each outcome has probability p=1/6 ≈ 0.167, resulting in information h ≈ 2.585 bits. This is higher than a coin flip because die outcomes are less predictable. **概念**: 对于公平的骰子，每个结果的概率p=1/6 ≈ 0.167，导致信息h ≈ 2.585比特。这比硬币翻转更高，因为骰子结果更不可预测。

- **ML Application**: Information theory enables quantifying the complexity of distributions. Higher entropy distributions require more information to specify, useful in sampling and compression algorithms. **机器学习应用**: 信息论能够量化分布的复杂性。高熵分布需要更多信息来指定，在采样和压缩算法中有用。

➡️ **Next**: `03_probability_vs_info.ipynb`

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from math import log2p = 1.0 / 6.0h = -log2(p)print('p(x)=%.3f, information: %.3f bits' % (p, h))
```

---

### Probability Vs Info

# 21 — Information and Entropy / 信息与熵

**Chapter 21 — File 3 of 6**

## Summary / 汇总

This notebook plots the relationship between probability and information. As probability increases, information decreases (inverse relationship). Certain events carry no information.

本笔记本绘制概率与信息之间的关系。随着概率增加，信息减少(反比关系)。确定事件不携带信息。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Plot Probability vs Information / 绘制概率对信息

```python
from math import log2from matplotlib import pyplot# list of probabilitiesprobs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]# calculate information for each probabilityinfo = [-log2(p) for p in probs]# plot probability vs informationpyplot.plot(probs, info, marker='.')pyplot.title('Probability vs Information')pyplot.xlabel('Probability')pyplot.ylabel('Information')pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: The function h(p) = -log₂(p) is decreasing: certain events (p=1.0) have h=0, while rare events (p→0) have h→∞. This inverse relationship is fundamental to information theory. **概念**: 函数h(p) = -log₂(p)是递减的：确定事件(p=1.0)具有h=0，而罕见事件(p→0)具有h→∞。这种反比关系是信息论的基础。

- **ML Application**: Understanding this relationship helps in designing loss functions. Low-probability correct predictions are penalized heavily, making the model work harder to predict rare classes. **机器学习应用**: 理解这种关系有助于设计损失函数。低概率正确预测受到严重处罚，使模型更努力地预测稀有类。

➡️ **Next**: `04_entropy_dice_roll.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

## Complete Code / 完整代码一览

```python
from math import log2from matplotlib import pyplotprobs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]info = [-log2(p) for p in probs]pyplot.plot(probs, info, marker='.')pyplot.title('Probability vs Information')pyplot.xlabel('Probability')pyplot.ylabel('Information')pyplot.show()
```

---

### Entropy Dice Roll

# 21 — Information and Entropy / 信息与熵

**Chapter 21 — File 4 of 6**

## Summary / 汇总

This notebook calculates entropy of a fair die (6 equally likely outcomes). Entropy H(X) = Σ p*log₂(p) measures the average information content or uncertainty of a distribution.

本笔记本计算公平骰子的熵(6个等可能结果)。熵H(X) = Σ p*log₂(p)衡量分布的平均信息内容或不确定性。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Entropy of Fair Die / 计算公平骰子的熵

```python
from math import log2# the number of events (outcomes)n = 6# probability of one eventp = 1.0 / n# calculate entropy: H = -sum(p * log2(p) for each outcome)# For uniform distribution, H = log2(n)entropy = -sum([p * log2(p) for _ in range(n)])# print the resultprint('entropy: %.3f bits' % entropy)
```

## Learning Notes / 学习笔记

- **Concept**: Entropy H(X) = -Σ p(x)log₂p(x) quantifies average uncertainty. For n equally likely events, H = log₂(n). Entropy is maximized for uniform distributions. **概念**: 熵H(X) = -Σ p(x)log₂p(x)量化平均不确定性。对于n个等可能事件，H = log₂(n)。均匀分布时熵最大。

- **ML Application**: Decision trees use information gain (parent entropy - weighted child entropy) to recursively split nodes. This greedy approach efficiently selects features that minimize final entropy. **机器学习应用**: 决策树使用信息增益(父节点熵 - 加权子节点熵)递归地分割节点。这种贪心方法有效地选择最小化最终熵的特征。

➡️ **Next**: `05_entropy_dice_roll_scipy.ipynb`

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from math import log2n = 6p = 1.0 / nentropy = -sum([p * log2(p) for _ in range(n)])print('entropy: %.3f bits' % entropy)
```

---

### Entropy Dice Roll Scipy

# 21 — Information and Entropy / 信息与熵

**Chapter 21 — File 5 of 6**

## Summary / 汇总

This notebook calculates entropy using scipy.stats.entropy. The scipy implementation is numerically stable and convenient for computing entropy of probability distributions.

本笔记本使用scipy.stats.entropy计算熵。scipy实现在数值上是稳定的，并且便于计算概率分布的熵。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Entropy Using SciPy / 使用SciPy计算熵

```python
from scipy.stats import entropy# discrete probabilities for a fair diep = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]# calculate entropy with base-2 logarithme = entropy(p, base=2)# print the resultprint('entropy: %.3f bits' % e)
```

## Learning Notes / 学习笔记

- **Concept**: scipy.stats.entropy handles edge cases and numerical stability automatically. It supports custom logarithm bases: base=2 for bits, base=e for nats, base=10 for dits. **概念**: scipy.stats.entropy自动处理边界情况和数值稳定性。它支持自定义对数基数：base=2用于比特，base=e用于纳特，base=10用于十进制。

- **ML Application**: In machine learning, you'll often compute entropy of label distributions for sanity checking or in loss functions. scipy.stats.entropy is the go-to reliable implementation. **机器学习应用**: 在机器学习中，你经常计算标签分布的熵进行健全性检查或在损失函数中。scipy.stats.entropy是可靠的首选实现。

➡️ **Next**: `06_probability_vs_entropy.ipynb`

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from scipy.stats import entropyp = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]e = entropy(p, base=2)print('entropy: %.3f bits' % e)
```

---

### Probability Vs Entropy

# 21 — Information and Entropy / 信息与熵

**Chapter 21 — File 6 of 6**

## Summary / 汇总

This notebook plots entropy of binary distributions against the probability mass of one class. Maximum entropy occurs at p=0.5 (balanced distribution).

本笔记本绘制二元分布的熵对一个类的概率质量。最大熵出现在p=0.5(平衡分布)。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Calculate and Plot Entropy of Binary Distributions / 计算并绘制二元分布的熵

```python
from math import log2from matplotlib import pyplot# calculate entropy (handle edge cases with small epsilon)def entropy(events, eps=1e-15):    return -sum([p * log2(p + eps) for p in events])# define probabilities for one classprobs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]# create binary probability distributions [p, 1-p]dists = [[p, 1.0 - p] for p in probs]# calculate entropy for each distributionents = [entropy(d) for d in dists]# plot probability distribution vs entropypyplot.plot(probs, ents, marker='.')pyplot.title('Probability Distribution vs Entropy')pyplot.xticks(probs, [str(d) for d in dists])pyplot.xlabel('Probability Distribution')pyplot.ylabel('Entropy (bits)')pyplot.show()
```

## Learning Notes / 学习笔记

- **Concept**: Binary entropy H(p) = -p*log₂(p) - (1-p)*log₂(1-p) is symmetric around p=0.5 with maximum value 1. Skewed distributions (p≈0 or p≈1) have low entropy. **概念**: 二元熵H(p) = -p*log₂(p) - (1-p)*log₂(1-p)在p=0.5周围对称，最大值为1。倾斜分布(p≈0或p≈1)具有低熵。

- **ML Application**: In imbalanced classification, skewed class distributions have low entropy, making naive classifiers competitive. This motivates the use of resampling, weighted losses, or ensemble methods. **机器学习应用**: 在不平衡分类中，倾斜的类分布具有低熵，使朴素分类器具有竞争力。这促进了重采样、加权损失或集合方法的使用。

➡️ **Next**: `../chapter_22/01_distributions.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |

## Complete Code / 完整代码一览

```python
from math import log2from matplotlib import pyplotdef entropy(events, ets=1e-15):    return -sum([p * log2(p + ets) for p in events])probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]dists = [[p, 1.0 - p] for p in probs]ents = [entropy(d) for d in dists]pyplot.plot(probs, ents, marker='.')pyplot.title('Probability Distribution vs Entropy')pyplot.xticks(probs, [str(d) for d in dists])pyplot.xlabel('Probability Distribution')pyplot.ylabel('Entropy (bits)')pyplot.show()
```

---

### Chapter Summary / 章节总结



---
