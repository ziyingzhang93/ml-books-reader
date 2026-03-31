# 概率论与机器学习 / Probability for Machine Learning
## Chapter 24

---

### Entropy



---

### Info Gain

# 24 — Information Gain and Decision Trees / 信息增益与决策树

**Chapter 24 — File 2 of 2**

## Summary / 汇总

This notebook calculates information gain for a feature split. IG = H(parent) - Σ(|S_i|/|S|)*H(S_i) measures how much a split reduces entropy.

本笔记本计算特征分割的信息增益。IG = H(parent) - Σ(|S_i|/|S|)*H(S_i)衡量分割减少熵的程度。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Calculate Information Gain / 计算信息增益

```python
from math import log2# calculate the entropy for the split in the datasetdef entropy(class0, class1):    return -(class0 * log2(class0) + class1 * log2(class1))# split of the main datasetclass0 = 13 / 20class1 = 7 / 20# calculate entropy before the change (parent entropy)s_entropy = entropy(class0, class1)print('Dataset Entropy: %.3f bits' % s_entropy)# split 1 (split via value1)s1_class0 = 7 / 8s1_class1 = 1 / 8# calculate the entropy of the first groups1_entropy = entropy(s1_class0, s1_class1)print('Group1 Entropy: %.3f bits' % s1_entropy)# split 2 (split via value2)s2_class0 = 6 / 12s2_class1 = 6 / 12# calculate the entropy of the second groups2_entropy = entropy(s2_class0, s2_class1)print('Group2 Entropy: %.3f bits' % s2_entropy)# calculate the information gain: IG = H(parent) - weighted average of child entropiesgain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)print('Information Gain: %.3f bits' % gain)
```

## Learning Notes / 学习笔记

- **Concept**: Information Gain IG = H(parent) - Σ(|S_i|/|S|)*H(S_i) quantifies the reduction in entropy from a split. Higher IG indicates better feature for splitting, reducing impurity. **概念**: 信息增益IG = H(parent) - Σ(|S_i|/|S|)*H(S_i)量化分割带来的熵减少。较高的IG表示更好的分割特征，减少不纯性。

- **ML Application**: Decision tree algorithms (ID3, C4.5, CART) use information gain or Gini impurity to greedily select splits. This recursive process builds trees that minimize leaf entropy. **机器学习应用**: 决策树算法(ID3, C4.5, CART)使用信息增益或Gini不纯性来贪心地选择分割。这个递归过程构建最小化叶熵的树。

➡️ **Next**: `../chapter_25/01_dataset.ipynb`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
from math import log2def entropy(class0, class1):    return -(class0 * log2(class0) + class1 * log2(class1))class0 = 13 / 20class1 = 7 / 20s_entropy = entropy(class0, class1)print('Dataset Entropy: %.3f bits' % s_entropy)s1_class0 = 7 / 8s1_class1 = 1 / 8s1_entropy = entropy(s1_class0, s1_class1)print('Group1 Entropy: %.3f bits' % s1_entropy)s2_class0 = 6 / 12s2_class1 = 6 / 12s2_entropy = entropy(s2_class0, s2_class1)print('Group2 Entropy: %.3f bits' % s2_entropy)gain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)print('Information Gain: %.3f bits' % gain)
```

---

### Chapter Summary / 章节总结

# Chapter 24: Information Gain

## Overview
This chapter explores **Information Gain**, a metric for measuring how much a feature split reduces uncertainty. Information gain is the foundation for building decision trees.

## Key Concepts
- **Dataset Entropy**: Uncertainty in class distribution
- **Conditional Entropy**: Remaining uncertainty after split
- **Information Gain**: Entropy reduction from splitting on a feature
- **Greedy Split Selection**: Build trees by maximizing IG at each node

## Evolution of Examples

### From Measurement to Application
1. **01_dataset_entropy.py**: Calculate entropy of class distribution
2. **02_information_gain_split.py**: Calculate IG for a feature split

## Dataset Entropy

### Definition
```
H(Y) = -Σₖ p(Y=k) × log₂(p(Y=k))
```
Entropy of target variable Y (class labels).

### Interpretation
**Average uncertainty in class predictions without any features**

### Example: Binary Classification
```
50 positive, 50 negative: H(Y) = 1 bit (maximum uncertainty)
90 positive, 10 negative: H(Y) ≈ 0.47 bits (less uncertainty)
100 positive, 0 negative: H(Y) = 0 bits (certain)
```

## Conditional Entropy After Split

### Definition
```
H(Y | X) = Σⱼ p(X=j) × H(Y | X=j)
```

Average entropy in class distribution after splitting on feature X.

### Breakdown
- Outer sum: Weight by proportion in each split
- Inner H: Entropy within each split (subset)

### Example: Binary Feature Split
```
Feature X ∈ {True, False}
When X=True: 40 positive, 10 negative → H = 0.92 bits
When X=False: 10 positive, 40 negative → H = 0.92 bits
Proportion: p(X=True) = 0.5, p(X=False) = 0.5

H(Y | X) = 0.5 × 0.92 + 0.5 × 0.92 = 0.92 bits
```

## Information Gain

### Definition
```
IG(Y, X) = H(Y) - H(Y | X)
```

**How much entropy decreases when we know feature X**

### Interpretation
- IG = 0: Feature provides no information (splits don't reduce uncertainty)
- IG > 0: Feature is useful (reduces uncertainty)
- Larger IG: Better split

### Example
```
Dataset: 50 positive, 50 negative
H(Y) = 1.0 bit

Feature A split:
  A=True: 45 pos, 5 neg → H = 0.47 bits
  A=False: 5 pos, 45 neg → H = 0.47 bits
  H(Y | A) = 0.5 × 0.47 + 0.5 × 0.47 = 0.47 bits
  IG(Y, A) = 1.0 - 0.47 = 0.53 bits

Feature B split:
  B=True: 30 pos, 30 neg → H = 1.0 bit
  B=False: 20 pos, 20 neg → H = 1.0 bit
  H(Y | B) = 0.5 × 1.0 + 0.5 × 1.0 = 1.0 bit
  IG(Y, B) = 1.0 - 1.0 = 0.0 bits
```
Feature A is much better (0.53 > 0.0)!

## Decision Tree Building

### Greedy Algorithm
```
1. Start with all samples at root
2. For each feature:
   - For each possible split value:
     - Calculate information gain
3. Choose feature and split with highest IG
4. Create two child nodes (one per split value)
5. Recursively repeat on each child until stopping criteria
```

### Stopping Criteria
- Node entropy = 0 (pure: all same class)
- Too few samples (min_samples_split)
- Max depth reached
- No feature improves IG

## Gain Ratio (Normalization)

### Problem with Information Gain
Features with many unique values get artificially high IG.

Example: Feature = Customer ID (each person different)
- Creates perfect splits (each person in own group)
- H(Y | ID) ≈ 0
- IG very high
- But not actually useful for prediction

### Gain Ratio Solution
```
GainRatio(Y, X) = IG(Y, X) / SplitInfo(X)
```

Where:
```
SplitInfo(X) = -Σⱼ p(X=j) × log₂(p(X=j))
```
Entropy of feature X itself (penalizes many unique values).

## Gini Index (Alternative)

### Definition
```
Gini(Y) = 1 - Σₖ p(Y=k)²
```

Measures impurity (probability of mislabeling random sample).

### Advantages over Entropy
- Faster to compute (no logarithm)
- Similar behavior to entropy
- Used by scikit-learn's DecisionTreeClassifier by default

## Comparison: Entropy vs Gini

| Metric | Formula | Range | Use |
|--------|---------|-------|-----|
| Entropy | -Σp log p | [0, log k] | Information theory |
| Gini | 1 - Σp² | [0, 1-(1/k)] | Faster, no log |
| Gain Ratio | IG / SplitInfo | [0, 1] | Handles many features |

## Limitations

### Greedy Nature
- Maximizes IG at each node locally
- May miss globally optimal tree
- NP-complete to find optimal tree

### Bias Toward High-Cardinality Features
- IG favors features with many unique values
- Gain Ratio / Gini helps but doesn't eliminate

### Instability
- Small data changes can drastically change tree
- Different split on one node affects entire subtree
- Random Forest aggregates multiple trees to reduce

## Key Takeaways
1. Information gain quantifies feature usefulness
2. IG = entropy reduction from knowing feature
3. Decision trees greedily maximize IG at each split
4. Gain Ratio normalizes IG to handle high-cardinality features
5. Gini index provides faster alternative to entropy
6. Information gain is the foundation for tree-based models
7. Used in: decision trees, random forests, gradient boosting

---
