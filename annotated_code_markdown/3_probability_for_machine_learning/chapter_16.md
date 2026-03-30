# 概率论与机器学习
## Chapter 16

---

### Fall Die

# 01 — Fall and Die Problem / 跌倒和死亡问题

**Chapter 16 — File 1 of 3**

## Summary

We use Bayes' theorem to calculate the probability that an elderly person who fell died, given their age group and mortality rate. This is a classic application of conditional probability in medical diagnosis.

我们使用贝叶斯定理计算跌倒的老年人死亡的概率，给定其年龄组和死亡率。这是医学诊断中条件概率的经典应用。

**Formula:**

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Define the Problem / 定义问题

```python
import numpy as np

# Problem: An elderly person fell. What's the probability they died?
# Given:
# P(died | elderly) = 0.10  (10% of elderly who fall die)
# P(elderly) = 0.15  (15% of population is elderly)
# P(died) = 0.05  (5% of all people die from falls)

P_died_given_elderly = 0.10  # P(Died | Elderly)
P_elderly = 0.15             # P(Elderly)
P_died = 0.05                # P(Died)

print(f'Fall and Die Problem / 跌倒和死亡问题')
print(f'=' * 70)
print(f'Given Information / 给定信息:')
print(f'P(Died | Elderly) = {P_died_given_elderly}')
print(f'P(Elderly) = {P_elderly}')
print(f'P(Died) = {P_died}')
print(f'\nQuestion / 问题:')
print(f'P(Elderly | Died) = ?')
```

## Step 2 — Apply Bayes' Theorem / 应用贝叶斯定理

```python
# Bayes' Theorem:
# P(Elderly | Died) = P(Died | Elderly) * P(Elderly) / P(Died)

P_elderly_given_died = (P_died_given_elderly * P_elderly) / P_died

print(f'\nBayes Theorem Application / 贝叶斯定理应用:')
print(f'=' * 70)
print(f'\nP(Elderly | Died) = P(Died | Elderly) × P(Elderly) / P(Died)')
print(f'\nSubstituting values / 代入值:')
print(f'P(Elderly | Died) = {P_died_given_elderly} × {P_elderly} / {P_died}')
print(f'P(Elderly | Died) = {P_died_given_elderly * P_elderly} / {P_died}')
print(f'P(Elderly | Died) = {P_elderly_given_died:.4f}')
print(f'\nInterpretation / 解释:')
print(f'Of all people who died from falls, {P_elderly_given_died*100:.2f}% were elderly')
print(f'在所有死于跌倒的人中，{P_elderly_given_died*100:.2f}%是老年人')
```

## Step 3 — Intuition Check / 直觉检查

```python
# Let's think about this intuitively
# Of 1000 elderly: 1000 * 0.10 = 100 die
# Of 1000 people: 1000 * 0.05 = 50 die total
# What fraction of the 50 deaths are elderly?

total_people = 10000
elderly = total_people * P_elderly
elderly_deaths = elderly * P_died_given_elderly
total_deaths = total_people * P_died
fraction_elderly_among_deaths = elderly_deaths / total_deaths

print(f'\nIntuitive Verification / 直觉验证:')
print(f'=' * 70)
print(f'Total population: {total_people}')
print(f'Elderly (15%): {elderly:.0f}')
print(f'Elderly deaths (10% of elderly): {elderly_deaths:.0f}')
print(f'Total deaths (5% of all): {total_deaths:.0f}')
print(f'\nFraction of deaths that are elderly: {fraction_elderly_among_deaths:.4f}')
print(f'This matches our calculation: {P_elderly_given_died:.4f}')
```

## Learning Notes / 学习笔记\n\n- **Concept**: Bayes' theorem reverses conditional probabilities. Given we observe a death, what's the probability the person was elderly? This is critical for diagnosis and classification.\n\n- **ML Application**: Bayes' theorem is fundamental to Naive Bayes classifiers, medical diagnosis systems, spam detection, and Bayesian inference in general.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Next / 下一步\n\n➡️ **Next**: `02_spam_detect.ipynb`

---

### Spam Detect

# 02 — Spam Detection / 垃圾邮件检测

**Chapter 16 — File 2 of 3**

## Summary

We use Bayes' theorem with the law of total probability to calculate P(spam|flagged). Given that an email is flagged as spam, what's the probability it's actually spam? This accounts for false positives from the spam filter.

我们使用贝叶斯定理和全概率法则计算P(垃圾|被标记)。给定电子邮件被标记为垃圾邮件，它实际是垃圾邮件的概率是多少？这考虑了垃圾邮件过滤器的假阳性。

**Formula:**

$$P(\text{spam}|\text{flagged}) = \frac{P(\text{flagged}|\text{spam}) \cdot P(\text{spam})}{P(\text{flagged})}$$

$$P(\text{flagged}) = P(\text{flagged}|\text{spam}) \cdot P(\text{spam}) + P(\text{flagged}|\text{not spam}) \cdot P(\text{not spam})$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Define Problem / 定义问题

```python
import numpy as np

# Email spam filter problem
# Given:
# P(spam) = 0.1  (10% of emails are spam)
# P(flagged | spam) = 0.95  (filter catches 95% of spam)
# P(flagged | not spam) = 0.05  (false positive rate: 5%)

P_spam = 0.1
P_not_spam = 1 - P_spam
P_flagged_given_spam = 0.95
P_flagged_given_not_spam = 0.05

print(f'Spam Detection Problem / 垃圾邮件检测问题')
print(f'=' * 70)
print(f'Given Information / 给定信息:')
print(f'P(Spam) = {P_spam}')
print(f'P(Not Spam) = {P_not_spam}')
print(f'P(Flagged | Spam) = {P_flagged_given_spam}')
print(f'P(Flagged | Not Spam) = {P_flagged_given_not_spam}')
print(f'\nQuestion / 问题:')
print(f'If an email is flagged, what is P(Spam | Flagged)?')
```

## Step 2 — Calculate Total Probability / 计算全概率

```python
# Law of Total Probability:
# P(Flagged) = P(Flagged|Spam)*P(Spam) + P(Flagged|NotSpam)*P(NotSpam)

P_flagged = (P_flagged_given_spam * P_spam) + (P_flagged_given_not_spam * P_not_spam)

print(f'\nLaw of Total Probability / 全概率法则:')
print(f'=' * 70)
print(f'\nP(Flagged) = P(Flagged|Spam)×P(Spam) + P(Flagged|NotSpam)×P(NotSpam)')
print(f'\nP(Flagged) = {P_flagged_given_spam}×{P_spam} + {P_flagged_given_not_spam}×{P_not_spam}')
print(f'P(Flagged) = {P_flagged_given_spam * P_spam} + {P_flagged_given_not_spam * P_not_spam}')
print(f'P(Flagged) = {P_flagged:.4f}')
```

## Step 3 — Apply Bayes' Theorem / 应用贝叶斯定理

```python
# Bayes' Theorem:
# P(Spam | Flagged) = P(Flagged | Spam) * P(Spam) / P(Flagged)

P_spam_given_flagged = (P_flagged_given_spam * P_spam) / P_flagged

print(f'\nBayes Theorem Application / 贝叶斯定理应用:')
print(f'=' * 70)
print(f'\nP(Spam | Flagged) = P(Flagged | Spam) × P(Spam) / P(Flagged)')
print(f'\nP(Spam | Flagged) = {P_flagged_given_spam} × {P_spam} / {P_flagged:.4f}')
print(f'P(Spam | Flagged) = {P_flagged_given_spam * P_spam:.4f} / {P_flagged:.4f}')
print(f'P(Spam | Flagged) = {P_spam_given_flagged:.4f}')
print(f'\nInterpretation / 解释:')
print(f'If an email is flagged, there is a {P_spam_given_flagged*100:.1f}% chance it is actually spam')
print(f'如果电子邮件被标记，它实际是垃圾邮件的概率是{P_spam_given_flagged*100:.1f}%')
```

## Step 4 — Intuitive Verification / 直觉验证

```python
# Out of 10000 emails:
total = 10000
spam = total * P_spam
not_spam = total * P_not_spam
spam_flagged = spam * P_flagged_given_spam
not_spam_flagged = not_spam * P_flagged_given_not_spam
total_flagged = spam_flagged + not_spam_flagged
actual_proportion = spam_flagged / total_flagged

print(f'\nIntuitive Verification / 直觉验证:')
print(f'=' * 70)
print(f'Out of {total:,} emails:')
print(f'  Spam: {spam:.0f}, Flagged: {spam_flagged:.0f}')
print(f'  Not Spam: {not_spam:.0f}, Flagged: {not_spam_flagged:.0f}')
print(f'\nTotal flagged: {total_flagged:.0f}')
print(f'Of these, {spam_flagged:.0f} are actually spam')
print(f'Proportion: {actual_proportion:.4f}')
print(f'\nThis matches our Bayes calculation: {P_spam_given_flagged:.4f}')
```

## Learning Notes / 学习笔记\n\n- **Concept**: Even with a good spam filter (95% detection, 5% false positive), if spam is rare (10%), most flagged emails are false positives! This is the base rate fallacy.\n\n- **ML Application**: Understanding this is critical for building reliable classifiers and interpreting their predictions, especially with imbalanced classes.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Next / 下一步\n\n➡️ **Next**: `03_lie_detector.ipynb`

---

### Lie Detector

# 03 — Lie Detector Problem / 谎言检测器问题

**Chapter 16 — File 3 of 3**

## Summary

We use Bayes' theorem with the complement rule to calculate P(lying|positive result). Given a positive lie detector result, what's the probability the person is actually lying? This demonstrates the importance of prior probabilities and the base rate fallacy.

我们使用贝叶斯定理和补集规则计算P(说谎|正结果)。给定积极的谎言检测器结果，一个人实际上说谎的概率是多少？这演示了先验概率和基本速率谬误的重要性。

**Formula:**

$$P(\text{lying}|\text{positive}) = \frac{P(\text{positive}|\text{lying}) \cdot P(\text{lying})}{P(\text{positive}|\text{lying}) \cdot P(\text{lying}) + P(\text{positive}|\text{not lying}) \cdot P(\text{not lying})}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Define Problem / 定义问题

```python
import numpy as np

# Lie detector test problem
# Given:
# P(positive | lying) = 0.9  (test catches 90% of liars)
# P(positive | not lying) = 0.2  (false positive rate: 20%)
# P(lying) = 0.1  (prior: assume 10% of suspects are actually lying)

P_positive_given_lying = 0.9
P_positive_given_not_lying = 0.2
P_lying = 0.1
P_not_lying = 1 - P_lying

print(f'Lie Detector Problem / 谎言检测器问题')
print(f'=' * 70)
print(f'Test Characteristics / 测试特征:')
print(f'P(Positive | Lying) = {P_positive_given_lying}  (90% sensitivity)')
print(f'P(Positive | Not Lying) = {P_positive_given_not_lying}  (20% false positive)')
print(f'\nPrior Beliefs / 先验信念:')
print(f'P(Lying) = {P_lying}')
print(f'P(Not Lying) = {P_not_lying}')
print(f'\nQuestion / 问题:')
print(f'If test is positive, what is P(Lying | Positive)?')
```

## Step 2 — Calculate Total Probability / 计算全概率

```python
# Total probability of positive result
# P(Positive) = P(Positive|Lying)*P(Lying) + P(Positive|NotLying)*P(NotLying)

P_positive = (P_positive_given_lying * P_lying) + (P_positive_given_not_lying * P_not_lying)

print(f'\nTotal Probability Calculation / 全概率计算:')
print(f'=' * 70)
print(f'\nP(Positive) = P(Pos|Lying)×P(Lying) + P(Pos|NotLying)×P(NotLying)')
print(f'\nP(Positive) = {P_positive_given_lying}×{P_lying} + {P_positive_given_not_lying}×{P_not_lying}')
print(f'P(Positive) = {P_positive_given_lying*P_lying:.3f} + {P_positive_given_not_lying*P_not_lying:.3f}')
print(f'P(Positive) = {P_positive:.3f}')
```

## Step 3 — Apply Bayes' Theorem / 应用贝叶斯定理

```python
# Bayes' Theorem:
# P(Lying | Positive) = P(Positive | Lying) * P(Lying) / P(Positive)

P_lying_given_positive = (P_positive_given_lying * P_lying) / P_positive

print(f'\nBayes Theorem Application / 贝叶斯定理应用:')
print(f'=' * 70)
print(f'\nP(Lying | Positive) = P(Pos | Lying) × P(Lying) / P(Positive)')
print(f'\nP(Lying | Positive) = {P_positive_given_lying} × {P_lying} / {P_positive:.3f}')
print(f'P(Lying | Positive) = {P_positive_given_lying * P_lying:.3f} / {P_positive:.3f}')
print(f'P(Lying | Positive) = {P_lying_given_positive:.4f}')
print(f'\nInterpretation / 解释:')
print(f'Even with a positive result, there is only a {P_lying_given_positive*100:.1f}% chance the person is lying!')
print(f'即使测试为正，该人说谎的概率仅为{P_lying_given_positive*100:.1f}%！')
```

## Step 4 — Base Rate Fallacy / 基本速率谬误

```python
# This demonstrates the base rate fallacy
# Many people would think a 90% accurate test means 90% probability when positive
# But the low base rate (10% are actually lying) means false positives dominate!

print(f'\nBase Rate Fallacy / 基本速率谬误:')
print(f'=' * 70)
print(f'\nIntuitive but WRONG reasoning:')
print(f'  "Test is 90% accurate, so positive = 90% chance lying"')
print(f'\nCorrect reasoning (Bayes):  {P_lying_given_positive*100:.1f}%')
print(f'\nWhy the difference?')
print(f'  Out of 1000 people:')
print(f'    100 are actually lying')
print(f'    900 are not lying')
print(f'\n  Test results:')
print(f'    Of 100 liars: {int(100 * P_positive_given_lying)} test positive (true positives)')
print(f'    Of 900 truthful: {int(900 * P_positive_given_not_lying)} test positive (false positives)')
print(f'\n  Total positive: {int(100 * P_positive_given_lying) + int(900 * P_positive_given_not_lying)}')
print(f'  True positives: {int(100 * P_positive_given_lying)}')
print(f'  Proportion: {int(100 * P_positive_given_lying) / (int(100 * P_positive_given_lying) + int(900 * P_positive_given_not_lying)):.4f}')
```

## Learning Notes / 学习笔记\n\n- **Concept**: The base rate fallacy is the failure to account for the prior probability. When the prior is low (few people are lying), false positives dominate even with a sensitive test.\n\n- **ML Application**: This is critical for medical diagnosis, fraud detection, and any classification task with imbalanced classes. Always consider base rates when interpreting model predictions.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

## Chapter 16 Complete / 第16章完成\n\nThis chapter covered Bayes' theorem applications: the foundation for probabilistic reasoning and machine learning classification.

---
