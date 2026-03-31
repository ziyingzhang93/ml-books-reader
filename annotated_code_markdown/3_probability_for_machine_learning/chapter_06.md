# 概率论与机器学习 / Probability for Machine Learning
## Chapter 06

---

### Birthday

# Chapter 06 — Birthday Problem / 生日悖论

**Chapter 06 — File 1 of 1**

## Summary

The birthday problem demonstrates the counterintuitive probability that in a group of people, two will share the same birthday. We calculate the probability of no collision and derive that with just 23 people, there's a 50% chance of a shared birthday.

生日问题展示了在一群人中，两个人生日相同的概率。我们计算无碰撞的概率，并推导出只需23人就有50%的概率发生生日重合。

**Formula:**

$$P(\text{no collision}) = \prod_{i=1}^{n-1} \frac{365-i}{365}$$

$$P(\text{collision}) = 1 - P(\text{no collision})$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Initialize Parameters / 初始化参数

We set up the calculation for groups of people from 1 to 30, assuming 365 days in a year.

```python
# Initialize parameters / 初始化参数
n = 30                    # Maximum group size / 最大人数
days = 365                # Days in a year / 一年的天数
p = 1.0                   # Probability of no collision / 无碰撞的概率
```

## Step 2 — Calculate Collision Probability / 计算碰撞概率

For each person added to the group, we multiply the probability that they have a different birthday from all previous people.

```python
# Calculate P(no collision) for increasing group sizes
# 计算递增的群体大小的无碰撞概率
# 生成整数序列 / Generate integer sequence
for i in range(1, n):
    # Available days for person i (excluding days taken by previous i-1 people)
    # 人i可用的天数（排除前i-1人占用的天数）
    av = days - i
    
    # Multiply probability: person i has different birthday
    # 乘以概率：第i个人有不同的生日
    p *= av / days
    
    # Collision probability = 1 - P(no collision)
    # 碰撞概率 = 1 - 无碰撞的概率
    collision_prob = (1 - p) * 100
    
    # 打印输出 / Print output
    print('n=%d, %d/%d, P(no collision)=%.3f%%, P(collision)=%.3f%%' 
          % (i+1, av, days, p*100, collision_prob))
```

## Step 3 — Find the Critical Point / 找到临界点

We determine when the probability of collision crosses 50%.

```python
# Find the first group size where collision probability > 50%
# 找到碰撞概率首次超过50%的最小人数
p = 1.0
# 生成整数序列 / Generate integer sequence
for i in range(1, n):
    av = days - i
    p *= av / days
    collision_prob = 1 - p
    
    if collision_prob >= 0.5:
        # 打印输出 / Print output
        print(f'\nCritical point: With {i+1} people, P(collision) = {collision_prob*100:.1f}%')
        # 打印输出 / Print output
        print(f'临界点：{i+1}人时，碰撞概率 = {collision_prob*100:.1f}%')
        break
```

## Learning Notes / 学习笔记

- **Concept**: The birthday paradox reveals how quickly collision probabilities grow. With 23 people, there's a 50.7% chance of a shared birthday—much higher than our intuition suggests (which expects ~1/365 ≈ 0.3%).
  
  **概念**：生日悖论揭示碰撞概率增长的速度。23人时，有50.7%的概率发生生日重合，远高于直觉预期（约1/365 ≈ 0.3%）。

- **ML Application**: This principle applies to hash collisions in data structures, duplicate detection in large datasets, and understanding false positive rates in statistical testing.
  
  **机器学习应用**：这个原理适用于数据结构中的哈希碰撞、大型数据集中的重复检测，以及理解统计测试中的假阳性率。

## Complete Code / 完整代码一览

```python
# Complete Birthday Problem Analysis / 完整的生日问题分析

# Initialize parameters / 初始化参数
n = 30                    # Maximum group size / 最大人数
days = 365                # Days in a year / 一年的天数
p = 1.0                   # Probability of no collision / 无碰撞的概率

# Calculate P(no collision) for increasing group sizes
# 计算递增的群体大小的无碰撞概率
# 打印输出 / Print output
print('Birthday Problem Analysis / 生日问题分析')
# 打印输出 / Print output
print('=' * 70)

# 生成整数序列 / Generate integer sequence
for i in range(1, n):
    # Available days for person i (excluding days taken by previous i-1 people)
    # 人i可用的天数（排除前i-1人占用的天数）
    av = days - i
    
    # Multiply probability: person i has different birthday
    # 乘以概率：第i个人有不同的生日
    p *= av / days
    
    # Collision probability = 1 - P(no collision)
    # 碰撞概率 = 1 - 无碰撞的概率
    collision_prob = (1 - p) * 100
    
    # 打印输出 / Print output
    print('n=%d, %d/%d, P(no collision)=%.3f%%, P(collision)=%.3f%%' 
          % (i+1, av, days, p*100, collision_prob))

# Find the critical point where P(collision) >= 50%
# 找到碰撞概率首次超过50%的临界点
# 打印输出 / Print output
print('\n' + '=' * 70)
p = 1.0
# 生成整数序列 / Generate integer sequence
for i in range(1, n):
    av = days - i
    p *= av / days
    collision_prob = 1 - p
    
    if collision_prob >= 0.5:
        # 打印输出 / Print output
        print(f'Critical Point / 临界点: {i+1} people, P(collision) = {collision_prob*100:.1f}%')
        break
```

---

### Chapter Summary / 章节总结

# Chapter 6: Birthday Problem

## Overview
This chapter explores **combinatorial probability** through the classic Birthday Problem—a striking example of how intuition fails when dealing with probabilities of events.

## Key Concepts
- **Combinatorial Counting**: Calculating probabilities using multiplication principle
- **Complement Rule**: It's easier to calculate the probability of "no shared birthday" then invert
- **Non-Intuitive Results**: With just 23 people, there's >50% chance of a shared birthday

## Problem Statement
What is the probability that in a group of n people, at least two share the same birthday?

## Solution Logic
Rather than count matching birthdays directly, count the complement:
- P(at least one match) = 1 - P(all different)
- P(all different) = (365/365) × (364/365) × (363/365) × ... × ((365-n+1)/365)

## Evolution of Examples
**01_birthday.py**: Single demo calculating probability for group sizes 1-30

## Key Takeaways
1. The probability reaches 50% at n=23 (counter-intuitive)
2. By n=30, probability exceeds 70%
3. This demonstrates why naive probability estimation is unreliable
4. Complement rule is a powerful tool for difficult counting problems

---
