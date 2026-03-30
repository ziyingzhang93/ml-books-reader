# 统计方法与机器学习
## Chapter 06

---

### Python Seed

# 06.01 — Python Random Seed / Python随机种子

**Chapter 06 — File 1 of 12**

## Summary / 摘要

**English:** This notebook demonstrates seeding the Python random number generator. Seeding ensures reproducibility: setting the same seed produces identical random sequences. This is essential for debugging, testing, and scientific replication.

**中文:** 本笔记本演示了Python随机数生成器的种子设置。设置种子确保可重现性：设置相同的种子会产生相同的随机序列。这对于调试、测试和科学复现至关重要。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Random Module / 导入随机模块

```python
# Import seed and random from Python's random module / 从Python的随机模块导入种子和随机
from random import seed
from random import random
```

## Step 2 — Set Seed and Generate First Sequence / 设置种子并生成第一个序列

```python
# Set seed to value 1 / 将种子设置为值1
# This initializes the pseudorandom generator to a known state / 这将伪随机生成器初始化为已知状态
seed(1)

# Generate three random numbers in [0, 1) / 在[0, 1)中生成三个随机数
print("First sequence with seed(1):")
print(random(), random(), random())
```

## Step 3 — Reset Seed and Verify Reproducibility / 重置种子并验证可重现性

```python
# Reset the seed to the same value / 将种子重置为相同值
seed(1)

# Generate three random numbers again / 再次生成三个随机数
# These will be identical to the first sequence / 这些将与第一个序列相同
print("\nSecond sequence with seed(1) again:")
print(random(), random(), random())

# Expected output: Both calls produce the same three values / 预期输出：两次调用产生相同的三个值
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** Pseudorandom number generators (PRNG) produce deterministic sequences that appear random but are fully determined by their seed. Setting a seed resets the PRNG to a specific state, ensuring the same seed always produces identical sequences. Different seeds produce different sequences. This allows controlled randomization—essential for experimentation, testing, and verification. In statistics, seed selection affects reproducibility but should not affect the statistical properties of independent experiments.

- **ML Application / 机器学习应用:** In machine learning, seeding is crucial for reproducible experiments. Setting a seed before data splitting, model initialization, and training enables others to replicate results exactly. When reporting model performance, practitioners should document seed values. Cross-validation with fixed seeds enables fair comparison across algorithms. In hyperparameter tuning and AutoML pipelines, seeds ensure consistency across random operations (shuffling, dropout, data augmentation).

➡️ **Next**: `02_python_random.ipynb` — Generate random floats in [0, 1)

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import random

# ===== Section 2: Generate First Random Sequence =====
# Set seed to initialize generator / 设置种子以初始化生成器
seed(1)

# Generate three random floats / 生成三个随机浮点数
print("First sequence with seed(1):")
print(random(), random(), random())

# ===== Section 3: Demonstrate Reproducibility =====
# Reset seed to same value / 将种子重置为相同值
seed(1)

# Generate again - should be identical / 再次生成 - 应该相同
print("\nSecond sequence with seed(1) again:")
print(random(), random(), random())
```

---

### Python Random

# 06.02 — Python Random Floats / Python随机浮点数

**Chapter 06 — File 2 of 12**

## Summary / 摘要

**English:** This notebook generates random floating-point numbers from the uniform distribution on [0, 1). The `random()` function is fundamental for Monte Carlo simulation, rejection sampling, and other stochastic algorithms.

**中文:** 本笔记本从[0, 1)上的均匀分布生成随机浮点数。`random()`函数对于蒙特卡洛模拟、拒绝采样和其他随机算法至关重要。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import seed and random functions / 导入种子和随机函数
from random import seed
from random import random
```

## Step 2 — Set Seed / 设置种子

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)
```

## Step 3 — Generate Random Floats / 生成随机浮点数

```python
# Generate 10 random numbers in range [0, 1) / 生成10个[0, 1)范围内的随机数
# Each call to random() returns a new random float / 每次调用random()返回一个新的随机浮点数
print("10 random floats from uniform distribution U[0,1):")
for i in range(10):
    value = random()
    print(f"  {i+1}: {value:.6f}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The uniform distribution on [0, 1) is fundamental in statistics and probability. Any continuous distribution can be sampled via the inverse transform method: generate $U \sim \text{Uniform}[0,1)$ and apply the inverse CDF, $X = F^{-1}(U)$. The returned values follow $X \sim F$. This enables sampling from arbitrary distributions. The uniformity property means each value in [0, 1) is equally likely (in the limit of large samples).

- **ML Application / 机器学习应用:** Uniform random numbers are essential in ML: data shuffling, train/test splitting, stochastic gradient descent (SGD) sampling, dropout regularization, and Monte Carlo estimation. Generating samples from other distributions (Gaussian, exponential, etc.) often begins with uniform random generation followed by transformation. Importance sampling, a variance reduction technique in deep learning, uses uniform randomness to weight samples according to their importance.

➡️ **Next**: `03_python_randint.ipynb` — Generate random integers

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import random

# ===== Section 2: Set Seed =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# ===== Section 3: Generate Random Floats =====
# Generate 10 random floats in [0, 1) / 生成10个[0, 1)中的随机浮点数
print("10 random floats from uniform distribution U[0,1):")
for i in range(10):
    value = random()
    print(f"  {i+1}: {value:.6f}")
```

---

### Python Randint

# 06.03 — Python Random Integers / Python随机整数

**Chapter 06 — File 3 of 12**

## Summary / 摘要

**English:** This notebook generates random integers from a discrete uniform distribution. The `randint(a, b)` function returns random integers in [a, b], useful for selecting indices, implementing randomized algorithms, and simulating discrete events.

**中文:** 本笔记本从离散均匀分布生成随机整数。`randint(a, b)`函数返回[a, b]中的随机整数，用于选择索引、实现随机算法和模拟离散事件。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import seed and randint functions / 导入种子和randint函数
from random import seed
from random import randint
```

## Step 2 — Set Seed / 设置种子

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)
```

## Step 3 — Generate Random Integers / 生成随机整数

```python
# Generate 10 random integers from [0, 10] / 从[0, 10]生成10个随机整数
# Note: randint(0, 10) includes both 0 and 10 / 注意：randint(0, 10)包括0和10
print("10 random integers from discrete uniform distribution U[0,10]:")
for i in range(10):
    value = randint(0, 10)
    print(f"  {i+1}: {value}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The discrete uniform distribution assigns equal probability to each integer in the range [a, b]. For $k = b - a + 1$ possible values, each has probability $1/k$. The mean is $(a + b)/2$ and variance is $(k^2 - 1)/12$. This distribution is fundamental for simulating die rolls, card draws, and other discrete random phenomena. Unlike continuous uniform, discrete uniform is defined on integers, making it natural for sampling indices or choices.

- **ML Application / 机器学习应用:** Random integer generation is essential in machine learning for sampling without replacement (e.g., selecting mini-batches in SGD), random forest feature selection, ensemble bagging, and stochastic data augmentation. In reinforcement learning, epsilon-greedy exploration uses random integers to select actions. Curriculum learning and hard example mining often use random sampling of indices to balance training. K-fold cross-validation uses random integers to assign folds to samples.

➡️ **Next**: `04_python_gauss.ipynb` — Generate random Gaussian values

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import randint

# ===== Section 2: Set Seed =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# ===== Section 3: Generate Random Integers =====
# Generate 10 random integers from [0, 10] / 从[0, 10]生成10个随机整数
print("10 random integers from discrete uniform distribution U[0,10]:")
for i in range(10):
    value = randint(0, 10)
    print(f"  {i+1}: {value}")
```

---

### Python Gauss

# 06.04 — Python Gaussian Distribution / Python高斯分布

**Chapter 06 — File 4 of 12**

## Summary / 摘要

**English:** This notebook generates random values from a Gaussian (normal) distribution. The `gauss(mu, sigma)` function samples from $N(\mu, \sigma)$, essential for simulation, hypothesis testing, and model initialization.

**中文:** 本笔记本从高斯（正态）分布生成随机值。`gauss(mu, sigma)`函数从$N(\mu, \sigma)$采样，对于模拟、假设检验和模型初始化至关重要。

**Formula / 公式:**
$$X \sim N(\mu, \sigma^2): f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import seed and gauss functions / 导入种子和gauss函数
from random import seed
from random import gauss
```

## Step 2 — Set Seed / 设置种子

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)
```

## Step 3 — Generate Gaussian Samples / 生成高斯样本

```python
# Generate 10 random samples from N(0, 1) - standard normal distribution / 从N(0, 1)生成10个随机样本 - 标准正态分布
# gauss(mu, sigma) returns samples from N(mu, sigma^2) / gauss(mu, sigma)返回来自N(mu, sigma^2)的样本
print("10 random values from Gaussian distribution N(0,1):")
for i in range(10):
    value = gauss(0, 1)
    print(f"  {i+1}: {value:.6f}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** The Gaussian distribution, parameterized by mean $\mu$ and standard deviation $\sigma$, is the most important distribution in statistics. It's characterized by its bell shape and the empirical rule: ~68% of values fall within $\pm 1\sigma$ of the mean, ~95% within $\pm 2\sigma$. The standard normal $N(0,1)$ is a reference; any Gaussian can be transformed to/from it via $Z = (X - \mu)/\sigma$. The Central Limit Theorem guarantees that sample means approach a Gaussian distribution regardless of the underlying distribution.

- **ML Application / 机器学习应用:** Gaussian random numbers are ubiquitous in ML: initializing neural network weights, sampling from variational autoencoders (VAE), Gaussian process regression, and probabilistic models. Many algorithms assume Gaussian errors (e.g., linear regression, Kalman filters). Bagging and dropout use pseudo-random sampling approximately following Gaussian distributions. Noise injection and data augmentation (adding Gaussian noise) improve model robustness. Confidence intervals and Bayesian inference heavily rely on Gaussian assumptions.

➡️ **Next**: `05_python_choice.ipynb` — Randomly select from a sequence

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import gauss

# ===== Section 2: Set Seed =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# ===== Section 3: Generate Gaussian Samples =====
# Generate 10 random samples from standard normal N(0,1) / 从标准正态N(0,1)生成10个随机样本
print("10 random values from Gaussian distribution N(0,1):")
for i in range(10):
    value = gauss(0, 1)
    print(f"  {i+1}: {value:.6f}")
```

---

### Python Sample

# 06.06 — Python Random Sample / Python随机样本

**Chapter 06 — File 6 of 12**

## Summary / 摘要

**English:** This notebook demonstrates random sampling without replacement. The `sample()` function selects a subset of specified size from a sequence, where each element appears at most once. This is essential for cross-validation, holdout testing, and unbiased data partitioning.

**中文:** 本笔记本演示了无放回随机采样。`sample()`函数从序列中选择指定大小的子集，每个元素最多出现一次。这对于交叉验证、留出测试和无偏数据分割至关重要。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import seed and sample functions / 导入种子和sample函数
from random import seed
from random import sample
```

## Step 2 — Set Seed and Create Sequence / 设置种子并创建序列

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Create a sequence of integers 0-19 / 创建0-19的整数序列
sequence = [i for i in range(20)]
print(f"Original sequence: {sequence}")
```

## Step 3 — Draw Sample Without Replacement / 无放回地抽样

```python
# Select a random subset of size 5 (without replacement) / 选择大小为5的随机子集（无放回）
# Each element can appear at most once in the sample / 样本中每个元素最多出现一次
# This is equivalent to a fair random partition / 这相当于公平的随机分割
subset = sample(sequence, 5)
print(f"\nRandom sample (size 5, no replacement): {subset}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** Sampling without replacement is a fundamental statistical technique. When selecting $k$ items from a population of $n$ without replacement, the number of possible samples is $\binom{n}{k} = n!/(k!(n-k)!)$. This constrains the probability space but is computationally efficient and ensures each item is unique. For large populations, sampling without replacement approaches sampling with replacement in terms of probability. The hypergeometric distribution governs the count of "success" items in samples without replacement, in contrast to the binomial distribution (with replacement).

- **ML Application / 机器学learning:** Sampling without replacement is critical in data science workflows: stratified K-fold cross-validation partitions data into non-overlapping folds; train/test splits use sampling to ensure disjoint sets; holdout validation uses sampling to create test sets. In imbalanced classification, stratified sampling preserves class ratios. Random forest uses bootstrap sampling (with replacement), while ensemble methods often use disjoint subsets (without replacement). Active learning and uncertainty sampling use careful selection strategies that depend on understanding the sampling distribution.

➡️ **Next**: `07_python_shuffle.ipynb` — Shuffle a sequence in place

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import sample

# ===== Section 2: Create Sequence =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Create sequence of integers / 创建整数序列
sequence = [i for i in range(20)]
print(f"Original sequence: {sequence}")

# ===== Section 3: Sample Without Replacement =====
# Draw 5 items without replacement / 无放回地抽取5个项目
subset = sample(sequence, 5)
print(f"\nRandom sample (size 5, no replacement): {subset}")
```

---

### Python Shuffle

# 06.07 — Python Shuffle / Python随机排列

**Chapter 06 — File 7 of 12**

## Summary / 摘要

**English:** This notebook demonstrates shuffling a sequence in place. The `shuffle()` function randomly permutes a list, modifying it in place. This is essential for data augmentation, randomizing order before train/test splits, and implementing randomized algorithms.

**中文:** 本笔记本演示了原地随机排列序列。`shuffle()`函数随机排列列表，就地修改它。这对于数据增强、在训练/测试分割前随机排序和实现随机算法至关重要。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import seed and shuffle functions / 导入种子和shuffle函数
from random import seed
from random import shuffle
```

## Step 2 — Set Seed and Create Sequence / 设置种子并创建序列

```python
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Create a sequence of integers 0-19 / 创建0-19的整数序列
sequence = [i for i in range(20)]
print(f"Original sequence: {sequence}")
```

## Step 3 — Shuffle Sequence In Place / 原地随机排列序列

```python
# Shuffle the sequence in place (modifies original list) / 原地排列序列（修改原始列表）
# shuffle() returns None and modifies the list directly / shuffle()返回None并直接修改列表
shuffle(sequence)

print(f"Shuffled sequence:   {sequence}")
print(f"\nNote: shuffle() modifies the list in place. A different seed would produce a different permutation.")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** A random shuffle generates one of the $n!$ possible permutations of a sequence, each with equal probability. The Fisher-Yates algorithm (used in Python) ensures uniformity by iterating from the end of the array, at each step swapping the current element with a randomly chosen earlier element. This runs in $O(n)$ time and produces an unbiased random permutation. Shuffles are fundamental to randomization techniques in statistics and computer science.

- **ML Application / 机器学习应用:** Shuffling is critical in machine learning: training data is shuffled before splitting to avoid ordering bias (e.g., data sorted by class); SGD requires shuffled mini-batches for unbiased gradient estimates; cross-validation shuffles indices before fold assignment; data augmentation often involves random shuffling of rows or columns. In recommendation systems, shuffling candidate lists prevents position bias. Curriculum learning can use shuffling to control the order of example presentation.

➡️ **Next**: `08_numpy_seed.ipynb` — Begin NumPy random functions

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from random import seed
from random import shuffle

# ===== Section 2: Create Sequence =====
# Set seed for reproducibility / 设置种子以保证可重现性
seed(1)

# Create sequence of integers / 创建整数序列
sequence = [i for i in range(20)]
print(f"Original sequence: {sequence}")

# ===== Section 3: Shuffle In Place =====
# Shuffle the sequence (modifies in place) / 随机排列序列（原地修改）
shuffle(sequence)

print(f"Shuffled sequence:   {sequence}")
print(f"\nNote: shuffle() modifies the list in place. A different seed would produce a different permutation.")
```

---

### Numpy Rand

# 06.09 — NumPy Uniform Random / NumPy均匀随机数

**Chapter 06 — File 9 of 12**

## Summary / 摘要

**English:** This notebook generates uniform random numbers in [0, 1) using NumPy's `rand()` function. Unlike Python's `random()`, NumPy's `rand()` returns arrays and is highly optimized for vectorized operations.

**中文:** 本笔记本使用NumPy的`rand()`函数生成[0, 1)中的均匀随机数。与Python的`random()`不同，NumPy的`rand()`返回数组并针对向量化操作进行了高度优化。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


## Step 1 — Import Libraries / 导入库

```python
# Import NumPy seed and random array functions / 导入NumPy种子和随机数组函数
from numpy.random import seed
from numpy.random import rand
```

## Step 2 — Set Seed / 设置种子

```python
# Set NumPy seed for reproducibility / 设置NumPy种子以保证可重现性
seed(1)
```

## Step 3 — Generate Uniform Random Array / 生成均匀随机数组

```python
# Generate 10 random floats from U[0,1) as a NumPy array / 生成10个来自U[0,1)的随机浮点数，以NumPy数组形式
# numpy.random.rand() returns array of specified shape / numpy.random.rand()返回指定形状的数组
values = rand(10)

print("10 random values from U[0,1) (NumPy array):")
print(values)
print(f"\nArray shape: {values.shape}")
print(f"Data type: {values.dtype}")
```

## Learning Notes / 学习笔记

- **Statistical Concept / 统计学概念:** NumPy's `rand()` generates samples from the standard uniform distribution $U[0,1)$. It can generate arrays of arbitrary shape efficiently. For a $m \times n$ array, `rand(m, n)` returns a matrix of random values. The underlying distribution is identical to Python's `random()`, but NumPy's implementation is vectorized and orders of magnitude faster. NumPy enables broadcasting operations on random arrays, making statistical computation rapid.

- **ML Application / 机器学习应用:** NumPy's `rand()` is standard for generating random features, initializing embeddings, and stochastic algorithms. In feature importance shuffling, random permutation matrices generated with `rand()` quantify feature contribution. Dropout layers use binary masks based on Bernoulli samples from `rand()`. Data augmentation often applies random affine transformations with parameters drawn from `rand()`. Large-scale simulations and Monte Carlo methods rely on NumPy's fast random generation for practical runtime.

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

➡️ **Next**: `10_numpy_randint.ipynb` — Generate random integers with NumPy

## Complete Code / 完整代码一览

```python
# ===== Section 1: Imports =====
from numpy.random import seed
from numpy.random import rand

# ===== Section 2: Set Seed =====
# Set NumPy seed for reproducibility / 设置NumPy种子以保证可重现性
seed(1)

# ===== Section 3: Generate Uniform Random Array =====
# Generate 10 random floats from U[0,1) / 从U[0,1)生成10个随机浮点数
values = rand(10)

print("10 random values from U[0,1) (NumPy array):")
print(values)
print(f"\nArray shape: {values.shape}")
print(f"Data type: {values.dtype}")
```

---

### Chapter Summary

# Chapter 6: Random Number Generation
# 第6章：随机数生成

## Theme | 主题
From reproducibility to generation to selection: mastering randomness in Python and NumPy.
从可重复性到生成到选择：掌握Python和NumPy中的随机性。

## Evolution Roadmap | 演变路线图
```
PYTHON STDLIB:
  seed() → Reproducibility
  └─ random() → Uniform [0,1)
     └─ randint(a,b) → Discrete Integers
        └─ gauss(μ,σ) → Gaussian Samples
           └─ choice(seq) → Element Selection
              └─ sample(seq,k) → k-Subset Selection
                 └─ shuffle(seq) → In-place Permutation

NUMPY EQUIVALENTS:
  seed() → Reproducibility
  └─ rand(shape) → Uniform [0,1)
     └─ randint(low,high,shape) → Discrete Integers
        └─ randn(shape) → Standard Normal
           └─ shuffle(arr) → In-place Permutation
```

## Progression Logic | 进度逻辑

### Stage 1: Reproducibility (再现性)
**English:** Set seed to ensure identical random sequences across runs, critical for debugging and research reproducibility.
**中文:** 设置种子以确保跨运行的相同随机序列，对调试和研究再现性至关重要。

### Stage 2: Uniform Generation (均匀生成)
**English:** Generate uniform random floats and integers as the foundation for all other distributions.
**中文:** 生成均匀随机浮点数和整数作为所有其他分布的基础。

### Stage 3: Distribution Sampling (分布采样)
**English:** Generate from specific distributions (Gaussian, etc.) for simulations and feature synthesis.
**中文:** 从特定分布（高斯等）生成，用于模拟和特征合成。

### Stage 4: Selection (选择)
**English:** Choose elements from sequences (choice), subsets (sample), and permutations (shuffle) for sampling and data augmentation.
**中文:** 从序列中选择元素（选择）、子集（样本）和排列（打乱）以进行采样和数据增强。

### Stage 5: NumPy Equivalents (NumPy等价物)
**English:** Vectorized NumPy functions offer 10-100x speedup for large-scale random generation.
**中文:** 向量化NumPy函数为大规模随机生成提供10-100倍加速。

## ML Relevance | ML相关性

1. **Reproducibility (再现性)**: Seeds enable reproducible ML pipelines and experiments.
2. **Data Augmentation (数据增强)**: Random sampling and shuffling create augmented training sets.
3. **Stochastic Methods (随机方法)**: Gradient descent, dropout, and simulated annealing depend on controlled randomness.
4. **Monte Carlo Simulation (蒙特卡洛模拟)**: Bootstrap and cross-validation require efficient random sampling.


---
