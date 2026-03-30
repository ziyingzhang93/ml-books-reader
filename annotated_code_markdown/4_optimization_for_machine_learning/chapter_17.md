# 机器学习优化方法
## Chapter 17

---

### Genetic Onemax

# 13 — Genetic Onemax / 13 Genetic Onemax

**Chapter 17 — File 1 of 2 / 第17章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **genetic algorithm search of the one max optimization problem**.

本脚本演示 **genetic algorithm search of the one max optimization problem**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — genetic algorithm search of the one max optimization problem

```python
from numpy.random import randint
from numpy.random import rand
```

---
## Step 2 — objective function

```python
def onemax(x):
	return -sum(x)
```

---
## Step 3 — tournament selection

```python
def selection(pop, scores, k=3):
```

---
## Step 4 — first random selection

```python
selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
```

---
## Step 5 — check if better (e.g. perform a tournament)

```python
if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
```

---
## Step 6 — crossover two parents to create two children

```python
def crossover(p1, p2, r_cross):
```

---
## Step 7 — children are copies of parents by default

```python
c1, c2 = p1.copy(), p2.copy()
```

---
## Step 8 — check for recombination

```python
if rand() < r_cross:
```

---
## Step 9 — select crossover point that is not on the end of the string

```python
pt = randint(1, len(p1)-2)
```

---
## Step 10 — perform crossover

```python
c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
```

---
## Step 11 — mutation operator

```python
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
```

---
## Step 12 — check for a mutation

```python
if rand() < r_mut:
```

---
## Step 13 — flip the bit

```python
bitstring[i] = 1 - bitstring[i]
```

---
## Step 14 — genetic algorithm

```python
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
```

---
## Step 15 — initial population of random bitstring

```python
pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
```

---
## Step 16 — keep track of best solution

```python
best, best_eval = 0, objective(pop[0])
```

---
## Step 17 — enumerate generations

```python
for gen in range(n_iter):
```

---
## Step 18 — evaluate all candidates in the population

```python
scores = [objective(c) for c in pop]
```

---
## Step 19 — check for new best solution

```python
for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
```

---
## Step 20 — select parents

```python
selected = [selection(pop, scores) for _ in range(n_pop)]
```

---
## Step 21 — create the next generation

```python
children = list()
		for i in range(0, n_pop, 2):
```

---
## Step 22 — get selected parents in pairs

```python
p1, p2 = selected[i], selected[i+1]
```

---
## Step 23 — crossover and mutation

```python
for c in crossover(p1, p2, r_cross):
```

---
## Step 24 — mutation

```python
mutation(c, r_mut)
```

---
## Step 25 — store for next generation

```python
children.append(c)
```

---
## Step 26 — replace population

```python
pop = children
	return [best, best_eval]
```

---
## Step 27 — define the total iterations

```python
n_iter = 100
```

---
## Step 28 — bits

```python
n_bits = 20
```

---
## Step 29 — define the population size

```python
n_pop = 100
```

---
## Step 30 — crossover rate

```python
r_cross = 0.9
```

---
## Step 31 — mutation rate

```python
r_mut = 1.0 / float(n_bits)
```

---
## Step 32 — perform the genetic algorithm search

```python
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---
## Learning Notes / 学习笔记

- **概念**: genetic algorithm search of the one max optimization problem 是机器学习中的常用技术。  
  *genetic algorithm search of the one max optimization problem is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Genetic Onemax / 13 Genetic Onemax
# Complete Code / 完整代码
# ===============================

# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand

# objective function
def onemax(x):
	return -sum(x)

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Genetic Continuous

# 21 — Genetic Continuous / 21 Genetic Continuous

**Chapter 17 — File 2 of 2 / 第17章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **genetic algorithm search for continuous function optimization**.

本脚本演示 **genetic algorithm search for continuous function optimization**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance


---
## Step 1 — genetic algorithm search for continuous function optimization

```python
from numpy.random import randint
from numpy.random import rand
```

---
## Step 2 — objective function

```python
def objective(x):
	return x[0]**2.0 + x[1]**2.0
```

---
## Step 3 — decode bitstring to numbers

```python
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
```

---
## Step 4 — extract the substring

```python
start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
```

---
## Step 5 — convert bitstring to a string of chars

```python
chars = ''.join([str(s) for s in substring])
```

---
## Step 6 — convert string to integer

```python
integer = int(chars, 2)
```

---
## Step 7 — scale integer to desired range

```python
value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
```

---
## Step 8 — store

```python
decoded.append(value)
	return decoded
```

---
## Step 9 — tournament selection

```python
def selection(pop, scores, k=3):
```

---
## Step 10 — first random selection

```python
selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
```

---
## Step 11 — check if better (e.g. perform a tournament)

```python
if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
```

---
## Step 12 — crossover two parents to create two children

```python
def crossover(p1, p2, r_cross):
```

---
## Step 13 — children are copies of parents by default

```python
c1, c2 = p1.copy(), p2.copy()
```

---
## Step 14 — check for recombination

```python
if rand() < r_cross:
```

---
## Step 15 — select crossover point that is not on the end of the string

```python
pt = randint(1, len(p1)-2)
```

---
## Step 16 — perform crossover

```python
c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
```

---
## Step 17 — mutation operator

```python
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
```

---
## Step 18 — check for a mutation

```python
if rand() < r_mut:
```

---
## Step 19 — flip the bit

```python
bitstring[i] = 1 - bitstring[i]
```

---
## Step 20 — genetic algorithm

```python
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
```

---
## Step 21 — initial population of random bitstring

```python
pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
```

---
## Step 22 — keep track of best solution

```python
best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
```

---
## Step 23 — enumerate generations

```python
for gen in range(n_iter):
```

---
## Step 24 — decode population

```python
decoded = [decode(bounds, n_bits, p) for p in pop]
```

---
## Step 25 — evaluate all candidates in the population

```python
scores = [objective(d) for d in decoded]
```

---
## Step 26 — check for new best solution

```python
for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
```

---
## Step 27 — select parents

```python
selected = [selection(pop, scores) for _ in range(n_pop)]
```

---
## Step 28 — create the next generation

```python
children = list()
		for i in range(0, n_pop, 2):
```

---
## Step 29 — get selected parents in pairs

```python
p1, p2 = selected[i], selected[i+1]
```

---
## Step 30 — crossover and mutation

```python
for c in crossover(p1, p2, r_cross):
```

---
## Step 31 — mutation

```python
mutation(c, r_mut)
```

---
## Step 32 — store for next generation

```python
children.append(c)
```

---
## Step 33 — replace population

```python
pop = children
	return [best, best_eval]
```

---
## Step 34 — define range for input

```python
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
```

---
## Step 35 — define the total iterations

```python
n_iter = 100
```

---
## Step 36 — bits per variable

```python
n_bits = 16
```

---
## Step 37 — define the population size

```python
n_pop = 100
```

---
## Step 38 — crossover rate

```python
r_cross = 0.9
```

---
## Step 39 — mutation rate

```python
r_mut = 1.0 / (float(n_bits) * len(bounds))
```

---
## Step 40 — perform the genetic algorithm search

```python
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))
```

---
## Learning Notes / 学习笔记

- **概念**: genetic algorithm search for continuous function optimization 是机器学习中的常用技术。  
  *genetic algorithm search for continuous function optimization is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Genetic Continuous / 21 Genetic Continuous
# Complete Code / 完整代码
# ===============================

# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define range for input
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))
# perform the genetic algorithm search
best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))
```

---

### Chapter Summary

# Chapter 17 Summary / 第17章总结

## Theme / 主题: Chapter 17 / Chapter 17

This chapter contains **2 code files** demonstrating chapter 17.

本章包含 **2 个代码文件**，演示Chapter 17。

---
## Evolution / 演化路线

  1. `13_genetic_onemax.ipynb` — Genetic Onemax
  2. `21_genetic_continuous.ipynb` — Genetic Continuous

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 17) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 17）是机器学习流水线中的基础构建块。

---
