# 机器学习优化方法 / Optimization for Machine Learning
## Chapter 19

---

### Diff Evolution

# 15 — Diff Evolution / 15 Diff Evolution

**Chapter 19 — File 1 of 2 / 第19章 — 第1个文件（共2个）**

---

## Summary / 总结

This script demonstrates **differential evolution search of the two-dimensional sphere objective function**.

本脚本演示 **differential evolution search of the two-dimensional sphere objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — differential evolution search of the two-dimensional sphere objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import min
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import around
```

---
## Step 2 — define objective function

```python
def obj(x):
    return x[0]**2.0 + x[1]**2.0
```

---
## Step 3 — define mutation operation

```python
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])
```

---
## Step 4 — define boundary check operation

```python
def check_bounds(mutated, bounds):
    # 获取长度 / Get length
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound
```

---
## Step 5 — define crossover operation

```python
def crossover(mutated, target, dims, cr):
```

---
## Step 6 — generate a uniform random value for every dimension

```python
p = rand(dims)
```

---
## Step 7 — generate trial vector by binomial crossover

```python
# 生成整数序列 / Generate integer sequence
trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
```

---
## Step 8 — initialise population of candidate solutions randomly within the specified bounds

```python
# 获取长度 / Get length
pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
```

---
## Step 9 — evaluate initial population of candidate solutions

```python
obj_all = [obj(ind) for ind in pop]
```

---
## Step 10 — find the best performing vector of initial population

```python
best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
```

---
## Step 11 — run iterations of the algorithm

```python
# 生成整数序列 / Generate integer sequence
for i in range(iter):
```

---
## Step 12 — iterate over all candidate solutions

```python
# 生成整数序列 / Generate integer sequence
for j in range(pop_size):
```

---
## Step 13 — choose three candidates, a, b and c, that are not the current one

```python
# 生成整数序列 / Generate integer sequence
candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
```

---
## Step 14 — perform mutation

```python
mutated = mutation([a, b, c], F)
```

---
## Step 15 — check that lower and upper bounds are retained after mutation

```python
mutated = check_bounds(mutated, bounds)
```

---
## Step 16 — perform crossover

```python
# 获取长度 / Get length
trial = crossover(mutated, pop[j], len(bounds), cr)
```

---
## Step 17 — compute objective function value for target vector

```python
obj_target = obj(pop[j])
```

---
## Step 18 — compute objective function value for trial vector

```python
obj_trial = obj(trial)
```

---
## Step 19 — perform selection

```python
if obj_trial < obj_target:
```

---
## Step 20 — replace the target vector with the trial vector

```python
pop[j] = trial
```

---
## Step 21 — store the new objective function value

```python
obj_all[j] = obj_trial
```

---
## Step 22 — find the best performing vector at each iteration

```python
best_obj = min(obj_all)
```

---
## Step 23 — store the lowest objective function value

```python
if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
```

---
## Step 24 — report progress at each iteration

```python
# 打印输出 / Print output
print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj]
```

---
## Step 25 — define population size

```python
pop_size = 10
```

---
## Step 26 — define lower and upper bounds for every dimension

```python
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
```

---
## Step 27 — define number of iterations

```python
iter = 100
```

---
## Step 28 — define scale factor for mutation

```python
F = 0.5
```

---
## Step 29 — define crossover rate for recombination

```python
cr = 0.7
```

---
## Step 30 — perform differential evolution

```python
solution = differential_evolution(pop_size, bounds, iter, F, cr)
# 打印输出 / Print output
print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))
```

---
## Learning Notes / 学习笔记

- **概念**: differential evolution search of the two-dimensional sphere objective function 是机器学习中的常用技术。  
  *differential evolution search of the two-dimensional sphere objective function is a common technique in machine learning.*

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
# Diff Evolution / 15 Diff Evolution
# Complete Code / 完整代码
# ===============================

# differential evolution search of the two-dimensional sphere objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import min
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import around

# define objective function
def obj(x):
    return x[0]**2.0 + x[1]**2.0

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])

# define boundary check operation
def check_bounds(mutated, bounds):
    # 获取长度 / Get length
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound

# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    # 生成整数序列 / Generate integer sequence
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    # 获取长度 / Get length
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # run iterations of the algorithm
    # 生成整数序列 / Generate integer sequence
    for i in range(iter):
        # iterate over all candidate solutions
        # 生成整数序列 / Generate integer sequence
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            # 生成整数序列 / Generate integer sequence
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            # 获取长度 / Get length
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # report progress at each iteration
            # 打印输出 / Print output
            print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj]

# define population size
pop_size = 10
# define lower and upper bounds for every dimension
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
# define number of iterations
iter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr)
# 打印输出 / Print output
print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))
```

---

➡️ **Next / 下一步**: File 2 of 2

---

### Plot Evaluation

# 18 — Plot Evaluation / 模型评估

**Chapter 19 — File 2 of 2 / 第19章 — 第2个文件（共2个）**

---

## Summary / 总结

This script demonstrates **differential evolution search of the two-dimensional sphere objective function**.

本脚本演示 **differential evolution search of the two-dimensional sphere objective function**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


---
## Step 1 — differential evolution search of the two-dimensional sphere objective function

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import min
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import around
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot
```

---
## Step 2 — define objective function

```python
def obj(x):
    return x[0]**2.0 + x[1]**2.0
```

---
## Step 3 — define mutation operation

```python
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])
```

---
## Step 4 — define boundary check operation

```python
def check_bounds(mutated, bounds):
    # 获取长度 / Get length
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound
```

---
## Step 5 — define crossover operation

```python
def crossover(mutated, target, dims, cr):
```

---
## Step 6 — generate a uniform random value for every dimension

```python
p = rand(dims)
```

---
## Step 7 — generate trial vector by binomial crossover

```python
# 生成整数序列 / Generate integer sequence
trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
```

---
## Step 8 — initialise population of candidate solutions randomly within the specified bounds

```python
# 获取长度 / Get length
pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
```

---
## Step 9 — evaluate initial population of candidate solutions

```python
obj_all = [obj(ind) for ind in pop]
```

---
## Step 10 — find the best performing vector of initial population

```python
best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
```

---
## Step 11 — initialise list to store the objective function value at each iteration

```python
obj_iter = list()
```

---
## Step 12 — run iterations of the algorithm

```python
# 生成整数序列 / Generate integer sequence
for i in range(iter):
```

---
## Step 13 — iterate over all candidate solutions

```python
# 生成整数序列 / Generate integer sequence
for j in range(pop_size):
```

---
## Step 14 — choose three candidates, a, b and c, that are not the current one

```python
# 生成整数序列 / Generate integer sequence
candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
```

---
## Step 15 — perform mutation

```python
mutated = mutation([a, b, c], F)
```

---
## Step 16 — check that lower and upper bounds are retained after mutation

```python
mutated = check_bounds(mutated, bounds)
```

---
## Step 17 — perform crossover

```python
# 获取长度 / Get length
trial = crossover(mutated, pop[j], len(bounds), cr)
```

---
## Step 18 — compute objective function value for target vector

```python
obj_target = obj(pop[j])
```

---
## Step 19 — compute objective function value for trial vector

```python
obj_trial = obj(trial)
```

---
## Step 20 — perform selection

```python
if obj_trial < obj_target:
```

---
## Step 21 — replace the target vector with the trial vector

```python
pop[j] = trial
```

---
## Step 22 — store the new objective function value

```python
obj_all[j] = obj_trial
```

---
## Step 23 — find the best performing vector at each iteration

```python
best_obj = min(obj_all)
```

---
## Step 24 — store the lowest objective function value

```python
if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # 添加元素到列表末尾 / Append element to list end
            obj_iter.append(best_obj)
```

---
## Step 25 — report progress at each iteration

```python
# 打印输出 / Print output
print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter]
```

---
## Step 26 — define population size

```python
pop_size = 10
```

---
## Step 27 — define lower and upper bounds for every dimension

```python
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
```

---
## Step 28 — define number of iterations

```python
iter = 100
```

---
## Step 29 — define scale factor for mutation

```python
F = 0.5
```

---
## Step 30 — define crossover rate for recombination

```python
cr = 0.7
```

---
## Step 31 — perform differential evolution

```python
solution = differential_evolution(pop_size, bounds, iter, F, cr)
# 打印输出 / Print output
print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))
```

---
## Step 32 — line plot of best objective function values

```python
pyplot.plot(solution[2], '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
```

---
## Learning Notes / 学习笔记

- **概念**: differential evolution search of the two-dimensional sphere objective function 是机器学习中的常用技术。  
  *differential evolution search of the two-dimensional sphere objective function is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot Evaluation / 模型评估
# Complete Code / 完整代码
# ===============================

# differential evolution search of the two-dimensional sphere objective function
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import rand
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy.random import choice
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import asarray
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import clip
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import argmin
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import min
# 导入NumPy数值计算库 / Import NumPy numerical computing library
from numpy import around
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
from matplotlib import pyplot

# define objective function
def obj(x):
    return x[0]**2.0 + x[1]**2.0

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])

# define boundary check operation
def check_bounds(mutated, bounds):
    # 获取长度 / Get length
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound

# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    # 生成整数序列 / Generate integer sequence
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    # 获取长度 / Get length
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # initialise list to store the objective function value at each iteration
    obj_iter = list()
    # run iterations of the algorithm
    # 生成整数序列 / Generate integer sequence
    for i in range(iter):
        # iterate over all candidate solutions
        # 生成整数序列 / Generate integer sequence
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            # 生成整数序列 / Generate integer sequence
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            # 获取长度 / Get length
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # 添加元素到列表末尾 / Append element to list end
            obj_iter.append(best_obj)
            # report progress at each iteration
            # 打印输出 / Print output
            print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return [best_vector, best_obj, obj_iter]

# define population size
pop_size = 10
# define lower and upper bounds for every dimension
bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
# define number of iterations
iter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr)
# 打印输出 / Print output
print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))

# line plot of best objective function values
pyplot.plot(solution[2], '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
```

---

### Chapter Summary / 章节总结

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **2 code files** demonstrating chapter 19.

本章包含 **2 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `15_diff_evolution.ipynb` — Diff Evolution
  2. `18_plot_evaluation.ipynb` — Plot Evaluation

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
