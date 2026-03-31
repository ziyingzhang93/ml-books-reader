# 机器学习微积分 / Calculus for Machine Learning
## Chapter 05

---

### Eval Limits

# 01 — Eval Limits / 模型评估

**Chapter 05 — File 1 of 1 / 第05章 — 第1个文件（共1个）**

---

## Summary / 总结

This script demonstrates **Eval Limits**.

本脚本演示 **模型评估**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def g(x):
    return 1-x

x = -1
epsilon = np.finfo(float).eps

# 打印输出 / Print output
print("Left limit is", g(x-epsilon))
# 打印输出 / Print output
print("Right limit is", g(x+epsilon))
```

---
## Learning Notes / 学习笔记

- **概念**: Eval Limits 是机器学习中的常用技术。  
  *Eval Limits is a common technique in machine learning.*

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
# Eval Limits / 模型评估
# Complete Code / 完整代码
# ===============================

# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

def g(x):
    return 1-x

x = -1
epsilon = np.finfo(float).eps

# 打印输出 / Print output
print("Left limit is", g(x-epsilon))
# 打印输出 / Print output
print("Right limit is", g(x+epsilon))
```

---

### Chapter Summary / 章节总结

# Chapter 05 Summary / 第05章总结

## Theme / 主题: Chapter 05 / Chapter 05

This chapter contains **1 code files** demonstrating chapter 05.

本章包含 **1 个代码文件**，演示Chapter 05。

---
## Evolution / 演化路线

  1. `01_eval_limits.ipynb` — Eval Limits

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 05) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 05）是机器学习流水线中的基础构建块。

---
