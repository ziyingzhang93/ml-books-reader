# Python ML
## Chapter 19

---

### Pytest

# 15 — Pytest / 15 Pytest

**Chapter 19 — File 5 of 9 / 第19章 — 第5个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Our code to be tested**.

本脚本演示 **Our code to be tested**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Our code to be tested

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height
```

---
## Step 2 — The test function to be executed by PyTest

```python
def test_normal_case():
    rectangle = Rectangle(2, 3)
    assert rectangle.get_area() == 6, "incorrect area"
```

---
## Learning Notes / 学习笔记

- **概念**: Our code to be tested 是机器学习中的常用技术。  
  *Our code to be tested is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pytest / 15 Pytest
# Complete Code / 完整代码
# ===============================

# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

# The test function to be executed by PyTest
def test_normal_case():
    rectangle = Rectangle(2, 3)
    assert rectangle.get_area() == 6, "incorrect area"
```

---

➡️ **Next / 下一步**: File 6 of 9

---

### Fixture

# 20 — Fixture / 20 Fixture

**Chapter 19 — File 7 of 9 / 第19章 — 第7个文件（共9个）**

---

## Summary / 总结

This script demonstrates **Our code to be tested**.

本脚本演示 **Our code to be tested**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — Step 1

```python
import pytest
```

---
## Step 2 — Our code to be tested

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

@pytest.fixture
def rectangle():
    return Rectangle(0, 0)

def test_negative_case(rectangle):
    print (rectangle.width)
    rectangle.set_width(-1)
    rectangle.set_height(2)
    assert rectangle.get_area() == -1, "incorrect negative output"
```

---
## Learning Notes / 学习笔记

- **概念**: Our code to be tested 是机器学习中的常用技术。  
  *Our code to be tested is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fixture / 20 Fixture
# Complete Code / 完整代码
# ===============================

import pytest

# Our code to be tested
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_area(self):
        return self.width * self.height

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

@pytest.fixture
def rectangle():
    return Rectangle(0, 0)

def test_negative_case(rectangle):
    print (rectangle.width)
    rectangle.set_width(-1)
    rectangle.set_height(2)
    assert rectangle.get_area() == -1, "incorrect negative output"
```

---

➡️ **Next / 下一步**: File 8 of 9

---

### Chapter Summary

# Chapter 19 Summary / 第19章总结

## Theme / 主题: Chapter 19 / Chapter 19

This chapter contains **9 code files** demonstrating chapter 19.

本章包含 **9 个代码文件**，演示Chapter 19。

---
## Evolution / 演化路线

  1. `05_unittest.ipynb` — Unittest
  2. `06_unittest.ipynb` — Unittest
  3. `10_unittest.ipynb` — Unittest
  4. `14_unittest.ipynb` — Unittest
  5. `15_pytest.ipynb` — Pytest
  6. `18_pytest.ipynb` — Pytest
  7. `20_fixture.ipynb` — Fixture
  8. `22_test_download.ipynb` — Test Download
  9. `23_test_download.ipynb` — Test Download

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 19) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 19）是机器学习流水线中的基础构建块。

---
