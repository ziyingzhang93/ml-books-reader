# 线性代数与机器学习
## Appendix 02

---

### Versions

```python
# Appendix 02.1 — Library Versions / 库版本

**Appendix 02 — Library Version Information / 附录02 — 库版本信息**

## Summary / 总结

Display the versions of all major libraries used in this course. Version information is important for reproducibility and troubleshooting.

显示本课程中使用的所有主要库的版本。版本信息对于可重复性和故障排除很重要。

## Why Version Tracking Matters / 为什么版本跟踪很重要

- **Reproducibility**: Others can recreate your environment with exact versions
- **Compatibility**: Different versions may have breaking changes
- **Bug Fixes**: Specific bugs may be fixed in certain versions
- **Best Practices**: Use version control (requirements.txt, environment.yml)
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Print Core Libraries / 打印核心库

```python
# Import all major libraries and display versions
# 导入所有主要库并显示版本

import numpy
import pandas
import matplotlib
import sklearn
import scipy
import sys

print("=" * 70)
print("LINEAR ALGEBRA FOR MACHINE LEARNING - LIBRARY VERSIONS")
print("=" * 70)
print()

# Python version
print(f"Python Version")
print(f"  {sys.version}")
print()

# Core scientific computing
print("Core Scientific Computing Libraries")
print(f"  NumPy:       {numpy.__version__}")
print(f"  SciPy:       {scipy.__version__}")
print(f"  Pandas:      {pandas.__version__}")
print()

# Machine Learning
print("Machine Learning Libraries")
print(f"  scikit-learn: {sklearn.__version__}")
print()

# Visualization
print("Visualization Libraries")
print(f"  Matplotlib:  {matplotlib.__version__}")
print()
```

## Step 2 — Check Optional Libraries / 检查可选库

```python
# Check for optional libraries used in the course
# 检查课程中使用的可选库

optional_libs = [
    'PIL',  # Image processing
    'seaborn',  # Visualization
    'plotly',  # Interactive plots
    'tensorflow',  # Deep learning
    'torch',  # PyTorch
    'requests',  # HTTP library
]

print("Optional Libraries (for extended functionality)")
for lib_name in optional_libs:
    try:
        lib = __import__(lib_name)
        version = lib.__version__ if hasattr(lib, '__version__') else 'installed'
        print(f"  {lib_name:15} {version}")
    except ImportError:
        print(f"  {lib_name:15} [not installed]")
print()
```

## Step 3 — Detailed Version Information / 详细版本信息

```python
# Display detailed information about each library
# 显示每个库的详细信息

print("=" * 70)
print("DETAILED LIBRARY INFORMATION")
print("=" * 70)
print()

print("NumPy (Numerical Computing)")
print(f"  Version: {numpy.__version__}")
print(f"  Purpose: Array operations, linear algebra")
print(f"  Used for: Core mathematical operations in course")
print()

print("Pandas (Data Manipulation)")
print(f"  Version: {pandas.__version__}")
print(f"  Purpose: Data structures and analysis")
print(f"  Used for: Working with World Bank and other tabular data")
print()

print("scikit-learn (Machine Learning)")
print(f"  Version: {sklearn.__version__}")
print(f"  Purpose: Machine learning algorithms and tools")
print(f"  Used for: PCA, SVM, preprocessing, distance metrics")
print()

print("Matplotlib (Visualization)")
print(f"  Version: {matplotlib.__version__}")
print(f"  Purpose: Static and interactive plotting")
print(f"  Used for: Data visualization throughout course")
print()

print("SciPy (Scientific Computing)")
print(f"  Version: {scipy.__version__}")
print(f"  Purpose: Advanced mathematical functions")
print(f"  Used for: Eigendecomposition, distance metrics")
print()
```

## Step 4 — System Information / 系统信息

```python
# Display system information
# 显示系统信息

import platform

print("=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)
print()

print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Machine Type:     {platform.machine()}")
print(f"Processor:        {platform.processor()}")
print()

# Check Python implementation
print(f"Python Implementation: {platform.python_implementation()}")
print(f"Python Compiler:       {platform.python_compiler()}")
print()
```

## Step 5 — Create Requirements File / 创建需求文件

```python
# Code to create a requirements.txt file for reproducibility
# 创建requirements.txt文件以实现可重复性的代码

example_code = '''
# Save this as requirements.txt in your project directory
# Then install with: pip install -r requirements.txt

# Core Dependencies
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=1.0.0

# Optional Dependencies
Pillow>=8.0.0        # Image processing
seaborn>=0.11.0      # Statistical visualization
jupyter>=1.0.0       # Jupyter notebooks
jupyterlab>=3.0.0    # Jupyter Lab

# Data Access
pandas-datareader>=0.10.0  # World Bank data download
requests>=2.25.0           # HTTP library
'''

print("Example requirements.txt (for reproducibility):")
print("=" * 70)
print(example_code)
print()
print("To create your environment:")
print("  1. Save above as 'requirements.txt'")
print("  2. Run: pip install -r requirements.txt")
print("  3. Share requirements.txt with others for reproducibility")
```

```python
## Step 6 — Check for Compatibility Issues / 检查兼容性问题
```

```python
# Check for known version compatibility issues
# 检查已知的版本兼容性问题

print("=" * 70)
print("VERSION COMPATIBILITY CHECK")
print("=" * 70)
print()

def check_version(module, min_version, name):
    """
    Check if module version meets minimum requirement
    检查模块版本是否满足最小要求
    """
    version = module.__version__
    min_parts = tuple(map(int, min_version.split('.')))
    version_parts = tuple(map(int, version.split('.')[:len(min_parts)]))
    
    if version_parts >= min_parts:
        status = "✓ OK"
    else:
        status = "⚠ UPDATE RECOMMENDED"
    
    print(f"{name:20} Current: {version:15} Minimum: {min_version:10} {status}")

print("Recommended minimum versions:")
check_version(numpy, '1.20.0', 'NumPy')
check_version(pandas, '1.3.0', 'Pandas')
check_version(sklearn, '0.24.0', 'scikit-learn')
check_version(matplotlib, '3.3.0', 'Matplotlib')
check_version(scipy, '1.5.0', 'SciPy')
print()
print("Note: If any show ⚠, consider updating for better compatibility")
```

```python
## Learning Notes / 学习笔记

- **Reproducibility**: Documenting library versions is essential for reproducible research. Small version changes can introduce breaking changes affecting your code.
  
  **可重复性**：记录库版本对于可重复研究至关重要。小的版本更改可能引入破坏性更改。

- **ML Application**: (1) Always pin versions in production (requirements.txt, Docker, conda env), (2) Use version control for both code and dependencies, (3) Regular updates maintain security and performance, but test thoroughly before updating production systems, (4) Document known incompatibilities between libraries, (5) For teaching/coursework, specify exact versions to ensure all students get same results.
  
  **ML应用**：(1) 在生产环境中始终固定版本，(2) 对代码和依赖项使用版本控制，(3) 定期更新以保持安全性和性能，(4) 文档说明库之间已知的不兼容性，(5) 对于教学，指定确切版本以确保结果一致。
```

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `SVM` | 支持向量机 | Support Vector Machine |
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

```python
# --- Import and Display All Versions / 导入并显示所有版本 ---
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import scipy
import sys
import platform

print("Python Version:")
print(f"  {sys.version}")
print()
print("Core Libraries:")
print(f"  NumPy:        {np.__version__}")
print(f"  Pandas:       {pd.__version__}")
print(f"  SciPy:        {scipy.__version__}")
print(f"  Matplotlib:   {matplotlib.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print()
print("System:")
print(f"  OS:           {platform.system()} {platform.release()}")
print(f"  Machine:      {platform.machine()}")
```

---

### Chapter Summary

# Appendix 02 Summary / 附录02总结：Library Versions

## Theme / 主题

This appendix documents the versions of libraries used in this book's code. It's a utility reference for reproducing the exact environment and ensuring code compatibility. Version information is crucial for debugging when code behavior differs.

本附录记录了本书代码中使用的库的版本。这是用于重现确切环境并确保代码兼容性的实用参考。当代码行为不同时，版本信息对于调试至关重要。

## Contents / 内容

```
Library Versions Used:
- NumPy: numerical arrays and linear algebra
- SciPy: scientific computing, sparse matrices, advanced linear algebra
- Scikit-learn: machine learning algorithms
- Matplotlib: visualization
- Jupyter: interactive notebooks
- Python: core language

Documented in: version check notebook
```

## Purpose / 目的

When code works on one machine but fails on another:
1. Check library versions using this reference
2. Install matching versions if needed
3. Report versions when asking for help (Stack Overflow, GitHub issues)

This appendix ensures reproducibility—a cornerstone of scientific computing and open-source code.

当代码在一台机器上工作但在另一台机器上失败时：
1. 使用此参考检查库版本
2. 如果需要，安装匹配的版本
3. 寻求帮助时报告版本（Stack Overflow、GitHub问题）

此附录确保了再现性——科学计算和开源代码的基石。

---
