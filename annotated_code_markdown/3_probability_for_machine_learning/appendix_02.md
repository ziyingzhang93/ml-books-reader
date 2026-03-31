# 概率论与机器学习 / Probability for Machine Learning
## Appendix 02

---

### Python Versions

# Appendix 02 — Library Versions / 库版本

**Chapter Appendix 02 — File 1 of 1**

## Summary / 汇总

This notebook prints the versions of key libraries used in the course. Version tracking is important for reproducibility.

本笔记本打印课程中使用的关键库的版本。版本跟踪对于可重复性很重要。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 可视化结果 / Visualize results


## Step 1 — Print Library Versions / 打印库版本

```python
# check library version numbers# scipyimport scipyprint('scipy: %s' % scipy.__version__)# numpyimport numpyprint('numpy: %s' % numpy.__version__)# matplotlibimport matplotlibprint('matplotlib: %s' % matplotlib.__version__)# pandasimport pandasprint('pandas: %s' % pandas.__version__)# statsmodelsimport statsmodelsprint('statsmodels: %s' % statsmodels.__version__)# scikit-learnimport sklearnprint('sklearn: %s' % sklearn.__version__)
```

## Learning Notes / 学习笔记

- **Concept**: Library versions matter for reproducibility. Updating libraries can change algorithm implementations, fixing bugs or introducing regressions that affect results. **概念**: 库版本对于可重复性很重要。更新库可以改变算法实现，修复错误或引入影响结果的回归。

- **ML Application**: In production, pin exact versions in requirements.txt or conda environment files. This ensures consistency across development, testing, and production environments. **机器学习应用**: 在生产中，在requirements.txt或conda环境文件中固定确切的版本。这确保了开发、测试和生产环境之间的一致性。

➡️ **Next**: `../README.md`

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `matplotlib` | 绑图库 | Plotting library |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |

## Complete Code / 完整代码一览

```python
# 打印输出 / Print output
import scipyprint('scipy: %s' % scipy.__version__)import numpyprint('numpy: %s' % numpy.__version__)import matplotlibprint('matplotlib: %s' % matplotlib.__version__)import pandasprint('pandas: %s' % pandas.__version__)import statsmodelsprint('statsmodels: %s' % statsmodels.__version__)import sklearnprint('sklearn: %s' % sklearn.__version__)
```

---

### Chapter Summary / 章节总结

# Appendix 2: Library Versions

## Overview
This appendix documents the software library versions used throughout the book, ensuring reproducibility and helping readers manage dependencies.

## Key Libraries

### NumPy
- **Role**: Numerical computing, arrays, random number generation
- **Chapters**: Used throughout book
- **Key functions**: random, mean, var, std, linspace

### SciPy
- **Role**: Statistical distributions, scientific computing
- **Chapters**: Ch.8-9 (distributions), Ch.21-22 (entropy/divergence), Ch.26-28 (metrics)
- **Key modules**: scipy.stats, scipy.spatial.distance

### Matplotlib
- **Role**: Data visualization, plotting distributions and curves
- **Chapters**: Used throughout for visualization
- **Key functions**: plt.plot, plt.hist, plt.show

### Scikit-learn
- **Role**: Machine learning algorithms, metrics, preprocessing
- **Chapters**: Ch.18 (Naive Bayes), Ch.25 (DummyClassifier), Ch.26-28 (metrics)
- **Key modules**: sklearn.naive_bayes, sklearn.dummy, sklearn.metrics, sklearn.calibration

### Pandas
- **Role**: Data manipulation, DataFrames
- **Chapters**: Ch.10, Ch.25-28 (data handling)
- **Key functions**: DataFrame, Series

### Keras/TensorFlow
- **Role**: Deep learning, neural networks
- **Chapters**: Ch.23 (cross-entropy loss), Ch.28 (probability calibration)
- **Key modules**: keras.losses

### PyTorch
- **Role**: Alternative deep learning framework
- **Chapters**: Ch.23 (cross-entropy loss)
- **Key modules**: torch.nn

## Minimum Requirements

For running all examples in the book:
- Python 3.6+
- NumPy 1.16+
- SciPy 1.1+
- Matplotlib 3.0+
- Scikit-learn 0.20+
- (Optional) Pandas 0.24+
- (Optional) TensorFlow 2.0+ or PyTorch 1.0+

## Installation

### Using pip
```bash
pip install numpy scipy matplotlib scikit-learn
pip install pandas  # Optional
pip install tensorflow  # Optional
pip install torch  # Optional
```

### Using conda
```bash
conda install numpy scipy matplotlib scikit-learn
conda install pandas  # Optional
conda install tensorflow  # Optional
conda install pytorch  # Optional
```

## Reproducibility

### Setting Random Seeds
For reproducible results:
```python
import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

### Environment
- All code tested on Linux (Ubuntu 20.04)
- Also compatible with macOS and Windows
- Version conflicts rare; any recent version of listed libraries should work

## Chapter-Specific Dependencies

| Chapter | Core Dependencies | Optional |
|---------|-------------------|----------|
| Ch.6 | NumPy | - |
| Ch.8-10 | NumPy, SciPy, Matplotlib | Pandas |
| Ch.13-16 | NumPy, Matplotlib | - |
| Ch.18 | NumPy, Scikit-learn, Matplotlib | - |
| Ch.19 | NumPy, Matplotlib | - |
| Ch.21-24 | NumPy, SciPy, Matplotlib | - |
| Ch.25-28 | NumPy, Scikit-learn, Matplotlib | Pandas |
| Ch.23 | NumPy, Matplotlib | TensorFlow/PyTorch |

## Troubleshooting

### Common Issues

**Import error: No module named 'numpy'**
- Solution: `pip install numpy`

**Matplotlib not showing plots**
- Solution: Use `%matplotlib inline` in Jupyter, or `plt.show()`

**Version incompatibility**
- Solution: Check package versions with `pip list`
- Use `pip install --upgrade package_name` to update

**TensorFlow installation fails**
- Solution: TensorFlow can be finicky. Consult official docs for your OS
- Note: TensorFlow only needed for Ch.23 optional section

## Verification

Check installation with:
```python
import numpy as np
import scipy
import matplotlib
import sklearn

print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

All should return version numbers without errors.

## Key Takeaways
1. Core dependencies: NumPy, SciPy, Matplotlib, Scikit-learn
2. Python 3.6+ required
3. TensorFlow/PyTorch optional (only for deep learning sections)
4. Use `pip` or `conda` for easy installation
5. Set random seeds for reproducibility
6. Check chapter-specific requirements before running examples

---
