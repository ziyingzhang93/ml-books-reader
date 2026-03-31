# Python 机器学习 / Python for Machine Learning
## Chapter 06

---

### Syntax

# 01 — Syntax / 01 Syntax

**Chapter 06 — File 1 of 21 / 第06章 — 第1个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Syntax**.

本脚本演示 **01 Syntax**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader as pdr
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader.wb

df = (
    pdr.wb
    .download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
    .reset_index()
    .filter(["country", "SP.POP.TOTL"])
    .groupby("country")
    .mean()
)
# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Syntax 是机器学习中的常用技术。  
  *Syntax is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Syntax / 01 Syntax
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader as pdr
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader.wb

df = (
    pdr.wb
    .download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
    .reset_index()
    .filter(["country", "SP.POP.TOTL"])
    .groupby("country")
    .mean()
)
# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 2 of 21

---

### Imperative

# 02 — Imperative / 02 Imperative

**Chapter 06 — File 2 of 21 / 第06章 — 第2个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Imperative**.

本脚本演示 **02 Imperative**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader as pdr
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader.wb

df = pdr.wb.download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
df = df.reset_index()
df = df.filter(["country", "SP.POP.TOTL"])
groups = df.groupby("country")
df = groups.mean()

# 打印输出 / Print output
print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Imperative 是机器学习中的常用技术。  
  *Imperative is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `groupby` | 分组聚合 | Group and aggregate |
| `pandas` | 数据分析库 | Data analysis library |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Imperative / 02 Imperative
# Complete Code / 完整代码
# ===============================

# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader as pdr
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas_datareader.wb

df = pdr.wb.download(indicator="SP.POP.TOTL", country="all", start=2000, end=2020)
df = df.reset_index()
df = df.filter(["country", "SP.POP.TOTL"])
groups = df.groupby("country")
df = groups.mean()

# 打印输出 / Print output
print(df)
```

---

➡️ **Next / 下一步**: File 3 of 21

---

### Readlog

# 03 — Readlog / 03 Readlog

**Chapter 06 — File 3 of 21 / 第06章 — 第3个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Read the log file, split into lines**.

本脚本演示 **Read the log file, split into lines**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Step 1

```python
import urllib.request
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — Read the log file, split into lines

```python
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()
```

---
## Step 3 — using regular expression to extract IP address and status code from a line

```python
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

# 打印输出 / Print output
print(ip_and_code(lines[0]))
```

---
## Learning Notes / 学习笔记

- **概念**: Read the log file, split into lines 是机器学习中的常用技术。  
  *Read the log file, split into lines is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Readlog / 03 Readlog
# Complete Code / 完整代码
# ===============================

import urllib.request
# 导入正则表达式模块 / Import regex module
import re

# Read the log file, split into lines
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

# 打印输出 / Print output
print(ip_and_code(lines[0]))
```

---

➡️ **Next / 下一步**: File 4 of 21

---

### Maxip

# 04 — Maxip / 04 Maxip

**Chapter 06 — File 4 of 21 / 第06章 — 第4个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Read the log file, split into lines**.

本脚本演示 **Read the log file, split into lines**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Step 1

```python
import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — Read the log file, split into lines

```python
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()
```

---
## Step 3 — using regular expression to extract IP address and status code from a line

```python
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)
```

---
## Step 4 — transform each line into (IP address, status code) pair

```python
ipcodepairs = map(ip_and_code, lines)
```

---
## Step 5 — keep only those with status code 404

```python
pairs404 = filter(is404, ipcodepairs)
```

---
## Step 6 — extract the IP address part from each pair

```python
ip404 = map(getIP, pairs404)
```

---
## Step 7 — count the occurrences, the result is a dictionary of IP addresses map to the count

```python
ipcount = collections.Counter(ip404)
```

---
## Step 8 — convert the (IP address, count) tuple into (count, IP address) order

```python
# 获取字典的键值对 / Get dict key-value pairs
countip = map(count_ip, ipcount.items())
```

---
## Step 9 — find the tuple with the maximum on the count

```python
# 打印输出 / Print output
print(max(countip))
```

---
## Learning Notes / 学习笔记

- **概念**: Read the log file, split into lines 是机器学习中的常用技术。  
  *Read the log file, split into lines is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Maxip / 04 Maxip
# Complete Code / 完整代码
# ===============================

import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re

# Read the log file, split into lines
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)

# transform each line into (IP address, status code) pair
ipcodepairs = map(ip_and_code, lines)
# keep only those with status code 404
pairs404 = filter(is404, ipcodepairs)
# extract the IP address part from each pair
ip404 = map(getIP, pairs404)
# count the occurrences, the result is a dictionary of IP addresses map to the count
ipcount = collections.Counter(ip404)
# convert the (IP address, count) tuple into (count, IP address) order
# 获取字典的键值对 / Get dict key-value pairs
countip = map(count_ip, ipcount.items())
# find the tuple with the maximum on the count
# 打印输出 / Print output
print(max(countip))
```

---

➡️ **Next / 下一步**: File 5 of 21

---

### Maxip

# 05 — Maxip / 05 Maxip

**Chapter 06 — File 5 of 21 / 第06章 — 第5个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Read the log file, split into lines**.

本脚本演示 **Read the log file, split into lines**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Step 1

```python
import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — Read the log file, split into lines

```python
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()
```

---
## Step 3 — using regular expression to extract IP address and status code from a line

```python
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)

ipcodepairs = [ip_and_code(x) for x in lines]
ip404 = [ip for ip,code in ipcodepairs if code=="404"]
ipcount = collections.Counter(ip404)
# 获取字典的键值对 / Get dict key-value pairs
countip = [(count,ip) for ip,count in ipcount.items()]
# 打印输出 / Print output
print(max(countip))
```

---
## Learning Notes / 学习笔记

- **概念**: Read the log file, split into lines 是机器学习中的常用技术。  
  *Read the log file, split into lines is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Maxip / 05 Maxip
# Complete Code / 完整代码
# ===============================

import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re

# Read the log file, split into lines
logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"
logfile = urllib.request.urlopen(logurl).read().decode("utf8")
lines = logfile.splitlines()

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

def is404(pair):
    return pair[1] == "404"
def getIP(pair):
    return pair[0]
def count_ip(count_item):
    ip, count = count_item
    return (count, ip)

ipcodepairs = [ip_and_code(x) for x in lines]
ip404 = [ip for ip,code in ipcodepairs if code=="404"]
ipcount = collections.Counter(ip404)
# 获取字典的键值对 / Get dict key-value pairs
countip = [(count,ip) for ip,count in ipcount.items()]
# 打印输出 / Print output
print(max(countip))
```

---

➡️ **Next / 下一步**: File 6 of 21

---

### Maxip

# 06 — Maxip / 06 Maxip

**Chapter 06 — File 6 of 21 / 第06章 — 第6个文件（共21个）**

---

## Summary / 总结

This script demonstrates **using regular expression to extract IP address and status code from a line**.

本脚本演示 **using regular expression to extract IP address and status code from a line**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
```

---
## Step 1 — Step 1

```python
import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re
```

---
## Step 2 — using regular expression to extract IP address and status code from a line

```python
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"

# 打印输出 / Print output
print(
    max(
        [(count,ip) for ip,count in
            collections.Counter([
                ip for ip, code in
                [ip_and_code(x) for x in
                     urllib.request.urlopen(logurl)
                     .read()
                     .decode("utf8")
                     .splitlines()
                ]
                if code=="404"
            # 获取字典的键值对 / Get dict key-value pairs
            ]).items()
        ]
    )
)
```

---
## Learning Notes / 学习笔记

- **概念**: using regular expression to extract IP address and status code from a line 是机器学习中的常用技术。  
  *using regular expression to extract IP address and status code from a line is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Maxip / 06 Maxip
# Complete Code / 完整代码
# ===============================

import collections
import urllib.request
# 导入正则表达式模块 / Import regex module
import re

# using regular expression to extract IP address and status code from a line
def ip_and_code(logline):
    m = re.match(r'([\d\.]+) .*? \[.*?\] ".*?" (\d+) ', logline)
    return (m.group(1), m.group(2))

logurl = "https://raw.githubusercontent.com/elastic/examples/master/" \
             "Common%20Data%20Formats/apache_logs/apache_logs"

# 打印输出 / Print output
print(
    max(
        [(count,ip) for ip,count in
            collections.Counter([
                ip for ip, code in
                [ip_and_code(x) for x in
                     urllib.request.urlopen(logurl)
                     .read()
                     .decode("utf8")
                     .splitlines()
                ]
                if code=="404"
            # 获取字典的键值对 / Get dict key-value pairs
            ]).items()
        ]
    )
)
```

---

➡️ **Next / 下一步**: File 7 of 21

---

### Count

# 07 — Count / 07 Count

**Chapter 06 — File 7 of 21 / 第06章 — 第7个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Count**.

本脚本演示 **07 Count**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

start = 0
step = 100
for i in itertools.count(start, step):
    # 打印输出 / Print output
    print(i)
    if i >= 1000:
        break
```

---
## Learning Notes / 学习笔记

- **概念**: Count 是机器学习中的常用技术。  
  *Count is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Count / 07 Count
# Complete Code / 完整代码
# ===============================

import itertools

start = 0
step = 100
for i in itertools.count(start, step):
    # 打印输出 / Print output
    print(i)
    if i >= 1000:
        break
```

---

➡️ **Next / 下一步**: File 8 of 21

---

### Cycle

# 08 — Cycle / 08 Cycle

**Chapter 06 — File 8 of 21 / 第06章 — 第8个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Cycle**.

本脚本演示 **08 Cycle**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

counter = 0
cyclic_list = [1, 2, 3, 4, 5]

for i in itertools.cycle(cyclic_list):
    # 打印输出 / Print output
    print(i)
    counter = counter+1
    if counter>10:
        break
```

---
## Learning Notes / 学习笔记

- **概念**: Cycle 是机器学习中的常用技术。  
  *Cycle is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cycle / 08 Cycle
# Complete Code / 完整代码
# ===============================

import itertools

counter = 0
cyclic_list = [1, 2, 3, 4, 5]

for i in itertools.cycle(cyclic_list):
    # 打印输出 / Print output
    print(i)
    counter = counter+1
    if counter>10:
        break
```

---

➡️ **Next / 下一步**: File 9 of 21

---

### Repeat

# 09 — Repeat / 09 Repeat

**Chapter 06 — File 9 of 21 / 第06章 — 第9个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Repeat**.

本脚本演示 **09 Repeat**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

for i in itertools.repeat(3,5):
    # 打印输出 / Print output
    print(i)
```

---
## Learning Notes / 学习笔记

- **概念**: Repeat 是机器学习中的常用技术。  
  *Repeat is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Repeat / 09 Repeat
# Complete Code / 完整代码
# ===============================

import itertools

for i in itertools.repeat(3,5):
    # 打印输出 / Print output
    print(i)
```

---

➡️ **Next / 下一步**: File 10 of 21

---

### Product

# 10 — Product / 10 Product

**Chapter 06 — File 10 of 21 / 第06章 — 第10个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Product**.

本脚本演示 **10 Product**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

x = [1, 2, 3]
y = ['A', 'B']
for t in itertools.product(x, y):
    # 打印输出 / Print output
    print(t)
```

---
## Learning Notes / 学习笔记

- **概念**: Product 是机器学习中的常用技术。  
  *Product is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Product / 10 Product
# Complete Code / 完整代码
# ===============================

import itertools

x = [1, 2, 3]
y = ['A', 'B']
for t in itertools.product(x, y):
    # 打印输出 / Print output
    print(t)
```

---

➡️ **Next / 下一步**: File 11 of 21

---

### Permutations

# 11 — Permutations / 11 Permutations

**Chapter 06 — File 11 of 21 / 第06章 — 第11个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Permutations**.

本脚本演示 **11 Permutations**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

x = [1, 2, 3]
for t in itertools.permutations(x):
    # 打印输出 / Print output
    print(t)
```

---
## Learning Notes / 学习笔记

- **概念**: Permutations 是机器学习中的常用技术。  
  *Permutations is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Permutations / 11 Permutations
# Complete Code / 完整代码
# ===============================

import itertools

x = [1, 2, 3]
for t in itertools.permutations(x):
    # 打印输出 / Print output
    print(t)
```

---

➡️ **Next / 下一步**: File 12 of 21

---

### Combinations

# 12 — Combinations / 12 Combinations

**Chapter 06 — File 12 of 21 / 第06章 — 第12个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Combinations**.

本脚本演示 **12 Combinations**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

x = ['A', 'B', 'C', 'D']
for t in itertools.combinations(x, 3):
    # 打印输出 / Print output
    print(t)
```

---
## Learning Notes / 学习笔记

- **概念**: Combinations 是机器学习中的常用技术。  
  *Combinations is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Combinations / 12 Combinations
# Complete Code / 完整代码
# ===============================

import itertools

x = ['A', 'B', 'C', 'D']
for t in itertools.combinations(x, 3):
    # 打印输出 / Print output
    print(t)
```

---

➡️ **Next / 下一步**: File 13 of 21

---

### Cwr

# 13 — Cwr / 13 Cwr

**Chapter 06 — File 13 of 21 / 第06章 — 第13个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Cwr**.

本脚本演示 **13 Cwr**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

x = ['A', 'B', 'C']
for t in itertools.combinations_with_replacement(x, 2):
    # 打印输出 / Print output
    print(t)
```

---
## Learning Notes / 学习笔记

- **概念**: Cwr 是机器学习中的常用技术。  
  *Cwr is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cwr / 13 Cwr
# Complete Code / 完整代码
# ===============================

import itertools

x = ['A', 'B', 'C']
for t in itertools.combinations_with_replacement(x, 2):
    # 打印输出 / Print output
    print(t)
```

---

➡️ **Next / 下一步**: File 14 of 21

---

### Accumulate

# 14 — Accumulate / 14 Accumulate

**Chapter 06 — File 14 of 21 / 第06章 — 第14个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Custom operator**.

本脚本演示 **Custom operator**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools
import operator
```

---
## Step 2 — Custom operator

```python
def my_operator(a, b):
    return a+b if a>5 else a-b

x = [2, 3, 4, -6]
mul_result = itertools.accumulate(x, operator.mul)
# 打印输出 / Print output
print("After mul operator", list(mul_result))
pow_result = itertools.accumulate(x, operator.pow)
# 打印输出 / Print output
print("After pow operator", list(pow_result))
my_operator_result = itertools.accumulate(x, my_operator)
# 打印输出 / Print output
print("After customized my_operator", list(my_operator_result))
```

---
## Learning Notes / 学习笔记

- **概念**: Custom operator 是机器学习中的常用技术。  
  *Custom operator is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Accumulate / 14 Accumulate
# Complete Code / 完整代码
# ===============================

import itertools
import operator

# Custom operator
def my_operator(a, b):
    return a+b if a>5 else a-b

x = [2, 3, 4, -6]
mul_result = itertools.accumulate(x, operator.mul)
# 打印输出 / Print output
print("After mul operator", list(mul_result))
pow_result = itertools.accumulate(x, operator.pow)
# 打印输出 / Print output
print("After pow operator", list(pow_result))
my_operator_result = itertools.accumulate(x, my_operator)
# 打印输出 / Print output
print("After customized my_operator", list(my_operator_result))
```

---

➡️ **Next / 下一步**: File 15 of 21

---

### Starmap

# 15 — Starmap / 15 Starmap

**Chapter 06 — File 15 of 21 / 第06章 — 第15个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Starmap**.

本脚本演示 **15 Starmap**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools
import operator

pair_list = [(1, 2), (4, 0.5), (5, 7), (100, 10)]

starmap_add_result = itertools.starmap(operator.add, pair_list)
# 打印输出 / Print output
print("Starmap add result: ", list(starmap_add_result))

x1 = [2, 3, 4, -6]
x2 = [4, 3, 2, 1]

# 将多个序列配对 / Pair multiple sequences
starmap_mul_result = itertools.starmap(operator.mul, zip(x1, x2))
# 打印输出 / Print output
print("Starmap mul result: ", list(starmap_mul_result))
```

---
## Learning Notes / 学习笔记

- **概念**: Starmap 是机器学习中的常用技术。  
  *Starmap is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Starmap / 15 Starmap
# Complete Code / 完整代码
# ===============================

import itertools
import operator

pair_list = [(1, 2), (4, 0.5), (5, 7), (100, 10)]

starmap_add_result = itertools.starmap(operator.add, pair_list)
# 打印输出 / Print output
print("Starmap add result: ", list(starmap_add_result))

x1 = [2, 3, 4, -6]
x2 = [4, 3, 2, 1]

# 将多个序列配对 / Pair multiple sequences
starmap_mul_result = itertools.starmap(operator.mul, zip(x1, x2))
# 打印输出 / Print output
print("Starmap mul result: ", list(starmap_mul_result))
```

---

➡️ **Next / 下一步**: File 16 of 21

---

### Filterfalse

# 16 — Filterfalse / 16 Filterfalse

**Chapter 06 — File 16 of 21 / 第06章 — 第16个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Filterfalse**.

本脚本演示 **16 Filterfalse**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import itertools

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_result = itertools.filterfalse(lambda x: x%2, my_list)
small_terms = itertools.filterfalse(lambda x: x>=5, my_list)
# 打印输出 / Print output
print('Even result:', list(even_result))
# 打印输出 / Print output
print('Less than 5:', list(small_terms))
```

---
## Learning Notes / 学习笔记

- **概念**: Filterfalse 是机器学习中的常用技术。  
  *Filterfalse is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Filterfalse / 16 Filterfalse
# Complete Code / 完整代码
# ===============================

import itertools

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_result = itertools.filterfalse(lambda x: x%2, my_list)
small_terms = itertools.filterfalse(lambda x: x>=5, my_list)
# 打印输出 / Print output
print('Even result:', list(even_result))
# 打印输出 / Print output
print('Less than 5:', list(small_terms))
```

---

➡️ **Next / 下一步**: File 17 of 21

---

### Lru

# 17 — Lru / 17 Lru

**Chapter 06 — File 17 of 21 / 第06章 — 第17个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Lru**.

本脚本演示 **17 Lru**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import functools

@functools.lru_cache()
def fib(n):
    global count
    count = count + 1
    return fib(n-2) + fib(n-1) if n>1 else 1

def fib_slow(n):
    global slow_count
    slow_count = slow_count + 1
    return fib_slow(n-2) + fib_slow(n-1) if n>1 else 1

count = 0
slow_count = 0
fib(30)
fib_slow(30)

# 打印输出 / Print output
print('With lru_cache total function evaluations: ', count)
# 打印输出 / Print output
print('Without lru_cache total function evaluations: ', slow_count)
```

---
## Learning Notes / 学习笔记

- **概念**: Lru 是机器学习中的常用技术。  
  *Lru is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Lru / 17 Lru
# Complete Code / 完整代码
# ===============================

import functools

@functools.lru_cache()
def fib(n):
    global count
    count = count + 1
    return fib(n-2) + fib(n-1) if n>1 else 1

def fib_slow(n):
    global slow_count
    slow_count = slow_count + 1
    return fib_slow(n-2) + fib_slow(n-1) if n>1 else 1

count = 0
slow_count = 0
fib(30)
fib_slow(30)

# 打印输出 / Print output
print('With lru_cache total function evaluations: ', count)
# 打印输出 / Print output
print('Without lru_cache total function evaluations: ', slow_count)
```

---

➡️ **Next / 下一步**: File 18 of 21

---

### Reduce

# 18 — Reduce / 18 Reduce

**Chapter 06 — File 18 of 21 / 第06章 — 第18个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Evaluates ((1+2)+3)+4**.

本脚本演示 **Evaluates ((1+2)+3)+4**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import functools
import operator
```

---
## Step 2 — Evaluates ((1+2)+3)+4

```python
list_sum = functools.reduce(operator.add, [1, 2, 3, 4])
# 打印输出 / Print output
print(list_sum)
```

---
## Step 3 — Evaluates (2^3)^4

```python
list_pow = functools.reduce(operator.pow, [2, 3, 4])
# 打印输出 / Print output
print(list_pow)
```

---
## Learning Notes / 学习笔记

- **概念**: Evaluates ((1+2)+3)+4 是机器学习中的常用技术。  
  *Evaluates ((1+2)+3)+4 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Reduce / 18 Reduce
# Complete Code / 完整代码
# ===============================

import functools
import operator

# Evaluates ((1+2)+3)+4
list_sum = functools.reduce(operator.add, [1, 2, 3, 4])
# 打印输出 / Print output
print(list_sum)

# Evaluates (2^3)^4
list_pow = functools.reduce(operator.pow, [2, 3, 4])
# 打印输出 / Print output
print(list_pow)
```

---

➡️ **Next / 下一步**: File 19 of 21

---

### Counter

# 19 — Counter / 19 Counter

**Chapter 06 — File 19 of 21 / 第06章 — 第19个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Counter**.

本脚本演示 **19 Counter**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import functools

def addcount(counter, element):
    if element not in counter:
        counter[element] = 1
    else:
        counter[element] += 1
    return counter

items = ["a", "b", "a", "c", "d", "c", "b", "a"]

counts = functools.reduce(addcount, items, {})
# 打印输出 / Print output
print(counts)
```

---
## Learning Notes / 学习笔记

- **概念**: Counter 是机器学习中的常用技术。  
  *Counter is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Counter / 19 Counter
# Complete Code / 完整代码
# ===============================

import functools

def addcount(counter, element):
    if element not in counter:
        counter[element] = 1
    else:
        counter[element] += 1
    return counter

items = ["a", "b", "a", "c", "d", "c", "b", "a"]

counts = functools.reduce(addcount, items, {})
# 打印输出 / Print output
print(counts)
```

---

➡️ **Next / 下一步**: File 20 of 21

---

### Power2

# 20 — Power2 / 20 Power2

**Chapter 06 — File 20 of 21 / 第06章 — 第20个文件（共21个）**

---

## Summary / 总结

This script demonstrates **Power2**.

本脚本演示 **20 Power2**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import functools
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

power_2 = functools.partial(np.power, 2)
# 打印输出 / Print output
print('2^4 =', power_2(4))
# 打印输出 / Print output
print('2^6 =', power_2(6))
```

---
## Learning Notes / 学习笔记

- **概念**: Power2 是机器学习中的常用技术。  
  *Power2 is a common technique in machine learning.*

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
# Power2 / 20 Power2
# Complete Code / 完整代码
# ===============================

import functools
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np

power_2 = functools.partial(np.power, 2)
# 打印输出 / Print output
print('2^4 =', power_2(4))
# 打印输出 / Print output
print('2^6 =', power_2(6))
```

---

➡️ **Next / 下一步**: File 21 of 21

---

### Mapreduce

# 21 — Mapreduce / 21 Mapreduce

**Chapter 06 — File 21 of 21 / 第06章 — 第21个文件（共21个）**

---

## Summary / 总结

This script demonstrates **All numbers from 1 to 20**.

本脚本演示 **All numbers from 1 to 20**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import functools
import operator
```

---
## Step 2 — All numbers from 1 to 20

```python
# 生成整数序列 / Generate integer sequence
input_list = list(range(20))
```

---
## Step 3 — Use map to see which numbers are divisible by 3

```python
bool_list = map(lambda x: 1 if x%3==0 else 0, input_list)
```

---
## Step 4 — Convert map object to list

```python
bool_list = list(bool_list)
# 打印输出 / Print output
print('bool_list =', bool_list)

total_divisible_3 = functools.reduce(operator.add, bool_list)
# 打印输出 / Print output
print('Total items divisible by 3 = ', total_divisible_3)
```

---
## Learning Notes / 学习笔记

- **概念**: All numbers from 1 to 20 是机器学习中的常用技术。  
  *All numbers from 1 to 20 is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Mapreduce / 21 Mapreduce
# Complete Code / 完整代码
# ===============================

import functools
import operator

# All numbers from 1 to 20
# 生成整数序列 / Generate integer sequence
input_list = list(range(20))
# Use map to see which numbers are divisible by 3
bool_list = map(lambda x: 1 if x%3==0 else 0, input_list)
# Convert map object to list
bool_list = list(bool_list)
# 打印输出 / Print output
print('bool_list =', bool_list)

total_divisible_3 = functools.reduce(operator.add, bool_list)
# 打印输出 / Print output
print('Total items divisible by 3 = ', total_divisible_3)
```

---

### Chapter Summary / 章节总结

# Chapter 06 Summary / 第06章总结

## Theme / 主题: Chapter 06 / Chapter 06

This chapter contains **21 code files** demonstrating chapter 06.

本章包含 **21 个代码文件**，演示Chapter 06。

---
## Evolution / 演化路线

  1. `01_syntax.ipynb` — Syntax
  2. `02_imperative.ipynb` — Imperative
  3. `03_readlog.ipynb` — Readlog
  4. `04_maxip.ipynb` — Maxip
  5. `05_maxip.ipynb` — Maxip
  6. `06_maxip.ipynb` — Maxip
  7. `07_count.ipynb` — Count
  8. `08_cycle.ipynb` — Cycle
  9. `09_repeat.ipynb` — Repeat
  10. `10_product.ipynb` — Product
  11. `11_permutations.ipynb` — Permutations
  12. `12_combinations.ipynb` — Combinations
  13. `13_cwr.ipynb` — Cwr
  14. `14_accumulate.ipynb` — Accumulate
  15. `15_starmap.ipynb` — Starmap
  16. `16_filterfalse.ipynb` — Filterfalse
  17. `17_lru.ipynb` — Lru
  18. `18_reduce.ipynb` — Reduce
  19. `19_counter.ipynb` — Counter
  20. `20_power2.ipynb` — Power2
  21. `21_mapreduce.ipynb` — Mapreduce

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 06) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 06）是机器学习流水线中的基础构建块。

---
