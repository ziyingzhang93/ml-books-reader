# Python ML
## Chapter 22

---

### Weather

# 01 — Weather / 01 Weather

**Chapter 22 — File 1 of 10 / 第22章 — 第1个文件（共10个）**

---

## Summary / 总结

This script demonstrates **The numbers are lat-lon of New York**.

本脚本演示 **The numbers are lat-lon of New York**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import requests
```

---
## Step 2 — The numbers are lat-lon of New York

```python
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
print(resp.status_code)
print(resp.text)
```

---
## Learning Notes / 学习笔记

- **概念**: The numbers are lat-lon of New York 是机器学习中的常用技术。  
  *The numbers are lat-lon of New York is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Weather / 01 Weather
# Complete Code / 完整代码
# ===============================

import requests

# The numbers are lat-lon of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)
print(resp.status_code)
print(resp.text)
```

---

➡️ **Next / 下一步**: File 2 of 10

---

### Fred

# 02 — Fred / 02 Fred

**Chapter 22 — File 2 of 10 / 第22章 — 第2个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Fred**.

本脚本演示 **02 Fred**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


---
## Step 1 — Step 1

```python
import io
import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE&cosd=2017-04-14&coed=2022-04-14"
resp = requests.get(URL)
if resp.status_code == 200:
   csvtext = resp.text
   csvbuffer = io.StringIO(csvtext)
   df = pd.read_csv(csvbuffer)
   print(df)
```

---
## Learning Notes / 学习笔记

- **概念**: Fred 是机器学习中的常用技术。  
  *Fred is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Fred / 02 Fred
# Complete Code / 完整代码
# ===============================

import io
import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10YIE&cosd=2017-04-14&coed=2022-04-14"
resp = requests.get(URL)
if resp.status_code == 200:
   csvtext = resp.text
   csvbuffer = io.StringIO(csvtext)
   df = pd.read_csv(csvbuffer)
   print(df)
```

---

➡️ **Next / 下一步**: File 3 of 10

---

### Temperature

# 07 — Temperature / 07 Temperature

**Chapter 22 — File 7 of 10 / 第22章 — 第7个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Reading temperature of New York**.

本脚本演示 **Reading temperature of New York**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import requests
from lxml import etree
from bs4 import BeautifulSoup
```

---
## Step 2 — Reading temperature of New York

```python
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)

if resp.status_code == 200:
```

---
## Step 3 — Using lxml

```python
dom = etree.HTML(resp.text)
    elements = dom.xpath("//span[@data-testid='TemperatureValue' and " \
                                   "contains(@class,'CurrentConditions')]")
    print(elements[0].text)
```

---
## Step 4 — Using BeautifulSoup

```python
soup = BeautifulSoup(resp.text, "lxml")
    elements = soup.select('span[data-testid="TemperatureValue"]' \
                               '[class^="CurrentConditions"]')
    print(elements[0].text)
```

---
## Learning Notes / 学习笔记

- **概念**: Reading temperature of New York 是机器学习中的常用技术。  
  *Reading temperature of New York is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Temperature / 07 Temperature
# Complete Code / 完整代码
# ===============================

import requests
from lxml import etree
from bs4 import BeautifulSoup

# Reading temperature of New York
URL = "https://weather.com/weather/today/l/40.75,-73.98"
resp = requests.get(URL)

if resp.status_code == 200:
    # Using lxml
    dom = etree.HTML(resp.text)
    elements = dom.xpath("//span[@data-testid='TemperatureValue' and " \
                                   "contains(@class,'CurrentConditions')]")
    print(elements[0].text)

    # Using BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml")
    elements = soup.select('span[data-testid="TemperatureValue"]' \
                               '[class^="CurrentConditions"]')
    print(elements[0].text)
```

---

➡️ **Next / 下一步**: File 8 of 10

---

### Yahoo

# 09 — Yahoo / 09 Yahoo

**Chapter 22 — File 9 of 10 / 第22章 — 第9个文件（共10个）**

---

## Summary / 总结

This script demonstrates **Read Yahoo home page**.

本脚本演示 **Read Yahoo home page**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
from lxml import etree
import requests
```

---
## Step 2 — Read Yahoo home page

```python
URL = "https://www.yahoo.com/"
resp = requests.get(URL)
dom = etree.HTML(resp.text)
```

---
## Step 3 — Print news headlines

```python
elements = dom.xpath("//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(etree.tostring(elem, method="text", encoding="unicode"))
```

---
## Learning Notes / 学习笔记

- **概念**: Read Yahoo home page 是机器学习中的常用技术。  
  *Read Yahoo home page is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Yahoo / 09 Yahoo
# Complete Code / 完整代码
# ===============================

from lxml import etree
import requests

# Read Yahoo home page
URL = "https://www.yahoo.com/"
resp = requests.get(URL)
dom = etree.HTML(resp.text)

# Print news headlines
elements = dom.xpath("//h3/a[u[@class='StretchedBox']]")
for elem in elements:
    print(etree.tostring(elem, method="text", encoding="unicode"))
```

---

➡️ **Next / 下一步**: File 10 of 10

---

### Chapter Summary

# Chapter 22 Summary / 第22章总结

## Theme / 主题: Chapter 22 / Chapter 22

This chapter contains **10 code files** demonstrating chapter 22.

本章包含 **10 个代码文件**，演示Chapter 22。

---
## Evolution / 演化路线

  1. `01_weather.ipynb` — Weather
  2. `02_fred.ipynb` — Fred
  3. `03_github.ipynb` — Github
  4. `04_wikipedia.ipynb` — Wikipedia
  5. `05_xpath.ipynb` — Xpath
  6. `06_beautifulsoup.ipynb` — Beautifulsoup
  7. `07_temperature.ipynb` — Temperature
  8. `08_pandas.ipynb` — Pandas
  9. `09_yahoo.ipynb` — Yahoo
  10. `10_selenium.ipynb` — Selenium

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 22) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 22）是机器学习流水线中的基础构建块。

---
