# Python 机器学习 / Python for Machine Learning
## Chapter 24

---

### Apple

# 01 — Apple / 01 Apple

**Chapter 24 — File 1 of 12 / 第24章 — 第1个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Reading Apple shares from Yahoo Finance server**.

本脚本演示 **Reading Apple shares from Yahoo Finance server**。

---
## Step 1 — Step 1

```python
import pandas_datareader as pdr
```

---
## Step 2 — Reading Apple shares from Yahoo Finance server

```python
shares_df = pdr.DataReader('AAPL', 'yahoo', start='2021-01-01', end='2021-12-31')
```

---
## Step 3 — Look at the data read

```python
print(shares_df)
```

---
## Learning Notes / 学习笔记

- **概念**: Reading Apple shares from Yahoo Finance server 是机器学习中的常用技术。  
  *Reading Apple shares from Yahoo Finance server is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Apple / 01 Apple
# Complete Code / 完整代码
# ===============================

import pandas_datareader as pdr

# Reading Apple shares from Yahoo Finance server
shares_df = pdr.DataReader('AAPL', 'yahoo', start='2021-01-01', end='2021-12-31')
# Look at the data read
print(shares_df)
```

---

➡️ **Next / 下一步**: File 2 of 12

---

### Multiple

# 02 — Multiple / 02 Multiple

**Chapter 24 — File 2 of 12 / 第24章 — 第2个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Multiple**.

本脚本演示 **02 Multiple**。

---
## Step 1 — Step 1

```python
import pandas_datareader as pdr

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo',
                                    start='2021-01-01', end='2021-12-31')
print(shares_multiple_df.head())
```

---
## Learning Notes / 学习笔记

- **概念**: Multiple 是机器学习中的常用技术。  
  *Multiple is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Multiple / 02 Multiple
# Complete Code / 完整代码
# ===============================

import pandas_datareader as pdr

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo',
                                    start='2021-01-01', end='2021-12-31')
print(shares_multiple_df.head())
```

---

➡️ **Next / 下一步**: File 3 of 12

---

### Plot

# 03 — Plot / 03 Plot

**Chapter 24 — File 3 of 12 / 第24章 — 第3个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **03 Plot**。

---
## Step 1 — Step 1

```python
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo',
                                    start='2021-01-01', end='2021-12-31')
print(shares_multiple_df)

def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', legend=''):
    "General routine for plotting time series data"
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

---
## Learning Notes / 学习笔记

- **概念**: Plot 是机器学习中的常用技术。  
  *Plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 03 Plot
# Complete Code / 完整代码
# ===============================

import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo',
                                    start='2021-01-01', end='2021-12-31')
print(shares_multiple_df)

def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', legend=''):
    "General routine for plotting time series data"
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

---

➡️ **Next / 下一步**: File 4 of 12

---

### Plot

# 04 — Plot / 04 Plot

**Chapter 24 — File 4 of 12 / 第24章 — 第4个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Plot**.

本脚本演示 **04 Plot**。

---
## Step 1 — Step 1

```python
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo', start='2021-01-01', end='2021-12-31')
print(shares_multiple_df)

def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', legend=''):
    "General routine for plotting time series data"
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

---
## Learning Notes / 学习笔记

- **概念**: Plot 是机器学习中的常用技术。  
  *Plot is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Plot / 04 Plot
# Complete Code / 完整代码
# ===============================

import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

companies = ['AAPL', 'MSFT', 'GE']
shares_multiple_df = pdr.DataReader(companies, 'yahoo', start='2021-01-01', end='2021-12-31')
print(shares_multiple_df)

def plot_timeseries_df(df, attrib, ticker_loc=1, title='Timeseries', legend=''):
    "General routine for plotting time series data"
    fig = plt.figure(figsize=(15,7))
    plt.plot(df[attrib], 'o-')
    _ = plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(ticker_loc))
    plt.title(title)
    plt.gca().legend(legend)
    plt.show()

plot_timeseries_df(shares_multiple_df.loc["2021-04-01":"2021-06-30"], "Close",
                   ticker_loc=3, title="Close price", legend=companies)
```

---

➡️ **Next / 下一步**: File 5 of 12

---

### Cpi

# 05 — Cpi / 05 Cpi

**Chapter 24 — File 5 of 12 / 第24章 — 第5个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Read data from FRED and print**.

本脚本演示 **Read data from FRED and print**。

---
## Step 1 — Step 1

```python
import pandas_datareader as pdr
import matplotlib.pyplot as plt
```

---
## Step 2 — Read data from FRED and print

```python
fred_df = pdr.DataReader(['CPIAUCSL','CPILFESL'], 'fred', "2010-01-01", "2021-12-31")
print(fred_df)
```

---
## Step 3 — Show in plot the data of 2019-2021

```python
fig = plt.figure(figsize=(15,7))
plt.plot(fred_df.loc["2019":], 'o-')
plt.xticks(rotation=90)
plt.legend(fred_df.columns)
plt.title("Consumer Price Index")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Read data from FRED and print 是机器学习中的常用技术。  
  *Read data from FRED and print is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Cpi / 05 Cpi
# Complete Code / 完整代码
# ===============================

import pandas_datareader as pdr
import matplotlib.pyplot as plt

# Read data from FRED and print
fred_df = pdr.DataReader(['CPIAUCSL','CPILFESL'], 'fred', "2010-01-01", "2021-12-31")
print(fred_df)

# Show in plot the data of 2019-2021
fig = plt.figure(figsize=(15,7))
plt.plot(fred_df.loc["2019":], 'o-')
plt.xticks(rotation=90)
plt.legend(fred_df.columns)
plt.title("Consumer Price Index")
plt.show()
```

---

➡️ **Next / 下一步**: File 6 of 12

---

### Worldbank

# 06 — Worldbank / 06 Worldbank

**Chapter 24 — File 6 of 12 / 第24章 — 第6个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Worldbank**.

本脚本演示 **06 Worldbank**。

---
## Step 1 — Step 1

```python
from pandas_datareader import wb

matches = wb.search('total.*population')
print(matches[["id","name"]])
```

---
## Learning Notes / 学习笔记

- **概念**: Worldbank 是机器学习中的常用技术。  
  *Worldbank is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Worldbank / 06 Worldbank
# Complete Code / 完整代码
# ===============================

from pandas_datareader import wb

matches = wb.search('total.*population')
print(matches[["id","name"]])
```

---

➡️ **Next / 下一步**: File 7 of 12

---

### Countries

# 07 — Countries / 07 Countries

**Chapter 24 — File 7 of 12 / 第24章 — 第7个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Countries**.

本脚本演示 **07 Countries**。

---
## Step 1 — Step 1

```python
import pandas_datareader.wb as wb

countries = wb.get_countries()
print(countries)
```

---
## Learning Notes / 学习笔记

- **概念**: Countries 是机器学习中的常用技术。  
  *Countries is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Countries / 07 Countries
# Complete Code / 完整代码
# ===============================

import pandas_datareader.wb as wb

countries = wb.get_countries()
print(countries)
```

---

➡️ **Next / 下一步**: File 8 of 12

---

### Population

# 08 — Population / 08 Population

**Chapter 24 — File 8 of 12 / 第24章 — 第8个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Get a list of 2-letter country code excluding aggregates**.

本脚本演示 **Get a list of 2-letter country code excluding aggregates**。

---
## Step 1 — Step 1

```python
import pandas_datareader.wb as wb
import pandas as pd
import matplotlib.pyplot as plt
```

---
## Step 2 — Get a list of 2-letter country code excluding aggregates

```python
countries = wb.get_countries()
countries = list(countries[countries.region != "Aggregates"]["iso2c"])
```

---
## Step 3 — Read countries' total population data (SP.POP.TOTL) in year 2020

```python
population_df = wb.download(indicator="SP.POP.TOTL", country=countries, start=2020, end=2020)
```

---
## Step 4 — Sort by population, then take top 25 countries, and make the index (i.e., countries) as a column

```python
population_df = (population_df.dropna()
                              .sort_values("SP.POP.TOTL")
                              .iloc[-25:]
                              .reset_index())
```

---
## Step 5 — Plot the population, in millions

```python
fig = plt.figure(figsize=(15,7))
plt.bar(population_df["country"], population_df["SP.POP.TOTL"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Get a list of 2-letter country code excluding aggregates 是机器学习中的常用技术。  
  *Get a list of 2-letter country code excluding aggregates is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Population / 08 Population
# Complete Code / 完整代码
# ===============================

import pandas_datareader.wb as wb
import pandas as pd
import matplotlib.pyplot as plt

# Get a list of 2-letter country code excluding aggregates
countries = wb.get_countries()
countries = list(countries[countries.region != "Aggregates"]["iso2c"])

# Read countries' total population data (SP.POP.TOTL) in year 2020
population_df = wb.download(indicator="SP.POP.TOTL", country=countries, start=2020, end=2020)

# Sort by population, then take top 25 countries, and make the index (i.e., countries) as a column
population_df = (population_df.dropna()
                              .sort_values("SP.POP.TOTL")
                              .iloc[-25:]
                              .reset_index())

# Plot the population, in millions
fig = plt.figure(figsize=(15,7))
plt.bar(population_df["country"], population_df["SP.POP.TOTL"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

---

➡️ **Next / 下一步**: File 9 of 12

---

### Json

# 09 — Json / 09 Json

**Chapter 24 — File 9 of 12 / 第24章 — 第9个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Create query URL for list of countries, by default only 50 entries returned per page**.

本脚本演示 **Create query URL for list of countries, by default only 50 entries returned per page**。

---
## Step 1 — Step 1

```python
import requests
```

---
## Step 2 — Create query URL for list of countries, by default only 50 entries returned per page

```python
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
```

---
## Step 3 — Expects HTTP status code 200 for correct query

```python
print(response.status_code)
```

---
## Step 4 — Get the response in JSON

```python
header, data = response.json()
print(header)
```

---
## Step 5 — Collect a list of 3-letter country code excluding aggregates

```python
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)
```

---
## Learning Notes / 学习笔记

- **概念**: Create query URL for list of countries, by default only 50 entries returned per page 是机器学习中的常用技术。  
  *Create query URL for list of countries, by default only 50 entries returned per page is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Json / 09 Json
# Complete Code / 完整代码
# ===============================

import requests

# Create query URL for list of countries, by default only 50 entries returned per page
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
# Expects HTTP status code 200 for correct query
print(response.status_code)
# Get the response in JSON
header, data = response.json()
print(header)
# Collect a list of 3-letter country code excluding aggregates
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)
```

---

➡️ **Next / 下一步**: File 10 of 12

---

### Webapi

# 12 — Webapi / 12 Webapi

**Chapter 24 — File 10 of 12 / 第24章 — 第10个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Create query URL for list of countries, by default only 50 entries returned per page**.

本脚本演示 **Create query URL for list of countries, by default only 50 entries returned per page**。

---
## Step 1 — Step 1

```python
import pandas as pd
import matplotlib.pyplot as plt
import requests
```

---
## Step 2 — Create query URL for list of countries, by default only 50 entries returned per page

```python
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
```

---
## Step 3 — Expects HTTP status code 200 for correct query

```python
print(response.status_code)
```

---
## Step 4 — Get the response in JSON

```python
header, data = response.json()
print(header)
```

---
## Step 5 — Collect a list of 3-letter country code excluding aggregates

```python
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)
```

---
## Step 6 — Create query URL for total population from all countries in 2020

```python
arguments = {
    "country": "all",
    "indicator": "SP.POP.TOTL",
    "date": 2020,
    "format": "json"
}
url = "http://api.worldbank.org/v2/country/{country}/" \
      "indicator/{indicator}?date={date}&format={format}&per_page=500"
query_population = url.format(**arguments)
response = requests.get(query_population)
print(response.status_code)
```

---
## Step 7 — Get the response in JSON

```python
header, population_data = response.json()
print(header)
```

---
## Step 8 — Filter for countries, not aggregates

```python
population = []
for item in population_data:
    if item["countryiso3code"] in countries:
        name = item["country"]["value"]
        population.append({"country":name, "population": item["value"]})
```

---
## Step 9 — Create DataFrame for sorting and filtering

```python
population = pd.DataFrame.from_dict(population)
population = population.dropna().sort_values("population").iloc[-25:]
```

---
## Step 10 — Plot bar chart

```python
fig = plt.figure(figsize=(15,7))
plt.bar(population["country"], population["population"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Create query URL for list of countries, by default only 50 entries returned per page 是机器学习中的常用技术。  
  *Create query URL for list of countries, by default only 50 entries returned per page is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Webapi / 12 Webapi
# Complete Code / 完整代码
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import requests


# Create query URL for list of countries, by default only 50 entries returned per page
url = "http://api.worldbank.org/v2/country/all?format=json&per_page=500"
response = requests.get(url)
# Expects HTTP status code 200 for correct query
print(response.status_code)
# Get the response in JSON
header, data = response.json()
print(header)
# Collect a list of 3-letter country code excluding aggregates
countries = [item["id"]
             for item in data
             if item["region"]["value"] != "Aggregates"]
print(countries)


# Create query URL for total population from all countries in 2020
arguments = {
    "country": "all",
    "indicator": "SP.POP.TOTL",
    "date": 2020,
    "format": "json"
}
url = "http://api.worldbank.org/v2/country/{country}/" \
      "indicator/{indicator}?date={date}&format={format}&per_page=500"
query_population = url.format(**arguments)
response = requests.get(query_population)
print(response.status_code)
# Get the response in JSON
header, population_data = response.json()
print(header)


# Filter for countries, not aggregates
population = []
for item in population_data:
    if item["countryiso3code"] in countries:
        name = item["country"]["value"]
        population.append({"country":name, "population": item["value"]})
# Create DataFrame for sorting and filtering
population = pd.DataFrame.from_dict(population)
population = population.dropna().sort_values("population").iloc[-25:]
# Plot bar chart
fig = plt.figure(figsize=(15,7))
plt.bar(population["country"], population["population"]/1e6)
plt.xticks(rotation=90)
plt.ylabel("Million Population")
plt.title("Population")
plt.show()
```

---

➡️ **Next / 下一步**: File 11 of 12

---

### Ar3

# 13 — Ar3 / 13 Ar3

**Chapter 24 — File 11 of 12 / 第24章 — 第11个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Predefined paramters**.

本脚本演示 **Predefined paramters**。

---
## Step 1 — Step 1

```python
import numpy as np
import matplotlib.pyplot as plt
```

---
## Step 2 — Predefined paramters

```python
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate
```

---
## Step 3 — Random initial values

```python
ar_data = list(np.random.randn(ar_n))
```

---
## Step 4 — Generate the rest of the values

```python
for i in range(length - ar_n):
    next_val = (ar_coeff @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)
```

---
## Step 5 — Plot the time series

```python
fig = plt.figure(figsize=(12,5))
plt.plot(ar_data)
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Predefined paramters 是机器学习中的常用技术。  
  *Predefined paramters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ar3 / 13 Ar3
# Complete Code / 完整代码
# ===============================

import numpy as np
import matplotlib.pyplot as plt

# Predefined paramters
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate

# Random initial values
ar_data = list(np.random.randn(ar_n))

# Generate the rest of the values
for i in range(length - ar_n):
    next_val = (ar_coeff @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)

# Plot the time series
fig = plt.figure(figsize=(12,5))
plt.plot(ar_data)
plt.show()
```

---

➡️ **Next / 下一步**: File 12 of 12

---

### Ar3

# 14 — Ar3 / 14 Ar3

**Chapter 24 — File 12 of 12 / 第24章 — 第12个文件（共12个）**

---

## Summary / 总结

This script demonstrates **Predefined paramters**.

本脚本演示 **Predefined paramters**。

---
## Step 1 — Step 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---
## Step 2 — Predefined paramters

```python
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate
```

---
## Step 3 — Random initial values

```python
ar_data = list(np.random.randn(ar_n))
```

---
## Step 4 — Generate the rest of the values

```python
for i in range(length - ar_n):
    next_val = (ar_coeff @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)
```

---
## Step 5 — Convert the data into a pandas DataFrame

```python
synthetic = pd.DataFrame({"AR(3)": ar_data})
synthetic.index = pd.date_range(start="2021-07-01", periods=len(ar_data), freq="D")
```

---
## Step 6 — Plot the time series

```python
fig = plt.figure(figsize=(12,5))
plt.plot(synthetic.index, synthetic.values)
plt.xticks(rotation=90)
plt.title("AR(3) time series")
plt.show()
```

---
## Learning Notes / 学习笔记

- **概念**: Predefined paramters 是机器学习中的常用技术。  
  *Predefined paramters is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ar3 / 14 Ar3
# Complete Code / 完整代码
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Predefined paramters
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 200                 # Number of data points to generate

# Random initial values
ar_data = list(np.random.randn(ar_n))

# Generate the rest of the values
for i in range(length - ar_n):
    next_val = (ar_coeff @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
    ar_data.append(next_val)

# Convert the data into a pandas DataFrame
synthetic = pd.DataFrame({"AR(3)": ar_data})
synthetic.index = pd.date_range(start="2021-07-01", periods=len(ar_data), freq="D")

# Plot the time series
fig = plt.figure(figsize=(12,5))
plt.plot(synthetic.index, synthetic.values)
plt.xticks(rotation=90)
plt.title("AR(3) time series")
plt.show()
```

---

### Chapter Summary / 章节总结

# Chapter 24 Summary / 第24章总结

## Theme / 主题: Chapter 24 / Chapter 24

This chapter contains **12 code files** demonstrating chapter 24.

本章包含 **12 个代码文件**，演示Chapter 24。

---
## Evolution / 演化路线

  1. `01_apple.ipynb` — Apple
  2. `02_multiple.ipynb` — Multiple
  3. `03_plot.ipynb` — Plot
  4. `04_plot.ipynb` — Plot
  5. `05_cpi.ipynb` — Cpi
  6. `06_worldbank.ipynb` — Worldbank
  7. `07_countries.ipynb` — Countries
  8. `08_population.ipynb` — Population
  9. `09_json.ipynb` — Json
  10. `12_webapi.ipynb` — Webapi
  11. `13_ar3.ipynb` — Ar3
  12. `14_ar3.ipynb` — Ar3

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 24) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 24）是机器学习流水线中的基础构建块。

---
