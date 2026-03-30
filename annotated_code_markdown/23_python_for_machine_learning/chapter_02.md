# Python ML
## Chapter 02

---

### Pretrained Model Image

# 08 — Pretrained Model Image / 图像处理

**Chapter 02 — File 2 of 4 / 第02章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the VGG16 model pre-trained on the ImageNet dataset**.

本脚本演示 **Load the VGG16 model pre-trained on the ImageNet dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
```

---
## Step 2 — Load the VGG16 model pre-trained on the ImageNet dataset

```python
vgg16_model = vgg16.VGG16(weights='imagenet')
```

---
## Step 3 — Read the arguments passed to the interpreter when invoking the script

```python
image_path = sys.argv[1]
top_guesses = sys.argv[2]
```

---
## Step 4 — Load the image, resized according to the model target size

```python
img_resized = image.load_img(image_path, target_size=(224, 224))
```

---
## Step 5 — Convert the image into an array

```python
img = image.img_to_array(img_resized)
```

---
## Step 6 — Display the image to check that it has been correctly resized

```python
plt.imshow(img.astype(np.uint8))
```

---
## Step 7 — Add in a dimension

```python
img = np.expand_dims(img, axis=0)
```

---
## Step 8 — Scale the pixel intensity values

```python
img = preprocess_input(img)
```

---
## Step 9 — Generate a prediction for the test image

```python
pred_vgg = vgg16_model.predict(img)
```

---
## Step 10 — Decode and print the top 3 predictions

```python
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the VGG16 model pre-trained on the ImageNet dataset 是机器学习中的常用技术。  
  *Load the VGG16 model pre-trained on the ImageNet dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Model Image / 图像处理
# Complete Code / 完整代码
# ===============================

import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the VGG16 model pre-trained on the ImageNet dataset
vgg16_model = vgg16.VGG16(weights='imagenet')

# Read the arguments passed to the interpreter when invoking the script
image_path = sys.argv[1]
top_guesses = sys.argv[2]

# Load the image, resized according to the model target size
img_resized = image.load_img(image_path, target_size=(224, 224))

# Convert the image into an array
img = image.img_to_array(img_resized)

# Display the image to check that it has been correctly resized
plt.imshow(img.astype(np.uint8))

# Add in a dimension
img = np.expand_dims(img, axis=0)

# Scale the pixel intensity values
img = preprocess_input(img)

# Generate a prediction for the test image
pred_vgg = vgg16_model.predict(img)

# Decode and print the top 3 predictions
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Pretrained Model Inputs

# 10 — Pretrained Model Inputs / 预训练模型

**Chapter 02 — File 3 of 4 / 第02章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the VGG16 model pre-trained on the ImageNet dataset**.

本脚本演示 **Load the VGG16 model pre-trained on the ImageNet dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Load the VGG16 model pre-trained on the ImageNet dataset

```python
vgg16_model = vgg16.VGG16(weights='imagenet')
```

---
## Step 2 — Load the image, resized according to the model target size

```python
img_resized = image.load_img(image_path, target_size=(224, 224))
```

---
## Step 3 — Convert the image into an array

```python
img = image.img_to_array(img_resized)
```

---
## Step 4 — Display the image to check that it has been correctly resized

```python
plt.imshow(img.astype(np.uint8))
```

---
## Step 5 — Add in a dimension

```python
img = np.expand_dims(img, axis=0)
```

---
## Step 6 — Scale the pixel intensity values

```python
img = preprocess_input(img)
```

---
## Step 7 — Generate a prediction for the test image

```python
pred_vgg = vgg16_model.predict(img)
```

---
## Step 8 — Decode and print the top 3 predictions

```python
print('Prediction:', decode_predictions(pred_vgg, top=top_guesses))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the VGG16 model pre-trained on the ImageNet dataset 是机器学习中的常用技术。  
  *Load the VGG16 model pre-trained on the ImageNet dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `model.predict` | 模型预测 | Model prediction |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Model Inputs / 预训练模型
# Complete Code / 完整代码
# ===============================

# Load the VGG16 model pre-trained on the ImageNet dataset
vgg16_model = vgg16.VGG16(weights='imagenet')

# Load the image, resized according to the model target size
img_resized = image.load_img(image_path, target_size=(224, 224))

# Convert the image into an array
img = image.img_to_array(img_resized)

# Display the image to check that it has been correctly resized
plt.imshow(img.astype(np.uint8))

# Add in a dimension
img = np.expand_dims(img, axis=0)

# Scale the pixel intensity values
img = preprocess_input(img)

# Generate a prediction for the test image
pred_vgg = vgg16_model.predict(img)

# Decode and print the top 3 predictions
print('Prediction:', decode_predictions(pred_vgg, top=top_guesses))
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Input

# 12 — Input / 12 Input

**Chapter 02 — File 4 of 4 / 第02章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Load the VGG16 model pre-trained on the ImageNet dataset**.

本脚本演示 **Load the VGG16 model pre-trained on the ImageNet dataset**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────┐
│  评估结果 Evaluate Results  │
└───────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

---
## Step 1 — Step 1

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
```

---
## Step 2 — Load the VGG16 model pre-trained on the ImageNet dataset

```python
vgg16_model = vgg16.VGG16(weights='imagenet')
```

---
## Step 3 — Ask the user for manual inputs

```python
image_path = input("Enter image path: ")
top_guesses = input("Enter number of top guesses: ")
```

---
## Step 4 — Load the image, resized according to the model target size

```python
img_resized = image.load_img(image_path, target_size=(224, 224))
```

---
## Step 5 — Convert the image into an array

```python
img = image.img_to_array(img_resized)
```

---
## Step 6 — Add in a dimension

```python
img = np.expand_dims(img, axis=0)
```

---
## Step 7 — Scale the pixel intensity values

```python
img = preprocess_input(img)
```

---
## Step 8 — Generate a prediction for the test image

```python
pred_vgg = vgg16_model.predict(img)
```

---
## Step 9 — Decode and print the top 3 predictions

```python
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))
```

---
## Learning Notes / 学习笔记

- **概念**: Load the VGG16 model pre-trained on the ImageNet dataset 是机器学习中的常用技术。  
  *Load the VGG16 model pre-trained on the ImageNet dataset is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `matplotlib` | 绑图库 | Plotting library |
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Input / 12 Input
# Complete Code / 完整代码
# ===============================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the VGG16 model pre-trained on the ImageNet dataset
vgg16_model = vgg16.VGG16(weights='imagenet')

# Ask the user for manual inputs
image_path = input("Enter image path: ")
top_guesses = input("Enter number of top guesses: ")

# Load the image, resized according to the model target size
img_resized = image.load_img(image_path, target_size=(224, 224))

# Convert the image into an array
img = image.img_to_array(img_resized)

# Add in a dimension
img = np.expand_dims(img, axis=0)

# Scale the pixel intensity values
img = preprocess_input(img)

# Generate a prediction for the test image
pred_vgg = vgg16_model.predict(img)

# Decode and print the top 3 predictions
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))
```

---

### Chapter Summary

# Chapter 02 Summary / 第02章总结

## Theme / 主题: Chapter 02 / Chapter 02

This chapter contains **4 code files** demonstrating chapter 02.

本章包含 **4 个代码文件**，演示Chapter 02。

---
## Evolution / 演化路线

  1. `03_pretrained_model.ipynb` — Pretrained Model
  2. `08_pretrained_model_image.ipynb` — Pretrained Model Image
  3. `10_pretrained_model_inputs.ipynb` — Pretrained Model Inputs
  4. `12_input.ipynb` — Input

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 02) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 02）是机器学习流水线中的基础构建块。

---
