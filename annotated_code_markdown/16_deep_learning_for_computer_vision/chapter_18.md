# CV深度学习
## Chapter 18

---

### Pretrained Classifier

# 04 — Pretrained Classifier / 预训练模型

**Chapter 18 — File 4 of 6 / 第18章 — 第4个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of using a pre-trained model as a classifier**.

本脚本演示 **example of using a pre-trained model as a classifier**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of using a pre-trained model as a classifier

```python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
```

---
## Step 2 — load an image from file

```python
image = load_img('dog.jpg', target_size=(224, 224))
```

---
## Step 3 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 4 — reshape data for the model

```python
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 5 — prepare the image for the VGG model

```python
image = preprocess_input(image)
```

---
## Step 6 — load the model

```python
model = VGG16()
```

---
## Step 7 — predict the probability across all output classes

```python
yhat = model.predict(image)
```

---
## Step 8 — convert the probabilities to class labels

```python
label = decode_predictions(yhat)
```

---
## Step 9 — retrieve the most likely result, e.g. highest probability

```python
label = label[0][0]
```

---
## Step 10 — print the classification

```python
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

---
## Learning Notes / 学习笔记

- **概念**: example of using a pre-trained model as a classifier 是机器学习中的常用技术。  
  *example of using a pre-trained model as a classifier is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Classifier / 预训练模型
# Complete Code / 完整代码
# ===============================

# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load an image from file
image = load_img('dog.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load the model
model = VGG16()
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

---

➡️ **Next / 下一步**: File 5 of 6

---

### Pretrained Feature Extractor

# 05 — Pretrained Feature Extractor / 特征工程

**Chapter 18 — File 5 of 6 / 第18章 — 第5个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of using the vgg16 model as a feature extraction model**.

本脚本演示 **example of using the vgg16 model as a feature extraction model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Step 1 — example of using the vgg16 model as a feature extraction model

```python
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
```

---
## Step 2 — load an image from file

```python
image = load_img('dog.jpg', target_size=(224, 224))
```

---
## Step 3 — convert the image pixels to a numpy array

```python
image = img_to_array(image)
```

---
## Step 4 — reshape data for the model

```python
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

---
## Step 5 — prepare the image for the VGG model

```python
image = preprocess_input(image)
```

---
## Step 6 — load model

```python
model = VGG16()
```

---
## Step 7 — remove the output layer

```python
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
```

---
## Step 8 — get extracted features

```python
features = model.predict(image)
print(features.shape)
```

---
## Step 9 — save to file

```python
dump(features, open('dog.pkl', 'wb'))
```

---
## Learning Notes / 学习笔记

- **概念**: example of using the vgg16 model as a feature extraction model 是机器学习中的常用技术。  
  *example of using the vgg16 model as a feature extraction model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `model.predict` | 模型预测 | Model prediction |
| `numpy` | 数值计算库 | Numerical computing library |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Feature Extractor / 特征工程
# Complete Code / 完整代码
# ===============================

# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
# load an image from file
image = load_img('dog.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# get extracted features
features = model.predict(image)
print(features.shape)
# save to file
dump(features, open('dog.pkl', 'wb'))
```

---

➡️ **Next / 下一步**: File 6 of 6

---

### Pretrained Model With New Output

# 06 — Pretrained Model With New Output / 预训练模型

**Chapter 18 — File 6 of 6 / 第18章 — 第6个文件（共6个）**

---

## Summary / 总结

This script demonstrates **example of tending the vgg16 model**.

本脚本演示 **example of tending the vgg16 model**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 定义模型结构 / Define model architecture


---
## Step 1 — example of tending the vgg16 model

```python
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
```

---
## Step 2 — load model without classifier layers

```python
model = VGG16(include_top=False, input_shape=(300, 300, 3))
```

---
## Step 3 — add new classifier layers

```python
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)
```

---
## Step 4 — define new model

```python
model = Model(inputs=model.inputs, outputs=output)
```

---
## Step 5 — summarize

```python
model.summary()
```

---
## Learning Notes / 学习笔记

- **概念**: example of tending the vgg16 model 是机器学习中的常用技术。  
  *example of tending the vgg16 model is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dense` | 全连接层（Keras） | Fully connected layer (Keras) |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Pretrained Model With New Output / 预训练模型
# Complete Code / 完整代码
# ===============================

# example of tending the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(300, 300, 3))
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()
# ...
```

---

### Chapter Summary

# Chapter 18 Summary / 第18章总结

## Theme / 主题: Chapter 18 / Chapter 18

This chapter contains **6 code files** demonstrating chapter 18.

本章包含 **6 个代码文件**，演示Chapter 18。

---
## Evolution / 演化路线

  1. `01_vgg_model.ipynb` — Vgg Model
  2. `02_inception_model.ipynb` — Inception Model
  3. `03_resnet_model.ipynb` — Resnet Model
  4. `04_pretrained_classifier.ipynb` — Pretrained Classifier
  5. `05_pretrained_feature_extractor.ipynb` — Pretrained Feature Extractor
  6. `06_pretrained_model_with_new_output.ipynb` — Pretrained Model With New Output

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 18) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 18）是机器学习流水线中的基础构建块。

---
