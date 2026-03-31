# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 03

---

### Tokenize



---

### Specialtoken

# 03 — Specialtoken / 03 Specialtoken

**Chapter 03 — File 2 of 7 / 第03章 — 第2个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Complete tokenization with special tokens**.

本脚本演示 **Complete tokenization with special tokens**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "I love machine learning!"
```

---
## Step 2 — Complete tokenization with special tokens

```python
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding="max_length",
    max_length=10,
    return_tensors="pt"
)

# 打印输出 / Print output
print("Full encoded sequence:")
# 将多个序列配对 / Pair multiple sequences
for token_id, token in zip(
    encoded["input_ids"][0],
    tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
):
    # 打印输出 / Print output
    print(f"{token}: {token_id}")
```

---
## Learning Notes / 学习笔记

- **概念**: Complete tokenization with special tokens 是机器学习中的常用技术。  
  *Complete tokenization with special tokens is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Specialtoken / 03 Specialtoken
# Complete Code / 完整代码
# ===============================

# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "I love machine learning!"

# Complete tokenization with special tokens
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    padding="max_length",
    max_length=10,
    return_tensors="pt"
)

# 打印输出 / Print output
print("Full encoded sequence:")
# 将多个序列配对 / Pair multiple sequences
for token_id, token in zip(
    encoded["input_ids"][0],
    tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
):
    # 打印输出 / Print output
    print(f"{token}: {token_id}")
```

---

➡️ **Next / 下一步**: File 3 of 7

---

### Sentiment



---

### Classification



---

### Analysis

# 06 — Analysis / 06 Analysis

**Chapter 03 — File 5 of 7 / 第03章 — 第5个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Remove extra whitespace and normalize**.

本脚本演示 **Remove extra whitespace and normalize**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTSentimentAnalyzer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()
        self.labels = ["NEGATIVE", "POSITIVE"]

    def preprocess_text(self, text):
```

---
## Step 2 — Remove extra whitespace and normalize

```python
text = " ".join(text.split())
```

---
## Step 3 — Tokenize with BERT-specific tokens

```python
inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
```

---
## Step 4 — Move to GPU if available

```python
# 获取字典的键值对 / Get dict key-value pairs
return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text):
```

---
## Step 5 — Prepare text for model

```python
inputs = self.preprocess_text(text)
```

---
## Step 6 — Get model predictions

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

---
## Step 7 — Convert to human-readable format

```python
prediction_dict = {
            "text": text,
            "sentiment": self.labels[probabilities.argmax().item()],
            "confidence": probabilities.max().item(),
            "probabilities": {
                label: prob.item()
                # 将多个序列配对 / Pair multiple sequences
                for label, prob in zip(self.labels, probabilities[0])
            }
        }
        return prediction_dict

def demonstrate_sentiment_analysis():
```

---
## Step 8 — Initialize analyzer

```python
analyzer = BERTSentimentAnalyzer()
```

---
## Step 9 — Test texts

```python
texts = [
        "This product completely transformed my workflow!",
        "Terrible experience, would not recommend.",
        "It's decent for the price, but nothing special."
    ]
```

---
## Step 10 — Analyze each text

```python
for text in texts:
        result = analyzer.predict(text)
        # 打印输出 / Print output
        print(f"\nText: {result['text']}")
        # 打印输出 / Print output
        print(f"Sentiment: {result['sentiment']}")
        # 打印输出 / Print output
        print(f"Confidence: {result['confidence']:.4f}")
        # 打印输出 / Print output
        print("Detailed probabilities:")
        # 获取字典的键值对 / Get dict key-value pairs
        for label, prob in result["probabilities"].items():
            # 打印输出 / Print output
            print(f"  {label}: {prob:.4f}")
```

---
## Step 11 — Running demonstration

```python
demonstrate_sentiment_analysis()
```

---
## Learning Notes / 学习笔记

- **概念**: Remove extra whitespace and normalize 是机器学习中的常用技术。  
  *Remove extra whitespace and normalize is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `torch.nn` | PyTorch 神经网络模块 | PyTorch neural network module |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Analysis / 06 Analysis
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BERTSentimentAnalyzer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()
        self.labels = ["NEGATIVE", "POSITIVE"]

    def preprocess_text(self, text):
        # Remove extra whitespace and normalize
        text = " ".join(text.split())

        # Tokenize with BERT-specific tokens
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move to GPU if available
        # 获取字典的键值对 / Get dict key-value pairs
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text):
        # Prepare text for model
        inputs = self.preprocess_text(text)

        # Get model predictions
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to human-readable format
        prediction_dict = {
            "text": text,
            "sentiment": self.labels[probabilities.argmax().item()],
            "confidence": probabilities.max().item(),
            "probabilities": {
                label: prob.item()
                # 将多个序列配对 / Pair multiple sequences
                for label, prob in zip(self.labels, probabilities[0])
            }
        }
        return prediction_dict

def demonstrate_sentiment_analysis():
    # Initialize analyzer
    analyzer = BERTSentimentAnalyzer()

    # Test texts
    texts = [
        "This product completely transformed my workflow!",
        "Terrible experience, would not recommend.",
        "It's decent for the price, but nothing special."
    ]

    # Analyze each text
    for text in texts:
        result = analyzer.predict(text)
        # 打印输出 / Print output
        print(f"\nText: {result['text']}")
        # 打印输出 / Print output
        print(f"Sentiment: {result['sentiment']}")
        # 打印输出 / Print output
        print(f"Confidence: {result['confidence']:.4f}")
        # 打印输出 / Print output
        print("Detailed probabilities:")
        # 获取字典的键值对 / Get dict key-value pairs
        for label, prob in result["probabilities"].items():
            # 打印输出 / Print output
            print(f"  {label}: {prob:.4f}")

# Running demonstration
demonstrate_sentiment_analysis()
```

---

➡️ **Next / 下一步**: File 6 of 7

---

### Ner

# 07 — Ner / 07 Ner

**Chapter 03 — File 6 of 7 / 第03章 — 第6个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Tokenize input text**.

本脚本演示 **Tokenize input text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BERTNamedEntityRecognizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()

    def recognize_entities(self, text):
```

---
## Step 2 — Tokenize input text

```python
inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
```

---
## Step 3 — Move inputs to device

```python
# 获取字典的键值对 / Get dict key-value pairs
inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

---
## Step 4 — Get predictions

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)
```

---
## Step 5 — Convert predictions to entities

```python
tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
```

---
## Step 6 — Extract entities

```python
entities = []
        current_entity = None
        # 将多个序列配对 / Pair multiple sequences
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token
            elif label == "O":
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                    current_entity = None
            if current_entity:
                # 添加元素到列表末尾 / Append element to list end
                entities.append(current_entity)

        return entities
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize input text 是机器学习中的常用技术。  
  *Tokenize input text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ner / 07 Ner
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BERTNamedEntityRecognizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()

    def recognize_entities(self, text):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move inputs to device
        # 获取字典的键值对 / Get dict key-value pairs
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)

        # Convert predictions to entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

        # Extract entities
        entities = []
        current_entity = None
        # 将多个序列配对 / Pair multiple sequences
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token
            elif label == "O":
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                    current_entity = None
            if current_entity:
                # 添加元素到列表末尾 / Append element to list end
                entities.append(current_entity)

        return entities
```

---

➡️ **Next / 下一步**: File 7 of 7

---

### Ner

# 08 — Ner / 08 Ner

**Chapter 03 — File 7 of 7 / 第03章 — 第7个文件（共7个）**

---

## Summary / 总结

This script demonstrates **Tokenize input text**.

本脚本演示 **Tokenize input text**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 定义模型结构 / Define model architecture


---
## Code Flow / 代码流程

```
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  🏗️ 定义模型 / Define Model
       │
       ▼
  📊 评估模型 / Evaluate Model
```

---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BERTNamedEntityRecognizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()

    def recognize_entities(self, text):
```

---
## Step 2 — Tokenize input text

```python
inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
```

---
## Step 3 — Move inputs to device

```python
# 获取字典的键值对 / Get dict key-value pairs
inputs = {k: v.to(self.device) for k, v in inputs.items()}
```

---
## Step 4 — Get predictions

```python
# 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)
```

---
## Step 5 — Convert predictions to entities

```python
tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]
```

---
## Step 6 — Extract entities

```python
entities = []
        current_entity = None
        # 将多个序列配对 / Pair multiple sequences
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token
            elif label == "O":
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                    current_entity = None
            if current_entity:
                # 添加元素到列表末尾 / Append element to list end
                entities.append(current_entity)

        return entities

def demonstrate_ner():
```

---
## Step 7 — Initialize recognizer

```python
ner = BERTNamedEntityRecognizer()
```

---
## Step 8 — Example text

```python
text = """
    Apple CEO Tim Cook announced new AI features at their headquarters
    in Cupertino, California. Microsoft and Google are also investing
    heavily in artificial intelligence research.
    """
```

---
## Step 9 — Get entities

```python
entities = ner.recognize_entities(text)
```

---
## Step 10 — Display results

```python
# 打印输出 / Print output
print("Found entities:")
    for entity in entities:
        # 打印输出 / Print output
        print(f"- {entity['text']} ({entity['type']})")
```

---
## Step 11 — Running demonstration

```python
demonstrate_ner()
```

---
## Learning Notes / 学习笔记

- **概念**: Tokenize input text 是机器学习中的常用技术。  
  *Tokenize input text is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `tokenizer` | 分词器：将文本切分为token | Tokenizer: split text into tokens |
| `transformer` | Transformer架构：基于注意力的模型 | Transformer: attention-based architecture |

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Ner / 08 Ner
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import AutoTokenizer, AutoModelForTokenClassification

class BERTNamedEntityRecognizer:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 切换到评估模式（禁用Dropout等） / Switch to eval mode (disable Dropout, etc.)
        self.model.eval()

    def recognize_entities(self, text):
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move inputs to device
        # 获取字典的键值对 / Get dict key-value pairs
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        # 禁用梯度计算（推理时节省内存） / Disable gradient computation (save memory during inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1)

        # Convert predictions to entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

        # Extract entities
        entities = []
        current_entity = None
        # 将多个序列配对 / Pair multiple sequences
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity:
                if token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    current_entity["text"] += " " + token
            elif label == "O":
                if current_entity:
                    # 添加元素到列表末尾 / Append element to list end
                    entities.append(current_entity)
                    current_entity = None
            if current_entity:
                # 添加元素到列表末尾 / Append element to list end
                entities.append(current_entity)

        return entities

def demonstrate_ner():
    # Initialize recognizer
    ner = BERTNamedEntityRecognizer()

    # Example text
    text = """
    Apple CEO Tim Cook announced new AI features at their headquarters
    in Cupertino, California. Microsoft and Google are also investing
    heavily in artificial intelligence research.
    """

    # Get entities
    entities = ner.recognize_entities(text)

    # Display results
    # 打印输出 / Print output
    print("Found entities:")
    for entity in entities:
        # 打印输出 / Print output
        print(f"- {entity['text']} ({entity['type']})")

# Running demonstration
demonstrate_ner()
```

---

### Chapter Summary / 章节总结



---
