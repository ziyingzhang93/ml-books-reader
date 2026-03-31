# HuggingFace Transformers NLP / NLP with HF Transformers
## Chapter 11

---

### Qna

# 01 — Qna / 01 Qna

**Chapter 11 — File 1 of 5 / 第11章 — 第1个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Qna**.

本脚本演示 **01 Qna**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing


---
## Step 1 — Step 1

```python
# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilbert-base-uncased-distilled-squad"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer,
                       device=device)
max_answer_length = 50
top_k = 3
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris, which is " \
          "known for its art, fashion, gastronomy and culture."
result = qa_pipeline(question=question, context=context,
                     max_answer_len=max_answer_length, top_k=top_k)
# 打印输出 / Print output
print(f"Question: {question}")
# 打印输出 / Print output
print(f"Context: {context}")
# 打印输出 / Print output
print(result)
```

---
## Learning Notes / 学习笔记

- **概念**: Qna 是机器学习中的常用技术。  
  *Qna is a common technique in machine learning.*

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
# Qna / 01 Qna
# Complete Code / 完整代码
# ===============================

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilbert-base-uncased-distilled-squad"

tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer,
                       device=device)
max_answer_length = 50
top_k = 3
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris, which is " \
          "known for its art, fashion, gastronomy and culture."
result = qa_pipeline(question=question, context=context,
                     max_answer_len=max_answer_length, top_k=top_k)
# 打印输出 / Print output
print(f"Question: {question}")
# 打印输出 / Print output
print(f"Context: {context}")
# 打印输出 / Print output
print(result)
```

---

➡️ **Next / 下一步**: File 2 of 5

---

### Chunking

# 02 — Chunking / 02 Chunking

**Chapter 11 — File 2 of 5 / 第11章 — 第2个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Initialize pipeline for simple queries and answer cache**.

本脚本演示 **Initialize pipeline for simple queries and answer cache**。

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
# 导入时间处理模块 / Import time module
import time
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5


class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
```

---
## Step 2 — Initialize pipeline for simple queries and answer cache

```python
self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word
```

---
## Step 3 — Add the last chunk if it's not empty

```python
if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
```

---
## Step 4 — Check cache

```python
cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
```

---
## Step 5 — Preprocess context into chunks

```python
context_chunks = self.preprocess_context(context, config.max_sequence_length)
```

---
## Step 6 — Get answers from all chunks

```python
answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)
```

---
## Step 7 — Return the best answer or indicate no answer found

```python
if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }
```

---
## Step 8 — Cache the result

```python
self.answer_cache[cache_key] = result
        return result


config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
context = """
The Python programming language was created by Guido van Rossum and was released in 1991.
Python is known for its simple syntax and readability. It has become one of the most
popular programming languages, especially in fields like data science and machine
learning.  The language is maintained by the Python Steering Council and developed by a
large community of contributors.
"""
questions = [
    "Who created Python?",
    "When was Python released?",
    "Why is Python popular?",
    "What is Python known for?"
]
for question in questions:
    start_time = time.time()
    answer = qa_system.get_answer(question, context, config)
    duration = time.time() - start_time
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
    # 打印输出 / Print output
    print(f"Duration: {duration:.2f}s")
    # 打印输出 / Print output
    print("-" * 50)
```

---
## Learning Notes / 学习笔记

- **概念**: Initialize pipeline for simple queries and answer cache 是机器学习中的常用技术。  
  *Initialize pipeline for simple queries and answer cache is a common technique in machine learning.*

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
# Chunking / 02 Chunking
# Complete Code / 完整代码
# ===============================

# 导入时间处理模块 / Import time module
import time
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5


class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)

        # Initialize pipeline for simple queries and answer cache
        self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word

      # Add the last chunk if it's not empty
      if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
        # Check cache
        cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # Preprocess context into chunks
        context_chunks = self.preprocess_context(context, config.max_sequence_length)

        # Get answers from all chunks
        answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)

        # Return the best answer or indicate no answer found
        if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }

        # Cache the result
        self.answer_cache[cache_key] = result
        return result


config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
context = """
The Python programming language was created by Guido van Rossum and was released in 1991.
Python is known for its simple syntax and readability. It has become one of the most
popular programming languages, especially in fields like data science and machine
learning.  The language is maintained by the Python Steering Council and developed by a
large community of contributors.
"""
questions = [
    "Who created Python?",
    "When was Python released?",
    "Why is Python popular?",
    "What is Python known for?"
]
for question in questions:
    start_time = time.time()
    answer = qa_system.get_answer(question, context, config)
    duration = time.time() - start_time
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
    # 打印输出 / Print output
    print(f"Duration: {duration:.2f}s")
    # 打印输出 / Print output
    print("-" * 50)
```

---

➡️ **Next / 下一步**: File 3 of 5

---

### Context

# 03 — Context / 03 Context

**Chapter 11 — File 3 of 5 / 第11章 — 第3个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Context**.

本脚本演示 **03 Context**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

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
import collections

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)
```

---
## Learning Notes / 学习笔记

- **概念**: Context 是机器学习中的常用技术。  
  *Context is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Context / 03 Context
# Complete Code / 完整代码
# ===============================

import collections

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)
```

---

➡️ **Next / 下一步**: File 4 of 5

---

### Context

# 04 — Context / 04 Context

**Chapter 11 — File 4 of 5 / 第11章 — 第4个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Initialize pipeline for simple queries and answer cache**.

本脚本演示 **Initialize pipeline for simple queries and answer cache**。

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
import collections
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5


class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
```

---
## Step 2 — Initialize pipeline for simple queries and answer cache

```python
self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word
```

---
## Step 3 — Add the last chunk if it's not empty

```python
if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
```

---
## Step 4 — Check cache

```python
cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
```

---
## Step 5 — Preprocess context into chunks

```python
context_chunks = self.preprocess_context(context, config.max_sequence_length)
```

---
## Step 6 — Get answers from all chunks

```python
answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)
```

---
## Step 7 — Return the best answer or indicate no answer found

```python
if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }
```

---
## Step 8 — Cache the result

```python
self.answer_cache[cache_key] = result
        return result

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)

context_manager = ContextManager(max_contexts=10)
context_manager.add_context("python", """
  Python is a high-level, interpreted programming language created by Guido van Rossum and
  released in 1991.  Python's design philosophy emphasizes code readability with its
  notable use of significant whitespace. Python features a dynamic type system and
  automatic memory management and supports multiple programming paradigms, including
  structured, object-oriented, and functional programming.
""")
context_manager.add_context("machine_learning", """
  Machine learning is a field of study that gives computers the ability to learn without
  being explicitly programmed. It is a branch of artificial intelligence based on the idea
  that systems can learn from data, identify patterns and make decisions with minimal
  human intervention.
""")

config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
question = "Who created Python?"
relevant_contexts = context_manager.search_relevant_context(question, top_k=1)
if relevant_contexts:
    relevance, context_id = relevant_contexts[0]
    context = context_manager.get_context(context_id)
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Most relevant context: {context_id} (relevance: {relevance:.2f})")
    # 打印输出 / Print output
    print(context)

    answer = qa_system.get_answer(question, context, config)
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
else:
    # 打印输出 / Print output
    print("No relevant context found.")
```

---
## Learning Notes / 学习笔记

- **概念**: Initialize pipeline for simple queries and answer cache 是机器学习中的常用技术。  
  *Initialize pipeline for simple queries and answer cache is a common technique in machine learning.*

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
# Context / 04 Context
# Complete Code / 完整代码
# ===============================

import collections
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5


class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)

        # Initialize pipeline for simple queries and answer cache
        self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word

      # Add the last chunk if it's not empty
      if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
        # Check cache
        cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # Preprocess context into chunks
        context_chunks = self.preprocess_context(context, config.max_sequence_length)

        # Get answers from all chunks
        answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)

        # Return the best answer or indicate no answer found
        if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }

        # Cache the result
        self.answer_cache[cache_key] = result
        return result

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)

context_manager = ContextManager(max_contexts=10)
context_manager.add_context("python", """
  Python is a high-level, interpreted programming language created by Guido van Rossum and
  released in 1991.  Python's design philosophy emphasizes code readability with its
  notable use of significant whitespace. Python features a dynamic type system and
  automatic memory management and supports multiple programming paradigms, including
  structured, object-oriented, and functional programming.
""")
context_manager.add_context("machine_learning", """
  Machine learning is a field of study that gives computers the ability to learn without
  being explicitly programmed. It is a branch of artificial intelligence based on the idea
  that systems can learn from data, identify patterns and make decisions with minimal
  human intervention.
""")

config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
question = "Who created Python?"
relevant_contexts = context_manager.search_relevant_context(question, top_k=1)
if relevant_contexts:
    relevance, context_id = relevant_contexts[0]
    context = context_manager.get_context(context_id)
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Most relevant context: {context_id} (relevance: {relevance:.2f})")
    # 打印输出 / Print output
    print(context)

    answer = qa_system.get_answer(question, context, config)
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
else:
    # 打印输出 / Print output
    print("No relevant context found.")
```

---

➡️ **Next / 下一步**: File 5 of 5

---

### Complete

# 05 — Complete / 05 Complete

**Chapter 11 — File 5 of 5 / 第11章 — 第5个文件（共5个）**

---

## Summary / 总结

This script demonstrates **Initialize pipeline for simple queries and answer cache**.

本脚本演示 **Initialize pipeline for simple queries and answer cache**。

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
import collections
# 导入时间处理模块 / Import time module
import time
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5

class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
```

---
## Step 2 — Initialize pipeline for simple queries and answer cache

```python
self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word
```

---
## Step 3 — Add the last chunk if it's not empty

```python
if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
```

---
## Step 4 — Check cache

```python
cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
```

---
## Step 5 — Preprocess context into chunks

```python
context_chunks = self.preprocess_context(context, config.max_sequence_length)
```

---
## Step 6 — Get answers from all chunks

```python
answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)
```

---
## Step 7 — Return the best answer or indicate no answer found

```python
if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }
```

---
## Step 8 — Cache the result

```python
self.answer_cache[cache_key] = result
        return result

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)


context_manager = ContextManager(max_contexts=10)
context_manager.add_context("python", """
  Python is a high-level, interpreted programming language created by Guido van Rossum and
  released in 1991. Python's design philosophy emphasizes code readability with its
  notable use of significant whitespace. Python features a dynamic type system and
  automatic memory management and supports multiple programming paradigms, including
  structured, object-oriented, and functional programming.
""")
context_manager.add_context("machine_learning", """
  Machine learning is a field of study that gives computers the ability to learn without
  being explicitly programmed. It is a branch of artificial intelligence based on the idea
  that systems can learn from data, identify patterns and make decisions with minimal
  human intervention.
""")

config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
question = "Who created Python?"
relevant_contexts = context_manager.search_relevant_context(question, top_k=1)
if relevant_contexts:
    relevance, context_id = relevant_contexts[0]
    context = context_manager.get_context(context_id)
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Most relevant context: {context_id} (relevance: {relevance:.2f})")
    # 打印输出 / Print output
    print(context)

    answer = qa_system.get_answer(question, context, config)
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
else:
    # 打印输出 / Print output
    print("No relevant context found.")
```

---
## Learning Notes / 学习笔记

- **概念**: Initialize pipeline for simple queries and answer cache 是机器学习中的常用技术。  
  *Initialize pipeline for simple queries and answer cache is a common technique in machine learning.*

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
# Complete / 05 Complete
# Complete Code / 完整代码
# ===============================

import collections
# 导入时间处理模块 / Import time module
import time
from dataclasses import dataclass

# 导入PyTorch深度学习框架 / Import PyTorch deep learning framework
import torch
# 导入HuggingFace Transformers库 / Import HuggingFace Transformers library
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline

@dataclass
class QAConfig:
    """Configuration for QA settings"""
    max_sequence_length: int = 512
    max_answer_length: int = 50
    top_k: int = 3
    threshold: float = 0.5

class QASystem:
    """Q&A system with chunking"""
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, model_name="distilbert-base-uncased-distilled-squad", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)

        # Initialize pipeline for simple queries and answer cache
        self.qa_pipeline = pipeline("question-answering", model=self.model,
                                    tokenizer=self.tokenizer, device=self.device)
        self.answer_cache = {}

    def preprocess_context(self, context, max_length=512):
      """Split long contexts into chunks below max_length"""
      chunks = []
      current_chunk = []
      current_length = 0

      for word in context.split():
          # 获取长度 / Get length
          if current_length + 1 + len(word) > max_length:
              # 添加元素到列表末尾 / Append element to list end
              chunks.append(" ".join(current_chunk))
              current_chunk = [word]
              # 获取长度 / Get length
              current_length = len(word)
          else:
              # 添加元素到列表末尾 / Append element to list end
              current_chunk.append(word)
              # 获取长度 / Get length
              current_length += 1 + len(word)  # length of space + word

      # Add the last chunk if it's not empty
      if current_chunk:
          # 添加元素到列表末尾 / Append element to list end
          chunks.append(" ".join(current_chunk))

      return chunks

    def get_answer(self, question, context, config):
        """Get answer with confidence score"""
        # Check cache
        cache_key = (question, context)
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # Preprocess context into chunks
        context_chunks = self.preprocess_context(context, config.max_sequence_length)

        # Get answers from all chunks
        answers = []
        for chunk in context_chunks:
            result = self.qa_pipeline(question=question,
                                      context=chunk,
                                      max_answer_len=config.max_answer_length,
                                      top_k=config.top_k)
            assert isinstance(result, list)
            for answer in result:
                if answer["score"] >= config.threshold:
                    # 添加元素到列表末尾 / Append element to list end
                    answers.append(answer)

        # Return the best answer or indicate no answer found
        if answers:
            best_answer = max(answers, key=lambda x: x["score"])
            result = {
                "answer": best_answer["answer"],
                "confidence": best_answer["score"],
            }
        else:
            result = {
                "answer": "No answer found",
                "confidence": 0.0,
            }

        # Cache the result
        self.answer_cache[cache_key] = result
        return result

class ContextManager:
    # 初始化：定义模型的所有层和参数 / Init: define all layers and parameters
    def __init__(self, max_contexts=10):
        self.contexts = collections.OrderedDict()
        self.max_contexts = max_contexts

    def add_context(self, context_id, context):
        """Add context with automatic cleanup"""
        # 获取长度 / Get length
        if len(self.contexts) >= self.max_contexts:
            self.contexts.popitem(last=False)
        self.contexts[context_id] = context

    def get_context(self, context_id):
        """Get context by ID"""
        return self.contexts.get(context_id)

    def search_relevant_context(self, question, top_k=3):
        """Search for relevant contexts based on relevance score"""
        relevant_contexts = []
        # 获取字典的键值对 / Get dict key-value pairs
        for context_id, context in self.contexts.items():
            relevance_score = self._calculate_relevance(question, context)
            # 添加元素到列表末尾 / Append element to list end
            relevant_contexts.append((relevance_score, context_id))
        return sorted(relevant_contexts, reverse=True)[:top_k]

    def _calculate_relevance(self, question, context):
        """Calculate relevance score between question and context.
        This is a simple counting the number of overlap words
        """
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        # 获取长度 / Get length
        return len(question_words.intersection(context_words)) / len(question_words)


context_manager = ContextManager(max_contexts=10)
context_manager.add_context("python", """
  Python is a high-level, interpreted programming language created by Guido van Rossum and
  released in 1991. Python's design philosophy emphasizes code readability with its
  notable use of significant whitespace. Python features a dynamic type system and
  automatic memory management and supports multiple programming paradigms, including
  structured, object-oriented, and functional programming.
""")
context_manager.add_context("machine_learning", """
  Machine learning is a field of study that gives computers the ability to learn without
  being explicitly programmed. It is a branch of artificial intelligence based on the idea
  that systems can learn from data, identify patterns and make decisions with minimal
  human intervention.
""")

config = QAConfig(max_sequence_length=512, max_answer_length=50, threshold=0.5)
qa_system = QASystem()
question = "Who created Python?"
relevant_contexts = context_manager.search_relevant_context(question, top_k=1)
if relevant_contexts:
    relevance, context_id = relevant_contexts[0]
    context = context_manager.get_context(context_id)
    # 打印输出 / Print output
    print(f"Question: {question}")
    # 打印输出 / Print output
    print(f"Most relevant context: {context_id} (relevance: {relevance:.2f})")
    # 打印输出 / Print output
    print(context)

    answer = qa_system.get_answer(question, context, config)
    # 打印输出 / Print output
    print(f"Answer: {answer['answer']}")
    # 打印输出 / Print output
    print(f"Confidence: {answer['confidence']:.2f}")
else:
    # 打印输出 / Print output
    print("No relevant context found.")
```

---

### Chapter Summary / 章节总结

# Chapter 11 Summary / 第11章总结

## Theme / 主题: Chapter 11 / Chapter 11

This chapter contains **5 code files** demonstrating chapter 11.

本章包含 **5 个代码文件**，演示Chapter 11。

---
## Evolution / 演化路线

  1. `01_qna.ipynb` — Qna
  2. `02_chunking.ipynb` — Chunking
  3. `03_context.ipynb` — Context
  4. `04_context.ipynb` — Context
  5. `05_complete.ipynb` — Complete

---
## ML Relevance / ML 关联

The techniques in this chapter (Chapter 11) are fundamental building blocks in machine learning pipelines.

本章技术（Chapter 11）是机器学习流水线中的基础构建块。

---
