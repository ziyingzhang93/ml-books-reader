# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 23

---

### Show Tar

# 23.1 — Read TAR Archive / 读取TAR档案

**Chapter 23 — File 1 of 2 / 第23章 — 第1个文件（共2个）**

## Summary / 总结

Demonstrates how to read and extract files from compressed tar.gz archives. This is useful when datasets are distributed as archived files.

演示如何从压缩的tar.gz档案中读取和提取文件。当数据集作为压缩文件分发时很有用。

## TAR Archive Basics / TAR档案基础

- TAR: Tape Archive format for grouping multiple files
- Compression options: .tar (uncompressed), .tar.gz (gzip), .tar.bz2 (bzip2)
- Common for distributing datasets in machine learning

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data


## Step 1 — Import Libraries / 导入库

```python
# Import tarfile module for reading archives
# 导入tarfile模块用于读取档案
import tarfile
# 导入操作系统接口 / Import OS interface
import os
```

## Step 2 — About TAR Archives / 关于TAR档案

```python
# TAR file information
# TAR文件信息
# 打印输出 / Print output
print("TAR Archive Information:")
# 打印输出 / Print output
print("="*50)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("TAR is a standard archive format in Unix/Linux systems.")
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Common extensions:")
# 打印输出 / Print output
print("  .tar       - Uncompressed TAR archive")
# 打印输出 / Print output
print("  .tar.gz    - TAR archive compressed with gzip")
# 打印输出 / Print output
print("  .tar.bz2   - TAR archive compressed with bzip2")
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Benefits:")
# 打印输出 / Print output
print("  - Preserves file structure and permissions")
# 打印输出 / Print output
print("  - Smaller file size with compression")
# 打印输出 / Print output
print("  - Universal support on Unix/Linux/Mac")
# 打印输出 / Print output
print("  - Partial extraction without full decompression")
```

## Step 3 — Open and List TAR Archive / 打开并列出TAR档案

```python
# Example code for reading TAR archives
# This example shows the pattern (requires actual tar file to run)
# 读取TAR档案的示例代码

example_code = '''
# Open a TAR archive
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    # List all files in the archive
    # 打印输出 / Print output
    print("Files in archive:")
    for member in tar.getmembers():
        # 打印输出 / Print output
        print(f"  {member.name} ({member.size} bytes)")
'''

# 打印输出 / Print output
print("Example: Opening TAR archive and listing contents")
# 打印输出 / Print output
print("="*50)
# 打印输出 / Print output
print(example_code)
```

## Step 4 — Extract Files / 提取文件

```python
# Example code for extracting files
# 提取文件的示例代码

example_code = '''
# Extract specific file from archive
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    # Extract a single file
    member = tar.getmember('data.csv')
    tar.extract(member, path='/extract/path')
    
    # Or extract all files
    tar.extractall(path='/extract/path')
'''

# 打印输出 / Print output
print("Example: Extracting files from archive")
# 打印输出 / Print output
print("="*50)
# 打印输出 / Print output
print(example_code)
```

## Step 5 — Read File Content from Archive / 从档案读取文件内容

```python
# Example code for reading file content without extracting
# 不提取就读取文件内容的示例代码

example_code = '''
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# Read CSV file directly from TAR archive
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    # Get file from archive
    member = tar.getmember('data.csv')
    # Read the file object
    file_obj = tar.extractfile(member)
    # Load with pandas
    # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
    df = pd.read_csv(file_obj)
    # 查看前几行数据（快速预览） / View first rows (quick preview)
    print(df.head())
'''

# 打印输出 / Print output
print("Example: Reading file content directly from archive")
# 打印输出 / Print output
print("="*50)
# 打印输出 / Print output
print(example_code)
```

```python
## Learning Notes / 学习笔记

- **Data Distribution**: Many ML datasets (Book Recommender, Large Scale datasets) use TAR format for efficient distribution. Understanding TAR archives is essential for data engineers and ML practitioners.
  
  **数据分发**：许多ML数据集使用TAR格式进行高效分发。理解TAR档案对数据工程师和ML从业者至关重要。

- **ML Application**: (1) TAR archives preserve directory structure, useful when datasets contain multiple organized files, (2) Compressed archives (tar.gz) save bandwidth for downloading large datasets, (3) You can read data directly from archives without extracting to disk, saving storage space for very large datasets.
  
  **ML应用**：(1) TAR档案保留目录结构，(2) 压缩档案节省带宽，(3) 可以直接从档案读取数据，节省存储空间。
```

➡️ **Next / 下一步**: `02_recommender.ipynb` — Building a book recommender system

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `head()` | 查看前几行数据 | View first few rows |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

```python
# --- Import Section / 导入部分 ---
import tarfile
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd

# --- Open TAR Archive / 打开TAR档案 ---
# Example pattern (requires actual tar file)
# with tarfile.open('archive.tar.gz', 'r:gz') as tar:
#     # List files
#     for member in tar.getmembers():
#         print(f"{member.name} ({member.size} bytes)")

# --- Read Data from Archive / 从档案读取数据 ---
# with tarfile.open('archive.tar.gz', 'r:gz') as tar:
#     member = tar.getmember('ratings.csv')
#     file_obj = tar.extractfile(member)
#     df = pd.read_csv(file_obj)
#     print(df.head())
```

---

### Recommender

```python
# 23.2 — Book Recommender System / 书籍推荐系统

**Chapter 23 — File 2 of 2 / 第23章 — 第2个文件（共2个）**

## Summary / 总结

Build a book recommender system using SVD-based collaborative filtering. Demonstrates how matrix decomposition can extract latent factors for personalized recommendations.

使用基于SVD的协同过滤构建书籍推荐系统。演示矩阵分解如何提取潜在因子进行个性化推荐。

## Recommender System Concept / 推荐系统概念

**Collaborative Filtering**: If user A and B rate books similarly, recommend books B liked to A
**SVD Approach**: Decompose user-book rating matrix to find latent features and user similarities
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 加载和准备数据 / Load and prepare data
- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance


---
## Code Flow / 代码流程

```
  📂 加载数据 / Load Data
       │
       ▼
  🔧 数据预处理 / Preprocess Data
       │
       ▼
  📊 评估模型 / Evaluate Model
```

## Step 1 — Import Libraries / 导入库

```python
# Import libraries for recommendation system
# 导入推荐系统库
import tarfile
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import TruncatedSVD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity
```

## Step 2 — Load Data from TAR Archive / 从TAR档案加载数据

```python
# Code pattern for loading book rating data from TAR archive
# TAR档案中加载书籍评分数据的代码模式

example_code = '''
# Extract ratings data
with tarfile.open('ratings.tar.gz', 'r:gz') as tar:
    # List contents
    for member in tar.getmembers():
        # 打印输出 / Print output
        print(f"{member.name}")
    
    # Load ratings
    member = tar.getmember('ratings.csv')
    file_obj = tar.extractfile(member)
    # 从CSV文件读取数据为DataFrame / Read CSV file into DataFrame
    ratings = pd.read_csv(file_obj)
    # 查看数据形状（行数, 列数） / Check data shape (rows, columns)
    print(f"Ratings shape: {ratings.shape}")
    # 查看前几行数据（快速预览） / View first rows (quick preview)
    print(ratings.head())
'''

# 打印输出 / Print output
print("Data Loading Pattern (Book Recommender Dataset):")
# 打印输出 / Print output
print("="*60)
# 打印输出 / Print output
print(example_code)
# 打印输出 / Print output
print()
# 打印输出 / Print output
print("Expected columns: user_id, book_id, rating")
```

## Step 3 — Create Rating Matrix / 创建评分矩阵

```python
# Code pattern for creating user-item matrix
# 创建用户项目矩阵的代码模式

example_code = '''
# Create user-book rating matrix
# Rows: users, Columns: books, Values: ratings
rating_matrix = ratings.pivot_table(
    index='user_id',
    columns='book_id',
    values='rating',
    fill_value=0  # Fill missing ratings with 0
)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Rating matrix shape: {rating_matrix.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"  Users: {rating_matrix.shape[0]}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"  Books: {rating_matrix.shape[1]}")
# 打印输出 / Print output
print(f"  Sparsity: {(rating_matrix == 0).sum().sum() / rating_matrix.size * 100:.1f}%")
'''

# 打印输出 / Print output
print("Rating Matrix Creation:")
# 打印输出 / Print output
print("="*60)
# 打印输出 / Print output
print(example_code)
```

```python
## Step 4 — Apply SVD for Latent Factors / 应用SVD提取潜在因子
```

```python
# SVD for collaborative filtering
# SVD用于协同过滤

example_code = '''
# Apply Truncated SVD to extract latent factors
# Decompose: R ≈ U * Sigma * V^T
# where U: user factors, V: book factors

n_factors = 50  # Number of latent factors
svd = TruncatedSVD(n_components=n_factors)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
user_factors = svd.fit_transform(rating_matrix)
book_factors = svd.components_.T

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"User factors shape: {user_factors.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Book factors shape: {book_factors.shape}")
# 打印输出 / Print output
print(f"\nExplained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
# 打印输出 / Print output
print(f"This means {svd.explained_variance_ratio_.sum()*100:.1f}% of rating variation is captured")
'''

# 打印输出 / Print output
print("SVD Factorization for Latent Factors:")
# 打印输出 / Print output
print("="*60)
# 打印输出 / Print output
print(example_code)
```

## Step 5 — Compute User Similarity / 计算用户相似性

```python
# Compute similarity between users
# 计算用户之间的相似性

example_code = '''
# Compute cosine similarity between users based on latent factors
# Users with similar latent factors have similar tastes
user_similarity = cosine_similarity(user_factors)

# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"User similarity matrix shape: {user_similarity.shape}")
# 打印输出 / Print output
print(f"\nSimilarity statistics:")
# 打印输出 / Print output
print(f"  Min similarity: {user_similarity[user_similarity < 1].min():.4f}")
# 打印输出 / Print output
print(f"  Max similarity: {user_similarity[user_similarity < 1].max():.4f}")

# Find most similar users
user_id = 0
similar_users = np.argsort(user_similarity[user_id])[-6:-1]  # Top 5 similar users
# 打印输出 / Print output
print(f"\nUsers most similar to user {user_id}:")
for similar_user in similar_users:
    sim_score = user_similarity[user_id, similar_user]
    # 打印输出 / Print output
    print(f"  User {similar_user}: similarity = {sim_score:.4f}")
'''

# 打印输出 / Print output
print("Computing User Similarity:")
# 打印输出 / Print output
print("="*60)
# 打印输出 / Print output
print(example_code)
```

## Step 6 — Generate Recommendations / 生成推荐

```python
# Generate recommendations
# 生成推荐

example_code = '''
def recommend_books(user_id, n_recommendations=5):
    """
    Recommend books for a user based on similar users' ratings
    基于相似用户的评分为用户推荐书籍
    """
    # Find similar users
    similar_users_idx = np.argsort(user_similarity[user_id])[-11:-1]  # Top 10
    
    # Get books rated by similar users that target user hasn't rated
    # 获取相似用户评分但目标用户未评分的书籍
    user_rated = set(np.nonzero(rating_matrix.iloc[user_id])[0])
    
    # Aggregate recommendations from similar users
    recommendations = {}
    for sim_user in similar_users_idx:
        sim_user_rated = np.nonzero(rating_matrix.iloc[sim_user])[0]
        for book_id in sim_user_rated:
            if book_id not in user_rated:
                if book_id not in recommendations:
                    recommendations[book_id] = []
                sim_score = user_similarity[user_id, sim_user]
                book_rating = rating_matrix.iloc[sim_user, book_id]
                # Weight rating by similarity score
                # 添加元素到列表末尾 / Append element to list end
                recommendations[book_id].append(sim_score * book_rating)
    
    # Average scores and rank
    # 计算均值 / Calculate mean
    rec_scores = {book: np.mean(scores) for book, scores in recommendations.items()}
    # 获取字典的键值对 / Get dict key-value pairs
    top_books = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    return top_books

# Get recommendations
user_id = 0
recommendations = recommend_books(user_id, n_recommendations=5)
# 打印输出 / Print output
print(f"Top 5 book recommendations for user {user_id}:")
# 同时获取索引和值 / Get both index and value
for rank, (book_id, score) in enumerate(recommendations, 1):
    # 打印输出 / Print output
    print(f"  {rank}. Book {book_id}: score = {score:.4f}")
'''

# 打印输出 / Print output
print("Recommendation Generation:")
# 打印输出 / Print output
print("="*60)
# 打印输出 / Print output
print(example_code)
```

## Learning Notes / 学习笔记

- **Math Essence**: SVD-based collaborative filtering decomposes the user-item rating matrix into latent factors. Users and items are represented in a low-dimensional space where similarity correlates with rating compatibility. This enables finding "hidden" patterns in user preferences.
  
  **数学本质**：基于SVD的协同过滤将用户项目评分矩阵分解为潜在因子。用户和项目在低维空间中表示，其中相似性与评分兼容性相关。

- **ML Application**: (1) Collaborative filtering works without explicit features, relying only on user-item interactions, (2) SVD reduces dimensionality (50 factors << thousands of books) while preserving rating patterns, (3) Recommendation quality depends on data sparsity - systems with sparse data use more advanced techniques (neural networks, deep learning), (4) This approach powers Netflix, Amazon, and Spotify recommendations at scale.
  
  **ML应用**：(1) 协同过滤无需显式特征，仅依赖用户项目交互，(2) SVD降低维度同时保留评分模式，(3) 推荐质量取决于数据稀疏性，(4) 此方法驱动Netflix、Amazon和Spotify的大规模推荐。

### Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `head()` | 查看前几行数据 | View first few rows |
| `np.mean` | 计算均值 | Calculate mean |
| `numpy` | 数值计算库 | Numerical computing library |
| `pandas` | 数据分析库 | Data analysis library |
| `read_csv` | 读取CSV文件 | Read CSV file |

➡️ **Next / 下一步**: `../chapter_24/01_eigenface.ipynb` — Eigenfaces for face recognition

## Complete Code / 完整代码一览

```python
# --- Import Section / 导入部分 ---
import tarfile
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
# 导入Pandas数据分析库 / Import Pandas data analysis library
import pandas as pd
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import TruncatedSVD
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data from Archive / 从档案加载数据 ---
# with tarfile.open('ratings.tar.gz', 'r:gz') as tar:
#     member = tar.getmember('ratings.csv')
#     file_obj = tar.extractfile(member)
#     ratings = pd.read_csv(file_obj)

# --- Create Rating Matrix / 创建评分矩阵 ---
# rating_matrix = ratings.pivot_table(
#     index='user_id',
#     columns='book_id',
#     values='rating',
#     fill_value=0
# )

# --- SVD and Recommendations / SVD和推荐 ---
# svd = TruncatedSVD(n_components=50)
# user_factors = svd.fit_transform(rating_matrix)
# user_similarity = cosine_similarity(user_factors)

# --- Find Similar Users and Recommend / 查找相似用户和推荐 ---
# user_id = 0
# similar_users = np.argsort(user_similarity[user_id])[-6:-1]
# print(f"Users similar to user {user_id}: {similar_users}")
```

---

### Chapter Summary / 章节总结

# Chapter 23 Summary / 第23章总结：Recommender System via SVD

## Theme / 主题

Recommender systems predict what users will like. The classic approach uses SVD to decompose a sparse user-item matrix into latent factors. Low-rank approximation recovers missing ratings. This chapter demonstrates SVD's power on real collaborative filtering data—from theory (SVD = U·Σ·V^T) to practice (predict unseen ratings).

推荐系统预测用户喜欢什么。经典方法使用SVD将稀疏的用户-项目矩阵分解为潜在因子。低秩逼近恢复缺失的评分。本章在真实协作过滤数据上演示了SVD的能力——从理论（SVD = U·Σ·V^T）到实践（预测看不见的评分）。

## Evolution / 演化路线

```
01_load_data.ipynb
    └─ Load ratings data archive (加载评分数据存档)
       User-item matrix: rows=users, cols=items, values=ratings
    
02_svd_factorization.ipynb
    └─ Decompose with SVD: A = U·Σ·V^T (使用SVD分解)
       U = user factors (潜在用户偏好)
       V = item factors (潜在项目特征)
       Σ = importance of each factor (因子重要性)
    
03_low_rank_approximation.ipynb
    └─ Keep top-k factors, truncate rest (保留前k个因子)
       A_k = U_k·Σ_k·V_k^T approximates A
       Recovers missing ratings via matrix completion
```

## Progression Logic / 进度逻辑

SVD-based recommendation progresses through **data → decomposition → prediction**:

1. **Data**: m × n user-item matrix (m users, n items)
   - Most entries are missing (sparse)
   - Observed entries are user ratings

2. **SVD**: A = U·Σ·V^T
   - U (m × m): user factors (how much each user likes each latent feature)
   - V (n × n): item factors (how much each item has each latent feature)
   - Σ: singular values (importance ranking)

3. **Low-rank approximation**: Keep top-k
   - Truncate small singular values (noise)
   - A_k = U_k·Σ_k·V_k^T ≈ A
   - Fills in missing entries: A_k[i,j] = predicted rating for user i, item j

4. **Prediction**: For unobserved pairs (user, item), use A_k[i,j]

The insight: **Latent factors capture user preferences and item properties**. User "action movie fan" + item "has explosions" → high rating.

SVD推荐通过**数据→分解→预测**进行：

1. **数据**：m × n用户-项目矩阵（m个用户，n个项目）
   - 大多数条目缺失（稀疏）
   - 观察的条目是用户评分

2. **SVD**：A = U·Σ·V^T
   - U（m × m）：用户因子（每个用户有多少喜欢每个潜在特征）
   - V（n × n）：项目因子（每个项目有多少有每个潜在特征）
   - Σ：奇异值（重要性排序）

3. **低秩逼近**：保留前k个
   - 截断小奇异值（噪声）
   - A_k = U_k·Σ_k·V_k^T ≈ A
   - 填入缺失条目：A_k[i,j] =用户i的预测评分，项目j

4. **预测**：对于未观察的对（用户、项目），使用A_k[i,j]

见解：**潜在因子捕获用户偏好和项目属性**。用户"动作电影迷" +项目"有爆炸" →高评分。

## ML Relevance / 机器学习相关性

In machine learning:
- **Collaborative filtering**:
  - Assumes: similar users like similar items
  - User-user: find k similar users, average their ratings
  - Item-item: find k similar items, average their ratings
  - Matrix factorization (SVD): latent factor approach (this chapter)

- **Why SVD works**:
  - User-item matrix is very sparse (users rate few items)
  - Low-rank structure: humans have ~k distinct preference types
  - SVD reveals this latent structure
  - Top-k truncation = noise filtering + dimensionality reduction

- **Latent factor interpretation**:
  - Factor 1: might be "comedy" (action=0, drama=0, comedy=1)
  - Factor 2: might be "violence" (high for action, low for family)
  - User vector: how much they like each factor
  - Item vector: how much it has each property
  - Prediction: user[k] × item[k] summed over all k factors

- **Advantages of SVD approach**:
  - Handles sparsity naturally (only stores nonzeros)
  - Scalable: O(mnk) for m users, n items, k factors
  - Interpretable: factors have semantic meaning
  - Flexible: can recommend to new users (cold start with side info)

- **Limitations**:
  - Cold start: new users/items with no ratings
  - Context ignored: doesn't use temporal (when), location, etc.
  - Diversity: tends to recommend obvious items

- **Real-world improvements**:
  - Regularization: L2 norm to prevent overfitting
  - Temporal dynamics: users' preferences change over time
  - Implicit feedback: clicks, views, not just explicit ratings
  - Neural approaches: deep learning for richer latent factors
  - Hybrid: combine content-based (item features) + collaborative

- **Netflix Prize example**:
  - Netflix used matrix factorization for recommendations
  - SVD-based approaches won or placed highly
  - Real-world proof that linear algebra powers ML

- **Modern alternatives**:
  - Neural collaborative filtering: learns factors via deep learning
  - Two-tower models: separate embeddings for users and items
  - Graph neural networks: model user-item bipartite graph
  - But SVD remains fast, interpretable baseline

**Key insight**: Recommender systems are matrix completion problems. Given sparse observations, fill in missing entries. SVD is the classical solution; neural approaches add nonlinearity and capacity.

在机器学习中：
- **协作过滤**：
  - 假设：相似的用户喜欢相似的项目
  - 用户-用户：找到k个相似用户，平均他们的评分
  - 项目-项目：找到k个相似项目，平均他们的评分
  - 矩阵因子分解（SVD）：潜在因子方法（本章）

- **为什么SVD有效**：
  - 用户-项目矩阵非常稀疏（用户评分很少项目）
  - 低秩结构：人类有~k种不同的偏好类型
  - SVD揭示了这种潜在结构
  - 前k个截断=噪声过滤+降维

- **潜在因子解释**：
  - 因子1：可能是"喜剧"（动作=0、戏剧=0、喜剧=1）
  - 因子2：可能是"暴力"（动作高，家庭低）
  - 用户向量：他们喜欢每个因子的程度
  - 项目向量：它有多少每个属性
  - 预测：user[k] × item[k]在所有k个因子中求和

- **SVD方法的优点**：
  - 自然处理稀疏性（仅存储非零值）
  - 可扩展：O(mnk)用于m个用户、n个项目、k个因子
  - 可解释：因子有语义意义
  - 灵活：可以向新用户推荐（冷启动与侧面信息）

- **限制**：
  - 冷启动：没有评分的新用户/项目
  - 上下文忽略：不使用时间（何时）、位置等
  - 多样性：倾向于推荐明显的项目

- **现实改进**：
  - 正则化：L2范数以防止过拟合
  - 时间动态：用户的偏好随时间变化
  - 隐式反馈：点击、浏览，而不仅仅是显式评分
  - 神经方法：深度学习以获得更丰富的潜在因子
  - 混合：结合基于内容（项目特征）+协作

- **Netflix奖励示例**：
  - Netflix使用矩阵因子分解进行推荐
  - 基于SVD的方法赢得或名列前茅
  - 线性代数如何驱动ML的现实证明

- **现代替代品**：
  - 神经协作过滤：通过深度学习学习因子
  - 双塔模型：用户和项目的单独嵌入
  - 图神经网络：模型用户-项目二部图
  - 但SVD保持快速、可解释的基线

**关键见解**：推荐系统是矩阵完成问题。给定稀疏观察，填入缺失条目。SVD是经典解决方案；神经方法添加非线性和容量。

---
