# 线性代数与机器学习 / Linear Algebra for Machine Learning
## Chapter 24

---

### Eigenface

```python
# 24.1 — Eigenfaces for Face Recognition / 特征脸用于人脸识别

**Chapter 24 — File 1 of 1 / 第24章 — 第1个文件（共1个）**

## Summary / 总结

Implement face recognition using eigenfaces (PCA-based approach). Demonstrates how PCA can extract the most discriminative features from high-dimensional image data and enable efficient face recognition.

使用特征脸（基于PCA的方法）实现人脸识别。演示PCA如何从高维图像数据中提取最有区别的特征并实现高效的人脸识别。

## Eigenfaces Concept / 特征脸概念

- **Idea**: Faces lie on a subspace; PCA extracts basis vectors (eigenfaces)
- **Advantage**: Compact representation; efficient matching
- **Dataset**: AT&T Face Database (from zip file; 40 subjects, 10 images each)
```

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 数据预处理 / Data preprocessing
- 评估模型效果 / Evaluate model performance
- 可视化结果 / Visualize results

## Code Flow / 代码流程

```
   
┌───────────────────────────────┐
│  划分训练/测试集 Split Train/Test  │
└───────────────────────────────┘
  │
  ▼
┌───────────────────┐
│  可视化 Visualize  │
└───────────────────┘
```

## Step 1 — Import Libraries / 导入库

```python
# Import libraries for face recognition
# 导入人脸识别库
import zipfile
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from PIL import Image
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import euclidean_distances
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt
```

## Step 2 — Load Face Data from ZIP / 从ZIP加载人脸数据

```python
# Code pattern for loading AT&T face database
# 加载AT&T人脸数据库的代码模式

example_code = '''
# Extract and load face images from ZIP archive
# 将多个序列配对 / Pair multiple sequences
def load_faces_from_zip(zip_path):
    """
    Load AT&T face database from ZIP file
    每个文件夹代表一个人，包含10个图像
    """
    faces = []
    labels = []
    subject_id = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List files in archive
        file_list = zf.namelist()
        
        # Group by person (folder)
        for person_dir in sorted(set(f.split('/')[0] for f in file_list)):
            # Load all images for this person
            person_images = [f for f in file_list if f.startswith(person_dir)]
            
            for img_path in sorted(person_images):
                if img_path.endswith(('.pgm', '.jpg', '.png')):
                    # Read image from ZIP
                    with zf.open(img_path) as f:
                        img = Image.open(f)
                        # Flatten to 1D vector
                        # 展平为一维数组 / Flatten to 1D array
                        face_vector = np.array(img).flatten()
                        # 添加元素到列表末尾 / Append element to list end
                        faces.append(face_vector)
                        # 添加元素到列表末尾 / Append element to list end
                        labels.append(subject_id)
            subject_id += 1
    
    # 创建NumPy数组 / Create NumPy array
    return np.array(faces), np.array(labels)

# Load the data
# 将多个序列配对 / Pair multiple sequences
X, y = load_faces_from_zip('att_faces.zip')
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"Face dataset shape: {X.shape}")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"  Samples: {X.shape[0]} face images")
# 查看数据形状（行数, 列数） / Check data shape (rows, columns)
print(f"  Features: {X.shape[1]} pixels")
# 找出唯一值 / Find unique values
print(f"  Subjects: {len(np.unique(y))}")
'''

# 打印输出 / Print output
print("Loading Face Data from ZIP:")
# 打印输出 / Print output
print("="*70)
# 打印输出 / Print output
print(example_code)
```

## Step 3 — Extract Eigenfaces using PCA / 使用PCA提取特征脸

```python
# Extract eigenfaces
# 提取特征脸

example_code = '''
# Apply PCA to extract eigenfaces
# Each eigenface is a direction of variation across all faces
n_components = 50  # Number of eigenfaces to keep
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=n_components)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
faces_pca = pca.fit_transform(X)

# 打印输出 / Print output
print(f"Explained variance by eigenfaces:")
cumsum = np.cumsum(pca.explained_variance_ratio_)
# 打印输出 / Print output
print(f"  First eigenface: {pca.explained_variance_ratio_[0]:.4f}")
# 打印输出 / Print output
print(f"  First 10 eigenfaces: {cumsum[9]:.4f}")
# 打印输出 / Print output
print(f"  First 50 eigenfaces: {cumsum[-1]:.4f}")

# Visualize eigenfaces as images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
# 同时获取索引和值 / Get both index and value
for i, ax in enumerate(axes.flat):
    # Reshape eigenface back to image
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    eigenface = pca.components_[i].reshape(46, 56)  # AT&T image size
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
    ax.axis('off')
plt.tight_layout()
# 显示图表 / Display the plot
plt.show()

# 打印输出 / Print output
print(f"\nEigenfaces represent fundamental face patterns")
# 打印输出 / Print output
print(f"Any face can be reconstructed as: face ≈ mean_face + Σ(weight_i × eigenface_i)")
'''

# 打印输出 / Print output
print("Extracting Eigenfaces (PCA):")
# 打印输出 / Print output
print("="*70)
# 打印输出 / Print output
print(example_code)
```

## Step 4 — Face Recognition by Distance / 通过距离的人脸识别

```python
# Face recognition using eigenface distance
# 使用特征脸距离的人脸识别

example_code = '''
def recognize_face(test_face, training_faces_pca, training_labels, pca_model):
    """
    Recognize a test face by finding closest match in training set
    通过在训练集中找到最接近的匹配来识别测试人脸
    """
    # Project test face to eigenface space
    # 改变数组形状（不改变数据） / Reshape array (data unchanged)
    test_face_pca = pca_model.transform(test_face.reshape(1, -1))
    
    # Compute distances to all training faces
    distances = euclidean_distances(test_face_pca, training_faces_pca)[0]
    
    # Find nearest neighbor
    nearest_idx = np.argmin(distances)
    nearest_label = training_labels[nearest_idx]
    nearest_distance = distances[nearest_idx]
    
    return nearest_label, nearest_distance

# Test on a face
test_idx = 0
test_face = X[test_idx]
true_label = y[test_idx]

# Recognize
predicted_label, distance = recognize_face(test_face, faces_pca, y, pca)

# 打印输出 / Print output
print(f"Test face: Subject {true_label}")
# 打印输出 / Print output
print(f"Predicted: Subject {predicted_label}")
# 打印输出 / Print output
print(f"Eigenface distance: {distance:.4f}")
# 打印输出 / Print output
print(f"Correct prediction: {true_label == predicted_label}")
'''

# 打印输出 / Print output
print("Face Recognition by Distance:")
# 打印输出 / Print output
print("="*70)
# 打印输出 / Print output
print(example_code)
```

## Step 5 — Evaluate Recognition Performance / 评估识别性能

```python
# Evaluate recognition accuracy
# 评估识别精度

example_code = '''
# Evaluate on all test images
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.model_selection import train_test_split
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics import accuracy_score, confusion_matrix

# Split data
# 划分训练集和测试集 / Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train PCA
# 主成分分析：降维，保留最重要的特征 / PCA: reduce dimensions, keep key features
pca = PCA(n_components=50)
# 拟合并转换数据（一步完成） / Fit and transform data (one step)
X_train_pca = pca.fit_transform(X_train)
# 用已拟合的模型转换数据 / Transform data with fitted model
X_test_pca = pca.transform(X_test)

# Recognize all test faces
predictions = []
for test_face in X_test:
    pred_label, _ = recognize_face(test_face, X_train_pca, y_train, pca)
    # 添加元素到列表末尾 / Append element to list end
    predictions.append(pred_label)

# Evaluate
# 计算准确率 = 正确预测数 / 总数 / Accuracy = correct predictions / total
accuracy = accuracy_score(y_test, predictions)
# 打印输出 / Print output
print(f"Face Recognition Accuracy: {accuracy:.2%}")
# 打印输出 / Print output
print(f"\nConfusion matrix (subset):")
# 生成混淆矩阵：展示预测对错分布 / Confusion matrix: show prediction error distribution
cm = confusion_matrix(y_test, predictions)
# 打印输出 / Print output
print(cm[:5, :5])  # Show top-left corner
'''

# 打印输出 / Print output
print("Evaluating Recognition Performance:")
# 打印输出 / Print output
print("="*70)
# 打印输出 / Print output
print(example_code)
```

```python
## Learning Notes / 学习笔记

- **Math Essence**: Eigenfaces exploits the fact that human faces occupy a subspace of image space. PCA finds this subspace, enabling compact representation (50 components << original pixel dimensions). Recognition is efficient: just compute distance in low-dimensional space.
  
  **数学本质**：特征脸利用人脸占据图像空间子空间的事实。PCA找到此子空间，实现紧凑表示。识别在低维空间中计算距离很有效。

- **ML Application**: (1) Eigenfaces was the leading face recognition approach before deep learning (1980s-2000s), (2) Still relevant for understanding dimensionality reduction and its practical applications, (3) Modern systems use CNNs (ConvNets) for better performance, but eigenfaces principles remain important, (4) Advantages: interpretable (can visualize eigenfaces), efficient. Disadvantages: lighting/pose sensitive, lower accuracy than modern methods.
  
  **ML应用**：(1) 特征脸在深度学习前是领先的人脸识别方法，(2) 仍与理解降维相关，(3) 现代系统使用CNNs获得更好性能，(4) 优势：可解释、高效。劣势：对光照/姿态敏感。
```

➡️ **Next / 下一步**: `../appendix_02/1_versions.ipynb` — Library versions

## Complete Code / 完整代码一览

---
## Glossary / 术语速查

| 术语 Term | 中文解释 | English |
|-----------|---------|---------|
| `Dataset` | 数据集基类，定义数据读取方式 | Base class defining how to read data |
| `Flatten` | 展平多维为一维 | Flatten multi-dim to 1D |
| `PCA` | 主成分分析，降维 | Principal Component Analysis, dimensionality reduction |
| `StandardScaler` | 标准化：均值=0，标准差=1 | Standardize: mean=0, std=1 |
| `accuracy_score` | 准确率：预测正确的比例 | Accuracy: proportion of correct predictions |
| `confusion_matrix` | 混淆矩阵：展示预测对错分布 | Confusion matrix: prediction error distribution |
| `fit_transform` | 拟合并转换数据 | Fit and transform data |
| `matplotlib` | 绑图库 | Plotting library |
| `np.array` | 创建NumPy数组 | Create NumPy array |
| `numpy` | 数值计算库 | Numerical computing library |
| `plt.show` | 显示图表 | Display plot |
| `plt.subplot` | 创建子图 | Create subplot |
| `predict` | 用训练好的模型做预测 | Make predictions with trained model |
| `train_test_split` | 划分训练集和测试集 | Split data into train/test sets |

```python
# --- Import Section / 导入部分 ---
import zipfile
# 导入NumPy数值计算库 / Import NumPy numerical computing library
import numpy as np
from PIL import Image
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.decomposition import PCA
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.preprocessing import StandardScaler
# 导入Scikit-learn机器学习库 / Import Scikit-learn ML library
from sklearn.metrics.pairwise import euclidean_distances
# 导入Matplotlib绑图库 / Import Matplotlib plotting library
import matplotlib.pyplot as plt

# --- Load Face Data from ZIP / 从ZIP加载人脸数据 ---
# with zipfile.ZipFile('att_faces.zip', 'r') as zf:
#     file_list = zf.namelist()
#     for filename in file_list[:5]:
#         print(filename)

# --- PCA for Eigenfaces / 特征脸的PCA ---
# pca = PCA(n_components=50)
# faces_pca = pca.fit_transform(X)
# print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# --- Face Recognition / 人脸识别 ---
# test_face_pca = pca.transform(test_face.reshape(1, -1))
# distances = euclidean_distances(test_face_pca, faces_pca)
# nearest = np.argmin(distances)
# print(f"Predicted person: {nearest}")
```

---

### Chapter Summary / 章节总结

# Chapter 24 Summary / 第24章总结：Eigenfaces for Face Recognition

## Theme / 主题

Eigenfaces is a foundational computer vision algorithm that uses eigendecomposition to recognize faces. By treating face images as high-dimensional vectors and computing their principal components (eigenfaces), we can represent any face as a linear combination of eigenfaces. This demonstrates how linear algebra—eigendecomposition in particular—enables real ML applications in a complete, end-to-end pipeline.

Eigenfaces是一个基础计算机视觉算法，使用特征分解来识别人脸。通过将人脸图像视为高维向量并计算其主成分（特征脸），我们可以将任何人脸表示为特征脸的线性组合。这演示了线性代数——特别是特征分解——如何在完整、端到端的管道中实现真实ML应用。

## Evolution / 演化路线

```
End-to-end Eigenfaces pipeline:

1. Load face images
   └─ Flatten 2D images to 1D vectors (784 → 1D)
   
2. Compute mean face
   └─ Subtract mean: center the data (subtract average face)
   
3. Compute covariance matrix
   └─ Cov = X^T · X / n (all pairwise face similarities)
   
4. Eigendecompose covariance
   └─ Find eigenvectors (eigenfaces) and eigenvalues
   
5. Project faces onto eigenfaces
   └─ Each face: X_pca = X · eigenfaces (represents as linear combo)
   
6. Recognition: compare projections
   └─ Find nearest neighbor in eigenface space → face ID
   
7. Evaluate: accuracy on test set
   └─ How many identities recognized correctly?
```

## Progression Logic / 进度逻辑

Eigenfaces follows the complete **ML pipeline** in one chapter:

1. **Data**: Real face images (28×28 pixels = 784D)
2. **Preprocessing**: Center data (subtract mean face)
3. **Feature extraction**: Eigendecomposition (PCA) → eigenfaces
4. **Representation**: Project each face onto eigenfaces
5. **Prediction**: Use 1-NN in eigenface space
6. **Evaluation**: Measure recognition accuracy

This is the complete journey: raw data → math → prediction → evaluation. Every concept from earlier chapters combines here.

Eigenfaces遵循**完整的ML管道**在一个章节中：

1. **数据**：真实人脸图像（28×28像素= 784D）
2. **预处理**：中心化数据（减去平均人脸）
3. **特征提取**：特征分解（PCA）→特征脸
4. **表示**：将每个人脸投影到特征脸上
5. **预测**：在特征脸空间中使用1-NN
6. **评估**：测量识别准确度

这是完整的旅程：原始数据→数学→预测→评估。之前所有章节的每个概念都在这里组合。

## ML Relevance / 机器学习相关性

In machine learning:
- **Face recognition with Eigenfaces**:
  - 1960s/70s method, foundational for computer vision
  - Uses PCA/eigendecomposition (not modern deep learning)
  - Modern methods use deep CNNs, but Eigenfaces explains the theory
  - Shows that linear algebra alone can solve non-trivial problems

- **Why it works**:
  - Faces have structure: all have eyes, nose, mouth in roughly same places
  - PCA captures this structure in eigenvectors (eigenfaces)
  - First eigenface: average face
  - Later eigenfaces: deviations (smiling, bald, thin, etc.)
  - Low-rank approximation: encode face using few eigenfaces

- **Representation**:
  - Raw image: 784 values (noisy, high-dimensional)
  - Eigenface representation: k values (compact, discriminative)
  - k << 784: compress 28×28 image to ~50D
  - 50D is manageable; 784D is hard for 1-NN

- **Recognition (1-NN in eigenface space)**:
  - Given test face: project to eigenface space → vector z_test
  - For each training face: project to eigenface space → z_train
  - Find argmin_train ||z_test - z_train||_2
  - Label: identity of nearest neighbor

- **Complete pipeline components**:
  - **Data cleaning**: normalize image intensities
  - **Centering**: subtract mean face (critical for PCA)
  - **Covariance**: X^T · X captures face structure
  - **Eigen-decomposition**: reveal principal modes of variation
  - **Dimensionality reduction**: keep ~50 eigenfaces (out of 784)
  - **Distance metric**: Euclidean in eigenface space
  - **Classification**: 1-NN (or k-NN)
  - **Evaluation**: accuracy, confusion matrix, per-class performance

- **Limitations**:
  - Lighting sensitive: shadows change appearance
  - Rotation/pose: faces at different angles fail
  - Occlusion: glasses, masks break recognition
  - Scale: resizing face slightly breaks it
  - Expression: happy vs. sad changes face

- **Modern improvements**:
  - Deep learning (CNNs): learn invariant features
  - 3D face models: handle pose/rotation
  - Data augmentation: artificially rotate/shade faces
  - Metric learning: learn embeddings optimized for recognition
  - Face detection + landmarks: normalize pose first

- **Why Eigenfaces still matters**:
  - Historical importance: first major success in face recognition
  - Educational: clean example of PCA + 1-NN
  - Interpretable: eigenfaces are actual face images (explainable)
  - Lightweight: can run on small devices
  - Baseline: compare modern methods against Eigenfaces

- **Complete mathematical view**:
  ```
  1. Center data: X' = X - mean_face
  2. Covariance: Cov = X'^T · X' / n
  3. Eigendecompose: Cov = Q · Λ · Q^T
  4. Eigenfaces: Q[:, :k] (first k eigenvectors as columns)
  5. Project: Z = X' · Q[:, :k]  (each face → k-dim vector)
  6. Classify: argmin_train ||z_test - z_train||_2
  ```

**Why it's the final chapter**: Eigenfaces brings together all key concepts—centering, covariance, eigendecomposition, dimensionality reduction, distance metrics, and 1-NN classification—into one cohesive, end-to-end ML system. It shows that understanding linear algebra deeply unlocks practical ML capabilities.

在机器学习中：
- **使用Eigenfaces的人脸识别**：
  - 1960年代/70年代方法，计算机视觉的基础
  - 使用PCA/特征分解（不是现代深度学习）
  - 现代方法使用深度CNN，但Eigenfaces解释理论
  - 表明仅线性代数可以解决非平凡问题

- **为什么有效**：
  - 人脸有结构：都有眼睛、鼻子、嘴在大约相同的位置
  - PCA在特征向量（特征脸）中捕获这个结构
  - 第一个特征脸：平均人脸
  - 后期特征脸：偏差（微笑、秃头、瘦弱等）
  - 低秩逼近：使用少数特征脸编码人脸

- **表示**：
  - 原始图像：784个值（嘈杂、高维）
  - 特征脸表示：k个值（紧凑、判别）
  - k << 784：将28×28图像压缩到~50D
  - 50D是可管理的；784D对于1-NN很难

- **识别（特征脸空间中的1-NN）**：
  - 给定测试人脸：投影到特征脸空间→向量z_test
  - 对于每个训练人脸：投影到特征脸空间→z_train
  - 找到argmin_train ||z_test - z_train||_2
  - 标签：最近邻的身份

- **完整管道分量**：
  - **数据清理**：规范化图像强度
  - **中心化**：减去平均人脸（对PCA至关重要）
  - **协方差**：X^T · X捕获人脸结构
  - **特征分解**：揭示变化的主要模式
  - **降维**：保留~50个特征脸（在784个中）
  - **距离度量**：特征脸空间中的欧几里得
  - **分类**：1-NN（或k-NN）
  - **评估**：准确度、混淆矩阵、每类性能

- **限制**：
  - 光线敏感：阴影改变外观
  - 旋转/姿态：不同角度的人脸失败
  - 遮挡：眼镜、口罩打破识别
  - 规模：稍微调整人脸大小会打破它
  - 表达：高兴vs.悲伤改变人脸

- **现代改进**：
  - 深度学习（CNN）：学习不变特征
  - 3D人脸模型：处理姿态/旋转
  - 数据增强：人工旋转/遮蔽人脸
  - 度量学习：学习优化识别的嵌入
  - 人脸检测+地标：先规范化姿态

- **为什么Eigenfaces仍然重要**：
  - 历史重要性：人脸识别的第一个主要成功
  - 教育：PCA + 1-NN的清晰示例
  - 可解释：特征脸是实际人脸图像（可解释）
  - 轻量级：可以在小设备上运行
  - 基线：将现代方法与Eigenfaces比较

- **完整数学视图**：
  ```
  1. 中心化数据：X' = X - mean_face
  2. 协方差：Cov = X'^T · X' / n
  3. 特征分解：Cov = Q · Λ · Q^T
  4. 特征脸：Q[:, :k]（第一个k个特征向量作为列）
  5. 投影：Z = X' · Q[:, :k]（每个人脸→k维向量）
  6. 分类：argmin_train ||z_test - z_train||_2
  ```

**为什么是最后一章**：Eigenfaces将所有关键概念——中心化、协方差、特征分解、降维、距离度量和1-NN分类——结合在一个内聚的、端到端的ML系统中。它表明深刻理解线性代数解锁实际的ML能力。

---
