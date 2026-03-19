# ===============================
# Summary / 总结
# ===============================
# This script demonstrates LU decomposition of a square matrix
# using SciPy, and verifies the result by reconstructing the
# original matrix from its factors P, L, U.
#
# 本脚本演示使用 SciPy 对方阵进行 LU 分解，
# 并通过 P·L·U 重构原矩阵来验证分解结果。

# ===============================
# Import libraries / 导入库
# ===============================
from numpy import array   # 创建数组 / create arrays
from scipy.linalg import lu  # LU 分解函数 / LU factorization

# ===============================
# Define a square matrix / 定义方阵
# ===============================
A = array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
print(A)

# ===============================
# LU factorization / LU 分解
# ===============================
# A = P · L · U
#   P: permutation matrix / 置换矩阵（行交换记录）
#   L: lower triangular   / 下三角矩阵
#   U: upper triangular   / 上三角矩阵
P, L, U = lu(A)
print(P)  # permutation matrix / 置换矩阵
print(L)  # lower triangular   / 下三角矩阵
print(U)  # upper triangular   / 上三角矩阵

# ===============================
# Reconstruct original matrix / 重构原矩阵
# ===============================
B = P @ L @ U  # P·L·U should equal A / P·L·U 应等于原矩阵 A
print(B)

# ===============================
# Learning Notes / 学习笔记
# ===============================
# - LU 分解将矩阵拆为「置换 × 下三角 × 上三角」，是高斯消元法的矩阵表达。
#   LU decomposition expresses Gaussian elimination in matrix form: A = P·L·U.
# - 在 ML 中常用于高效求解线性方程组 Ax=b 和计算行列式。
#   In ML it enables efficient solving of linear systems Ax=b and determinant computation.
