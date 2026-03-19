# LU decomposition
from numpy import array
from scipy.linalg import lu
# define a square matrix
A = array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]])
print(A)
# factorize
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
# alternative syntax in Python 3.5: B = P @ L @ U
B = P.dot(L).dot(U)
print(B)
