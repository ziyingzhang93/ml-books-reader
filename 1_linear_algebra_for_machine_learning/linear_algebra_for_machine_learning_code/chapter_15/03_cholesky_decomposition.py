# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define symmetrical matrix
A = array([
	[2, 1, 1],
	[1, 2, 1],
	[1, 1, 2]])
print(A)
# factorize
L = cholesky(A)
print(L)
# reconstruct
# alternative syntax in Python 3.5: B = L @ L.T
B = L.dot(L.T)
print(B)
