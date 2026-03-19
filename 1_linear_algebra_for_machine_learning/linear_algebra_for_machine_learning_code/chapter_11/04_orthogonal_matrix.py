# orthogonal matrix
from numpy import array
from numpy.linalg import inv
# define orthogonal matrix
Q = array([
	[1, 0],
	[0, -1]])
print(Q)
# inverse equivalence
V = inv(Q)
print(Q.T)
print(V)
# identity equivalence
# alternative syntax in Python 3.5: I = Q @ Q.T
I = Q.dot(Q.T)
print(I)
