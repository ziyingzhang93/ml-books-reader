# enumerate columns in a numpy array
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# step through columns
for col in range(data.shape[1]):
	print(data[:, col])
