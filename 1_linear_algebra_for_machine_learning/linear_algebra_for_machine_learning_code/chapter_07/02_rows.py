# enumerate rows in a numpy array
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# step through rows
for row in range(data.shape[0]):
	print(data[row, :])
