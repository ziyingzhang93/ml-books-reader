# sum values row-wise
from numpy import asarray
# define data as a list
data = [[1,2,3], [4,5,6]]
# convert to a numpy array
data = asarray(data)
# summarize the array content
print(data)
# sum data by row
result = data.sum(axis=1)
# summarize the result
print(result)
