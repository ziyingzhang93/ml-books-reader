import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generating a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)

# Plotting the dataset
plt.scatter(x[:, 0], x[:, 1])
plt.show()
