import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()
