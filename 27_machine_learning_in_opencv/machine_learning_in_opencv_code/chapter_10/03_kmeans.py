import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=5, cluster_std=1.5, random_state=10)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the input data
compactness, y_pred, centers = cv2.kmeans(data=x.astype(np.float32), K=5, bestLabels=None,
                                          criteria=criteria, attempts=10,
                                          flags=cv2.KMEANS_RANDOM_CENTERS)

# Plot the data clusters, each having a different color, together with the
# corresponding cluster centers
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.scatter(centers[:, 0], centers[:, 1], c='red')
plt.show()
