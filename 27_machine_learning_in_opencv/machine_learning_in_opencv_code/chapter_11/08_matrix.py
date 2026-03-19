import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Cluster labels
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Print confusion matrix
print(confusion_matrix(labels_true, labels_pred))
