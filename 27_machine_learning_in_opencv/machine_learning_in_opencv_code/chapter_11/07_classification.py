import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Check the shape of the 'imgs' array
print(imgs.shape)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the image data
_, clusters, centers = cv2.kmeans(data=imgs.astype(np.float32), K=10, bestLabels=None,
                                  criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Visualize the cluster centers
fig, ax = plt.subplots(2, 5)

for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

plt.show()

# Cluster labels
labels = np.array([2, 0, 7, 5, 1, 4, 6, 9, 3, 8])

labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
print("Accuracy: {0:.2f}%".format(accuracy))
