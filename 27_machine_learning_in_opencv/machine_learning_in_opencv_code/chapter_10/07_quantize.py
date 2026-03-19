import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('Images/bricks.jpg')

# Convert it from BGR to RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image to an Mx3 array
img_data = img_RGB.reshape(-1, 3)

# Find the number of unique RGB values
print(len(np.unique(img_data, axis=0)), 'unique RGB values out of',
      img_data.shape[0], 'pixels')

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the pixel values
compactness, labels, centers = cv2.kmeans(data=img_data.astype(np.float32), K=5,
                                          bestLabels=None, criteria=criteria, attempts=10,
                                          flags=cv2.KMEANS_RANDOM_CENTERS)

# Apply the RGB values of the cluster centers to all pixel labels
colors = centers[labels].reshape(-1, 3)

# Find the number of unique RGB values
print(len(np.unique(colors, axis=0)), 'unique RGB values out of',
      img_data.shape[0], 'pixels')

# Reshape array to the original image shape
img_colors = colors.reshape(img_RGB.shape)

# Display the quantized image
plt.imshow(img_colors.astype(np.uint8))
plt.show()
