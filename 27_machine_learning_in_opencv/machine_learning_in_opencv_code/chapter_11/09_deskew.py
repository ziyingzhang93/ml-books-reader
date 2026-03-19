import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from sklearn.metrics import confusion_matrix


def deskew_image(img):
    # Calculate the image moments
    img_moments = cv2.moments(img)

    # Moment m02 indicates how much the pixel intensities are spread out along the
    # vertical axis, mu11 is the central moment or the weight average intensity
    if abs(img_moments['mu02']) > 1e-2:
        # Calculate the image skew
        img_skew = (img_moments['mu11'] / img_moments['mu02'])

        # Generate the transformation matrix: We are here tweaking slightly the
        # approximation of vertical translation due to skew by making use of a
        # scaling factor of 0.6, because we empirically found that this value
        # worked better for this application
        m = np.float32([[1, img_skew, -0.6 * img.shape[0] * img_skew], [0, 1, 0]])

        # Apply the transformation matrix to the image
        img_deskew = cv2.warpAffine(src=img, M=m, dsize=img.shape,
                                    flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    else:
        # If the vertical spread of pixel intensities is small, return a copy of the
        # original image
        img_deskew = img.copy()

    return img_deskew

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# De-skew all dataset images
imgs_deskewed = np.zeros(imgs.shape)

for i in range(imgs_deskewed.shape[0]):
    new = deskew_image(imgs[i, :].reshape(20, 20))
    imgs_deskewed[i, :] = new.reshape(1, -1)

# Specify the algorithm's termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)

# Run the k-means clustering algorithm on the de-skewed image data
_, clusters, centers = cv2.kmeans(data=imgs_deskewed.astype(np.float32), K=10,
                                  bestLabels=None, criteria=criteria, attempts=10,
                                  flags=cv2.KMEANS_RANDOM_CENTERS)

# Reshape array into 20x20 images
imgs_centers = centers.reshape(-1, 20, 20)

# Visualize the cluster centers
fig, ax = plt.subplots(2, 5)

for i, center in zip(ax.flat, imgs_centers):
    i.imshow(center)

plt.show()

# Cluster labels
labels = np.array([9, 5, 6, 4, 2, 3, 7, 8, 1, 0])

labels_pred = np.zeros(labels_true.shape, dtype='int')

# Re-order the cluster labels
for i in range(10):
    mask = clusters.ravel() == i
    labels_pred[mask] = labels[i]

# Calculate the algorithm's accuracy
accuracy = (np.sum(labels_true == labels_pred) / labels_true.size) * 100

# Print the accuracy
print("Accuracy: {0:.2f}%".format(accuracy))

# Print confusion matrix
print(confusion_matrix(labels_true, labels_pred))
