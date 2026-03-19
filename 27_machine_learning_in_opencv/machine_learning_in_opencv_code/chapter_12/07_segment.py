import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

# Load data from text file
data = np.loadtxt("Data/Skin_NonSkin.txt", dtype=int)

# Select the BGR values from the loaded data
BGR = data[:, :3]

# Convert to RGB by swapping the array columns
RGB = BGR.copy()
RGB[:, [2, 0]] = RGB[:, [0, 2]]

# Convert RGB values to HSV
HSV = rgb_to_hsv(RGB.reshape(RGB.shape[0], -1, 3) / 255)
HSV = HSV.reshape(RGB.shape[0], 3)

# Select only the hue values
hue = HSV[:, 0] * 360

# Select the labels from the loaded data
labels = data[:, -1]

# Create a new Normal Bayes Classifier
norm_bayes = cv2.ml.NormalBayesClassifier_create()

# Train the classifier on the hue values
norm_bayes.train(hue.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
# Load a test image
face_img = cv2.imread("Images/face.jpg")

# Reshape the image into a three-column array
face_BGR = face_img.reshape(-1, 3)

# Convert to RGB by swapping the array columns
face_RGB = face_BGR.copy()
face_RGB[:, [2, 0]] = face_RGB[:, [0, 2]]

# Convert from RGB to HSV
face_HSV = rgb_to_hsv(face_RGB.reshape(face_RGB.shape[0], -1, 3) / 255)
face_HSV = face_HSV.reshape(face_RGB.shape[0], 3)

# Select only the hue values
face_hue = face_HSV[:, 0] * 360

# Display the hue image
plt.imshow(face_hue.reshape(face_img.shape[0], face_img.shape[1]))
plt.show()

# Generate a prediction from the trained classifier
ret, labels_pred, output_probs = norm_bayes.predictProb(face_hue.astype(np.float32))

# Reshape array into the input image size and choose the skin-labelled pixels
skin_mask = labels_pred.reshape(face_img.shape[0], face_img.shape[1], 1) == 1

# Display the segmented image
plt.imshow(skin_mask, cmap='gray')
plt.show()
