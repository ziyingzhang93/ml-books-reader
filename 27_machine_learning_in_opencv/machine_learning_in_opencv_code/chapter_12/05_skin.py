import cv2
import numpy as np
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
