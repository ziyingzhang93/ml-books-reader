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

# Choose the skin-labelled hue values
skin = hue[labels == 1]

# Compute their histogram
hist, bin_edges = np.histogram(skin, range=[0, 360], bins=360)

# Display the computed histogram
plt.bar(bin_edges[:-1], hist, width=4)
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.title('Histogram of the hue values of skin pixels')
plt.show()
