import cv2
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the full training image
img, sub_imgs = split_images('Images/digits.png', 20)

# Show entire image to check that the correct image has been loaded
cv2.imshow('Training image', img)
cv2.waitKey(0)

# Show one sample to check that the sub-images have been correctly split
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)

# Split the dataset into training and testing
train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, 0.5)

# Convert the training and testing images into feature vectors using the HOG technique
train_hog = hog_descriptors(train_imgs)
test_hog = hog_descriptors(test_imgs)
