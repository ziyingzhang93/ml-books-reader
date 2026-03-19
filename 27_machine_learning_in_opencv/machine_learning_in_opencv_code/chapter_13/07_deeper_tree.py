import cv2
import numpy as np
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
        split_data(20, sub_imgs, 0.8)

# Convert the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Create an empty random forest
rtrees_digits = cv2.ml.RTrees_create()

# Read the default parameter values
print('Default tree depth:', rtrees_digits.getMaxDepth())
print('Default termination criteria:', rtrees_digits.getTermCriteria())

# Change the default parameter values
rtrees_digits.setMaxDepth(15)
rtrees_digits.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                               100,
                               0.01))

# Train the random forest
rtrees_digits.train(digits_train_hog.astype(np.float32), cv2.ml.ROW_SAMPLE,
                    digits_train_labels)

# Predict the target labels of the testing data
_, digits_test_pred = rtrees_digits.predict(digits_test_hog)

# Compute and print the achieved accuracy
accuracy_digits = (np.sum(digits_test_pred.astype(int) == digits_test_labels)
                    / digits_test_labels.size) * 100
print('New tree depth:', rtrees_digits.getMaxDepth())
print('New termination criteria:', rtrees_digits.getTermCriteria())
print('Accuracy:', accuracy_digits, '%')
