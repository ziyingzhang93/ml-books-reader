import cv2
import numpy as np
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors


# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
        split_data(20, sub_imgs, 0.8)

# Create a new SVM
svm_digits = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(cv2.ml.SVM_RBF)
svm_digits.setType(cv2.ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-6))

# Converting the image data into HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)
digits_test_hog = hog_descriptors(digits_test_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog.astype(np.float32), cv2.ml.ROW_SAMPLE,
                 digits_train_labels)

# Predict labels for the testing data
_, digits_test_pred = svm_digits.predict(digits_test_hog.astype(np.float32))

# Compute and print the achieved accuracy
accuracy_digits = (np.sum(digits_test_pred.astype(int) == digits_test_labels)
                    / digits_test_labels.size) * 100
print('Accuracy:', accuracy_digits, '%')
