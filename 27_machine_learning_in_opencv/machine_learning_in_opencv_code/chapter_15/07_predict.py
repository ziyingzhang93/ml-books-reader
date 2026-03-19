import cv2
import numpy as np
from digits_dataset import split_images, split_data

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, digits_train_labels, digits_test_imgs, digits_test_labels = \
    split_data(20, sub_imgs, 0.8)

# Create an empty logistic regression model
lr_digits = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr_digits.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr_digits.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr_digits.setMiniBatchSize(400)

# Set the number of iterations
lr_digits.setIterations(10)

# Train the logistic regressor on the set of training data
lr_digits.train(digits_train_imgs.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                digits_train_labels.astype(np.float32))

# Predict the target labels of the testing data
_, y_pred = lr_digits.predict(digits_test_imgs.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred[:, 0] == digits_test_labels[:, 0]) / digits_test_labels.size) * 100
print('Accuracy:', accuracy, '%')
