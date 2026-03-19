import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and testing datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()

# Create an empty logistic regression model
lr = cv2.ml.LogisticRegression_create()

# Check the default training method
print('Training Method:', lr.getTrainMethod())

# Set the training method to mini-batch gradient descent and the size of the mini-batch
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(5)

# Set the number of iterations
lr.setIterations(10)

# Train the logistic regressor on the set of training data
lr.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train.astype(np.float32))

# Print the learned coefficients
print(lr.get_learnt_thetas())

# Predict the target labels of the testing data
_, y_pred = lr.predict(x_test.astype(np.float32))

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')

# Plot the groundtruth and predicted class labels
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax1.set_title('Groundtruth testing data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
ax2.set_title('Predicted testing data')
plt.show()

# Print the groundtruth and predicted class labels of the testing data
print('Groundtruth class labels:', y_test)
print('Predicted class labels:  ', y_pred[:, 0].astype(int))
