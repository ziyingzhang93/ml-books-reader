import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new SVM
svm = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
svm.setGamma(0.1)

# Train the SVM on the set of training data
svm.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

x_bound, y_bound = np.meshgrid(np.arange(x_test[:, 0].min() - 1, x_test[:, 0].max() + 1, 0.05),
                               np.arange(x_test[:, 1].min() - 1, x_test[:, 1].max() + 1, 0.05))

bound_points = np.column_stack((x_bound.reshape(-1, 1), y_bound.reshape(-1, 1)))
_, bound_pred = svm.predict(bound_points.astype(np.float32))

# Plot the test set
plt.contourf(x_bound, y_bound, bound_pred.reshape(x_bound.shape), cmap=plt.cm.coolwarm)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.show()