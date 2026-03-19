import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Plot the dataset
plt.scatter(x[:, 0], x[:, 1], c=y_true)
plt.show()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Create a new Normal Bayes Classifier
norm_bayes = cv2.ml.NormalBayesClassifier_create()

# Train the classifier on the training data
norm_bayes.train(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, y_train)

# Generate a prediction from the trained classifier
ret, y_pred, y_probs = norm_bayes.predictProb(x_test.astype(np.float32))

# Plot the class predictions
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)
plt.show()
