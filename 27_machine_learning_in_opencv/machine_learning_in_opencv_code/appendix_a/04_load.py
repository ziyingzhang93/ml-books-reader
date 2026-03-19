import cv2
from sklearn.datasets import make_blobs
from sklearn import model_selection as ms

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=8, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Load a trained SVM
svm = cv2.ml.SVM_load("rbf_svm.dat")

# Predict the target labels of the testing data
_, y_pred = svm.predict(x_test.astype('float32'))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
