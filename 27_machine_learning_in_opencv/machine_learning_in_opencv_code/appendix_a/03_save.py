import cv2
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
svm.train(x_train.astype('float32'), cv2.ml.ROW_SAMPLE, y_train)

# Save the trained model
svm.save("rbf_svm.dat")

# Predict the target labels of the testing data
_, y_pred = svm.predict(x_test.astype('float32'))

# Compute and print the achieved accuracy
accuracy = (sum(y_pred[:, 0].astype(int) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
