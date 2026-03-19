import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn.datasets import make_blobs

# Generate a dataset of 2D data points and their groundtruth labels
x, y_true = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=15)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
    ms.train_test_split(x, y_true, test_size=0.2, random_state=10)

# Plot the training and test datasets
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
ax1.set_title('Training data')
ax2.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
ax2.set_title('Testing data')
plt.show()
