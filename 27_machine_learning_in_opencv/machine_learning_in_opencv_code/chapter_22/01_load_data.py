import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)

# Check visually
fig, ax = plt.subplots(4, 5, sharex=True, sharey=True)
idx = np.random.randint(len(X_train), size=4*5).reshape(4,5)
for i in range(4):
    for j in range(5):
        ax[i][j].imshow(X_train[idx[i][j]], cmap="gray")
plt.show()
