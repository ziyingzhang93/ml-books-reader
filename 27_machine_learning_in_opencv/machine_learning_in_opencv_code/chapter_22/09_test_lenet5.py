import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Load the frozen model in OpenCV
net = cv2.dnn.readNetFromONNX('lenet5.onnx')

# Prepare input image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
correct = 0
wrong = 0
for i in range(len(X_test)):
    img = X_test[i]
    label = y_test[i]

    blob = cv2.dnn.blobFromImage(img, 1.0, (28, 28))

    # Run inference
    net.setInput(blob)
    output = net.forward()
    prediction = np.argmax(output)
    if prediction == label:
        correct += 1
    else:
        wrong += 1

print("count of test samples:", len(X_test))
print("accuracy:", (correct/(correct+wrong)))
