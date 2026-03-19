#!/usr/bin/env python

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)

# LeNet5 model
model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])

# Reshape data to shape of (n_sample, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype('float32')
X_test = np.expand_dims(X_test, axis=3).astype('float32')

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Training
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
          callbacks=[earlystopping])

model.save("lenet5.h5")
