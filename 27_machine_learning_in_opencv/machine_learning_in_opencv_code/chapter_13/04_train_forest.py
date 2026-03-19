import csv
import cv2
import numpy as np
from sklearn import model_selection as ms

# Function to load the dataset
def load_csv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset

# Function to convert a string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = np.float32(row[column].strip())

# Load the dataset from text file
data = load_csv('Data/data_banknote_authentication.txt')

# Convert the dataset string numbers to float
for i in range(len(data[0])):
    str_column_to_float(data, i)

# Convert list to array
data = np.array(data)

# Separate the dataset samples from the groundtruth
samples = data[:, :4]
target = data[:, -1, np.newaxis].astype(np.int32)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = \
        ms.train_test_split(samples, target, test_size=0.2, random_state=10)

# Create an empty random forest
rtrees = cv2.ml.RTrees_create()

# Train the random forest
rtrees.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

# Predict the target labels of the testing data
_, y_pred = rtrees.predict(x_test)

# Compute and print the achieved accuracy
accuracy = (np.sum(y_pred.astype(np.int32) == y_test) / y_test.size) * 100
print('Accuracy:', accuracy, '%')
