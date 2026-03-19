import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the full training image
img, sub_imgs = split_images('Images/digits.png', 20)

# Check that the correct image has been loaded
cv2.imshow('Training image', img)
cv2.waitKey(0)

# Check that the sub-images have been correctly split
cv2.imshow('Sub-image', sub_imgs[0, 0, :, :].reshape(20, 20))
cv2.waitKey(0)

# Define different training-testing splits
ratio = [0.5, 0.7, 0.9]

for i in ratio:
    # Split the dataset into training and testing
    train_imgs, train_labels, test_imgs, test_labels = split_data(20, sub_imgs, i)

    # Convert the training and testing images into feature vectors using the HOG technique
    train_hog = hog_descriptors(train_imgs)
    test_hog = hog_descriptors(test_imgs)

    # Initiate a kNN classifier and train it on the training data
    knn = cv2.ml.KNearest_create()
    knn.train(train_hog, cv2.ml.ROW_SAMPLE, train_labels)

    # Initiate a dictionary to hold the ratio and accuracy values
    accuracy_dict = {}

    # Populate the dictionary with the keys corresponding to the values of 'k'
    keys = range(3, 16)

    for k in keys:
        # Test the kNN classifier on the testing data
        ret, result, neighbours, dist = knn.findNearest(test_hog, k)

        # Compute the accuracy and print it
        accuracy = (np.sum(result == test_labels) / test_labels.size) * 100
        print("Accuracy: {0:.2f}%, Training: {1:.0f}%, k: {2}".format(accuracy, i*100, k))

        # Populate the dictionary with the values corresponding to the accuracy
        accuracy_dict[k] = accuracy

    # Plot the accuracy values against the value of 'k'
    plt.plot(accuracy_dict.keys(), accuracy_dict.values(),
             marker='o', label=str(i*100)+'%')
    plt.title('Accuracy of the $k-nearest neighbors model')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

plt.show()
