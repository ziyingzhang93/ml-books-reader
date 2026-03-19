import os
import pickle
import numpy as np

def load_images(path):
    # Create empty lists to store the images and labels
    imgs = []
    labels = []

    # Iterate over the dataset's files
    for batch in range(5):
        # Specify the path to the training data
        train_path_batch = os.path.join(path, 'data_batch_' + str(batch + 1))

        # Extract the training images and labels from the dataset files
        train_imgs_batch, train_labels_batch = extract_data(train_path_batch)

        # Store the training images
        imgs.append(train_imgs_batch)
        train_imgs = np.array(imgs).reshape(-1, 3072)

        # Store the training labels
        labels.append(train_labels_batch)
        train_labels = np.array(labels).reshape(-1, 1)

    # Specify the path to the testing data
    test_path_batch = path + 'test_batch'

    # Extract the testing images and labels from the dataset files
    test_imgs, test_labels = extract_data(test_path_batch)
    test_labels = np.array(test_labels)[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels

def extract_data(path):
    # Open pickle file and return a dictionary
    with open(path, 'rb') as fo:
        loaded_dict = pickle.load(fo, encoding='bytes')

    # Extract the dictionary values
    dict_values = list(loaded_dict.values())

    # Extract the images and labels
    imgs = dict_values[2]
    labels = dict_values[1]

    return imgs, labels
