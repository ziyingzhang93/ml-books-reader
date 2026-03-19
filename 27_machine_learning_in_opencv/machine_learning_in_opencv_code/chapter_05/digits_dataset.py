import cv2
import numpy as np

def split_images(img_name, img_size):
    # Load the full image from the specified file
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    # Find the number of sub-images on each row and column according to their size
    num_rows = img.shape[0] / img_size
    num_cols = img.shape[1] / img_size

    # Split the full image horizontally and vertically into sub-images
    sub_imgs = [np.hsplit(row, num_cols) for row in np.vsplit(img, num_rows)]

    return img, np.array(sub_imgs)

def split_data(img_size, sub_imgs, ratio):
    # Compute the partition between the training and testing data
    partition = int(sub_imgs.shape[1] * ratio)

    # Split dataset into training and test sets
    train = sub_imgs[:, :partition, :, :]
    test = sub_imgs[:, partition:sub_imgs.shape[1], :, :]

    # Flatten each image into a one-dimensional vector
    train_imgs = train.reshape(-1, img_size ** 2)
    test_imgs = test.reshape(-1, img_size ** 2)

    # Create the groundtruth labels
    labels = np.arange(10)
    train_labels = np.repeat(labels, train_imgs.shape[0] / labels.shape[0])[:, np.newaxis]
    test_labels = np.repeat(labels, test_imgs.shape[0] / labels.shape[0])[:, np.newaxis]

    return train_imgs, train_labels, test_imgs, test_labels
