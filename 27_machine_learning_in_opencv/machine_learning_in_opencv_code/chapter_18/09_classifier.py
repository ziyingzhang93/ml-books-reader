import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data
from feature_extraction import hog_descriptors

# Load the digits image
img, sub_imgs = split_images('Images/digits.png', 20)

# Obtain training and testing datasets from the digits image
digits_train_imgs, _, digits_test_imgs, _ = split_data(20, sub_imgs, 0.8)

# Create an empty list to store the random numbers
rand_nums = []

# Seed the random number generator for repeatability
random.seed(10)

# Choose 25 random digits from the testing dataset
for i in range(0, digits_test_imgs.shape[0], int(digits_test_imgs.shape[0] / 25)):
    # Generate a random integer
    rand = random.randint(i, int(digits_test_imgs.shape[0] / 25) + i - 1)
    # Append it to the list
    rand_nums.append(rand)

# Shuffle the order of the generated random integers
random.shuffle(rand_nums)

# Read the image data corresponding to the random integers
rand_test_imgs = digits_test_imgs[rand_nums, :]

# Initialize an array to hold the test image
test_img = np.zeros((100, 100), dtype=np.uint8)

# Start a sub-image counter
img_count = 0

# Iterate over the test image
for i in range(0, test_img.shape[0], 20):
    for j in range(0, test_img.shape[1], 20):
        # Populate the test image with the chosen digits
        test_img[i:i + 20, j:j + 20] = rand_test_imgs[img_count].reshape(20, 20)
        # Increment the sub-image counter
        img_count += 1

# Display the test image
plt.imshow(test_img, cmap='gray')
plt.show()

# Generate labels: First 1/10 of all samples are positive samples and the reset are
# negative samples
digits_train_labels = np.ones((digits_train_imgs.shape[0], 1), dtype=int)
digits_train_labels[int(digits_train_labels.shape[0]/10):digits_train_labels.shape[0]] = 0

# Create a new SVM
svm_digits = cv2.ml.SVM_create()

# Set the SVM kernel to RBF
svm_digits.setKernel(cv2.ml.SVM_RBF)
svm_digits.setType(cv2.ml.SVM_C_SVC)
svm_digits.setGamma(0.5)
svm_digits.setC(12)
svm_digits.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-6))

# Convert the training images to HOG descriptors
digits_train_hog = hog_descriptors(digits_train_imgs)

# Train the SVM on the set of training data
svm_digits.train(digits_train_hog, cv2.ml.ROW_SAMPLE, digits_train_labels)

# Create an empty list to store the matching patch coordinates
positive_patches = []

# Define the stride to shift with
stride = 5

# Iterate over the test image
for i in range(0, test_img.shape[0] - 20 + stride, stride):
    for j in range(0, test_img.shape[1] - 20 + stride, stride):
        # Crop a patch from the test image
        patch = test_img[i:i + 20, j:j + 20].reshape(1, 400)
        # Convert the image patch into HOG descriptors
        patch_hog = hog_descriptors(patch)
        # Predict the target label of the image patch
        _, patch_pred = svm_digits.predict(patch_hog.astype(np.float32))
        # If a match is found, store its coordinate values
        if patch_pred == 1:
            positive_patches.append((i, j))

# Convert the list to an array
positive_patches = np.array(positive_patches)

# Iterate over the match coordinates and draw their bounding box
for i in range(positive_patches.shape[0]):
    cv2.rectangle(test_img, (positive_patches[i, 1], positive_patches[i, 0]),
                  (positive_patches[i, 1] + 20, positive_patches[i, 0] + 20), 255, 1)

# Display the test image
plt.imshow(test_img, cmap='gray')
plt.show()
