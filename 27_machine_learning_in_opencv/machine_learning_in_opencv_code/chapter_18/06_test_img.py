import random
import numpy as np
import matplotlib.pyplot as plt
from digits_dataset import split_images, split_data

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
