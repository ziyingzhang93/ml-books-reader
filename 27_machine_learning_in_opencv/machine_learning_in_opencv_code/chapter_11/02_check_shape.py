from digits_dataset import split_images, split_data

# Load the digits image and divide it into sub-images
img, sub_imgs = split_images('Images/digits.png', 20)

# Create the groundtruth labels
imgs, labels_true, _, _ = split_data(20, sub_imgs, 1.0)

# Check the shape of the 'imgs' array
print(imgs.shape)
