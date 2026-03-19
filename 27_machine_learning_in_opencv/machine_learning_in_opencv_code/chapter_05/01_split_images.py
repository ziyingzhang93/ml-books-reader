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
