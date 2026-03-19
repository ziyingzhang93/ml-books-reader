import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Images/Dog.jpg')
plt.imshow(img)
plt.title('Displaying image using Matplotlib')
plt.show()
