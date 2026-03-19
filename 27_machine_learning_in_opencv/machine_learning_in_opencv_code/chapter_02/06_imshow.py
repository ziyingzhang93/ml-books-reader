import cv2

img = cv2.imread('Images/Dog.jpg')
cv2.imshow('Displaying image using OpenCV', img)
cv2.waitKey(0)
