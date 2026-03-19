import cv2

img = cv2.imread('Images/Dog.jpg')
cv2.imwrite("output.jpg", img)
