import cv2

img_gray = cv2.imread('Images/Dog.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
