import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
