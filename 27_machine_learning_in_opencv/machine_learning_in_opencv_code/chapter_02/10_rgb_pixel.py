import cv2

img = cv2.imread('Images/Dog.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb[0, 0])
