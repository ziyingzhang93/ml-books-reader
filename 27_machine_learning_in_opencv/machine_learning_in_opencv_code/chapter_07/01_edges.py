import cv2

# Load the image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny method
edges = cv2.Canny(gray, 150, 300)

# Display the image with corners
img[edges == 255] = (255,0,0)
cv2.imshow('Canny Edges', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
