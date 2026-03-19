import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT and SURF detectors
sift = cv2.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# Detect key points and compute descriptors
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
keypoints_surf, descriptors_surf = surf.detectAndCompute(img, None)
