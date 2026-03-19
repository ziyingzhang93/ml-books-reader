import cv2

# Load the image and convery to grayscale
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector with 30 keypoints
orb = cv2.ORB_create(30)

# Detect key points and compute descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)
for x in keypoints:
    print("({:.2f},{:.2f}) = size {:.2f} angle {:.2f}".format(
            x.pt[0], x.pt[1], x.size, x.angle))

img_kp = cv2.drawKeypoints(img, keypoints, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
