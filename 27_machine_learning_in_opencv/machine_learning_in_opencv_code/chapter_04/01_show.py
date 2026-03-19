import cv2

imgfile = "Hooded_mountain_tanager_(Buthraupis_montana_cucullata)_Caldas.jpg"

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cv2.imshow("bird", img)
cv2.waitKey(0)
