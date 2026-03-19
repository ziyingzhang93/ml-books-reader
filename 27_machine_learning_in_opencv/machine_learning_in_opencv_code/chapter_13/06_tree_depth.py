import cv2

rtrees_digits = cv2.ml.RTrees_create()
print('Default tree depth:', rtrees_digits.getMaxDepth())
print('Default termination criteria:', rtrees_digits.getTermCriteria())
