import cv2

lr = cv2.ml.LogisticRegression_create()
print('Training Method:', lr.getTrainMethod())
