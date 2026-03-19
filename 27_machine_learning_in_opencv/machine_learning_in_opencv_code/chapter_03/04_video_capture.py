import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error establishing connection")

ret, frame = capture.read()

if ret:
    cv2.imshow('Displaying image frames from a webcam', frame)
    cv2.waitKey(0)
