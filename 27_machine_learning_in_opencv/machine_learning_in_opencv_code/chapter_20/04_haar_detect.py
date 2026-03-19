import os
import cv2

# Photo https://unsplash.com/photos/people-walking-on-sidewalk-during-daytime-GBkAx9qUeus
filename = 'people2.jpg'

# Load the Haar cascade for face detection
filepath = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(filepath)

# Read the input image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                      minSize=(20, 20))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

# Display the result
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
