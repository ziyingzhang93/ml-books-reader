import cv2

image = 'oxford-iiit-pet/images/Abyssinian_88.jpg'
model = 'cat_detect/cascade.xml'

classifier = cv2.CascadeClassifier(model)
img = cv2.imread(image)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform object detection
objects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))

# Draw rectangles around detected objects
for (x, y, w, h) in objects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
