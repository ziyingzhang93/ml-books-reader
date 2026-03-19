import cv2

# Create video capture object
capture = cv2.VideoCapture(0)

# Check that a camera connection has been established
if not capture.isOpened():
    print("Error establishing connection")

while capture.isOpened():
    # Read an image frame
    ret, frame = capture.read()

    # If an image frame has been grabbed, display it
    if ret:
        cv2.imshow('Displaying image frames from a webcam', frame)

    # If the Esc key is pressed, terminate the while loop
    if cv2.waitKey(25) == 27:
        break

# Release the video capture and close the display window
capture.release()
cv2.destroyAllWindows()
