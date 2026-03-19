import cv2

# Create video capture object
capture = cv2.VideoCapture('Videos/Iceland2.mp4')

# Check that a camera connection has been established
if not capture.isOpened():
    print("Error opening video file")
else:
    # Get video properties and print them
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print("Image frame width: ", int(frame_width))
    print("Image frame height: ", int(frame_height))
    print("Frame rate: ", int(fps))

while capture.isOpened():
    # Read an image frame
    ret, frame = capture.read()

    # If an image frame has been grabbed, display it
    if ret:
        cv2.imshow('Displaying image frames from video file', frame)

    # If the Esc key is pressed, terminate the while loop
    if cv2.waitKey(25) == 27:
        break

# Release the video capture and close the display window
capture.release()
cv2.destroyAllWindows()
