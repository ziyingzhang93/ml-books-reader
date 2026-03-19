import cv2

try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480))
    assert writer.isOpened()
    print("Supported")
except:
    print("Not supported")
