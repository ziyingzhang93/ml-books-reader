# OpenCV ML
## Chapter 03

---

### Video Esc

# 07 — Video Esc / 07 Video Esc

**Chapter 03 — File 2 of 4 / 第03章 — 第2个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Create video capture object**.

本脚本演示 **Create video capture object**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
```

---
## Step 2 — Create video capture object

```python
capture = cv2.VideoCapture(0)
```

---
## Step 3 — Check that a camera connection has been established

```python
if not capture.isOpened():
    print("Error establishing connection")

while capture.isOpened():
```

---
## Step 4 — Read an image frame

```python
ret, frame = capture.read()
```

---
## Step 5 — If an image frame has been grabbed, display it

```python
if ret:
        cv2.imshow('Displaying image frames from a webcam', frame)
```

---
## Step 6 — If the Esc key is pressed, terminate the while loop

```python
if cv2.waitKey(25) == 27:
        break
```

---
## Step 7 — Release the video capture and close the display window

```python
capture.release()
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Create video capture object 是机器学习中的常用技术。  
  *Create video capture object is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Video Esc / 07 Video Esc
# Complete Code / 完整代码
# ===============================

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
```

---

➡️ **Next / 下一步**: File 3 of 4

---

### Video File

# 10 — Video File / 10 Video File

**Chapter 03 — File 3 of 4 / 第03章 — 第3个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Create video capture object**.

本脚本演示 **Create video capture object**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
```

---
## Step 2 — Create video capture object

```python
capture = cv2.VideoCapture('Videos/Iceland2.mp4')
```

---
## Step 3 — Check that a camera connection has been established

```python
if not capture.isOpened():
    print("Error opening video file")
else:
```

---
## Step 4 — Get video properties and print them

```python
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print("Image frame width: ", int(frame_width))
    print("Image frame height: ", int(frame_height))
    print("Frame rate: ", int(fps))

while capture.isOpened():
```

---
## Step 5 — Read an image frame

```python
ret, frame = capture.read()
```

---
## Step 6 — If an image frame has been grabbed, display it

```python
if ret:
        cv2.imshow('Displaying image frames from video file', frame)
```

---
## Step 7 — If the Esc key is pressed, terminate the while loop

```python
if cv2.waitKey(25) == 27:
        break
```

---
## Step 8 — Release the video capture and close the display window

```python
capture.release()
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Create video capture object 是机器学习中的常用技术。  
  *Create video capture object is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Video File / 10 Video File
# Complete Code / 完整代码
# ===============================

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
```

---

➡️ **Next / 下一步**: File 4 of 4

---

### Video Gray

# 11 — Video Gray / 11 Video Gray

**Chapter 03 — File 4 of 4 / 第03章 — 第4个文件（共4个）**

---

## Summary / 总结

This script demonstrates **Create video capture object**.

本脚本演示 **Create video capture object**。

---
## Background / 背景导读

**本文件主要内容 / What this file covers:**

- 演示核心概念和API用法 / Demonstrate core concepts and API usage


---
## Step 1 — Step 1

```python
import cv2
```

---
## Step 2 — Create video capture object

```python
capture = cv2.VideoCapture('Videos/Iceland2.mp4')
```

---
## Step 3 — Check that a camera connection has been established

```python
if not capture.isOpened():
    print("Error opening video file")
else:
```

---
## Step 4 — Get video properties and print them

```python
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print("Image frame width: ", int(frame_width))
    print("Image frame height: ", int(frame_height))
    print("Frame rate: ", int(fps))

while capture.isOpened():
```

---
## Step 5 — Read an image frame

```python
ret, frame = capture.read()
```

---
## Step 6 — If an image frame has been grabbed, display it

```python
if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Displaying image frames from video file', frame)
```

---
## Step 7 — If the Esc key is pressed, terminate the while loop

```python
if cv2.waitKey(25) == 27:
        break
```

---
## Step 8 — Release the video capture and close the display window

```python
capture.release()
cv2.destroyAllWindows()
```

---
## Learning Notes / 学习笔记

- **概念**: Create video capture object 是机器学习中的常用技术。  
  *Create video capture object is a common technique in machine learning.*

- **ML 应用**: 本示例展示了如何在实践中应用该技术。  
  *This example shows how to apply the technique in practice.*

---
## Complete Code / 完整代码一览

Below is the full code for quick reference. / 以下是完整代码，供快速参考。

```python
# ===============================
# Video Gray / 11 Video Gray
# Complete Code / 完整代码
# ===============================

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Displaying image frames from video file', frame)

    # If the Esc key is pressed, terminate the while loop
    if cv2.waitKey(25) == 27:
        break

# Release the video capture and close the display window
capture.release()
cv2.destroyAllWindows()
```

---
