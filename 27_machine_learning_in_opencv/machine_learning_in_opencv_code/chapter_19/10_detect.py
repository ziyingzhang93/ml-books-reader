import pathlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile: str) -> dict:
    """read the Pascal VOC XML and return (filename, object name, bounding box)
    where bounding box is a vector of (xmin, ymin, xmax, ymax). The pixel
    coordinates are 1-based.
    """
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []
            }
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes

# load the SVM
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

svm = cv2.ml.SVM_load('svm_model.yml')
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog.setSVMDetector(svm.getSupportVectors()[0])

# Run the SVM on each image
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    # read image and the groundtruth
    img = cv2.imread(str(img_src / ann["filename"]))
    bbox = ann["objects"][0]
    start_point = (bbox["xmin"], bbox["ymin"])
    end_point = (bbox["xmax"], bbox["ymax"])
    # detect and draw
    locations, scores = hog.detectMultiScale(img)
    x, y, w, h = locations[np.argmax(scores.flatten())]
    cv2.rectangle(img, start_point, end_point, (0,0,255), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 5)
    cv2.imshow(f"{ann['filename']}: {ann['objects'][0]['name']}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:  # ESC key
        break
