import pathlib
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def read_voc_xml(xmlfile):
    """read the Pascal VOC XML"""
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []}
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

def make_square(xmin, xmax, ymin, ymax):
    """Shrink the bounding box to square shape"""
    xcenter = (xmax + xmin) // 2
    ycenter = (ymax + ymin) // 2
    halfdim = min(xmax-xmin, ymax-ymin) // 2
    xmin, xmax = xcenter-halfdim, xcenter+halfdim
    ymin, ymax = ycenter-halfdim, ycenter+halfdim
    return xmin, xmax, ymin, ymax

# Define HOG parameters
winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9

num_samples = 1000

# Load your dataset and corresponding bounding box annotations
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

# collect samples by cropping the images from dataset
positive = []
negative = []

# collect positive and negative samples
for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    # cat for positive samples, other for negative samples
    if ann["objects"][0]["name"] == "cat":
        if len(positive) <= num_samples:
            # adjust the bounding box to square
            box = ann["objects"][0]
            xmin, xmax, ymin, ymax = box["xmin"], box["xmax"], box["ymin"], box["ymax"]
            xmin, xmax, ymin, ymax = make_square(xmin, xmax, ymin, ymax)
            # crop a positive sample
            img = cv2.imread(str(img_src / ann["filename"]))
            sample = img[ymin:ymax, xmin:xmax]
            sample = cv2.resize(sample, winSize)
            positive.append(sample)
    else:
        if len(negative) <= num_samples:
            # random bounding box: at least the target size to avoid scaling up
            height, width = img.shape[:2]
            boxsize = random.randint(winSize[0], min(height, width))
            x = random.randint(0, width-boxsize)
            y = random.randint(0, height-boxsize)
            sample = img[y:y+boxsize, x:x+boxsize]
            assert tuple(sample.shape[:2]) == (boxsize, boxsize)
            sample = cv2.resize(sample, winSize)
            negative.append(sample)

images = positive + negative
labels = ([1] * len(positive)) + ([0] * len(negative))

# Create the HOG descriptor and the HOG from each image
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
data = []
for img in images:
    features = hog.compute(img)
    data.append(features.flatten())

# Convert data and labels to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-8))

svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# Save the SVM model
svm.save('svm_model.yml')
print(svm.getSupportVectors())
