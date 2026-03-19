import pathlib
import xml.etree.ElementTree as ET

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

# Read Pascal VOC and write data
base_path = pathlib.Path("oxford-iiit-pet")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

negative = []
positive = []
for xmlfile in ann_src.glob("*.xml"):
    # load xml
    ann = read_voc_xml(str(xmlfile))
    if ann['objects'][0]['name'] == 'dog':
        # negative sample (dog)
        negative.append(str(img_src / ann['filename']))
    else:
        # positive sample (cats)
        bbox = []
        for obj in ann['objects']:
            x = obj['xmin']
            y = obj['ymin']
            w = obj['xmax'] - obj['xmin']
            h = obj['ymax'] - obj['ymin']
            bbox.append(f"{x} {y} {w} {h}")
        line = f"{str(img_src/ann['filename'])} {len(bbox)} {' '.join(bbox)}"
        positive.append(line)

# write the output to `negative.dat` and `postiive.dat`
with open("negative.dat", "w") as fp:
    fp.write("\n".join(negative))

with open("positive.dat", "w") as fp:
    fp.write("\n".join(positive))
