import os
import xml.etree.ElementTree as ET

# Dataset annotation path
ANNOTATIONS_PATH = "./dataset_raw/annotations"


def parse_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {}

    # Image filename
    filename = root.find("filename").text
    data["filename"] = filename

    objects = []

    # Read all annotated objects
    for obj in root.findall("object"):

        label = obj.find("name").text

        bndbox = obj.find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        objects.append({
            "label": label,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    data["objects"] = objects

    return data


# ---------------- TEST ---------------- #

xml_files = os.listdir(ANNOTATIONS_PATH)

first_xml = xml_files[0]

xml_path = os.path.join(ANNOTATIONS_PATH, first_xml)

parsed_data = parse_xml(xml_path)

print("\nParsed Annotation:\n")

print(parsed_data)