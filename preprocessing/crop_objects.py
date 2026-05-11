import os
import cv2
import xml.etree.ElementTree as ET
import uuid

IMAGE_PATH = "./dataset_raw/images"
ANNOTATION_PATH = "./dataset_raw/annotations"

HELMET_OUTPUT = "./dataset_processed/helmet"
NO_HELMET_OUTPUT = "./dataset_processed/no_helmet"

os.makedirs(HELMET_OUTPUT, exist_ok=True)
os.makedirs(NO_HELMET_OUTPUT, exist_ok=True)

# ✅ Load ALL annotation files
xml_files = sorted([
    f for f in os.listdir(ANNOTATION_PATH)
    if f.endswith(".xml")
])

count = 0

for xml_file in xml_files:

    xml_path = os.path.join(ANNOTATION_PATH, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find("filename").text
    image_path = os.path.join(IMAGE_PATH, image_filename)

    img = cv2.imread(image_path)
    if img is None:
        continue

    for obj in root.findall("object"):

        label = obj.find("name").text
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        cropped = img[ymin:ymax, xmin:xmax]

        # ✅ quality checks
        if cropped.size == 0:
            continue
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue

        # ✅ label mapping
        if label == "helmet":
            save_dir = HELMET_OUTPUT
        elif label == "head":
            save_dir = NO_HELMET_OUTPUT
        else:
            continue

        filename = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), cropped)
        count += 1

print(f"\nTotal Cropped Images Created: {count}")
print("Helmet crops:", len(os.listdir(HELMET_OUTPUT)))
print("No-helmet crops:", len(os.listdir(NO_HELMET_OUTPUT)))
