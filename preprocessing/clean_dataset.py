import os
import cv2

IMAGE_PATH = "./dataset_raw/images"
ANNOTATION_PATH = "./dataset_raw/annotations"

images = os.listdir(IMAGE_PATH)
annotations = os.listdir(ANNOTATION_PATH)

print(f"Total Images: {len(images)}")
print(f"Total Annotations: {len(annotations)}")
# Remove file extensions
image_names = set([
    os.path.splitext(img)[0]
    for img in images
])

annotation_names = set([
    os.path.splitext(xml)[0]
    for xml in annotations
])

# Find missing pairs
missing_annotations = image_names - annotation_names
missing_images = annotation_names - image_names

print("\nMissing Annotation Files:")
print(len(missing_annotations))

print("\nMissing Image Files:")
print(len(missing_images))

corrupted_images = []

for image_file in images:

    image_path = os.path.join(IMAGE_PATH, image_file)

    img = cv2.imread(image_path)

    if img is None:
        corrupted_images.append(image_file)

print("\nCorrupted Images:")
print(len(corrupted_images))
import xml.etree.ElementTree as ET

class_counts = {}

for xml_file in annotations:

    xml_path = os.path.join(ANNOTATION_PATH, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):

        label = obj.find("name").text

        if label not in class_counts:
            class_counts[label] = 0

        class_counts[label] += 1

print("\nClass Distribution:\n")

for label, count in class_counts.items():
    print(f"{label}: {count}")