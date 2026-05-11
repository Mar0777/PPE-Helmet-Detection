import cv2
import os
import xml.etree.ElementTree as ET

IMAGE_PATH = "./dataset_raw/images"
ANNOTATION_PATH = "./dataset_raw/annotations"

# Load first XML file
xml_file = os.listdir(ANNOTATION_PATH)[0]

xml_path = os.path.join(ANNOTATION_PATH, xml_file)

tree = ET.parse(xml_path)
root = tree.getroot()

print(root.find("filename").text)
image_filename = root.find("filename").text

image_path = os.path.join(IMAGE_PATH, image_filename)

img = cv2.imread(image_path)

print(type(img))
print(img.shape)
# Get first object only
# Draw ALL objects
for obj in root.findall("object"):

    label = obj.find("name").text

    bndbox = obj.find("bndbox")

    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    # Draw rectangle
    cv2.rectangle(
        img,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2
    )

    # Draw label
    cv2.putText(
        img,
        label,
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
label = obj.find("name").text

bndbox = obj.find("bndbox")

xmin = int(bndbox.find("xmin").text)
ymin = int(bndbox.find("ymin").text)
xmax = int(bndbox.find("xmax").text)
ymax = int(bndbox.find("ymax").text)

print(label)
print(xmin, ymin, xmax, ymax)
# Draw rectangle
cv2.rectangle(
    img,
    (xmin, ymin),
    (xmax, ymax),
    (0, 255, 0),
    2
)

# Draw label text
cv2.putText(
    img,
    label,
    (xmin, ymin - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# Show image
cv2.imshow("Bounding Box", img)

cv2.waitKey(0)
cv2.destroyAllWindows()