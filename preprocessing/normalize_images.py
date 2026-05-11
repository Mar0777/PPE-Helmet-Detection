import os
import cv2

INPUT_DIRS = [
    "./dataset_processed/helmet",
    "./dataset_processed/no_helmet"
]

OUTPUT_SIZE = (224, 224)

print("Starting resize process...")
for folder in INPUT_DIRS:

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        resized = cv2.resize(img, OUTPUT_SIZE)

        cv2.imwrite(img_path, resized)

print("Resizing completed successfully.")