import os
import cv2
import numpy as np
import random

NO_HELMET_PATH = "./dataset_processed/no_helmet"
OUTPUT_PATH = "./dataset_processed/no_helmet_aug"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting augmentation for no_helmet class...")

def horizontal_flip(img):
    return cv2.flip(img, 1)

def adjust_brightness(img):
    value = random.randint(-40, 40)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # convert to int32 to avoid overflow
    v = v.astype(np.int32)

    v = np.clip(v + value, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def rotate(img):
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def save_image(img, path):
    cv2.imwrite(path, img)


images = os.listdir(NO_HELMET_PATH)

count = 0
target = len(images)  # we will double it

for img_name in images:

    img_path = os.path.join(NO_HELMET_PATH, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # original copy (keep it)
    save_image(img, os.path.join(OUTPUT_PATH, f"orig_{img_name}"))

    # augmentation 1
    aug1 = horizontal_flip(img)
    save_image(aug1, os.path.join(OUTPUT_PATH, f"flip_{img_name}"))

    # augmentation 2
    aug2 = adjust_brightness(img)
    save_image(aug2, os.path.join(OUTPUT_PATH, f"bright_{img_name}"))

    # augmentation 3
    aug3 = rotate(img)
    save_image(aug3, os.path.join(OUTPUT_PATH, f"rot_{img_name}"))

    count += 1

print(f"Augmentation completed. Base images processed: {count}")