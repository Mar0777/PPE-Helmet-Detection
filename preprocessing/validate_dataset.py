import os
import cv2

HELMET = "./dataset_final/helmet"
NO_HELMET = "./dataset_final/no_helmet"

def check_folder(path):
    corrupted = 0
    total = 0

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)

        total += 1
        if img is None:
            corrupted += 1

    return total, corrupted

h_total, h_bad = check_folder(HELMET)
n_total, n_bad = check_folder(NO_HELMET)

print("Helmet ->", h_total, "corrupted:", h_bad)
print("No-Helmet ->", n_total, "corrupted:", n_bad)