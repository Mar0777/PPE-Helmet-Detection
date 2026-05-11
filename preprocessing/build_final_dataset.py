import os
import shutil

SOURCE_HELMET = "./dataset_processed/helmet"
SOURCE_NO_HELMET = "./dataset_processed/no_helmet"
SOURCE_NO_HELMET_AUG = "./dataset_processed/no_helmet_aug"

DEST_HELMET = "./dataset_final/helmet"
DEST_NO_HELMET = "./dataset_final/no_helmet"

os.makedirs(DEST_HELMET, exist_ok=True)
os.makedirs(DEST_NO_HELMET, exist_ok=True)

print("Starting final dataset build...")
import os
import shutil

helmet_files = os.listdir(SOURCE_HELMET)

print(f"Copying helmet images: {len(helmet_files)}")

count = 0

for file in helmet_files:

    src = os.path.join(SOURCE_HELMET, file)
    dst = os.path.join(DEST_HELMET, file)

    shutil.copy2(src, dst)
    count += 1

print(f"Helmet copied: {count}")
# ---------- NO_HELMET (original + augmented) ---------- #

no_helmet_original = os.listdir(SOURCE_NO_HELMET)
no_helmet_aug = os.listdir(SOURCE_NO_HELMET_AUG)

all_no_helmet = no_helmet_original + no_helmet_aug

print(f"\nMerging no_helmet images:")
print(f"Original: {len(no_helmet_original)}")
print(f"Augmented: {len(no_helmet_aug)}")
print(f"Total to copy: {len(all_no_helmet)}")

count = 0

for file in all_no_helmet:

    # determine correct source folder
    if file in no_helmet_original:
        src = os.path.join(SOURCE_NO_HELMET, file)
    else:
        src = os.path.join(SOURCE_NO_HELMET_AUG, file)

    dst = os.path.join(DEST_NO_HELMET, file)

    shutil.copy2(src, dst)
    count += 1

print(f"No_helmet copied: {count}")
