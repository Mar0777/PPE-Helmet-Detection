import os
import random
import shutil



SOURCE_HELMET = "./dataset_final/helmet"
SOURCE_NO_HELMET = "./dataset_final/no_helmet"

helmet_files = os.listdir(SOURCE_HELMET)
no_helmet_files = os.listdir(SOURCE_NO_HELMET)

# Shuffle files randomly
random.shuffle(helmet_files)
random.shuffle(no_helmet_files)

print(f"Helmet images: {len(helmet_files)}")
print(f"No-helmet images: {len(no_helmet_files)}")
# ---------- SPLIT CALCULATIONS ---------- #

def split_indices(total):

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    return train_end, val_end


helmet_train_end, helmet_val_end = split_indices(len(helmet_files))
no_train_end, no_val_end = split_indices(len(no_helmet_files))

print("\nHelmet Split:")
print("Train:", helmet_train_end)
print("Val:", helmet_val_end - helmet_train_end)
print("Test:", len(helmet_files) - helmet_val_end)

print("\nNo-Helmet Split:")
print("Train:", no_train_end)
print("Val:", no_val_end - no_train_end)
print("Test:", len(no_helmet_files) - no_val_end)
def copy_files(files, source_dir, dest_dir):

    for file in files:

        src = os.path.join(source_dir, file)
        dst = os.path.join(dest_dir, file)

        shutil.copy2(src, dst)

# ---------- HELMET ---------- #

helmet_train = helmet_files[:helmet_train_end]
helmet_val = helmet_files[helmet_train_end:helmet_val_end]
helmet_test = helmet_files[helmet_val_end:]

copy_files(helmet_train, SOURCE_HELMET, "./dataset_split/train/helmet")
copy_files(helmet_val, SOURCE_HELMET, "./dataset_split/val/helmet")
copy_files(helmet_test, SOURCE_HELMET, "./dataset_split/test/helmet")

print("\nHelmet dataset copied.")

# ---------- NO_HELMET ---------- #

no_train = no_helmet_files[:no_train_end]
no_val = no_helmet_files[no_train_end:no_val_end]
no_test = no_helmet_files[no_val_end:]

copy_files(no_train, SOURCE_NO_HELMET, "./dataset_split/train/no_helmet")
copy_files(no_val, SOURCE_NO_HELMET, "./dataset_split/val/no_helmet")
copy_files(no_test, SOURCE_NO_HELMET, "./dataset_split/test/no_helmet")

print("No-helmet dataset copied.")

print("\nDataset splitting completed successfully.")