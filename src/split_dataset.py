import os
import random
import shutil

IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"

VAL_IMG_DIR = "dataset/images/val"
VAL_LBL_DIR = "dataset/labels/val"

SPLIT = 0.2


def main():
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    random.shuffle(images)

    val_count = int(len(images) * SPLIT)
    val_files = images[:val_count]

    for f in val_files:
        shutil.move(os.path.join(IMG_DIR, f), os.path.join(VAL_IMG_DIR, f))

        label = f.replace(".png", ".txt")
        shutil.move(os.path.join(LBL_DIR, label), os.path.join(VAL_LBL_DIR, label))


if __name__ == "__main__":
    main()