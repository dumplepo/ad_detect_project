import os
import cv2
import random
import numpy as np

IMG_DIR = r"dataset/images/train"
LBL_DIR = r"dataset/labels/train"
OUT_DIR = r"dataset/preview"

os.makedirs(OUT_DIR, exist_ok=True)

files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]
img_name = random.choice(files)

img_path = os.path.join(IMG_DIR, img_name)
lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(lbl_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    coords = list(map(float, parts[1:]))

    pts = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * w)
        y = int(coords[i+1] * h)
        pts.append([x, y])

    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, (0,255,0), 2)

out_path = os.path.join(OUT_DIR, img_name)
cv2.imwrite(out_path, img)

print(f"Saved preview: {out_path}")