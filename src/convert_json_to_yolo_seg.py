import os
import json
import zlib
import base64
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
from pathlib import Path

# =========================================================
# Paths
# =========================================================
RAW_IMAGE_DIR = "data/raw/image"
RAW_ANN_DIR = "data/raw/annotation"

OUT_IMAGE_TRAIN = "data/processed/image/train"
OUT_IMAGE_VAL   = "data/processed/image/val"
OUT_LABEL_TRAIN = "data/processed/label/train"
OUT_LABEL_VAL   = "data/processed/label/val"

VAL_SPLIT = 0.2
CLASS_ID = 0   # billboard class

# =========================================================
# Create output folders
# =========================================================
for p in [OUT_IMAGE_TRAIN, OUT_IMAGE_VAL, OUT_LABEL_TRAIN, OUT_LABEL_VAL]:
    os.makedirs(p, exist_ok=True)

# =========================================================
# Decode Supervisely bitmap mask
# =========================================================
def decode_bitmap(data_str):
    compressed = base64.b64decode(data_str)
    decompressed = zlib.decompress(compressed)

    png_array = np.frombuffer(decompressed, dtype=np.uint8)
    mask = cv2.imdecode(png_array, cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise ValueError("Failed to decode bitmap mask.")

    # Supervisely masks may contain RGBA
    if len(mask.shape) == 3:
        mask = mask[:, :, 3]

    mask = (mask > 0).astype(np.uint8)
    return mask

# =========================================================
# Convert mask -> YOLO polygon segments
# =========================================================
def mask_to_yolo_segments(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []

    for contour in contours:
        contour = contour.squeeze()

        if len(contour.shape) != 2 or len(contour) < 3:
            continue

        polygon = []
        for x, y in contour:
            polygon.extend([
                x / img_w,
                y / img_h
            ])

        segments.append(polygon)

    return segments

# =========================================================
# Collect images
# =========================================================
image_files = sorted([
    f for f in os.listdir(RAW_IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

random.seed(42)
random.shuffle(image_files)

split_idx = int(len(image_files) * (1 - VAL_SPLIT))
train_files = image_files[:split_idx]
val_files   = image_files[split_idx:]

print(f"Total images: {len(image_files)}")
print(f"Train images: {len(train_files)}")
print(f"Val images:   {len(val_files)}")

# =========================================================
# Process dataset
# =========================================================
def process_files(file_list, out_img_dir, out_lbl_dir):
    for img_name in tqdm(file_list):
        img_path = os.path.join(RAW_IMAGE_DIR, img_name)

        # -------------------------------------------------
        # Find annotation file
        # -------------------------------------------------
        base_name = Path(img_name).stem

        ann_candidates = [
            os.path.join(RAW_ANN_DIR, base_name + ".json"),
            os.path.join(RAW_ANN_DIR, img_name + ".json"),
        ]

        ann_path = None
        for candidate in ann_candidates:
            if os.path.exists(candidate):
                ann_path = candidate
                break

        if ann_path is None:
            print(f"Skipping {img_name}: annotation not found")
            continue

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping {img_name}: failed to read image")
            continue

        img_h, img_w = image.shape[:2]

        # -------------------------------------------------
        # Read annotation
        # -------------------------------------------------
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        label_lines = []

        # -------------------------------------------------
        # Process objects
        # -------------------------------------------------
        for obj in ann.get("objects", []):

            if "bitmap" not in obj:
                continue

            bitmap = obj["bitmap"]

            try:
                local_mask = decode_bitmap(bitmap["data"])
            except Exception as e:
                print(f"Skipping object in {img_name}: decode error: {e}")
                continue

            origin_x, origin_y = bitmap["origin"]
            h, w = local_mask.shape

            # Create full-image mask
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            x2 = min(origin_x + w, img_w)
            y2 = min(origin_y + h, img_h)

            full_mask[origin_y:y2, origin_x:x2] = local_mask[:y2-origin_y, :x2-origin_x]

            segments = mask_to_yolo_segments(full_mask, img_w, img_h)

            for seg in segments:
                if len(seg) >= 6:  # at least 3 points
                    line = f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in seg)
                    label_lines.append(line)

        # -------------------------------------------------
        # Save label file
        # -------------------------------------------------
        label_path = os.path.join(out_lbl_dir, base_name + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        # -------------------------------------------------
        # Copy image
        # -------------------------------------------------
        shutil.copy(img_path, os.path.join(out_img_dir, img_name))

# =========================================================
# Run conversion
# =========================================================
process_files(train_files, OUT_IMAGE_TRAIN, OUT_LABEL_TRAIN)
process_files(val_files, OUT_IMAGE_VAL, OUT_LABEL_VAL)

print("\nConversion complete.")
print("YOLO labels saved to:")
print(" - data/processed/label/train")
print(" - data/processed/label/val")