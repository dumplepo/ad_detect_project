import os
import json
import zlib
import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import shutil

TRAIN_IMAGE_DIR = r"data/train/image"
TRAIN_JSON_DIR = r"data/train/annotation"

VAL_IMAGE_DIR = r"data/test/image"
VAL_JSON_DIR = r"data/test/annotation"

OUT_IMG_TRAIN = r"dataset/images/train"
OUT_LBL_TRAIN = r"dataset/labels/train"
OUT_IMG_VAL = r"dataset/images/val"
OUT_LBL_VAL = r"dataset/labels/val"

os.makedirs(OUT_IMG_TRAIN, exist_ok=True)
os.makedirs(OUT_LBL_TRAIN, exist_ok=True)
os.makedirs(OUT_IMG_VAL, exist_ok=True)
os.makedirs(OUT_LBL_VAL, exist_ok=True)


def decode_bitmap(bitmap_data, origin, img_h, img_w):
    decoded = zlib.decompress(base64.b64decode(bitmap_data))
    mask_img = Image.open(BytesIO(decoded)).convert("L")
    mask = np.array(mask_img)

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    x0, y0 = origin
    h, w = mask.shape

    full_mask[y0:y0+h, x0:x0+w] = mask
    return full_mask


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) >= 3:
            poly = approx.reshape(-1, 2)
            polygons.append(poly)

    return polygons


def process(image_dir, json_dir, out_img_dir, out_lbl_dir):
    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        img_h = data["size"]["height"]
        img_w = data["size"]["width"]

        label_lines = []

        for obj in data["objects"]:
            bitmap = obj["bitmap"]["data"]
            origin = obj["bitmap"]["origin"]

            mask = decode_bitmap(bitmap, origin, img_h, img_w)
            polygons = mask_to_polygon(mask)

            for poly in polygons:
                norm_points = []
                for x, y in poly:
                    norm_points.extend([x / img_w, y / img_h])

                line = "0 " + " ".join(map(str, norm_points))
                label_lines.append(line)

        txt_name = file.replace(".json", ".txt")
        with open(os.path.join(out_lbl_dir, txt_name), "w") as f:
            f.write("\n".join(label_lines))

        img_name = file.replace(".json", ".jpg")
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(out_img_dir, img_name)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)


process(TRAIN_IMAGE_DIR, TRAIN_JSON_DIR, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
process(VAL_IMAGE_DIR, VAL_JSON_DIR, OUT_IMG_VAL, OUT_LBL_VAL)

print("Conversion complete.")