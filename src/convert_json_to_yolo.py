import os
import json
import zlib
import base64
import numpy as np
import cv2
import shutil
from tqdm import tqdm

# =========================
# PATHS
# =========================
INPUT_IMG_DIR = "data/train/image"
INPUT_JSON_DIR = "data/train/annotation"

OUT_IMG_DIR = "dataset/images/train"
OUT_LABEL_DIR = "dataset/labels/train"

# =========================
# SETUP
# =========================
def ensure_dirs():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)


# =========================
# FIND JSON (from image name)
# =========================
def find_json(image_name):
    base = os.path.splitext(image_name)[0]

    for ext in [".json", ".png.json", ".jpg.json"]:
        path = os.path.join(INPUT_JSON_DIR, base + ext)
        if os.path.exists(path):
            return path

    return None


# =========================
# DECODE BITMAP
# =========================
def decode_bitmap_to_mask(bitmap):
    try:
        z = zlib.decompress(base64.b64decode(bitmap["data"]))
        nparr = np.frombuffer(z, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return mask
    except:
        return None


# =========================
# RECONSTRUCT FULL MASK
# =========================
def reconstruct_mask(bitmap, img_h, img_w):
    mask = decode_bitmap_to_mask(bitmap)

    if mask is None:
        return None

    origin_x, origin_y = bitmap["origin"]
    h, w = mask.shape

    full = np.zeros((img_h, img_w), dtype=np.uint8)

    y2 = min(origin_y + h, img_h)
    x2 = min(origin_x + w, img_w)

    full[origin_y:y2, origin_x:x2] = mask[: y2 - origin_y, : x2 - origin_x]

    return full


# =========================
# MASK → POLYGONS
# =========================
def mask_to_polygons(mask):
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for c in contours:
        if len(c) < 3:
            continue

        poly = c.squeeze()

        if len(poly.shape) != 2:
            continue

        polygons.append(poly)

    return polygons


# =========================
# NORMALIZE
# =========================
def normalize_polygon(poly, w, h):
    return [(x / w, y / h) for x, y in poly]


# =========================
# PROCESS IMAGE (MAIN LOGIC)
# =========================
def process_one(image_path):
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    json_path = find_json(image_name)

    if json_path is None:
        print(f"❌ Missing JSON for {image_name}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image {image_name}")
        return

    img_h, img_w = img.shape[:2]

    with open(json_path) as f:
        data = json.load(f)

    lines = []

    for obj in data["objects"]:
        if obj.get("geometryType") != "bitmap":
            continue

        mask = reconstruct_mask(obj["bitmap"], img_h, img_w)

        if mask is None or mask.sum() == 0:
            continue

        polygons = mask_to_polygons(mask)

        for poly in polygons:
            norm_poly = normalize_polygon(poly, img_w, img_h)

            flat = [coord for point in norm_poly for coord in point]

            if len(flat) < 6:
                continue

            line = "0 " + " ".join(map(str, flat))
            lines.append(line)

    if not lines:
        print(f"⚠️ No labels for {image_name}")
        return

    # copy image
    shutil.copy(image_path, os.path.join(OUT_IMG_DIR, image_name))

    # write label
    label_path = os.path.join(OUT_LABEL_DIR, base_name + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    images = [
        f for f in os.listdir(INPUT_IMG_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Found {len(images)} images")

    for img in tqdm(images):
        process_one(os.path.join(INPUT_IMG_DIR, img))


if __name__ == "__main__":
    main()