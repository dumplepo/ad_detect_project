import cv2
import json
import base64
import zlib
import numpy as np
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/segment/billboard_seg2/weights/best.pt"
IMAGE_PATH = "data/test/2.png"
OUTPUT_JSON = "data/test/2.json"


# =========================
# ENCODE MASK → JSON FORMAT
# =========================
def encode_bitmap(mask):
    # encode mask as PNG
    success, buffer = cv2.imencode(".png", mask.astype(np.uint8))
    if not success:
        return None

    # compress + base64
    compressed = zlib.compress(buffer.tobytes())
    b64 = base64.b64encode(compressed).decode("utf-8")

    return b64


# =========================
# MAIN
# =========================
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)

image = cv2.imread(IMAGE_PATH)
img_h, img_w = image.shape[:2]

objects = []

for r in results:
    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()  # (N, H, W)

    for mask in masks:
        # binary mask
        mask = (mask > 0.5).astype(np.uint8) * 255

        # find bounding box (crop)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = mask[y_min:y_max+1, x_min:x_max+1]

        encoded = encode_bitmap(cropped)

        if encoded is None:
            continue

        obj = {
            "id": int(np.random.randint(1e9)),
            "classId": 0,
            "description": "",
            "geometryType": "bitmap",
            "labelerLogin": "model",
            "createdAt": "",
            "updatedAt": "",
            "tags": [],
            "classTitle": "billboard",
            "bitmap": {
                "data": encoded,
                "origin": [int(x_min), int(y_min)]
            }
        }

        objects.append(obj)


# =========================
# FINAL JSON
# =========================
output = {
    "description": "",
    "tags": [],
    "size": {
        "height": img_h,
        "width": img_w
    },
    "objects": objects
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=4)

print(f"✅ Saved: {OUTPUT_JSON}")