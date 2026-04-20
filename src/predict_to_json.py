import os
import json
import zlib
import base64
import random
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

TEST_IMAGE_DIR = r"data/test/image"
OUTPUT_JSON_DIR = r"data/output_json"
MODEL_PATH = r"runs/segment/billboard_seg/weights/best.pt"

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)


def encode_bitmap(mask):
    ys, xs = np.where(mask > 0)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = mask[y_min:y_max+1, x_min:x_max+1]

    pil_img = Image.fromarray(cropped)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    png_data = buffer.getvalue()

    compressed = zlib.compress(png_data)
    encoded = base64.b64encode(compressed).decode("utf-8")

    return encoded, [int(x_min), int(y_min)]


model = YOLO(MODEL_PATH)

images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith((".jpg", ".png"))]
img_name = random.choice(images)

img_path = os.path.join(TEST_IMAGE_DIR, img_name)
img = cv2.imread(img_path)

h, w = img.shape[:2]

results = model(img_path)[0]

objects = []

if results.masks is not None:
    masks = results.masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 0.5).astype(np.uint8) * 255

        bitmap_data, origin = encode_bitmap(mask)

        obj = {
            "id": i + 1,
            "classId": 1,
            "description": "",
            "geometryType": "bitmap",
            "labelerLogin": "yolov8-seg",
            "createdAt": "",
            "updatedAt": "",
            "tags": [],
            "classTitle": "billboard",
            "bitmap": {
                "data": bitmap_data,
                "origin": origin
            }
        }

        objects.append(obj)

output = {
    "description": "",
    "tags": [],
    "size": {
        "height": h,
        "width": w
    },
    "objects": objects
}

json_path = os.path.join(OUTPUT_JSON_DIR, img_name.replace(".jpg", ".json"))
with open(json_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved JSON: {json_path}")