import cv2
import json
import numpy as np
import zlib
import base64

# -----------------------------
# File paths
# -----------------------------
image_path = "data/test/2.png"
json_path = "data/test/2.json"
replacement_path = "data/new/new.jpg"
output_path = "output_billboard_tiled.png"

# -----------------------------
# Load images
# -----------------------------
image = cv2.imread(image_path)
replacement = cv2.imread(replacement_path)

if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")
if replacement is None:
    raise FileNotFoundError(f"Cannot load replacement image: {replacement_path}")

output = image.copy()

# -----------------------------
# Decode bitmap mask
# -----------------------------
def decode_bitmap_mask(encoded_bitmap):
    compressed = base64.b64decode(encoded_bitmap)
    decompressed = zlib.decompress(compressed)
    mask = np.frombuffer(decompressed, dtype=np.uint8)
    mask = cv2.imdecode(mask, cv2.IMREAD_UNCHANGED)
    return mask

# -----------------------------
# Create tiled ad image
# -----------------------------
def create_tiled_image(ad_img, target_h, target_w):
    ad_h, ad_w = ad_img.shape[:2]

    # Scale ad image to target height while preserving aspect ratio
    scale = target_h / ad_h
    new_w = int(ad_w * scale)

    ad_resized = cv2.resize(ad_img, (new_w, target_h))

    # Tile horizontally
    tiled = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = 0
    while x < target_w:
        end_x = min(x + new_w, target_w)
        width_to_copy = end_x - x
        tiled[:, x:end_x] = ad_resized[:, :width_to_copy]
        x += new_w

    return tiled

# -----------------------------
# Load annotations
# -----------------------------
with open(json_path, "r") as f:
    data = json.load(f)

# -----------------------------
# Replace billboard with tiled ad
# -----------------------------
for obj in data["objects"]:
    bitmap_data = obj["bitmap"]["data"]
    origin_x, origin_y = obj["bitmap"]["origin"]

    mask = decode_bitmap_mask(bitmap_data)

    # Extract alpha mask
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        alpha_mask = mask[:, :, 3]
    else:
        alpha_mask = mask

    h, w = alpha_mask.shape[:2]

    # Create tiled ad matching billboard box size
    tiled_ad = create_tiled_image(replacement, h, w)

    # Region of interest
    roi = output[origin_y:origin_y+h, origin_x:origin_x+w]

    # Apply mask
    mask_binary = alpha_mask > 0
    roi[mask_binary] = tiled_ad[mask_binary]

# -----------------------------
# Save result
# -----------------------------
cv2.imwrite(output_path, output)

print(f"Saved result: {output_path}")

cv2.imshow("Tiled Billboard Replacement", output)
cv2.waitKey(0)
cv2.destroyAllWindows()