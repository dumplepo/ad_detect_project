import cv2
import numpy as np
import os
from glob import glob

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_FOLDER = "data/train/mask/"
OUTPUT_FOLDER = "output_boundaries"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# LOAD IMAGES
# ----------------------------
image_paths = glob(os.path.join(IMAGE_FOLDER, "*.*"))

if len(image_paths) == 0:
    raise ValueError("No images found in data/train/mask/")

# ----------------------------
# PROCESS EACH IMAGE
# ----------------------------
for img_path in image_paths:
    # Read grayscale image
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"Skipping invalid image: {img_path}")
        continue

    # ----------------------------
    # BINARY THRESHOLD (white vs black)
    # ----------------------------
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # ----------------------------
    # FIND CONTOURS (BOUNDARIES)
    # ----------------------------
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert grayscale to BGR for visualization
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    all_even_points = []

    # ----------------------------
    # PROCESS EACH CONTOUR
    # ----------------------------
    for cnt in contours:
        # Approximate contour to reduce points (important!)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Flatten points
        points = approx.reshape(-1, 2)

        # Ensure even number of coordinates
        if len(points) % 2 != 0:
            points = np.vstack([points, points[-1]])

        # Save as flat list [x1,y1,x2,y2,...]
        flat_points = points.flatten().tolist()
        all_even_points.append(flat_points)

        # ----------------------------
        # DRAW BOUNDARY (GREEN LINES)
        # ----------------------------
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

    # ----------------------------
    # SAVE RESULT IMAGE
    # ----------------------------
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_FOLDER, f"bound_{filename}")
    cv2.imwrite(save_path, output)

    # ----------------------------
    # PRINT POINTS
    # ----------------------------
    print(f"\nImage: {filename}")
    for i, pts in enumerate(all_even_points):
        print(f"Contour {i} points (even-length): {len(pts)}")
        print(pts)

    # ----------------------------
    # SHOW RESULT
    # ----------------------------
    cv2.imshow("Boundary Detection", output)
    cv2.waitKey(0)

cv2.destroyAllWindows()