from ultralytics import YOLO

# load your trained model
model = YOLO("runs/segment/billboard_seg2/weights/best.pt")

# run inference
results = model("data/test/test.jpg", save=True)

print("Done. Check runs/segment/predict/")