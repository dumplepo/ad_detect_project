from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="billboard_seg"
)