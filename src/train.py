from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")  # pretrained segmentation model

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu"
    )

if __name__ == "__main__":
    main()