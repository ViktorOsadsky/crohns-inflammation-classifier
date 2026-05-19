from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs",
    name="distal_ileum_yolov8s"
)