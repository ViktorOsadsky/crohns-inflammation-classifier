from ultralytics import YOLO

model = YOLO(
    "runs/detect/distal_ileum_yolov8s/weights/best.pt"
)

metrics = model.val(
    data="yolo_dataset/dataset.yaml"
)

print(metrics)


results = model.predict(
    source="yolo_dataset/images/val",
    save=True,
    conf=0.25
)