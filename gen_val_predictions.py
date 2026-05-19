from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

model = YOLO("runs/detect/distal_ileum_yolov8s/weights/best.pt")

img_path = "yolo_dataset/images/val"
label_path = "yolo_dataset/labels/val"

output_dir = "val_predictions"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(img_path):

    image_file = os.path.join(img_path, img_name)
    label_file = os.path.join(label_path, img_name.replace(".png", ".txt"))

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f.readlines():
                _, x, y, bw, bh = map(float, line.split())

                x *= w
                y *= h
                bw *= w
                bh *= h

                x1 = x - bw / 2
                y1 = y - bh / 2

                ax.add_patch(plt.Rectangle(
                    (x1, y1),
                    bw,
                    bh,
                    fill=False,
                    edgecolor="green",
                    linewidth=2
                ))

    results = model.predict(image_file, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes.xywh:
            x, y, bw, bh = box.tolist()

            x1 = x - bw / 2
            y1 = y - bh / 2

            ax.add_patch(plt.Rectangle(
                (x1, y1),
                bw,
                bh,
                fill=False,
                edgecolor="red",
                linewidth=2
            ))

    ax.set_title(img_name)
    ax.axis("off")

    save_path = os.path.join(output_dir, img_name.replace(".png", ".jpg"))

    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

print(f"Saved overlays to: {output_dir}")