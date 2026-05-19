import os
import csv
import shutil
import random
from sklearn.model_selection import train_test_split

CSV_FILE = "bboxes.csv"
IMAGE_FOLDER = "total without marks/total w_o marks"

OUTPUT_FOLDER = "yolo_dataset"
VAL_SPLIT = 0.2
RANDOM_SEED = 42

folders = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT_FOLDER, folder), exist_ok=True)

annotations = []

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        filename = row[0]
        x_center = row[1]
        y_center = row[2]
        width = row[3]
        height = row[4]
        annotations.append([
            filename,
            x_center,
            y_center,
            width,
            height
        ])

train_annotations, val_annotations = train_test_split(
    annotations,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED
)

def process_split(annotation_list, split_name):
    for ann in annotation_list:
        filename, x_c, y_c, w, h = ann
        image_src = os.path.join(IMAGE_FOLDER, filename)

        image_dst = os.path.join(
            OUTPUT_FOLDER,
            "images",
            split_name,
            filename
        )

        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)

        label_filename = os.path.splitext(filename)[0] + ".txt"

        label_path = os.path.join(
            OUTPUT_FOLDER,
            "labels",
            split_name,
            label_filename
        )

        with open(label_path, "w") as f:
            f.write(f"0 {x_c} {y_c} {w} {h}")

process_split(train_annotations, "train")
process_split(val_annotations, "val")

yaml_text = """
path: yolo_dataset

train: images/train
val: images/val

names:
  0: distal_ileum
"""

with open(os.path.join(OUTPUT_FOLDER, "dataset.yaml"), "w") as f:
    f.write(yaml_text)

print("YOLO dataset generated successfully.")