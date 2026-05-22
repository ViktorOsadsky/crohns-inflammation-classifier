import cv2
import os
import csv
import matplotlib.pyplot as plt


img_folder = "Control without mark"
csv_file = "extractedbboxes.csv"

annotations = {}

with open(csv_file, "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        filename = row[0]
        x_c, y_c, w, h = map(float, row[1:])

        if filename not in annotations:
            annotations[filename] = []

        annotations[filename].append((x_c, y_c, w, h))

for img_name in os.listdir(img_folder):

    image_path = os.path.join(img_folder, img_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_img, w_img = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(img_name)
    ax.axis("off")

    if img_name in annotations:

        for (x_c, y_c, w, h) in annotations[img_name]:

            x_c *= w_img
            y_c *= h_img
            w *= w_img
            h *= h_img

            x_min = int(x_c - w / 2)
            y_min = int(y_c - h / 2)

            rect = plt.Rectangle(
                (x_min, y_min),
                w,
                h,
                fill=False,
                edgecolor="lime",
                linewidth=2
            )

            ax.add_patch(rect)

    plt.show()