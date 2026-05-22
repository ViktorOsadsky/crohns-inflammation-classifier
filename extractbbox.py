import os
import cv2
import csv
import numpy as np

IMAGE_FOLDER = "Control with mark"
OUTPUT_CSV = "extractedbboxes.csv"

with open(OUTPUT_CSV, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "x_center",
        "y_center",
        "width",
        "height"
    ])
    for filename in os.listdir(IMAGE_FOLDER):

        image_path = os.path.join(IMAGE_FOLDER, filename)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print(f"No bbox found: {filename}")
            continue

        contour = max(contours, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(contour)

        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h

        writer.writerow([
            filename,
            x_center,
            y_center,
            width,
            height
        ])

        print(f"Processed: {filename}")

print(f"\nSaved bbox data to: {OUTPUT_CSV}")