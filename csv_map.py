import cv2
import os
import csv
import matplotlib.pyplot as plt


csv_file = "bboxes.csv"
map_file = "mapped_bboxes.csv"

annotations = []

with open(csv_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    i = 0
    for row in reader:
        annotations.append(row)
        filename = row[0]
        i += 1
        row[0] = i

    
    with open(map_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x_center", "y_center", "width", "height"])
        writer.writerows(annotations)