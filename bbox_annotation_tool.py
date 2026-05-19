import cv2
import os
import csv
import matplotlib.pyplot as plt


img_folder = "total with marks/total with marks/"
img_files = os.listdir(img_folder)
annotations = []
rect_patches = []

start_point = None
current_image_name = None
image = None

fig, ax = None, None

def on_press(event):
    global start_point

    if event.xdata is None or event.ydata is None:
        return

    start_point = (int(event.xdata), int(event.ydata))

def on_move(event):
    global rect_patch, start_point

    if start_point is None or event.xdata is None or event.ydata is None:
        return

    x1, y1 = start_point
    x2, y2 = int(event.xdata), int(event.ydata)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    w = x_max - x_min
    h = y_max - y_min

    if hasattr(on_move, "rect") and on_move.rect:
        on_move.rect.remove()

    on_move.rect = plt.Rectangle(
        (x_min, y_min),
        w,
        h,
        fill=False,
        edgecolor="red",
        linewidth=1
    )

    ax.add_patch(on_move.rect)
    fig.canvas.draw_idle()

on_move.rect = None

def on_release(event):
    global start_point

    if start_point is None or event.xdata is None or event.ydata is None:
        return

    x1, y1 = start_point
    x2, y2 = int(event.xdata), int(event.ydata)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    w = x_max - x_min
    h = y_max - y_min

    x_center = (x_min + w / 2) / image.shape[1]
    y_center = (y_min + h / 2) / image.shape[0]
    w_n = w / image.shape[1]
    h_n = h / image.shape[0]

    annotations.append([
        current_image_name,
        x_center,
        y_center,
        w_n,
        h_n
    ])

    rect = plt.Rectangle(
        (x_min, y_min),
        w,
        h,
        fill=False,
        edgecolor="red",
        linewidth=1
    )
    ax.add_patch(rect)
    rect_patches.append(rect)

    start_point = None
    fig.canvas.draw_idle()

def on_key(event):
    global annotations, rect_patches

    if event.key == "ctrl+z":

        if len(annotations) == 0:
            return

        annotations.pop()

        last_rect = rect_patches.pop()
        last_rect.remove()

        fig.canvas.draw_idle()

        print("Undo last annotation")

for img_name in img_files:

    current_image_name = img_name
    image_path = os.path.join(img_folder, img_name)

    image = cv2.imread(image_path)
    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(img_name)
    ax.axis("off")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

csv_file = "bboxes.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x_center", "y_center", "width", "height"])
    writer.writerows(annotations)

print("Saved:", csv_file)