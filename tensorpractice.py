import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

csv_path = "full_df.csv"
df = pd.read_csv(csv_path)


images_right = []
labels_right = []

images_left = []
labels_left = []

for index, row in df.iterrows():

    path = f"preprocessed_images/{row["filename"]}"

    if os.path.exists(path):
        img_info = cv2.imread(path)
        img_info = cv2.resize(img_info, (224, 224))
        img_info = img_info / 255.0

        if "right" in row["filename"]:
            images_right.append(img_info)

            label_right = row["labels"][2]
            labels_right.append(label_right)
        else:
            images_left.append(img_info)

            label_left = row["labels"][2]
            labels_left.append(label_left)

# tests to check that the arrays were filled correctly
print(len(images_right), len(labels_right))
print(len(images_left), len(labels_left))
print(labels_left)

# tests to check if the images are loaded properly

"""
img_info_right = (images_right[0] * 255).astype('uint8')
right_rgb = cv2.cvtColor(img_info_right, cv2.COLOR_BGR2RGB)

img_info_left = (images_left[0] * 255).astype('uint8')
left_rgb = cv2.cvtColor(img_info_left, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(left_rgb)
axes[1].imshow(right_rgb)

plt.show()
"""