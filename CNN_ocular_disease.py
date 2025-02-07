import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import LabelEncoder

csv_path = "full_df.csv"
df = pd.read_csv(csv_path)

# print(df[["Right-Diagnostic Keywords", "labels"]])
images = []
labels = []


for index, row in df.iterrows():

    path = f"preprocessed_images/{row["filename"]}"

    if os.path.exists(path):
        img_info = cv2.imread(path)
        img_info = cv2.resize(img_info, (256, 256))
        img_info = img_info / 255.0
        img_info = img_info.astype(np.float32)  # Make sure images are float32

        images.append(img_info)

        label = row["labels"][2]
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print(labels)

dataset = tf.data.Dataset.from_tensor_slices((images, labels))

dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# testing to see if the images and labels are of proper shape
for batch_images, batch_labels in dataset.take(1):  # take(1) to get just one batch
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch images dtype: {batch_images.dtype}")
    print(f"Batch labels dtype: {batch_labels.dtype}")

    # test to see if image is loaded properly
    first_image = cv2.cvtColor(batch_images[0].numpy(), cv2.COLOR_BGR2RGB)
    plt.imshow(first_image)
    plt.show()

# tests to check if the images are loaded properly
"""
img_info_right = (images[0] * 255).astype('uint8')
right_rgb = cv2.cvtColor(img_info_right, cv2.COLOR_BGR2RGB)

plt.imshow(right_rgb)
plt.show()

img_info_right = (images_right[0] * 255).astype('uint8')
right_rgb = cv2.cvtColor(img_info_right, cv2.COLOR_BGR2RGB)

img_info_left = (images_left[0] * 255).astype('uint8')
left_rgb = cv2.cvtColor(img_info_left, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(left_rgb)
axes[1].imshow(right_rgb)

plt.show()
"""
# tests to check that the arrays were filled correctly