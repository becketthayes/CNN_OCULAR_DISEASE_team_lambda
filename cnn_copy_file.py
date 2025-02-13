import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def visualizer_helper(images, labels):
    combined_array = list(zip(images, labels))
    np.random.shuffle(combined_array)

    data_visualizer = iter(combined_array)
    fig, axes = plt.subplots(2, 4, figsize = (20, 8))
    index_row = 0
    index_col = 0
    labels_stored = []
    while index_row < 2:
        curr_data = next(data_visualizer)
        if curr_data[1] in labels_stored:
            continue
        img = cv2.cvtColor(curr_data[0], cv2.COLOR_BGR2RGB)
        axes[index_row, index_col].imshow(img)
        axes[index_row, index_col].title.set_text(curr_data[1])
        if index_col < 3:
            index_col += 1
        else:
            index_row += 1
            index_col = 0
        labels_stored.append(curr_data[1])

    plt.show()

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

#visualizer_helper(images, labels)

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
    """
    first_image = cv2.cvtColor(batch_images[0].numpy(), cv2.COLOR_BGR2RGB)
    plt.imshow(first_image)
    plt.show()
    """

len_train = 160
len_val = 20
len_test = 20

train_data = dataset.take(len_train)
val_data = dataset.skip(len_train).take(len_val)
test_data = dataset.skip(len_train+len_val).take(len_test)

class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(zip(np.unique(labels), class_weights))

"""
resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(256, 256, 3),
                                                  pooling='avg',
                                                  classes=8,
                                                  weights="imagenet")
for layer in pretrained_model.layers:
    layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(64, activation="relu"))
resnet_model.add(Dense(8, activation="softmax"))
resnet_model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet_model.summary()
resnet_model.fit(train_data, epochs=20, validation_data=val_data, class_weight = class_weights)
"""
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.35))
model.add(Dense(8, activation="softmax"))

model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, epochs=70, validation_data=val_data, class_weight = class_weights)
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

# tests to check that the arrays were filled correctly

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size_).take(test_size)
"""

