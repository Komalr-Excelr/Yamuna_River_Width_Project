import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = (256, 256)
N_CLASSES = 4  # background, water, riverbank, land
BATCH_SIZE = 4
EPOCHS = 20

# Data loader
def load_image(path, mask=False):
    img = Image.open(path).resize(IMG_SIZE)
    arr = np.array(img)
    if mask:
        arr = arr.astype(np.uint8)
    else:
        arr = arr / 255.0
    return arr

def data_generator(img_dir, mask_dir, batch_size):
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    while True:
        idxs = np.random.permutation(len(img_files))
        for i in range(0, len(img_files), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            imgs = [load_image(os.path.join(img_dir, img_files[j])) for j in batch_idxs]
            masks = [load_image(os.path.join(mask_dir, mask_files[j]), mask=True) for j in batch_idxs]
            imgs = np.stack(imgs)
            masks = np.stack(masks)
            masks = keras.utils.to_categorical(masks, num_classes=N_CLASSES)
            yield imgs, masks

def get_steps(img_dir, batch_size):
    return int(np.ceil(len(os.listdir(img_dir)) / batch_size))

# U-Net model
def unet(input_shape, n_classes):
    inputs = keras.Input(shape=input_shape)
    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)
    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)
    # Decoder
    u3 = layers.UpSampling2D()(b)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)
    u1 = layers.UpSampling2D()(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(16, 3, activation='relu', padding='same')(c6)
    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(c6)
    return keras.Model(inputs, outputs)

# Paths
data_dir = 'data/dataset'
train_img = os.path.join(data_dir, 'train/images')
train_mask = os.path.join(data_dir, 'train/masks')
val_img = os.path.join(data_dir, 'val/images')
val_mask = os.path.join(data_dir, 'val/masks')

# Generators
train_gen = data_generator(train_img, train_mask, BATCH_SIZE)
val_gen = data_generator(val_img, val_mask, BATCH_SIZE)
steps_train = get_steps(train_img, BATCH_SIZE)
steps_val = get_steps(val_img, BATCH_SIZE)

# Model
model = unet((*IMG_SIZE, 3), N_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=steps_val
)

# Save model
model.save('unet_river_segmentation.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_history.png')
plt.close()
