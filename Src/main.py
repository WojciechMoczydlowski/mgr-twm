import numpy as np
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import os
from skimage import transform
from PIL import Image
from keras import backend as K
from skimage.io import imread, imsave, imread_collection, concatenate_images
from matplotlib import pyplot as plt


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou


image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
test_images = tf.keras.preprocessing.image.DirectoryIterator(
    directory="./Data/Validation/Images",
    image_data_generator=image_generator,
    shuffle=False,
)

test_masks = tf.keras.preprocessing.image.DirectoryIterator(
    directory="./Data/Validation/Masks",
    image_data_generator=image_generator,
    shuffle=False,
)


model = load_model(
    "./Models/road_mapper_final.h5",
    custom_objects={
        "soft_dice_loss": soft_dice_loss,
        "iou_coef": iou_coef,
        "dice_coef_loss": dice_coef,
        "dice_loss": dice_coef,
    },
)

# evaluation = model.evaluate(test_images, test_masks)
predictions = model.predict(test_images, verbose=1)

thresh_val = 0.1
predicton_threshold = (predictions > thresh_val).astype(np.uint8)

ix = random.randint(0, len(predictions))
num_samples = 10

images, _ = next(test_images)
masks, _ = next(test_masks)

for i in range(31):
    index = i
    path = os.path.join("validation", f"{index}")
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imsave(
        f"validation/{index}/prediction.tiff", np.squeeze(predictions[i][:, :, 0])
    )
    plt.imsave(
        f"validation/{index}/image.tiff",
        images[i].astype(np.uint8),
    )
    plt.imsave(
        f"validation/{index}/mask.tiff",
        masks[i].astype(np.uint8),
    )

    image = images[0]
    pred = predictions[i]
