import imagesize
import numpy as np
import cv2

from tqdm import tqdm

from utx.statistics import images
from utx import imagex
from utx import osx
from utx.plot import plot
import matplotlib.pyplot as plt

import tensorflow as tf


SIZE = 2340//2, 1650//2


def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess() -> None:
    paths = osx.list_files_relative_recursively("validated_data", where=lambda p: p.endswith(".png"))
    dims = np.array([imagesize.get(path) for path in paths]).T  # [2, N]
    aspect_ratio = dims[0] / dims[1]
    plt.hist(aspect_ratio)
    plt.show()
    indices = np.argwhere(aspect_ratio > 1).reshape(-1)
    for i in tqdm(indices, desc="Adjusting Aspect Ratios"):
        img = imagex.load(paths[i])
        imagex.save(np.rot90(img), paths[i])

    print(images.get_image_stats(paths))


def get_model() -> tf.keras.Model:
    inp = tf.keras.Input((*SIZE, 3))
    out = tf.keras.layers.Conv2D(16, 5, strides=2)(inp)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Conv2D(16, 5, strides=2)(out)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(1)(out)
    out = tf.keras.layers.Activation("sigmoid")(out)

    return tf.keras.Model(inp, out)


def get_data(validation: bool = False):
    return tf.keras.utils.image_dataset_from_directory(
        "./validated_data",
        image_size=SIZE,
        interpolation="bilinear",
        color_mode="rgb",
        batch_size=32,
        labels="inferred",
        label_mode="binary",
        class_names=["no_title", "title"],
        validation_split=.3,
        subset="validation" if validation else "training",
        shuffle=not validation,
        seed=42
    ).map(
        lambda img, lbl: (tf.image.per_image_standardization(img), lbl)
    )


def train():
    model = get_model()

    model.compile(
        optimizer="adam",
        loss="bce",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.F1Score()
        ]
    )

    model.summary()

    train_gen = get_data()
    val_gen = get_data(validation=True)

    model.fit(
        x=train_gen,
        validation_data=val_gen,
        class_weight={0: .2, 1: .8},
        epochs=10
    )


if __name__ == '__main__':
    # preprocess()
    train()
