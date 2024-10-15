import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from utx.plx import plx
from utx.imgx import load


def main():
    file = "validated_data/title/00_C Direktion_0.png"
    img = load(file)
    img_dil = cv2.dilate(255 - img, np.ones((20, 20)))
    img_dil = cv2.cvtColor(img_dil, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dx, dy = cv2.spatialGradient(img)

    ratio = np.abs(dx).sum() / np.abs(dy).sum()
    print(ratio)
    if not 0.85 < ratio < 1.15:
        if ratio > 1.15:  # landscape
            img = np.rot90(img)
    plx.imshow(img).show()
    plt.figure(figsize=(16, 16))
    plt.subplot(321)
    plt.imshow(img, cmap="gray")
    plt.subplot(322)
    plt.imshow(np.repeat(np.abs(dy).sum(axis=1, keepdims=True), 600, axis=1), cmap="gray")
    plt.subplot(323)
    plt.imshow(np.repeat(np.abs(dx).sum(axis=0, keepdims=True), 600, axis=0), cmap="gray")
    plt.subplot(324)
    plt.imshow(img, cmap="gray")
    plt.subplot(325)
    plt.imshow(np.abs(dx), cmap="gray")
    plt.title("dx")
    plt.subplot(326)
    plt.title("dy")
    plt.imshow(np.abs(dy), cmap="gray")
    plt.show()
    plt.plot(np.median(img_dil.sum(axis=1)) - img_dil.sum(axis=1))
    plt.show()
    plt.imshow(img[img_dil.sum(axis=1) < np.median(img_dil.sum(axis=1)), :], cmap="gray")
    plt.show()

    print()


if __name__ == '__main__':
    main()