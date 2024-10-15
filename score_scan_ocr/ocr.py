from typing import List, Tuple, Union

import pandas as pd
from PIL import Image
import pytesseract as tes
from pytesseract import Output
import pypdfium2 as pdfium
import cv2
import numpy as np

from difflib import get_close_matches

from scipy.stats import gaussian_kde
from scipy import optimize

from utx.osutil import file_name, list_files_relative
from utx.color import Colors

from score_scan_ocr import defs, utils


def preprocess(file_path: str) -> Image:
    print(file_path)
    first_page = pdfium.PdfDocument(file_path).get_page(0)
    width, height = first_page.get_size()
    img = first_page.render_to(
        pdfium.BitmapConv.pil_image,
        scale=4, crop=(0, height*.75, 0, 0)
    )
    # img.save("imgs/"+file_name(file_path)+".jpg")
    return img


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df



def image_as_binary(img: np.ndarray) -> np.ndarray:
    color_distr = gaussian_kde(img.reshape(-1)/255)
    threshold = optimize.fmin(color_distr, .5, disp=False)
    return img > threshold * 255


def get_font_size(box: np.ndarray) -> np.ndarray:
    box = rgb2gray(box.copy())
    min_line = image_as_binary(box).mean(axis=0) < .95
    # remove white columns on the borders (so letter starts and ends at the borders)
    min_line = min_line[min_line.argmin():-(min_line[::-1].argmin() + 1)]
    # return np.mean(
    diff = np.diff(  # compute the length of the black/white regions
            np.where(  # select indices where pixels change from black to white and vice versa
                min_line[:-1] != min_line[1:]
            )[0]  # returns 1d-tuple
        )[::2]  # only take the black area lengths
    # )  # use the median as most spaces will be between chars and not whitespaces
    return diff.mean()


def plot_boxes(img: Image, boxes: pd.DataFrame, file: str) -> None:
    img = img.convert("RGB")
    img = np.array(img)[:, :, ::-1].copy()
    # d = tes.image_to_data(img, output_type=Output.DICT)
    max_box_height = (boxes["height"]/img.shape[0]).max()
    # print(img.shape)
    blocks = boxes["block_num"].max()

    for i, box in boxes.iterrows():
        (x, y, w, h) = (box['left'], box['top'], box['width'], box['height'])
        size = get_font_size(img[y:y+h, x:x+w])
        cv2.imwrite(f"boxes/{i}_box_{size:.2f}.png", img[y:y+h, x:x+w])
        c = int(box['height']/(max_box_height*img.shape[0])*255)
        # print(c, box["height"], max_box_height, box["height"]/img.shape[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, c*0, int(box["block_num"]/blocks*255)), 2)
        # cv2.putText(img, f"{box['conf']: .2f} - {box['text']}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, c, 0), 1)
        # cv2.putText(img, f"{box['block_num']}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, c, 0), 2)
        cv2.putText(img, f"{size:.2f}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, c, 0), 2)

    # cv2.imshow('img', img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img[:, :, ::-1])
    # plt.savefig(f"pred/mp_{file_name(file)}.jpg")

    cv2.imwrite(f"pred/{file_name(file)}.jpg", img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


def plot_groups(img: Image, groups: List[defs.Group], file: str) -> None:
    img = img.convert("RGB")
    img = np.array(img)[:, :, ::-1].copy()

    for group, color in zip(groups, Colors.gen(len(groups))):
        box = group.joint_box()
        cv2.rectangle(img, (box.left, box.top), (box.right, box.bottom), color.bgr, 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def best(original: List[defs.Group], weighting=List[float], *sortings: List[defs.Group]) -> Tuple[int, defs.Group]:
    scores = [0] * len(original)
    for i, sorting in enumerate(sortings):
        weight = weighting[i]
        for x, group in enumerate(sorting):
            idx = original.index(group)
            scores[idx] += (len(sorting) - x) * weight
    amax = np.argmax(np.array(scores))
    return int(amax), original[amax]


def parse_file(file_path: Union[str, Image.Image]) -> defs.Document:
    img = file_path if isinstance(file_path, Image.Image) else preprocess(file_path)
    # run ocr and remove artifacts
    data_df = tes.image_to_data(
        img, output_type=Output.DATAFRAME
    ).dropna()
    data_df["text"] = data_df["text"].astype(str).str.replace("[^a-zA-z0-9 ]", "", regex=True)
    data_df = data_df[data_df["text"].str.strip() != ""]

    if not len(data_df):
        return defs.Document.empty()

    groups = defs.Group.get_groups_from_df(data_df)
    # plot bounding boxes of the detected text
    # plot_groups(img, groups, file_path)
    plot_boxes(img, data_df, file_path)

    # todo extract title

    highest = sorted(groups, key=lambda g: g.joint_box().top)
    tallest = sorted(groups, key=lambda g: g.mean_h, reverse=True)
    widest = sorted(groups, key=lambda g: g.joint_box().width, reverse=True)
    leftest = sorted(groups, key=lambda g: g.joint_box().left)

    idx, title_group = best(groups, [1., 1.5, 2], highest, widest, tallest)
    groups = groups[:idx:]
    # todo extract subtitle

    # todo extract arrangeur

    # todo extract instrument
    # for g in groups:

    # construct [potential instruments] + [empty instrument] and pick the first one.
    # If no Instruments found the dummy is used.
    instrument = (utils.filter_none([defs.Instrument.parse(g.text) for g in groups]) + [defs.Instrument("")])[0]

    # todo extract composer

    # todo extract number


    # print(title_group.text)
    # print(instrument)
    return defs.Document(title_group.text, "", instrument, "", "", "")


if __name__ == '__main__':
    files = list_files_relative("samples", where=lambda f: f != "solution.txt")
    files.sort()
    for file in files:
        print(parse_file(file), end="\n\n")
    # parse_file("samples/Bond...James Bond 542.pdf")