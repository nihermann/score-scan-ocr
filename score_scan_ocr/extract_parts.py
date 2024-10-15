import difflib
from difflib import get_close_matches, SequenceMatcher
from typing import Optional, Tuple

import PIL
import re

import matplotlib.pyplot as plt
import pandas as pd

import pypdfium2 as pdfium
import pytesseract as tes
from pytesseract import Output

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import numpy as np
import cv2

from tqdm import tqdm

from argparse import ArgumentParser
import os

import defs
from score_scan_ocr.pdfs.handle import Pdf
from utils import rnd_chars, count_all_pdf_pages, plot_page_count

from utx.plx import plx
from utx.plx import ishow
from utx.colorx import Colors


def split(path: str, title_keyword: str, scale: float = 4) -> None:
    """
    Splits parts out of a collective pdf and saves each part into a separate pdf with its instrument name.
    :param path: path to the collective pdf.
    :param title_keyword: keyword that is clearly visible on every title page of every part.
    :param scale: image upscale factor to improve ocr.
    """
    pdf = Pdf.from_file(path)
    doc: Optional[Pdf] = None
    list_of_files_with_blank_pages = []
    num_removed_pages = 0
    # use file name as prefix: path/to/file.pdf -> file
    prefix = path.split("/")[-1].split(".")[0]

    for i, page in tqdm(enumerate(pdf), desc="Pages parsed", total=len(pdf)):
        # also checks if page is empty
        page_img, low_res_img = get_top25_as_img(page, scale)
        if is_empty_page(low_res_img):
            num_removed_pages += 1
            if doc.filename:
                list_of_files_with_blank_pages.append(doc.filename)
            continue

        # if keyword was found this page should be a title page.
        data_df = tes.image_to_data(
            remove_stamps(page_img.copy(), th=40), output_type=Output.DATAFRAME
        ).dropna()

        # sum all found text together separated by a space.
        all_text = (data_df.text.astype(str).str.lower() + " ").sum()

        if title_keyword in all_text:
        # if has_stamp(page_img, hue=173):

            # if there was a document open we save it and subsequently create a new one.
            if doc is not None:
                doc.save()

            # extract instrument from the title page and generate the filename.
            filename = retrieve_filename(data_df, prefix)
            doc = Pdf(filename)

        if doc is not None:
            doc.add_page_from_other_pdf(pdf, page, i)

    # save the last part too.
    doc.save()

    if len(list_of_files_with_blank_pages):
        print("\n" + "=" * 50)
        print("Please check the following files for missing pages:")
        print(*list_of_files_with_blank_pages, sep="\n")

    n = count_all_pdf_pages("splits/")
    if len(pdf) - num_removed_pages - n != 0:
        print("\n" + "=" * 50)
        print(Colors.red.c(
            f"Files or pages went missing! Num of pages in the original pdf: {len(pdf)}, removed: {num_removed_pages}, found {n} in splits/."))
        print(Colors.red.c(f"-> {len(pdf) - num_removed_pages - n} pages are missing!"))

    plot_page_count("splits/")


def get_top25_as_img(page: pdfium.PdfPage, scale: float = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops the top 25% out of some pdf page.
    :param page: page to extract the top.
    :param scale: up scale factor.
    :return: the crop as a numpy array in scaled and low resolution.
    """
    width, height = page.get_size()
    low_res_img = page.render_to(pdfium.BitmapConv.pil_image, scale=0.5, crop=(0, height*.75, 0, 0))

    return np.array(page.render_to(
        pdfium.BitmapConv.pil_image,
        scale=scale, crop=(0, height*.75, 0, 0)
    )), np.array(low_res_img)


def is_empty_page(img: np.ndarray) -> bool:
    """
    Checks if a page is (mostly) empty.
    :param img: image of a page.
    :return: bool indicating if it's empty or not.
    """
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    dx, dy = cv2.spatialGradient(img)

    xm, ym = np.abs(dx).mean(), np.abs(dy).mean()
    if min(xm, ym) < 6:
        plx.imshow(img).title(f"dx: {xm:.3f} ; dy: {ym:.3f}").show()
        return True
    return False


def has_stamp(img: np.ndarray, hue: int) -> bool:
    return (np.abs(img - np.array([19, 141, 132])) < 10).all(2).sum() > 400


def remove_stamps(img: np.ndarray, th: int = 40) -> np.ndarray:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[img_hsv[..., 1] > th, :] = 255
    return img


def retrieve_filename(data_df: pd.DataFrame, prefix: str) -> str:
    """
    Retrieves the filename by attempting to find the instrument, its tune and part no.
    :param data_df: ocr results of the current page.
    :param prefix: prefix of the file before the instrument specifications.
    :return: the filename - it's an x followed by random chars in case nothing could be found.
    """
    instr = "x" + rnd_chars(10)
    try:
        instr = find_instrument(data_df)
    except defs.InstrumentNotFoundException:
        # print(f"Failed to retrieve the instrument name - find it and rename it under {instr}")
        pass
    filename = f"splits/{prefix} - {instr}.pdf"
    while os.path.exists(filename):
        filename = filename.split(".")[0] + rnd_chars(1) + ".pdf"
    return filename


def find_instrument(data_df: pd.DataFrame) -> defs.Instrument:
    """
    Groups ocr hitboxes and anticipates the leftmost group to contain the instrument.
    :param data_df: DataFrame containing the ocr results.
    :return: The text of the anticipated instrument group.
    """
    data_df = data_df[data_df["text"].astype(str).str.strip() != ""]
    pd.set_option('mode.chained_assignment', None)
    data_df.text = data_df.text.str.replace("/", "")
    if not len(data_df):
        raise Exception("Are you sure this is a title page? No text found on this crop.")

    groups = defs.Group.get_groups_from_df(data_df)
    max_score = 0
    best = None
    stop_words = set(stopwords.words("german") + stopwords.words("english"))
    for group in groups:
        text = re.sub(r'[^\w\s]', '', group.text)
        names = [word for word in word_tokenize(text, "german") if not word.lower() in stop_words]
        text = " ".join(names)

        for name in names:
            potential = get_close_matches(name, list(defs.Inverse(defs.Instruments).keys()), n=1)
            if not potential:
                continue

            score = SequenceMatcher(None, name, potential[0]).ratio()
            if score > max_score:
                max_score = SequenceMatcher(None, name, potential[0]).ratio()
                best = text

    return defs.Instrument.parse(best)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        required=True, type=str,
        help="Please specify the path to the pdf document containing all parts."
    )
    parser.add_argument(
        "-kw", "--keyword",
        required=True, type=str,
        help="Provide one word that is clearly readable on every title page of every part."
    )
    parser.add_argument(
        "-sc", "--scale",
        default=4, type=float,
        help="Upscaling the image can increase the ocr accuracy but computation takes longer. Default=4"
    )
    args = parser.parse_args()
    os.makedirs("./splits", exist_ok=True)

    split(args.file, args.keyword, args.scale)