import tempfile
from typing import Tuple, List

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

import pypdfium2 as pdfium

import os


class Pdf:
    def __init__(self, filename: str = None) -> None:
        self.handler = pdfium.PdfDocument.new()
        self.filename = filename

    def save(self, filename: str = None) -> None:
        """
        Saves the pdf to file. Filename can be omitted if self.filename was set before.
        :param filename: name/path to save the pdf to.
        """
        if filename is None and self.filename:
            filename = self.filename
        with open(filename, "wb") as buffer:
            self.handler.save(buffer)

    @classmethod
    def from_two_page_doc_to_single_page_horizontally_cut(cls, file_path: str, rotate180: bool = False, shiftl: int = 0, shiftr: int = 0):
        """
        Parse a scan of two pages per page by splitting them in the middle.
        :param file_path: path to the pdf document.
        :param rotate180: optionally rotate the image by 180 deg.
        :param shiftl: int - shift the left image by some pixels. The border will be replicated.
        :param shiftr: int - shift the right image by some pixels. The border will be replicated.
        :return: an pdf document with twice the number of pages containing the single pages.
        """
        # convert pdf to list of images
        double_pages = convert_from_path(file_path, dpi=300)

        # compute the cutoff to obtain equally sized pages
        cutoff_left = int(np.ceil(double_pages[0].width * 0.5))
        cutoff_right = int(double_pages[0].width * 0.5)

        def shift(arr: np.ndarray, n: int):
            """
            Shifts an image horizontally by repeating the borders.
            :param arr: 2D-array to be shifted
            :param n: amount of shift in pixels.
            :return: the shifted array
            """
            if n == 0:
                return arr
            if n < 0:
                bg = np.repeat(np.expand_dims(arr[:, -1, :], axis=1), repeats=arr.shape[1], axis=1)
                bg[:, :n] = arr[:, -n:]
            else:
                bg = np.repeat(np.expand_dims(arr[:, 0, :], axis=1), repeats=arr.shape[1], axis=1)
                bg[:, n:] = arr[:, :-n]
            return bg

        pages = []
        for page in tqdm(double_pages):
            page = np.array(page)

            if rotate180:
                page = np.rot90(page, k=2)

            pages += [
                Image.fromarray(shift(page[:, :cutoff_left], shiftl), "RGB"),
                Image.fromarray(shift(page[:, cutoff_right:], shiftr), "RGB")
            ]

        fname = "temp/" + str(np.random.random())[2:] + ".pdf"

        pages[0].save(
            fname, "PDF", resolution=300.0, save_all=True, append_images=pages[1:]
        )

        pdf = cls.from_file(fname)
        os.remove(fname)
        return pdf

    @classmethod
    def from_file(cls, path: str):
        """
        Load pdf from file.
        :param path: path to file.
        :return: Pdf object.
        """
        pdf = cls()
        pdf.handler = pdfium.PdfDocument(path)
        return pdf

    def add_page_from_other_pdf(self, old, page: pdfium.PdfPage, i: int) -> None:
        """
        Add a pdf page from old to new.
        :param new: Pdf document to add the page to.
        :param old: the Pdf document where the page originally belongs to.
        :param page: the page instance to copy to new.
        :param i: index of page to copy in the old document.
        """
        xpage = old.handler.page_as_xobject(i, self.handler)
        npage = self.handler.new_page(*page.get_size())
        npage.insert_object(xpage.as_pageobject())
        npage.generate_content()

    def page_as_img(self, i: int, scale: int = 4, only_top: bool = True) -> np.ndarray:
        width, height = self.handler[i].get_size()
        return np.array(self.handler[i].render_to(
            pdfium.BitmapConv.pil_image,
            scale=scale, crop=(0, height * .75 * int(only_top), 0, 0)))

    def __iter__(self):
        return self.handler.__iter__()

    def __len__(self):
        return len(self.handler)


def split_pdf(pdf_path, breaks: List[Tuple[int, str]]) -> None:
    """
    Splits pdf in multiple files.
    :param pdf_path: path to joint pdf
    :param breaks: tuple of (#pages: int, file_name: str)
    """
    pdf = Pdf.from_file(pdf_path)
    new_pdf = Pdf()
    npages, nname = breaks.pop(0)
    for i, page in enumerate(pdf):
        npages -= 1
        new_pdf.add_page_from_other_pdf(pdf, page, i)
        if npages == 0:
            new_pdf.save(nname)
            new_pdf = Pdf()
            if len(breaks):
                npages, nname = breaks.pop(0)


if __name__ == '__main__':
    # (
    #     Pdf.from_two_page_doc_to_single_page_horizontally_cut(
    #         "Images - Oriola.pdf",
    #         rotate180=False, shiftl=-30, shiftr=14
    #     )
    #     .save("../Images - Oriola.pdf")
    # )
    split_pdf(
        "Armenian Dances - Reed Alfred - Fagott 2.pdf",
        [
            (4, "Armenian Dances - Reed Alfred - Fagott 2s.pdf"),
            (3, "Armenian Dances - Reed Alfred - Kb Fagott.pdf")
        ]
    )
