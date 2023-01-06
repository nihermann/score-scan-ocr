import PIL
import pypdfium2 as pdfium
import pytesseract as tes
from pytesseract import Output
from argparse import ArgumentParser
import os

import defs
from utils import rnd_chars


def get_top25_as_img(page: pdfium.PdfPage, scale: float = 4) -> PIL.Image.Image:
    """
    Crops the top 25% out of some pdf page.
    :param page: page to extract the top.
    :param scale: up scale factor.
    :return: the crop as a PIL.Image
    """
    width, height = page.get_size()
    return page.render_to(
        pdfium.BitmapConv.pil_image,
        scale=scale, crop=(0, height*.75, 0, 0)
    )


def contains_keyword(img: PIL.Image.Image, kw: str) -> bool:
    """
    Checks if some keyword is found in some image with ocr. Case is ignored.
    :param img: image to perform ocr on.
    :param kw: string to search for.
    :return: bool whether kw was found in the image or not.
    """
    text = tes.image_to_string(img)
    return kw.lower() in text.lower()


def add_page(new: pdfium.PdfDocument, old: pdfium.PdfDocument, page: pdfium.PdfPage, i: int) -> None:
    """
    Add a pdf page from old to new.
    :param new: Pdf document to add the page to.
    :param old: the Pdf document where the page originally belongs to.
    :param page: the page instance to copy to new.
    :param i: index of page to copy in the old document.
    """
    xpage = old.page_as_xobject(i, new)
    npage = new.new_page(*page.get_size())
    npage.insert_object(xpage.as_pageobject())
    npage.generate_content()


def find_instrument(img) -> defs.Instrument:
    """
    Groups ocr hitboxes and anticipates the leftmost group to contain the instrument.
    :param img: of the document to perform ocr on.
    :return: The text of the anticipated instrument group.
    """
    data_df = tes.image_to_data(
        img, output_type=Output.DATAFRAME
    ).dropna()
    data_df = data_df[data_df["text"].astype(str).str.strip() != ""]
    if not len(data_df):
        raise Exception("Are you sure this is a title page? No text found on this crop.")

    groups = defs.Group.get_groups_from_df(data_df)
    groups.sort(key=lambda g: g.joint_box().left)
    return defs.Instrument(groups[0].text)


def split(path: str, title_keyword: str, scale: float = 4) -> None:
    """
    Splits parts out of a collective pdf and saves each part into a separate pdf with its instrument name.
    :param path: path to the collective pdf.
    :param title_keyword: keyword that is clearly visible on every title page of every part.
    :param scale: image upscale factor to improve ocr.
    """
    pdf = pdfium.PdfDocument(path)
    doc: pdfium.PdfDocument = None
    doc_title = None
    # use file name as prefix: path/to/file.pdf -> file
    prefix = path.split("/")[-1].split(".")[0]
    for i, page in enumerate(pdf):
        print(i)
        page_img = get_top25_as_img(page, scale)

        # if keyword was found this page should be a new title page.
        if contains_keyword(page_img, title_keyword):
            # if there was a document open we save it and subsequently create a new one.
            if doc is not None:
                with open(doc_title, "wb") as buffer:
                    doc.save(buffer)
                print("Saved", doc_title)

            # extract instrument from the title page and generate the filename.
            instr = rnd_chars(10)
            try:
                instr = find_instrument(page_img)
            except Exception:
                print(f"Failed to retrieve the instrument name - find it and rename it under {instr}")
            doc_title = f"splits/{prefix} - {instr}.pdf"
            while os.path.exists(doc_title):
                doc_title = doc_title.split(".")[0] + rnd_chars(1) + ".pdf"
            doc = pdfium.PdfDocument.new()

        # add page to document either way.
        add_page(doc, pdf, page, i)

    # save the last part too.
    with open(doc_title, "wb") as buffer:
        doc.save(buffer)


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
