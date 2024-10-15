import os
import random
import string

import pandas as pd
from PyPDF2 import PdfMerger, PdfReader
from pdfrw import PdfReader as PReader
import pypdfium2 as pdfium
import matplotlib.pyplot as plt


def caps_to_capitalized_nouns(string: str):
    """
    For example "TENOR SAXOPHONE" -> "Tenor Saxophone"
    :param string: string to modify.
    :return: modified string.
    """
    return " ".join([(word[0] + word[1:].lower()) if len(word) > 1 else word for word in string.split(" ")])


def filter_none(l: list):
    return [e for e in l if e is not None]


def add_metadata(file, metadata) -> None:
    with open(file, "rb") as f, open("temp.pdf", "wb") as out:
        merger = PdfMerger()
        merger.append(f)
        merger.add_metadata(metadata)
        merger.write(out)
    os.renames("temp.pdf", file)


def read_metadata(file) -> dict:
    with open(file, "rb") as f:
        return PdfReader(file).metadata


def count_all_pdf_pages(path: str):
    n = 0
    for file in [f for f in os.listdir(path) if f.endswith(".pdf")]:
        n += PReader(f"splits/{file}").numPages
    # print("A total of", n, "pages were found in splits/")
    return n


def plot_page_count(path: str):
    df = pd.DataFrame(columns=["name", "page_count"])
    for i, file in enumerate(filter(lambda n: n.endswith(".pdf"), os.listdir(path))):
        df.loc[i, :] = pd.Series({
            "name": file[:-4].split(" - ")[-1],
            "page_count": PReader(f"splits/{file}").numPages
        })
    (df
     .sort_values(by="page_count")
     .plot(x="name", y="page_count", kind="barh")
     )
    plt.title("Page Counts per File")
    plt.tight_layout()
    plt.show()


def rnd_chars(n: int) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


def get_pdf_pages_as_lowres_img(file_spec: str, scale: float = .4):
    pdf = pdfium.PdfDocument(file_spec)
    return [page.render_to(pdfium.BitmapConv.pil_image, scale=scale) for page in pdf]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # count_all_pdf_pages()
    plot_page_count("splits")
