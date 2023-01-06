import os
import random
import string

from PyPDF2 import PdfMerger, PdfReader
from pdfrw import PdfReader as PReader


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


def count_pages():
    n = 0
    for file in [f for f in os.listdir("splits") if f.endswith(".pdf")]:
        n += PReader(f"splits/{file}").numPages
    print("A total of", n, "pages were found in splits/")


def rnd_chars(n: int) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


if __name__ == '__main__':
    count_pages()