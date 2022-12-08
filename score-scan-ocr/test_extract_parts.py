from utx import osutil
import pypdfium2 as pdfium


def test():
    paths = osutil.list_files_relative("splits", where=lambda f: f.endswith(".pdf"))
    for path in paths:
        pdf = pdfium.PdfDocument(path)
        assert len(pdf) == 2, "Incorrect page number"
        pdf.close()


if __name__ == '__main__':
    test()