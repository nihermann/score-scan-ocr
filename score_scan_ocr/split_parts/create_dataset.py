import os
from utx import osx
from pdf2image import convert_from_path
from tqdm import tqdm


def create_dataset(base_path: str, to_path: str) -> None:
    """
    Skims recursively through all pdfs in the base_path, expecting that the first page of a pdf is
    always a title page and the remaining ones no title pages.
    :param base_path: path to recursively skim through
    :param to_path: path to save the images to
    :return:
    """
    os.makedirs(osx.join_paths(to_path, "title"), exist_ok=True)
    os.makedirs(osx.join_paths(to_path, "no_title"), exist_ok=True)

    for path in tqdm(osx.list_files_relative_recursively(base_path, lambda p: p.endswith(".pdf"))):
        try:
            images = convert_from_path(path)
            title_page = images.pop(0)

            title_page.save(osx.join_paths(to_path, "title/" + osx.file_name(path) + ".png"))
            for i, non_title_page in enumerate(images):
                non_title_page.save(
                    osx.join_paths(to_path, "no_title/" + osx.file_name(path) + f"_{i}.png")
                )
        except Exception as e:
            print(f"Failed for {path} with {e}.")


if __name__ == '__main__':
    create_dataset("../../nas_archive", "data/")
