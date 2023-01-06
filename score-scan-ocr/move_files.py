import os
import argparse

from defs import CustomTranslation, Instrument
from utils import add_metadata, read_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "-p",
        "--root",
        required=True, type=str,
        help="Absolute path to root directory")

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print where files are moved."
    )

    args = parser.parse_args()

    for file in [f for f in os.listdir("splits") if f.endswith(".pdf")]:
        path = os.path.join("splits", file)
        new_metadata = Instrument.from_filename(file).metadata()
        add_metadata(path, new_metadata)
        instr = read_metadata(path).get("/Instrument")

        if instr is not None and instr in CustomTranslation:
            dest = os.path.join(
                args.root,
                CustomTranslation.dest_folder(instr),
                file[:file.rfind(" - ")],
                file)
            m = read_metadata(path)
            print(f"\n{m.get('/Instrument')} {m.get('/Tune')} {m.get('/Part')}")
            print(file, "->", dest.strip(args.root))
            if input("Press x to abort: ") == "x":
                continue
            os.renames(path, dest)
            if args.verbose:
                print(file, "->", dest.strip(args.root))

    for file in [f for f in os.listdir("splits") if f.endswith(".pdf")]:
        path = os.path.join("splits", file)
        m = read_metadata(path)
        print(f"\n{m.get('/Instrument')} {m.get('/Tune')} {m.get('/Part')}", path)