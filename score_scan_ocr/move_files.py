import os
import argparse

from defs import CustomTranslation, Instrument, InstrumentNotFoundException
from utils import add_metadata, read_metadata

from utx.colorx import Colors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "-p",
        "--root",
        required=True, type=str,
        help="Absolute path to root directory")

    parser.add_argument(
        "-r", "--raw",
        action="store_true",
        help="Preserves all file names and is not unifying/interpreting them."
    )

    parser.add_argument(
        "-d", "--dry",
        action="store_true",
        help="Only show changes without applying them."
    )

    args = parser.parse_args()

    # todo comment
    for file in [f for f in os.listdir("splits") if f.endswith(".pdf")]:
        path = os.path.join("splits", file)
        instrument = Instrument.from_filename(file, should_raise=False)
        add_metadata(path, instrument.metadata())
        instr = read_metadata(path).get("/Instrument")

        if instr is not None and instr in CustomTranslation:
            dest = os.path.join(
                args.root,
                CustomTranslation.dest_folder(instr),
                file[:file.rfind(" - ")],
                file if args.raw else f"{file[:file.rfind(' - ')]} - {instrument}.pdf")
            m = read_metadata(path)
            print(f"\n{m.get('/Instrument')} {m.get('/Tune')} {m.get('/Part')}", "->", dest.strip(args.root))
            if not args.dry:
                os.renames(path, dest)
        else:
            print(Colors.red.c(f"\nSkipped i: {instrument} | p: {path}"))

    print("\n" + "="*50)

    for file in [f for f in os.listdir("splits") if f.endswith(".pdf")]:
        path = os.path.join("splits", file)
        m = read_metadata(path)
        print(f"\n{m.get('/Instrument')} {m.get('/Tune')} {m.get('/Part')}", path)