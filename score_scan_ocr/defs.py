import os
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Dict, List, Set, Callable, Iterable, Union

import numpy as np
import pandas as pd

import utils

# import score_scan_ocr.prefs as prefs


class InstrumentNotFoundException(Exception):
    pass


class InstrPrefs:
    def __init__(self, file_path: str = "templates/custom_translation.csv"):
        self.file_path = "/".join(__file__.split("/")[:-1]) + "/" + file_path
        self._data = pd.read_csv(self.file_path)

        missing_instrs = set(Instruments.keys()).difference(self._data["Instrument"])
        self._data = self._data.set_index("Instrument")
        for instr in missing_instrs:
            self._data.loc[instr] = {"Nickname": "", "Folder": ""}

        # Order and get rid of invalid user extensions
        self._data = self._data.loc[list(Instruments.keys())]
        self._data.to_csv(self.file_path)

    def __contains__(self, item):
        return item in self._data.index

    def nickname(self, instrument) -> str:
        if instrument not in self:
            return instrument
        name = self._data.loc[instrument]["Nickname"]
        if pd.isna(name):
            name = instrument
        return name

    def dest_folder(self, instrument) -> str:
        return self._data.loc[instrument]["Folder"]

    def nickname_to_id(self, nickname: str) -> str:
        data = self._data.reset_index().set_index("Nickname")
        return data.loc[nickname]["Instrument"] if nickname in data.index else nickname


class Instrument:
    def __init__(self, string: str, should_raise: bool = True):
        string = utils.caps_to_capitalized_nouns(string)
        string, self.tune = self._extract_tune(string)
        string, self.part = self._extract_parts(string.lower())
        self.name = self._extract_name(string, should_raise=should_raise)

    @classmethod
    def from_filename(cls, name: str, should_raise: bool = True):
        instr = name.split(" - ")[-1]
        if instr.endswith(".pdf"):
            instr = instr[:-4]
        rest, tune = cls._extract_tune(instr)
        rest, part = cls._extract_parts(rest)
        instr = CustomTranslation.nickname_to_id(rest.strip())
        return cls(f"{instr} {tune if tune else ''} {part if part else ''}".strip(), should_raise=should_raise)

    @classmethod
    def parse(cls, string: str, should_raise: bool = True):
        return cls(string, should_raise=should_raise)

    @staticmethod
    def _extract_tune(name: str):
        tunes = list(Inverse(Tunes).keys())
        hits = [re.findall(fr"(.*\s|^){tune}(\s.*|$)", name) for tune in tunes]
        mask = [len(hit) for hit in hits]
        if sum(mask) == 1:
            idx = np.argmax(mask)
            id_tune = Inverse(Tunes)[tunes[idx]]
            add_remainder = lambda left, right: left.strip() + right.rstrip()
            return add_remainder(*hits[idx][0]), id_tune
        return name, None

    @staticmethod
    def _extract_parts(name: str):
        match = re.search(r"(\d(,|$))+", name)
        if match:
            start, end = match.span()
            remainder = name[:start] + name[start + end:]
            return remainder.strip(), name[start:end].strip()
        return name, None

    @staticmethod
    def _extract_name(name: str, should_raise: bool = True):
        matches = get_close_matches(name, list(Inverse(Instruments).keys()))
        if not matches:
            if should_raise:
                raise InstrumentNotFoundException()
            return name
        return Inverse(Instruments)[matches[0]]

    def metadata(self):
        return {
            "/Instrument": self.name,
            "/Tune": self.tune if self.tune else "",
            "/Part": self.part if self.part else ""
        }

    def __str__(self):
        return CustomTranslation.nickname(self.name) + (f" {self.tune}" if self.tune else "") + (f" {self.part}" if self.part else "")

    def __repr__(self):
        return str(self)


@dataclass
class Document:
    title: str
    subtitle: str
    instrument: Union[Instrument, str]
    arrangeur: str
    composer: str
    number: str

    def __post_init__(self):
        if isinstance(self.instrument, str):
            self.instrument = Instrument(self.instrument)

    @classmethod
    def empty(cls):
        return cls("", "", "", "", "", "")


def parse_solutions(solution_doc_path: str = "samples/solution.txt") -> List[Document]:
    docs = []
    with open(solution_doc_path, "r") as f:
        lines = f.read().split("\n")
        for i in range(0, len(lines), 6):
            docs.append(
                Document(*lines[i:i + 6])
            )
            if docs[-1].instrument:
                docs[-1].instrument = "saxophone"
            # print(docs[-1])
    return docs


class Box:
    def __init__(self, data_row: pd.Series):
        self.width = data_row.width
        self.height = data_row.height

        self.left = data_row.left
        self.right = data_row.left + data_row.width
        self.top = data_row.top
        self.bottom = data_row.top + data_row.height

        self.mid = data_row.top + int(data_row.height / 2)

    def overlaps(self, other):
        if other.right < self.left or self.right < other.left:
            return False
        if self.bottom < other.top or other.bottom < self.top:
            return False
        return True

    def rdist(self, other):
        return np.sqrt((self.mid - other.mid) ** 2 + (self.right - other.left) ** 2)

    @classmethod
    def from_mid_right(cls, box, height, width):
        return Box(pd.Series({
            "height": height,
            "width": width,
            "left": box.right,
            "top": box.mid - height / 2
        }))

    def __repr__(self):
        return f"Box({self.left}, {self.top}, {self.width}, {self.height})"

    @property
    def ltrb(self) -> np.ndarray:
        return np.array([self.left, self.top, self.right, self.bottom])


class PredBox(Box):
    def __init__(self, data_row: pd.Series):
        super().__init__(data_row)

        self.text = data_row.text

    def __str__(self):
        return self.text + super().__repr__()

    def __repr__(self):
        return str(self)


class Group:
    def __init__(self):
        self.boxes: List[PredBox] = []
        self.text: str = ""

    def __len__(self):
        return len(self.boxes)

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.boxes)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.text == other.text

    @staticmethod
    def get_groups_from_df(boxes_df: pd.DataFrame):
        boxes = [PredBox(row) for _, row in boxes_df.iterrows()]
        # boxes.sort(key=lambda b: b.left + int(boxes_df["height"].mean() * b.top/300)*300)
        groups = []
        while boxes:
            group = Group()
            boxes = group.group(boxes)
            groups.append(group)
        return groups

    def group(self, box_pool: List[PredBox]) -> List[PredBox]:
        self.boxes = [box_pool.pop(0)]
        self.text += self.boxes[-1].text
        i = 0
        while i < len(box_pool):
            box0 = self.boxes[-1]
            # create a square that continues box0 with (h*1.5, h*1.5) as hitbox to find the next
            hitbox = Box.from_mid_right(box0, self.median_h * 1.5, self.median_h * 1.5)
            dists, idxs = [], []
            for j, box in enumerate(box_pool):
                if hitbox.overlaps(box):
                    dists.append(box0.rdist(box))
                    idxs.append(j)

            # If no successor is found the group is done
            if len(dists) == 0:
                break

            min_j = np.argmin(dists)
            box = box_pool.pop(idxs[min_j])
            self.text += " " + box.text
            self.boxes.append(box)

        return box_pool

    @property
    def mean_h(self):
        return sum([box.height for box in self.boxes]) / len(self.boxes)

    @property
    def median_h(self):
        return np.median(np.array([box.height for box in self.boxes]))

    def joint_box(self) -> Box:
        ltrb = np.array([box.ltrb for box in self.boxes])
        l, t = np.min(ltrb, 0)[:2]
        r, b = np.max(ltrb, 0)[2:]
        return Box(pd.Series({
            "height": int(b - t),
            "width": int(r - l),
            "left": l,
            "top": t
        }))


Instruments: Dict[str, Set[str]] = {
    # Wood
    "Piccolo": {"Piccolo"},
    "Flute": {"Flute", "Flöte", "Fl"},
    "Oboe": {"Oboe"},
    "Englishhorn": {"Englishhorn"},
    "Clarinet": {"Clarinet", "Klarinette", "Klar"},
    "Bassoon": {"Bassoon", "Fagott", "Fg"},
    "Saxophone": {"Saxophone", "Sax", "Saxofon"},
    # Wind
    "Trumpet": {"Trumpet", "Trompete", "Trp"},
    "Cornet": {"Cornet", "Kornett"},
    "Bugle": {"Bugle", "Flügelhorn"},
    "Horn": {"Horn", "Cor", "Cors", "Corn", "Corno", "Waldhorn"},
    "Trombone": {"Trombone", "Posaune", "Pos"},
    "Euphonium": {"Euphonium", "Euph"},
    "Baritone": {"Baritone"},
    "Tenorhorn": {"Tenorhoorn", "Tenorhorn"},
    "Tuba": {"Tuba"},
    # Strings
    "Violin": {"Violin", "Geige"},
    "Viola": {"Viola", "Bratsche"},
    "Cello": {"Cello", "Violoncello"},
    "String Bass": {"String Bass", "Kontrabass"},
    # Rhythm
    "Timpani": {"Timpani", "Pauke"},
    "Percussion": {"Percussion", "Perc", "Perc."},
    "Tambourine": {"Tambourine", "Tamb"},
    "Bass Drum & Cymbals": {"Bass Drum & Cymbals"},
    "Drumset": {"Drumset", "Set", "Schlagzeug"},
    "Mallets": {"Mallets"},
    "Xylophone": {"Xylophone", "Xylo"},
    "Marimbaphone": {"Marimbaphone", "Marimba"},
    "Glockenspiel": {"Glockenspiel", "Bells"},
    "Harp": {"Harfe", "Harp"},
    "Piano": {"Piano", "Klavier", "Keys", "Keyboard"},
    # Choir
    "Soprano": {"Soprano", "Sopran"},
    "Choir": {"Choir", "Chor"}
}


def add_variations(key: str, variations: List[str], drop: bool = False):
    values = Instruments.pop(key) if drop else Instruments[key]
    for var in variations:
        Instruments[f"{var} {key}"] = {f"{var} {syn}" for syn in values}


add_variations("Flute", ["Alt", "Tenor", "Bass"])
add_variations("Clarinet", ["Solo", "Bass", "Contrabass", "Alt"])
add_variations("Saxophone", ["Bass", "Tenor", "Alto", "Baritone", "Soprano"], drop=True)
add_variations("Bassoon", ["Contrabass"])

add_variations("Trumpet", ["Solo", "Piccolo", "Bass"])
add_variations("Trombone", ["Bass"])

add_variations("Soprano", ["Solo"])


CustomTranslation = InstrPrefs()


Inverse: Callable[[Dict[str, Iterable[str]]], Dict[str, str]] = lambda dictionary: {
    value: key for key, values in dictionary.items() for value in values
}

InstrumentSynonyms = list(Inverse(Instruments).keys())

Tunes: Dict[str, Set[str]] = {
    "Es": {"Es", "Eb"},
    "B": {"B", "Bb"},
    "F": {"F"},
    "C": {"C"},
    "A": {"A"}
}

if __name__ == '__main__':
    names = list(map(lambda x: x.split(" - ")[-1].split(".")[0], os.listdir("splits/")))
    names.sort()

    instr = [Instrument(name) for name in names]
    parse_solutions()
