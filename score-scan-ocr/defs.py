from typing import Dict, List
import pandas as pd
import numpy as np


class Document:
    title: str
    subtitle: str
    instrument: str
    arrangeur: str
    composer: str
    number: str

    def __init__(self, title: str, subtitle: str, insturment: str, arrangeur: str, composer: str, number: str):
        self.title = title
        self.subtitle = subtitle
        self.instrument = insturment
        self.arrangeur = arrangeur
        self.composer = composer
        self.number = number

    @classmethod
    def empty(cls):
        return cls("", "", "", "", "", "")

    def __repr__(self):
        return f"""
        {self.title} {self.number}(
            {self.subtitle}
            {self.instrument}
            Arr: {self.arrangeur}
            {self.composer}
        )
        """


def parse_solutions(solution_doc_path: str = "samples/solution.txt") -> List[Document]:
    docs = []
    with open(solution_doc_path, "r") as f:
        lines = f.read().split("\n")
        for i in range(0, len(lines), 6):
            docs.append(
                Document(*lines[i:i+6])
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

        self.mid = data_row.top + int(data_row.height/2)

    def overlaps(self, other):
        if other.right < self.left or self.right < other.left:
            return False
        if self.bottom < other.top or other.bottom < self.top:
            return False
        return True

    def rdist(self, other):
        return np.sqrt((self.mid - other.mid)**2 + (self.right - other.left)**2)

    @classmethod
    def from_mid_right(cls, box, height, width):
        return Box(pd.Series({
            "height": height,
            "width": width,
            "left": box.right,
            "top": box.mid - height/2
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
            hitbox = Box.from_mid_right(box0, self.median_h*1.5, self.median_h *1.5)
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


Instruments: Dict[str, set] = {
    "en": {
        # Wood
        "Piccolo",
        "Flute",
        "Oboe",
        "Clarinet",
        "Bassoon",
        "Saxophone",
        # Wind
        "Trumpet",
        "Horn",
        "Trombone",
        "Euphonium",
        "Tuba",
        # Strings
        "Viola",
        "Cello",
        "String Bass",
        # Rhythm
        "Timpani",
        "Percussion",
        "Drumset",
        "Mallets"
    },
    "de": {

    }
}


if __name__ == '__main__':
    parse_solutions()