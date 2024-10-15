import pandas as pd
from score_scan_ocr import defs


class InstrPrefs:
    def __init__(self, file_path: str = "templates/custom_translation.csv"):
        self.file_path = "/".join(__file__.split("/")[:-1]) + "/" + file_path
        self._data = pd.read_csv(self.file_path)

        missing_instrs = set(defs.Instruments.keys()).difference(self._data["Instrument"])
        self._data = self._data.set_index("Instrument")
        for instr in missing_instrs:
            self._data.loc[instr] = {"Nickname": "", "Folder": ""}

        # Order and get rid of invalid user extensions
        self._data = self._data.loc[list(defs.Instruments.keys())]
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


if __name__ == '__main__':
    a = InstrPrefs()
