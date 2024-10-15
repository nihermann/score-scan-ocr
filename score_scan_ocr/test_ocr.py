from difflib import SequenceMatcher
from typing import Dict

from utx import osutil, clsx
import pandas as pd

import defs
import ocr


def check_input(file: str, document: defs.Document) -> Dict[str, float]:
    pred_doc = ocr.parse_file(file)
    print("="*50, "\nTesting file:", file)
    scores = {}
    for attr, (pred, actual) in clsx.get_all_attrs([pred_doc, document]).items():
        scores[attr] = SequenceMatcher(None, pred, actual.lower()).ratio()
        print(f"""
    {attr}:
        pred   - {pred}
        actual - {actual}
    {scores[attr]*100: .3f}%
        """)
    perc_correct = sum(scores.values())/len(scores.values()) * 100
    print(f"Reached mean score of {perc_correct: .3f}%")
    print("="*50, "\n")
    return scores


def ttest_samples():
    paths = osutil.list_files_relative("samples", where=lambda f: f != "solution.txt")
    paths.sort()
    docs = defs.parse_solutions("samples/solution.txt")

    stats = []
    for path, doc in zip(paths, docs):
        stats.append(check_input(path, doc))
    stats = pd.DataFrame(stats) * 100
    print("\nMEAN TOTAL SCORE PER CATEGORY")
    print("------------------------------")
    print(str.join(" %\n", str(stats.mean().round(3)).split("\n")[:-1]) + " %")

    print(f"\nTOTAL SCORES ABOVE 70%: \t\t {(stats.mean(1) > 70).sum()}")
    print(f"REACHED TOTAL MEAN SCORE OF: \t{stats.mean().mean(): .3f}%")


if __name__ == '__main__':
    ttest_samples()