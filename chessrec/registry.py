"""Maps names (used by the scripts' command-line options) to stage implementations.

To add one, write a function that loads it and add it to the right dict.
Import heavy libraries like PyTorch inside that function. "oracle" maps to
None, i.e. the answer is read from the label.
"""

from __future__ import annotations

from typing import Callable

from chessrec import config

MODELS_DIR = config.ROOT / "experiments" / "models"  # git-ignored


def _tiny(task: str):
    from chessrec.baselines import TinyImageClassifier
    return TinyImageClassifier.load(MODELS_DIR / f"tiny_{task}.joblib")


def _hog_svm(task: str):
    from chessrec.classical import SvmClassifier
    return SvmClassifier.load(MODELS_DIR / f"hog_svm_{task}.joblib")


LOCALISERS: dict[str, Callable] = {
    "oracle": lambda: None,
}

OCCUPANCY_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("occupancy"),
    "hog_svm": lambda: _hog_svm("occupancy"),
}

PIECE_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("pieces"),
    "hog_svm": lambda: _hog_svm("pieces"),
}
