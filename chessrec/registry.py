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

def _cnn(task: str, variant: str = ""):
    """CNN trained by scripts/train_cnn.py; variant matches its --variant option."""
    from chessrec.cnn import CNNClassifier  # imports PyTorch
    suffix = f"_{variant}" if variant else ""
    return CNNClassifier.load(MODELS_DIR / f"cnn_{task}{suffix}.pt")


LOCALISERS: dict[str, Callable] = {
    "oracle": lambda: None,
}

OCCUPANCY_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("occupancy"),
    "cnn": lambda: _cnn("occupancy"),
    "cnn_scratch": lambda: _cnn("occupancy", "scratch"),
    "cnn_noaug": lambda: _cnn("occupancy", "noaug"),
}

PIECE_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("pieces"),
    "cnn": lambda: _cnn("pieces"),
    "cnn_scratch": lambda: _cnn("pieces", "scratch"),
    "cnn_noaug": lambda: _cnn("pieces", "noaug"),
}
