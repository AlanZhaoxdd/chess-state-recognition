
"""Names for the available stage implementations.

The command-line scripts pick implementations by name, e.g.
``python scripts/evaluate.py --occupancy tiny --pieces oracle``.

To add your implementation, write a function that builds it (loading any
trained model from :data:`MODELS_DIR`) and add it to the right dictionary::

    def _hog_svm_pieces():
        from chessrec.hog_svm import HogSvmClassifier
        return HogSvmClassifier.load(MODELS_DIR / "hog_svm_pieces.joblib")

    PIECE_CLASSIFIERS["hog_svm"] = _hog_svm_pieces

Import heavy dependencies (PyTorch, ...) inside the function, so that the
scripts still start for people who have not installed them.

``"oracle"`` is always available and maps to ``None``: the pipeline then
reads that stage's answer from the ground-truth label.
"""

from __future__ import annotations

from typing import Callable

from chessrec import config

#: Where trained models are stored (git-ignored).
MODELS_DIR = config.ROOT / "experiments" / "models"


def _tiny(task: str):
    from chessrec.baselines import TinyImageClassifier
    return TinyImageClassifier.load(MODELS_DIR / f"tiny_{task}.joblib")


def _hough_localiser():
    from chessrec.localisation import ChessboardLocaliser
    return ChessboardLocaliser()


LOCALISERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "hough": _hough_localiser,
}

OCCUPANCY_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("occupancy"),
}

PIECE_CLASSIFIERS: dict[str, Callable] = {
    "oracle": lambda: None,
    "tiny": lambda: _tiny("pieces"),
}
