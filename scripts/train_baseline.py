"""Fit the tiny-image k-NN baseline for occupancy and piece classification.

Reads the crops written by ``scripts/export_squares.py`` and stores the two
classifiers in ``experiments/models/tiny_{occupancy,pieces}.joblib``.

Usage::

    python scripts/train_baseline.py                     # first 300 training boards
    python scripts/train_baseline.py --split val --max-boards 100
"""

from __future__ import annotations

import argparse

from chessrec import config
from chessrec.baselines import TinyImageClassifier
from chessrec.data import load_square_crops
from chessrec.registry import MODELS_DIR

#: Tiny-image size (width, height) per task; piece crops are twice as tall.
SIZES = {"occupancy": (16, 16), "pieces": (16, 32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=config.SPLITS, default="train",
                        help="split to take the training crops from (default: train)")
    parser.add_argument("--max-boards", type=int, default=300,
                        help="use only the first N boards; k-NN keeps every "
                             "example in memory (default: 300)")
    parser.add_argument("-k", type=int, default=5, help="neighbours that vote (default: 5)")
    args = parser.parse_args()

    config.set_seed()
    for task, size in SIZES.items():
        crops, labels, boards = load_square_crops(task, args.split, args.max_boards)
        classifier = TinyImageClassifier(size=size, k=args.k).fit(crops, labels)
        path = MODELS_DIR / f"tiny_{task}.joblib"
        classifier.save(path)
        print(f"{task}: fitted on {len(crops)} crops from {len(set(boards))} "
              f"{args.split} boards -> {path}")


if __name__ == "__main__":
    main()
