"""Evaluation metrics, shared by everyone so numbers are comparable.

Typical use::

    evaluator = Evaluator()
    for sample in dataset:
        result = recognise(sample.load_image(), truth=sample, ...)
        evaluator.add(sample, result)
    print(format_report(evaluator.summary()))

What is measured
----------------
* **Localisation** -- mean corner error in pixels and the share of boards
  whose four corners are all within a tolerance of the true ones.
* **Occupancy** -- accuracy, precision and recall over all squares
  ("occupied" is the positive class).
* **Pieces** -- on squares that are occupied *and* were judged occupied:
  overall accuracy, per-class recall and the 12x12 confusion matrix. Report
  per-class numbers: pawns dominate, so overall accuracy alone flatters.
* **Whole board** -- wrong squares per board, and the share of boards with
  no mistake / at most one mistake. This is the metric that matters to a
  user of the system.
"""

from __future__ import annotations

import numpy as np

from chessrec import geometry
from chessrec.data import BoardSample
from chessrec.fen import EMPTY, PIECE_NAMES, SYMBOL_TO_ID
from chessrec.pipeline import Recognition

#: A board counts as localised if every corner is within this fraction of
#: the image width of its true position (12 px for a 1200 px wide photo,
#: roughly a quarter of a square).
CORNER_TOLERANCE = 0.01


def corner_errors(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Distance in pixels between matching corners, shape (4,).

    Both sets are put in the same order first, so the input order is free.
    """
    difference = geometry.sort_corners(predicted) - geometry.sort_corners(true)
    return np.linalg.norm(difference, axis=1)


def _ratio(numerator: float, denominator: float) -> float:
    """Division that returns NaN instead of failing when nothing was counted."""
    return float(numerator) / denominator if denominator else float("nan")


class Evaluator:
    """Accumulates results board by board and summarises them."""

    def __init__(self, corner_tolerance: float = CORNER_TOLERANCE):
        self.corner_tolerance = corner_tolerance
        self.corner_error: list[float] = []     # mean error of each found board
        self.localised: list[bool] = []         # one entry per board
        self.board_errors: list[int] = []       # wrong squares per board
        self.occupancy = np.zeros((2, 2), dtype=int)     # [true, predicted]
        self.pieces = np.zeros((12, 12), dtype=int)      # [true, predicted]

    def add(self, truth: BoardSample, result: Recognition, image_width: int | None = None) -> None:
        """Record the result for one board.

        Args:
            truth: the ground-truth label.
            result: what the pipeline predicted.
            image_width: width of the photo in pixels, for the localisation
                tolerance. Read from the image file if not given.
        """
        # Localisation
        if result.corners is None:
            self.localised.append(False)
        else:
            if image_width is None:
                image_width = truth.load_image().shape[1]
            errors = corner_errors(result.corners, truth.corners)
            self.corner_error.append(float(errors.mean()))
            self.localised.append(bool(errors.max() <= self.corner_tolerance * image_width))

        # Squares
        true_grid, predicted_grid = truth.grid, np.asarray(result.grid)
        self.board_errors.append(int((true_grid != predicted_grid).sum()))

        true_occupied = true_grid != EMPTY
        predicted_occupied = predicted_grid != EMPTY
        for t in (0, 1):
            for p in (0, 1):
                self.occupancy[t, p] += int(((true_occupied == t) & (predicted_occupied == p)).sum())

        for t, p in zip(true_grid[true_occupied & predicted_occupied],
                        predicted_grid[true_occupied & predicted_occupied]):
            self.pieces[SYMBOL_TO_ID[t], SYMBOL_TO_ID[p]] += 1

    def summary(self) -> dict:
        """All metrics as a JSON-serialisable dictionary."""
        n = len(self.board_errors)
        errors = np.array(self.board_errors)
        (tn, fp), (fn, tp) = self.occupancy
        per_class = {
            name: _ratio(self.pieces[i, i], self.pieces[i].sum())
            for i, name in enumerate(PIECE_NAMES)
        }
        return {
            "boards": n,
            "localisation": {
                "success_rate": _ratio(sum(self.localised), n),
                "mean_corner_error_px": _ratio(sum(self.corner_error), len(self.corner_error)),
                "tolerance_fraction_of_width": self.corner_tolerance,
            },
            "occupancy": {
                "accuracy": _ratio(tp + tn, self.occupancy.sum()),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "confusion_true_rows": self.occupancy.tolist(),
            },
            "pieces": {
                "accuracy": _ratio(np.trace(self.pieces), self.pieces.sum()),
                "mean_class_recall": float(np.nanmean(list(per_class.values())))
                if self.pieces.sum() else float("nan"),
                "per_class_recall": per_class,
                "confusion_true_rows": self.pieces.tolist(),
            },
            "board": {
                "square_accuracy": _ratio(64 * n - errors.sum(), 64 * n),
                "mean_errors_per_board": _ratio(errors.sum(), n),
                "boards_no_error": _ratio((errors == 0).sum(), n),
                "boards_at_most_one_error": _ratio((errors <= 1).sum(), n),
            },
        }


def format_report(summary: dict) -> str:
    """Readable text version of :meth:`Evaluator.summary`."""
    loc, occ, pcs, board = (summary[k] for k in ("localisation", "occupancy", "pieces", "board"))
    lines = [
        f"Boards evaluated            {summary['boards']}",
        "",
        "Localisation",
        f"  success rate              {loc['success_rate']:.1%}",
        f"  mean corner error         {loc['mean_corner_error_px']:.2f} px",
        "Occupancy (all squares)",
        f"  accuracy                  {occ['accuracy']:.2%}",
        f"  precision / recall        {occ['precision']:.2%} / {occ['recall']:.2%}",
        "Pieces (squares correctly judged occupied)",
        f"  accuracy                  {pcs['accuracy']:.2%}",
        f"  mean per-class recall     {pcs['mean_class_recall']:.2%}",
    ]
    lines += [f"    {name:<14}          {recall:.1%}"
              for name, recall in pcs["per_class_recall"].items()]
    lines += [
        "Whole board",
        f"  square accuracy           {board['square_accuracy']:.2%}",
        f"  mean errors per board     {board['mean_errors_per_board']:.2f}",
        f"  boards with no error      {board['boards_no_error']:.1%}",
        f"  boards with <= 1 error    {board['boards_at_most_one_error']:.1%}",
    ]
    return "\n".join(lines)
