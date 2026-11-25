"""How much do the classifiers suffer when the board corners are slightly wrong?

The true corners are moved by random Gaussian noise (standard deviation sigma
pixels in x and in y, independently for each corner) and the full pipeline is
run with these corners. This mimics a localiser that finds roughly the right
board but misses the exact corners, which is what package A's errors look like.
The noise is seeded per board, so every classifier sees exactly the same
corners and the numbers are comparable.

Results go to experiments/runs/robustness_<split>_<name>.{json,md}; copy them to
experiments/results/ if the report uses them.

Usage::

    python scripts/classical_robustness.py
    python scripts/classical_robustness.py --sigmas 0 4 8 12 --name hog_svm
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tqdm import tqdm

from chessrec import config, registry
from chessrec.data import RenderDataset
from chessrec.evaluation import Evaluator
from chessrec.pipeline import recognise

RUNS_DIR = config.ROOT / "experiments" / "runs"


class FixedCorners:
    """Localiser that returns whatever corners it was last given."""

    def __init__(self) -> None:
        self.corners: np.ndarray | None = None

    def localise(self, image: np.ndarray) -> np.ndarray | None:
        """Ignore the image and return the stored corners."""
        return self.corners


def noisy_corners(corners: np.ndarray, sigma: float, board: int) -> np.ndarray:
    """True corners plus Gaussian noise; the same (board, sigma) gives the same noise."""
    rng = np.random.default_rng([config.SEED, board, int(round(sigma * 100))])
    return np.asarray(corners, dtype=np.float32) + rng.normal(0, sigma, (4, 2)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="val")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0, 4, 8, 12],
                        help="noise standard deviations in pixels (default: 0 4 8 12)")
    parser.add_argument("--occupancy", choices=registry.OCCUPANCY_CLASSIFIERS, default="hog_svm")
    parser.add_argument("--pieces", choices=registry.PIECE_CLASSIFIERS, default="hog_svm")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N boards")
    parser.add_argument("--name", default=None,
                        help="name of the saved result (default: <occupancy>_<pieces>)")
    args = parser.parse_args()

    config.set_seed()
    occupancy = registry.OCCUPANCY_CLASSIFIERS[args.occupancy]()
    pieces = registry.PIECE_CLASSIFIERS[args.pieces]()
    localiser = FixedCorners()
    samples = list(RenderDataset(args.split))[:args.limit]
    evaluators = {sigma: Evaluator() for sigma in args.sigmas}

    for board, sample in enumerate(tqdm(samples, desc="evaluate", unit="board")):
        image = sample.load_image()  # read once, reuse for every noise level
        for sigma in args.sigmas:
            localiser.corners = noisy_corners(sample.corners, sigma, board)
            result = recognise(image, localiser=localiser, occupancy=occupancy,
                               pieces=pieces, truth=sample)
            evaluators[sigma].add(sample, result, image_width=image.shape[1])

    rows = []
    for sigma, evaluator in evaluators.items():
        s = evaluator.summary()
        rows.append({"sigma_px": sigma,
                     "mean_corner_error_px": s["localisation"]["mean_corner_error_px"],
                     "localisation_success": s["localisation"]["success_rate"],
                     "occupancy_accuracy": s["occupancy"]["accuracy"],
                     "pieces_accuracy": s["pieces"]["accuracy"],
                     "pieces_mean_class_recall": s["pieces"]["mean_class_recall"],
                     "square_accuracy": s["board"]["square_accuracy"],
                     "mean_errors_per_board": s["board"]["mean_errors_per_board"],
                     "boards_no_error": s["board"]["boards_no_error"]})

    header = ("| Noise sigma (px) | Mean corner error (px) | Corners within 1% | Occupancy acc. "
              "| Piece acc. | Square acc. | Errors per board | Boards with no error |")
    lines = [header, "|" + "---|" * 8]
    for r in rows:
        lines.append(f"| {r['sigma_px']:g} | {r['mean_corner_error_px']:.1f} "
                     f"| {r['localisation_success']:.1%} | {r['occupancy_accuracy']:.2%} "
                     f"| {r['pieces_accuracy']:.2%} | {r['square_accuracy']:.2%} "
                     f"| {r['mean_errors_per_board']:.2f} | {r['boards_no_error']:.1%} |")
    print("\n" + "\n".join(lines))

    name = args.name or f"{args.occupancy}_{args.pieces}"
    stem = RUNS_DIR / f"robustness_{args.split}_{name}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    setup = {"split": args.split, "occupancy": args.occupancy, "pieces": args.pieces,
             "limit": args.limit, "noise": "Gaussian, per corner and axis, seeded per board"}
    stem.with_suffix(".json").write_text(json.dumps({"setup": setup, "results": rows}, indent=2),
                                         encoding="utf-8")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {stem}.json and .md")


if __name__ == "__main__":
    main()
