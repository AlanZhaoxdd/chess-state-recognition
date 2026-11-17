"""Recognise the position in one photo and print it as FEN.

Stages left at "oracle" need --label. Use --black if the photo is taken
from Black's side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chessrec import registry
from chessrec.data import load_sample
from chessrec.pipeline import recognise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", type=Path, help="photo of a chessboard")
    parser.add_argument("--label", type=Path, default=None,
                        help="dataset label (.json), needed by oracle stages")
    parser.add_argument("--localiser", choices=registry.LOCALISERS, default="oracle")
    parser.add_argument("--occupancy", choices=registry.OCCUPANCY_CLASSIFIERS, default="oracle")
    parser.add_argument("--pieces", choices=registry.PIECE_CLASSIFIERS, default="oracle")
    parser.add_argument("--black", action="store_true",
                        help="the camera is on Black's side (ignored with --label)")
    args = parser.parse_args()

    uses_oracle = "oracle" in (args.localiser, args.occupancy, args.pieces)
    if uses_oracle and args.label is None:
        parser.error("oracle stages need --label; choose real implementations "
                     "for all three stages to run on an unlabelled photo")

    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not read image {args.image}")

    result = recognise(
        image,
        localiser=registry.LOCALISERS[args.localiser](),
        occupancy=registry.OCCUPANCY_CLASSIFIERS[args.occupancy](),
        pieces=registry.PIECE_CLASSIFIERS[args.pieces](),
        white_turn=not args.black,
        truth=load_sample(args.label) if args.label else None,
    )
    if result.corners is None:
        raise SystemExit("No chessboard found in the photo.")
    print(result.fen)
    print("\n".join(" ".join(row) for row in result.grid))


if __name__ == "__main__":
    main()
