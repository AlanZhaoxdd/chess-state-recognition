"""Training crops cut with deliberately noisy board corners (work package B).

Works like export_squares.py, except that each board's true corners are moved
by Gaussian noise (sigma pixels in x and y, per corner) before warping. Every
crop is labelled with what its shifted cell really shows, using the same rule
the pipeline uses when it is scored. Training on these crops as well as the
clean ones teaches the classifiers to cope with a localiser that misses the
corners by a few pixels.

Writes data/squares_jitter<sigma>/{occupancy,pieces}/<split>/<class>/*.png;
pass that folder to train_classical.py with --extra-squares-dir.

Usage::

    python scripts/export_jittered_squares.py --sigma 8
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from chessrec import config, geometry
from chessrec.data import BoardSample, RenderDataset
from chessrec.fen import EMPTY, OCCUPANCY_NAMES, PIECE_NAMES, SYMBOL_TO_NAME
from chessrec.pipeline import _truth_under_cells  # same labelling rule as evaluation


def save_png(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image)  # cv2.imwrite fails on non-ASCII paths
    if not ok:
        raise OSError(f"Could not encode {path}")
    buffer.tofile(path)


def jitter(corners: np.ndarray, sigma: float, board: int, copy: int) -> np.ndarray:
    """True corners plus seeded Gaussian noise, reproducible per (board, copy)."""
    rng = np.random.default_rng([config.SEED, board, copy, int(round(sigma * 100))])
    return np.asarray(corners, dtype=np.float32) + rng.normal(0, sigma, (4, 2)).astype(np.float32)


def export_board(task: tuple[int, BoardSample, float, int, Path, Path]) -> tuple[int, int]:
    """Save the jittered crops of one board; returns (occupancy crops, piece crops)."""
    board, sample, sigma, copies, occupancy_dir, pieces_dir = task
    image = sample.load_image()
    n_occupancy = n_pieces = 0
    for copy in range(copies):
        corners = jitter(sample.corners, sigma, board, copy)
        warped = geometry.warp_board(image, corners)
        truth = _truth_under_cells(sample, corners)  # [row, col] by image cell
        for row in range(8):
            for col in range(8):
                symbol = truth[row, col]
                name = f"{sample.sample_id}-j{copy}_r{row}c{col}.png"
                occupied = symbol != EMPTY
                save_png(occupancy_dir / OCCUPANCY_NAMES[occupied] / name,
                         geometry.crop_occupancy(warped, row, col))
                n_occupancy += 1
                if occupied:
                    save_png(pieces_dir / SYMBOL_TO_NAME[symbol] / name,
                             geometry.crop_piece(warped, row, col))
                    n_pieces += 1
    return n_occupancy, n_pieces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="train")
    parser.add_argument("--sigma", type=float, default=8.0,
                        help="corner noise standard deviation in pixels (default: 8)")
    parser.add_argument("--copies", type=int, default=1,
                        help="jittered versions of each board (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N boards")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = config.DATA_DIR / f"squares_jitter{args.sigma:g}"
    occupancy_dir = root / "occupancy" / args.split
    pieces_dir = root / "pieces" / args.split
    for folder, classes in ((occupancy_dir, OCCUPANCY_NAMES), (pieces_dir, PIECE_NAMES)):
        shutil.rmtree(folder, ignore_errors=True)
        for name in classes:
            (folder / name).mkdir(parents=True)

    samples = list(RenderDataset(args.split))[:args.limit]
    tasks = [(i, s, args.sigma, args.copies, occupancy_dir, pieces_dir)
             for i, s in enumerate(samples)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            counts = list(tqdm(pool.map(export_board, tasks, chunksize=8), total=len(tasks),
                               desc=f"export {args.split} sigma={args.sigma:g}", unit="board"))
    else:
        counts = [export_board(t) for t in tqdm(tasks, desc="export", unit="board")]
    print(f"{args.split}: {len(samples)} boards x {args.copies} -> "
          f"{sum(c[0] for c in counts)} occupancy crops, {sum(c[1] for c in counts)} piece crops "
          f"in {root}")


if __name__ == "__main__":
    main()
