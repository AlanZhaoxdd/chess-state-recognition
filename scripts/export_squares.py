"""Cut every rendered board into per-square crops for the classifiers.

Uses the ground-truth corners, so the classifiers don't depend on board
localisation. Writes data/squares/{occupancy,pieces}/<split>/<class>/*.png
(one folder per class, so ImageFolder can read it).
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
from chessrec.fen import EMPTY, OCCUPANCY_NAMES, PIECE_NAMES, SYMBOL_TO_NAME, square_name


def save_png(path: Path, image: np.ndarray) -> None:
    # cv2.imwrite can't handle non-ASCII paths on Windows
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    buffer.tofile(path)


def export_board(task: tuple[BoardSample, Path, Path]) -> tuple[int, int]:
    """Save all crops of one board. Returns (number of occupancy crops, number of piece crops)."""
    sample, occupancy_dir, pieces_dir = task
    warped = geometry.warp_board(sample.load_image(), sample.corners)
    grid = sample.grid
    n_pieces = 0

    for row in range(8):
        for col in range(8):
            # chess square shown in this image cell
            g_row, g_col = (row, col) if sample.white_turn else (7 - row, 7 - col)
            symbol = grid[g_row, g_col]
            name = f"{sample.sample_id}_{square_name(g_row, g_col)}.png"

            occupied = symbol != EMPTY
            save_png(occupancy_dir / OCCUPANCY_NAMES[occupied] / name,
                     geometry.crop_occupancy(warped, row, col))
            if occupied:
                save_png(pieces_dir / SYMBOL_TO_NAME[symbol] / name,
                         geometry.crop_piece(warped, row, col))
                n_pieces += 1
    return 64, n_pieces


def export_split(split: str, limit: int | None, workers: int) -> None:
    dataset = RenderDataset(split)
    samples = list(dataset)[:limit]

    occupancy_dir = config.SQUARES_DIR / "occupancy" / split
    pieces_dir = config.SQUARES_DIR / "pieces" / split
    # start from empty folders so old crops don't mix in
    for folder, classes in ((occupancy_dir, OCCUPANCY_NAMES), (pieces_dir, PIECE_NAMES)):
        shutil.rmtree(folder, ignore_errors=True)
        for name in classes:
            (folder / name).mkdir(parents=True)

    tasks = [(sample, occupancy_dir, pieces_dir) for sample in samples]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            counts = list(tqdm(pool.map(export_board, tasks, chunksize=8),
                               total=len(tasks), desc=f"export {split}", unit="board"))
    else:
        counts = [export_board(t) for t in tqdm(tasks, desc=f"export {split}", unit="board")]

    n_occupancy = sum(c[0] for c in counts)
    n_pieces = sum(c[1] for c in counts)
    print(f"{split}: {len(samples)} boards -> {n_occupancy} occupancy crops, "
          f"{n_pieces} piece crops")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", choices=config.SPLITS, default=None,
                        help="splits to export (default: all that are downloaded)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only export the first N boards of each split")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel worker processes (default: 4)")
    args = parser.parse_args()

    splits = args.splits or [s for s in config.SPLITS if (config.RENDER_DIR / s).is_dir()]
    if not splits:
        raise SystemExit("No data found. Run `python scripts/download_data.py` first.")
    for split in splits:
        export_split(split, args.limit, args.workers)


if __name__ == "__main__":
    main()
