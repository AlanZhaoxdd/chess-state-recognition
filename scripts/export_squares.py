"""Cut every rendered board into per-square crops for the classifiers.

Uses the *ground-truth* corners from the labels, so the classifiers can be
developed and evaluated independently of board localisation.

Output (one PNG per square, class = folder name, so it works directly with
``torchvision.datasets.ImageFolder`` and is easy to walk for HOG/LBP)::

    data/squares/
        occupancy/<split>/<empty|occupied>/<board id>_<square>.png   100x100
        pieces/<split>/<white_pawn|...|black_king>/<board id>_<square>.png   200x100 (HxW)

Occupancy has a crop for all 64 squares; pieces only for occupied squares.

Usage::

    python scripts/export_squares.py                      # every split that is downloaded
    python scripts/export_squares.py --splits val --limit 20
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
    """Write a PNG; works with non-ASCII paths on Windows, unlike cv2.imwrite."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    buffer.tofile(path)


def export_board(task: tuple[BoardSample, Path, Path]) -> tuple[int, int]:
    """Export all crops of one board. Returns (#occupancy, #piece) crops."""
    sample, occupancy_dir, pieces_dir = task
    warped = geometry.warp_board(sample.load_image(), sample.corners)
    grid = sample.grid
    n_pieces = 0

    for row in range(8):          # image cells, see chessrec.geometry
        for col in range(8):
            # Which chess square does this image cell show?
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
    # Start from empty folders so stale crops from an earlier run cannot mix in.
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
