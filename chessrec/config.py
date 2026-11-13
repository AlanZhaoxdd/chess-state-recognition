"""Paths and constants shared by every stage of the pipeline.

Change a value here and every stage picks it up; do not hard-code paths or
crop sizes anywhere else.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Repository root (the folder that contains ``chessrec/``).
ROOT = Path(__file__).resolve().parent.parent

#: Where datasets live. Override with the ``CHESSREC_DATA`` environment
#: variable if you keep the data on another drive.
DATA_DIR = Path(os.environ.get("CHESSREC_DATA", ROOT / "data"))

#: Rendered boards: ``RENDER_DIR/<split>/<id>.png`` + ``<id>.json``.
RENDER_DIR = DATA_DIR / "render"

#: Exported per-square crops, see ``scripts/export_squares.py``.
SQUARES_DIR = DATA_DIR / "squares"

#: Dataset splits, exactly as published with the dataset.
SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------
# Geometry of the warped (top-down) board
# --------------------------------------------------------------------------

#: Side length of one square in the warped image, in pixels.
SQUARE_SIZE = 50

#: Side length of the 8x8 board in the warped image.
BOARD_SIZE = 8 * SQUARE_SIZE

#: Border kept around the board in the warped image, so that tall pieces on
#: the far ranks and the board edge are not cut off.
WARP_MARGIN = 4 * SQUARE_SIZE

#: Side length of the whole warped image.
WARP_SIZE = BOARD_SIZE + 2 * WARP_MARGIN

#: Size (height, width) of one occupancy crop: the square plus half a square
#: of context on every side.
OCCUPANCY_CROP_SIZE = (2 * SQUARE_SIZE, 2 * SQUARE_SIZE)

#: Size (height, width) of one piece crop: taller than wide, because pieces
#: stand above their square in a perspective view.
PIECE_CROP_SIZE = (4 * SQUARE_SIZE, 2 * SQUARE_SIZE)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

#: Default random seed. Use it everywhere so results can be reproduced.
SEED = 3204


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if it is installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
