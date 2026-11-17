"""Paths and constants shared by all stages. Don't hard-code these elsewhere."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# set CHESSREC_DATA to keep the data on another drive
DATA_DIR = Path(os.environ.get("CHESSREC_DATA", ROOT / "data"))
RENDER_DIR = DATA_DIR / "render"     # <split>/<id>.png + <id>.json
SQUARES_DIR = DATA_DIR / "squares"   # written by scripts/export_squares.py
SPLITS = ("train", "val", "test")

# geometry of the warped top-down board, in pixels
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
WARP_MARGIN = 4 * SQUARE_SIZE        # border so tall pieces on far ranks are not cut off
WARP_SIZE = BOARD_SIZE + 2 * WARP_MARGIN

# crop sizes (height, width); piece crops are taller because pieces stand above their square
OCCUPANCY_CROP_SIZE = (2 * SQUARE_SIZE, 2 * SQUARE_SIZE)
PIECE_CROP_SIZE = (4 * SQUARE_SIZE, 2 * SQUARE_SIZE)

SEED = 3204


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
