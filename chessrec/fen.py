"""Class names and FEN <-> 8x8 grid conversion.

A grid is an 8x8 array of FEN symbols with "." for empty. grid[0, 0] is a8 and
grid[7, 0] is a1 (White at the bottom). Only the piece-placement part of FEN
is used, the rest can't be read from a photo.
"""

from __future__ import annotations

import numpy as np

EMPTY = "."

# the index in this tuple is the class id used by the piece classifiers
PIECE_SYMBOLS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")

# same order as PIECE_SYMBOLS; used as folder names because "P" and "p" clash on Windows/macOS
PIECE_NAMES = (
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king",
)

OCCUPANCY_NAMES = ("empty", "occupied")

SYMBOL_TO_ID = {symbol: i for i, symbol in enumerate(PIECE_SYMBOLS)}
SYMBOL_TO_NAME = dict(zip(PIECE_SYMBOLS, PIECE_NAMES))
NAME_TO_SYMBOL = dict(zip(PIECE_NAMES, PIECE_SYMBOLS))

FILES = "abcdefgh"


def fen_to_grid(fen: str) -> np.ndarray:
    """FEN string (or just its placement field) -> 8x8 grid. Raises ValueError if invalid."""
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen!r}")

    grid = np.full((8, 8), EMPTY, dtype="<U1")
    for row, rank in enumerate(ranks):
        col = 0
        for char in rank:
            if char.isdigit():
                col += int(char)  # run of empty squares
            elif char in SYMBOL_TO_ID:
                if col >= 8:
                    raise ValueError(f"Rank {8 - row} is too long: {fen!r}")
                grid[row, col] = char
                col += 1
            else:
                raise ValueError(f"Unknown FEN symbol {char!r} in {fen!r}")
        if col != 8:
            raise ValueError(f"Rank {8 - row} has {col} squares: {fen!r}")
    return grid


def grid_to_fen(grid: np.ndarray) -> str:
    """8x8 grid -> piece-placement field of a FEN string."""
    grid = np.asarray(grid)
    if grid.shape != (8, 8):
        raise ValueError(f"Grid must be 8x8, got {grid.shape}")

    ranks = []
    for row in grid:
        rank, empty_run = "", 0
        for cell in row:
            if cell == EMPTY:
                empty_run += 1
                continue
            if cell not in SYMBOL_TO_ID:
                raise ValueError(f"Unknown piece symbol {cell!r}")
            if empty_run:
                rank += str(empty_run)
                empty_run = 0
            rank += str(cell)
        if empty_run:
            rank += str(empty_run)
        ranks.append(rank)
    return "/".join(ranks)


def square_name(row: int, col: int) -> str:
    """Name of grid[row, col], e.g. "e4"."""
    return f"{FILES[col]}{8 - row}"
