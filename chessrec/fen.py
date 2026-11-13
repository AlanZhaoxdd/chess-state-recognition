"""Class names and conversion between FEN and an 8x8 grid.

Grid convention used across the whole project
---------------------------------------------
A board is an 8x8 NumPy array of single-character strings:

* ``grid[0, 0]`` is square **a8**, ``grid[0, 7]`` is **h8**,
  ``grid[7, 0]`` is **a1**: the board as printed in a chess book with
  White at the bottom.
* Each cell holds a FEN piece symbol (``"P"`` white pawn ... ``"k"`` black
  king) or :data:`EMPTY` for an empty square.

Only the piece-placement field of a FEN string is used; side to move,
castling rights and so on cannot be read from a photo.
"""

from __future__ import annotations

import numpy as np

#: Symbol used for an empty square in a grid.
EMPTY = "."

#: FEN symbols of the 12 piece classes. The index in this tuple is the class
#: id used by every piece classifier.
PIECE_SYMBOLS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")

#: Readable class names, in the same order as :data:`PIECE_SYMBOLS`. Used as
#: folder names because "P" and "p" would collide on Windows and macOS.
PIECE_NAMES = (
    "white_pawn", "white_knight", "white_bishop",
    "white_rook", "white_queen", "white_king",
    "black_pawn", "black_knight", "black_bishop",
    "black_rook", "black_queen", "black_king",
)

#: Occupancy class names; the index is the class id.
OCCUPANCY_NAMES = ("empty", "occupied")

SYMBOL_TO_ID = {symbol: i for i, symbol in enumerate(PIECE_SYMBOLS)}
SYMBOL_TO_NAME = dict(zip(PIECE_SYMBOLS, PIECE_NAMES))
NAME_TO_SYMBOL = dict(zip(PIECE_NAMES, PIECE_SYMBOLS))

FILES = "abcdefgh"


def fen_to_grid(fen: str) -> np.ndarray:
    """Convert a FEN string (or just its placement field) to an 8x8 grid.

    Raises:
        ValueError: if the placement field does not describe 8 ranks of 8
            squares or contains an unknown symbol.
    """
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(ranks)}: {fen!r}")

    grid = np.full((8, 8), EMPTY, dtype="<U1")
    for row, rank in enumerate(ranks):
        col = 0
        for char in rank:
            if char.isdigit():
                col += int(char)  # a digit is a run of empty squares
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
    """Convert an 8x8 grid to the piece-placement field of a FEN string."""
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
    """Algebraic name (``"e4"``) of the grid cell ``grid[row, col]``."""
    return f"{FILES[col]}{8 - row}"
