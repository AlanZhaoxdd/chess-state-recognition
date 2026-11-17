"""Interfaces for the three pipeline stages.

You don't need to inherit from these, any class with the right method works.
All images are uint8 arrays in BGR order (OpenCV).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BoardLocaliser(Protocol):
    def localise(self, image: np.ndarray) -> np.ndarray | None:
        """Return the 4 corners of the playing area as a (4, 2) array of (x, y)
        in any order, or None if no board is found."""
        ...


class OccupancyClassifier(Protocol):
    def predict(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, 100, 100, 3) from geometry.crop_occupancy.
        Returns (N,) with 0 = empty, 1 = occupied."""
        ...


class PieceClassifier(Protocol):
    def predict(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, 200, 100, 3) from geometry.crop_piece.
        Returns (N,) class ids 0-11, the index in fen.PIECE_SYMBOLS."""
        ...
