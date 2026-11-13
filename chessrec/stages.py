"""The three interfaces a pipeline stage has to implement.

Any object with the right method works; you do not need to inherit from
these classes. They exist to document the contract and for type checkers.

All images are OpenCV-style ``uint8`` arrays in **BGR** channel order.
To plug a new implementation into the pipeline, register it in
:mod:`chessrec.registry`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BoardLocaliser(Protocol):
    """Finds the chessboard in a photo."""

    def localise(self, image: np.ndarray) -> np.ndarray | None:
        """Locate the four outer corners of the 8x8 playing area.

        Args:
            image: the photo, shape (H, W, 3).

        Returns:
            Array of shape (4, 2) with (x, y) pixel coordinates, in any
            order, or ``None`` if no board was found.
        """
        ...


class OccupancyClassifier(Protocol):
    """Decides whether squares are empty or occupied."""

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """Classify a batch of occupancy crops.

        Args:
            crops: shape (N, 100, 100, 3), produced by
                :func:`chessrec.geometry.crop_occupancy`.

        Returns:
            Array of shape (N,): 0 = empty, 1 = occupied
            (the index in :data:`chessrec.fen.OCCUPANCY_NAMES`).
        """
        ...


class PieceClassifier(Protocol):
    """Identifies the piece on occupied squares."""

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """Classify a batch of piece crops.

        Args:
            crops: shape (N, 200, 100, 3), produced by
                :func:`chessrec.geometry.crop_piece`.

        Returns:
            Array of shape (N,) with class ids 0..11
            (the index in :data:`chessrec.fen.PIECE_SYMBOLS`).
        """
        ...
