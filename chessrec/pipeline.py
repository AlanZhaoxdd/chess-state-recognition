"""The end-to-end pipeline: photo -> board position.

:func:`recognise` chains the three stages (see :mod:`chessrec.stages`).
Any stage can be replaced by an *oracle* that reads the answer from the
ground-truth label instead, by passing ``None`` for it. That lets us

* run the pipeline before every stage is implemented, and
* measure one stage at a time (e.g. real classifiers on perfectly
  localised boards, or a real localiser followed by perfect classifiers).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chessrec import config, geometry
from chessrec.data import BoardSample
from chessrec.fen import EMPTY, PIECE_SYMBOLS, grid_to_fen
from chessrec.stages import BoardLocaliser, OccupancyClassifier, PieceClassifier


@dataclass(frozen=True)
class Recognition:
    """Result of running the pipeline on one photo."""

    #: Detected corners, shape (4, 2), or ``None`` if no board was found.
    corners: np.ndarray | None
    #: Predicted position as an 8x8 grid (all empty if no board was found).
    grid: np.ndarray

    @property
    def fen(self) -> str:
        """Predicted position as the piece-placement field of a FEN string."""
        return grid_to_fen(self.grid)


def _truth_under_cells(truth: BoardSample, corners: np.ndarray) -> np.ndarray:
    """What each image cell *really* shows, given possibly inexact corners.

    This is what a perfect classifier would answer. The centre of every
    image cell is mapped back into the photo with the predicted corners and
    then onto the true board with the true corners. If the predicted corners
    are exact, this is simply the true position; if localisation is off by a
    square, the answer is shifted accordingly, and cells that fall outside
    the real board are empty.

    Returns:
        8x8 array of symbols indexed by image cell ``[row, col]``.
    """
    s, m = config.SQUARE_SIZE, config.WARP_MARGIN
    rows, cols = np.mgrid[0:8, 0:8]
    centres = np.stack([m + s * (cols + 0.5), m + s * (rows + 0.5)], axis=-1)
    centres = centres.reshape(-1, 1, 2).astype(np.float32)

    predicted_to_photo = np.linalg.inv(geometry.board_homography(corners))
    photo_to_true = geometry.board_homography(truth.corners)
    mapped = cv2.perspectiveTransform(centres, photo_to_true @ predicted_to_photo)
    true_cells = np.floor((mapped.reshape(-1, 2) - m) / s).astype(int)

    true_view = geometry.to_image_view(truth.grid, truth.white_turn)
    result = np.full(64, EMPTY, dtype="<U1")
    for i, (col, row) in enumerate(true_cells):
        if 0 <= row < 8 and 0 <= col < 8:
            result[i] = true_view[row, col]
    return result.reshape(8, 8)


def recognise(
    image: np.ndarray,
    *,
    localiser: BoardLocaliser | None = None,
    occupancy: OccupancyClassifier | None = None,
    pieces: PieceClassifier | None = None,
    white_turn: bool = True,
    truth: BoardSample | None = None,
) -> Recognition:
    """Recognise the position shown in a photo.

    Args:
        image: the photo (H x W x 3, BGR).
        localiser, occupancy, pieces: the stage implementations. ``None``
            selects the oracle for that stage, which requires ``truth``.
        white_turn: True if the camera is on White's side (see
            :mod:`chessrec.geometry`). Ignored when ``truth`` is given; the
            label's value is used instead.
        truth: the ground-truth label, needed only by oracle stages.

    Returns:
        The detected corners and the predicted position.
    """
    if truth is None and None in (localiser, occupancy, pieces):
        raise ValueError("Oracle stages (None) need the ground-truth label `truth`.")
    if truth is not None:
        white_turn = truth.white_turn

    # 1. Board localisation
    corners = truth.corners if localiser is None else localiser.localise(image)
    if corners is None:
        return Recognition(None, np.full((8, 8), EMPTY, dtype="<U1"))
    corners = np.asarray(corners, dtype=np.float32)

    # 2. Square extraction
    warped = geometry.warp_board(image, corners)
    occupancy_crops, piece_crops = geometry.crop_all(warped)  # [row, col]
    true_view = _truth_under_cells(truth, corners) if truth is not None else None

    # 3. Occupancy classification
    if occupancy is None:
        occupied = true_view != EMPTY
    else:
        flat = occupancy_crops.reshape(64, *occupancy_crops.shape[2:])
        occupied = np.asarray(occupancy.predict(flat)).astype(bool).reshape(8, 8)

    # 4. Piece classification, only on squares judged occupied
    view = np.full((8, 8), EMPTY, dtype="<U1")
    if occupied.any():
        if pieces is None:
            symbols = true_view[occupied]
            # A square wrongly judged occupied has no true piece. It is an
            # error whatever we answer; a pawn (the most common piece) keeps
            # the output a valid grid.
            symbols = np.where(symbols == EMPTY, "P", symbols)
        else:
            class_ids = np.asarray(pieces.predict(piece_crops[occupied]))
            symbols = np.array(PIECE_SYMBOLS)[class_ids]
        view[occupied] = symbols

    # 5. Back from image cells to the chess grid
    return Recognition(corners, geometry.to_image_view(view, white_turn).copy())
