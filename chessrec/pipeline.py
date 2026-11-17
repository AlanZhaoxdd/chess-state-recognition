"""Full pipeline: photo -> board position.

Passing None for a stage uses an oracle that reads the answer from the label,
so each stage can be tested on its own before the others are finished.
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
    """Pipeline output: corners (None if no board found) and the 8x8 grid."""

    corners: np.ndarray | None
    grid: np.ndarray

    @property
    def fen(self) -> str:
        return grid_to_fen(self.grid)


def _truth_under_cells(truth: BoardSample, corners: np.ndarray) -> np.ndarray:
    """What each image cell really shows when the corners may be slightly off.

    Maps each cell centre back to the photo with the predicted corners, then
    onto the true board. Cells that land outside the real board are empty.
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
    """Recognise the position in a BGR photo.

    A stage set to None uses the oracle, which needs `truth`. If `truth` is
    given, its white_turn is used instead of the argument.
    """
    if truth is None and None in (localiser, occupancy, pieces):
        raise ValueError("Oracle stages (None) need the ground-truth label `truth`.")
    if truth is not None:
        white_turn = truth.white_turn

    # 1. find the board
    corners = truth.corners if localiser is None else localiser.localise(image)
    if corners is None:
        return Recognition(None, np.full((8, 8), EMPTY, dtype="<U1"))
    corners = np.asarray(corners, dtype=np.float32)

    # 2. warp and crop
    warped = geometry.warp_board(image, corners)
    occupancy_crops, piece_crops = geometry.crop_all(warped)
    true_view = _truth_under_cells(truth, corners) if truth is not None else None

    # 3. occupancy
    if occupancy is None:
        occupied = true_view != EMPTY
    else:
        flat = occupancy_crops.reshape(64, *occupancy_crops.shape[2:])
        occupied = np.asarray(occupancy.predict(flat)).astype(bool).reshape(8, 8)

    # 4. pieces, only on squares judged occupied
    view = np.full((8, 8), EMPTY, dtype="<U1")
    if occupied.any():
        if pieces is None:
            symbols = true_view[occupied]
            # a square wrongly judged occupied has no true piece; it's an error
            # either way, so fill in a pawn to keep the grid valid
            symbols = np.where(symbols == EMPTY, "P", symbols)
        else:
            class_ids = np.asarray(pieces.predict(piece_crops[occupied]))
            symbols = np.array(PIECE_SYMBOLS)[class_ids]
        view[occupied] = symbols

    # 5. image cells -> chess grid
    return Recognition(corners, geometry.to_image_view(view, white_turn).copy())
