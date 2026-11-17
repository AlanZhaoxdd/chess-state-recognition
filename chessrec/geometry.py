"""Perspective warp of the board and cropping of the 64 squares.

The cropping idea follows Wölflein & Arandjelović (2021): warp the board to a
top-down view, then take a small crop for occupancy and a taller crop for the
piece, since a piece stands above its square in perspective.

After warping, image cell (0, 0) is the top-left square and the bottom of the
image is nearest the camera. With White to move cell (r, c) shows grid[r, c];
with Black to move it shows grid[7 - r, 7 - c]. to_image_view() converts.
"""

from __future__ import annotations

import cv2
import numpy as np

from chessrec import config


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left.

    Takes the two points with the smallest y as the top pair, then sorts each
    pair by x. Works as long as the camera looks at the board from a side.
    """
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected corners of shape (4, 2), got {corners.shape}")

    by_y = corners[np.argsort(corners[:, 1], kind="stable")]
    top, bottom = by_y[:2], by_y[2:]
    top_left, top_right = top[np.argsort(top[:, 0], kind="stable")]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0], kind="stable")]
    return np.stack([top_left, top_right, bottom_right, bottom_left])


def board_homography(corners: np.ndarray) -> np.ndarray:
    """Homography from photo pixels to the warped top-down image."""
    lo = config.WARP_MARGIN
    hi = config.WARP_MARGIN + config.BOARD_SIZE
    destination = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
    return cv2.getPerspectiveTransform(sort_corners(corners), destination)


def warp_board(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the photo so the board is an axis-aligned square (WARP_SIZE x WARP_SIZE)."""
    size = (config.WARP_SIZE, config.WARP_SIZE)
    return cv2.warpPerspective(image, board_homography(corners), size)


def to_image_view(grid: np.ndarray, white_turn: bool) -> np.ndarray:
    """Convert between chess grid and image cells (rotate 180 if Black to move).

    Applying it twice gives the input back, so it works both ways.
    """
    grid = np.asarray(grid)
    return grid if white_turn else grid[::-1, ::-1]


def crop_occupancy(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop for empty/occupied: the square plus half a square on every side."""
    s, m = config.SQUARE_SIZE, config.WARP_MARGIN
    y0 = m + s * row - s // 2
    x0 = m + s * col - s // 2
    return warped[y0:y0 + 2 * s, x0:x0 + 2 * s].copy()


def crop_piece(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop for piece classification, size PIECE_CROP_SIZE.

    In the warped view pieces lean away from the camera (up, and out towards
    the nearer side edge), so the crop extends upwards and sideways. Crops on
    the left half are mirrored so every piece leans the same way.
    """
    s, m = config.SQUARE_SIZE, config.WARP_MARGIN
    out_h, out_w = config.PIECE_CROP_SIZE

    # far rows need more room above: 1 square (nearest) to 3 squares (farthest)
    extra_up = 1.0 + 2.0 * (7 - row) / 7
    left_half = col < 4
    distance = (3 - col) if left_half else (col - 4)  # 0 = centre, 3 = edge
    extra_side = 0.25 + 0.75 * distance / 3

    y0 = int(m + s * (row - extra_up))
    y1 = m + s * (row + 1)
    if left_half:
        x0, x1 = int(m + s * (col - extra_side)), m + s * (col + 1)
    else:
        x0, x1 = m + s * col, int(m + s * (col + 1 + extra_side))

    crop = warped[y0:y1, x0:x1]
    if left_half:
        crop = cv2.flip(crop, 1)  # mirror so the lean is always to the right

    canvas = np.zeros((out_h, out_w, 3), dtype=warped.dtype)
    h, w = crop.shape[:2]
    canvas[out_h - h:, :w] = crop
    return canvas


def crop_all(warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All 64 occupancy and piece crops, each indexed [row, col] by image cell."""
    occupancy = np.stack([
        np.stack([crop_occupancy(warped, r, c) for c in range(8)]) for r in range(8)
    ])
    pieces = np.stack([
        np.stack([crop_piece(warped, r, c) for c in range(8)]) for r in range(8)
    ])
    return occupancy, pieces
