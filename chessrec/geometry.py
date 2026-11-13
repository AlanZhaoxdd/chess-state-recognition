"""Perspective warp of the board and cropping of its 64 squares.

The cropping scheme follows the idea of Wölflein & Arandjelović,
"Determining Chess Game State from an Image" (J. Imaging, 2021): warp the
board to a top-down view, then cut a small crop for occupancy and a taller
crop for piece recognition, because a piece stands *above* its square in a
perspective view.

Image-grid convention
---------------------
After warping, the board is an 8x8 grid of *image cells*: cell ``(0, 0)`` is
the top-left square of the warped image and cell ``(7, 7)`` the bottom-right.
The bottom of the warped image is the side nearest the camera.

Which chess square an image cell shows depends on where the camera stands.
The dataset is rendered from the perspective of the player to move, so:

* White to move: image cell ``(r, c)`` shows ``grid[r, c]``
* Black to move: image cell ``(r, c)`` shows ``grid[7 - r, 7 - c]``

(``grid`` as defined in :mod:`chessrec.fen`.) Use :func:`to_image_view` to
convert between the two; it is its own inverse.
"""

from __future__ import annotations

import cv2
import numpy as np

from chessrec import config


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """Order four corner points as top-left, top-right, bottom-right, bottom-left.

    "Top" and "left" refer to the photo. The two points with the smallest y
    are the top pair, and each pair is then ordered by x. This is valid as
    long as the camera looks at the board from one of its sides, which holds
    for the dataset and for ordinary photos of a game.

    Args:
        corners: array of shape (4, 2) with (x, y) pixel coordinates in any
            order.

    Returns:
        float32 array of shape (4, 2) in the order TL, TR, BR, BL.
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
    """3x3 homography that maps photo pixels to the warped top-down image.

    The four board corners land on the corners of a ``BOARD_SIZE`` square
    that sits ``WARP_MARGIN`` pixels inside the warped image.
    """
    lo = config.WARP_MARGIN
    hi = config.WARP_MARGIN + config.BOARD_SIZE
    destination = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=np.float32)
    return cv2.getPerspectiveTransform(sort_corners(corners), destination)


def warp_board(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a photo so that the board becomes an axis-aligned square.

    Args:
        image: the photo, as read by OpenCV (H x W x 3, BGR).
        corners: the four outer corners of the 8x8 playing area, shape (4, 2),
            any order.

    Returns:
        Image of shape (WARP_SIZE, WARP_SIZE, 3). The playing area occupies
        the central ``BOARD_SIZE`` square; areas outside the photo are black.
    """
    size = (config.WARP_SIZE, config.WARP_SIZE)
    return cv2.warpPerspective(image, board_homography(corners), size)


def to_image_view(grid: np.ndarray, white_turn: bool) -> np.ndarray:
    """Rotate between the chess grid and the image-cell grid.

    With White to move the two coincide; with Black to move the camera sits
    on the other side, so the grid is rotated by 180 degrees. Applying the
    function twice gives back the input, so it converts in both directions.
    """
    grid = np.asarray(grid)
    return grid if white_turn else grid[::-1, ::-1]


def crop_occupancy(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop image cell ``(row, col)`` for empty-vs-occupied classification.

    The crop is the square plus half a square of context on every side, so
    it also sees the base of a piece that leans out of its square.

    Returns:
        Array of shape ``OCCUPANCY_CROP_SIZE + (3,)``.
    """
    s, m = config.SQUARE_SIZE, config.WARP_MARGIN
    y0 = m + s * row - s // 2
    x0 = m + s * col - s // 2
    return warped[y0:y0 + 2 * s, x0:x0 + 2 * s].copy()


def crop_piece(warped: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop image cell ``(row, col)`` for piece classification.

    In the warped view a piece is smeared away from the camera: upwards, and
    sideways towards the nearer left/right edge of the board. The crop
    therefore

    * extends upwards by 1 square (nearest rank) up to 3 squares (farthest
      rank),
    * extends outwards by up to 1 square on the left half (to the left) or
      the right half (to the right),
    * is mirrored on the left half so every piece leans the same way, and
    * is placed bottom-left in a fixed-size black canvas.

    Returns:
        Array of shape ``PIECE_CROP_SIZE + (3,)``.
    """
    s, m = config.SQUARE_SIZE, config.WARP_MARGIN
    out_h, out_w = config.PIECE_CROP_SIZE

    # Far rows (small ``row``) need more head-room than near rows.
    extra_up = 1.0 + 2.0 * (7 - row) / 7
    # Columns far from the centre line lean further outwards.
    left_half = col < 4
    distance = (3 - col) if left_half else (col - 4)  # 0 (centre) .. 3 (edge)
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
    """Crop all 64 image cells of a warped board.

    Returns:
        ``(occupancy, pieces)`` with shapes ``(8, 8) + OCCUPANCY_CROP_SIZE +
        (3,)`` and ``(8, 8) + PIECE_CROP_SIZE + (3,)``, indexed by image cell
        ``[row, col]``.
    """
    occupancy = np.stack([
        np.stack([crop_occupancy(warped, r, c) for c in range(8)]) for r in range(8)
    ])
    pieces = np.stack([
        np.stack([crop_piece(warped, r, c) for c in range(8)]) for r in range(8)
    ])
    return occupancy, pieces
