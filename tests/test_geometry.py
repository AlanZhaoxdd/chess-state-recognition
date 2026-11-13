"""Geometry tests on a synthetic board, so they run without the dataset."""

import cv2
import numpy as np
import pytest

from chessrec import config, geometry

S, M = config.SQUARE_SIZE, config.WARP_MARGIN


def cell_colour(row: int, col: int) -> tuple[int, int, int]:
    """A distinct, easily recognised colour for each of the 64 cells."""
    return (40 + 25 * row, 40 + 25 * col, 200)


def make_photo():
    """Render a flat 8x8 board and view it in perspective.

    Returns the "photo" and the four board corners in photo coordinates,
    ordered TL, TR, BR, BL.
    """
    flat = np.zeros((400, 400, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            flat[50 * r:50 * (r + 1), 50 * c:50 * (c + 1)] = cell_colour(r, c)

    flat_corners = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
    # Far edge narrower than the near edge, like a real photo.
    photo_corners = np.float32([[300, 200], [700, 210], [860, 600], [140, 590]])
    H = cv2.getPerspectiveTransform(flat_corners, photo_corners)
    photo = cv2.warpPerspective(flat, H, (1000, 700), flags=cv2.INTER_NEAREST)
    return photo, photo_corners


def test_sort_corners_any_input_order():
    expected = np.float32([[300, 200], [700, 210], [860, 600], [140, 590]])
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = expected[rng.permutation(4)]
        np.testing.assert_array_equal(geometry.sort_corners(shuffled), expected)


def test_sort_corners_rejects_bad_shape():
    with pytest.raises(ValueError):
        geometry.sort_corners(np.zeros((3, 2)))


def test_warp_puts_every_cell_in_place():
    photo, corners = make_photo()
    warped = geometry.warp_board(photo, corners[[2, 0, 3, 1]])  # shuffled
    assert warped.shape == (config.WARP_SIZE, config.WARP_SIZE, 3)
    for r in range(8):
        for c in range(8):
            centre = warped[M + S * r + S // 2, M + S * c + S // 2]
            assert tuple(int(v) for v in centre) == cell_colour(r, c), (r, c)


def test_crop_shapes_and_content():
    photo, corners = make_photo()
    warped = geometry.warp_board(photo, corners)
    occupancy, pieces = geometry.crop_all(warped)
    assert occupancy.shape == (8, 8, *config.OCCUPANCY_CROP_SIZE, 3)
    assert pieces.shape == (8, 8, *config.PIECE_CROP_SIZE, 3)

    for r in range(8):
        for c in range(8):
            # Occupancy crop: the cell itself is centred in the crop.
            centre = occupancy[r, c, S, S]
            assert tuple(int(v) for v in centre) == cell_colour(r, c), (r, c)

            # Piece crop: the cell sits at the bottom of the canvas. On the
            # left half it is mirrored, which puts the cell bottom-left too.
            base = pieces[r, c, -S // 2, S // 2]
            assert tuple(int(v) for v in base) == cell_colour(r, c), (r, c)


def test_to_image_view():
    grid = np.arange(64).reshape(8, 8)
    assert geometry.to_image_view(grid, white_turn=True) is grid
    flipped = geometry.to_image_view(grid, white_turn=False)
    assert flipped[0, 0] == grid[7, 7] and flipped[7, 0] == grid[0, 7]
    np.testing.assert_array_equal(
        geometry.to_image_view(flipped, white_turn=False), grid)
