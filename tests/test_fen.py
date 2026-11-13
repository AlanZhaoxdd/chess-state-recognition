import numpy as np
import pytest

from chessrec.fen import EMPTY, fen_to_grid, grid_to_fen, square_name

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_start_position_layout():
    grid = fen_to_grid(START + " w KQkq - 0 1")  # extra FEN fields are ignored
    assert grid.shape == (8, 8)
    assert grid[0, 0] == "r" and grid[0, 4] == "k"  # a8, e8
    assert grid[7, 0] == "R" and grid[7, 3] == "Q"  # a1, d1
    assert (grid[2:6] == EMPTY).all()


@pytest.mark.parametrize("fen", [
    START,
    "8/8/8/8/8/8/8/8",
    "r3k2r/8/2n5/8/4P3/8/8/R3K2R",
    "1Q6/5p1k/6pp/8/3b4/7P/5PP1/6K1",
])
def test_round_trip(fen):
    assert grid_to_fen(fen_to_grid(fen)) == fen


@pytest.mark.parametrize("fen", [
    "8/8/8/8/8/8/8",            # 7 ranks
    "9/8/8/8/8/8/8/8",          # rank too long
    "7/8/8/8/8/8/8/8",          # rank too short
    "x7/8/8/8/8/8/8/8",         # unknown symbol
])
def test_invalid_fen_is_rejected(fen):
    with pytest.raises(ValueError):
        fen_to_grid(fen)


def test_grid_to_fen_rejects_bad_shape():
    with pytest.raises(ValueError):
        grid_to_fen(np.full((8, 7), EMPTY))


def test_square_name():
    assert square_name(0, 0) == "a8"
    assert square_name(7, 7) == "h1"
    assert square_name(4, 4) == "e4"
