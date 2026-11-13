"""Pipeline and metric tests on a synthetic board (no dataset needed)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from chessrec import config
from chessrec.data import BoardSample
from chessrec.evaluation import Evaluator, corner_errors, format_report
from chessrec.fen import EMPTY, SYMBOL_TO_ID, fen_to_grid
from chessrec.pipeline import recognise

FEN = "r3k2r/pp4pp/8/3q4/4P3/8/PP4PP/R3K2R"
CORNERS = np.float32([[300, 200], [700, 210], [860, 600], [140, 590]])  # TL TR BR BL
S = config.SQUARE_SIZE


def make_sample(white_turn: bool):
    """A fake photo in which every *image cell* has a unique colour."""
    flat = np.zeros((400, 400, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            flat[50 * r:50 * (r + 1), 50 * c:50 * (c + 1)] = (10 + r, 10 + c, 200)
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]), CORNERS)
    photo = cv2.warpPerspective(flat, H, (1000, 700), flags=cv2.INTER_NEAREST)
    sample = BoardSample("0000", Path("unused.png"), FEN, CORNERS.copy(), white_turn)
    return photo, sample


class CellReader:
    """Fake classifier that recovers the image cell from the crop's colour
    and answers from the true position: a perfect classifier that really
    looks at the pixels it is given."""

    def __init__(self, sample: BoardSample, pixel: tuple[int, int], pieces: bool):
        grid = sample.grid if sample.white_turn else sample.grid[::-1, ::-1]
        self.view, self.pixel, self.pieces = grid, pixel, pieces

    def predict(self, crops):
        answers = []
        for crop in crops:
            b, g, _ = crop[self.pixel]
            symbol = self.view[int(b) - 10, int(g) - 10]
            answers.append(SYMBOL_TO_ID[symbol] if self.pieces else int(symbol != EMPTY))
        return np.array(answers)


class FixedLocaliser:
    def __init__(self, corners):
        self.corners = corners

    def localise(self, image):
        return self.corners


@pytest.mark.parametrize("white_turn", [True, False])
def test_all_oracle_reproduces_the_position(white_turn):
    photo, sample = make_sample(white_turn)
    result = recognise(photo, truth=sample)
    assert result.fen == FEN


@pytest.mark.parametrize("white_turn", [True, False])
def test_real_stages_receive_the_right_crops(white_turn):
    """Classifiers that read the pixels must also reproduce the position,
    which checks cropping, batching and the black-side rotation."""
    photo, sample = make_sample(white_turn)
    result = recognise(
        photo,
        localiser=FixedLocaliser(CORNERS[[1, 3, 0, 2]]),          # any order
        occupancy=CellReader(sample, (S, S), pieces=False),        # crop centre
        pieces=CellReader(sample, (-S // 2, S // 2), pieces=True),  # crop base
        white_turn=white_turn,
    )
    assert result.fen == FEN


def test_oracle_classifiers_follow_a_wrong_localisation():
    """Corners shifted by one square to the right: perfect classifiers then
    see the board shifted by one file, and the last file falls off it."""
    photo, sample = make_sample(True)
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]), CORNERS)
    shifted = cv2.perspectiveTransform(
        np.float32([[[50, 0]], [[450, 0]], [[450, 400]], [[50, 400]]]), H).reshape(4, 2)

    result = recognise(photo, localiser=FixedLocaliser(shifted), truth=sample)
    true = fen_to_grid(FEN)
    np.testing.assert_array_equal(result.grid[:, :7], true[:, 1:])
    assert (result.grid[:, 7] == EMPTY).all()


def test_missing_board_gives_empty_grid():
    photo, sample = make_sample(True)
    result = recognise(photo, localiser=FixedLocaliser(None), truth=sample)
    assert result.corners is None and result.fen == "8/8/8/8/8/8/8/8"


def test_oracle_without_truth_is_rejected():
    photo, _ = make_sample(True)
    with pytest.raises(ValueError):
        recognise(photo)


def test_corner_errors_ignore_order():
    moved = CORNERS + np.float32([3, 4])  # every corner 5 px away
    np.testing.assert_allclose(corner_errors(moved[[2, 0, 3, 1]], CORNERS), 5.0, atol=1e-4)


def test_evaluator_counts():
    photo, sample = make_sample(True)
    evaluator = Evaluator()

    # Board 1: perfect.
    evaluator.add(sample, recognise(photo, truth=sample), image_width=1000)

    # Board 2: one missed piece, one invented piece, one confused piece.
    result = recognise(photo, truth=sample)
    grid = result.grid.copy()
    grid[0, 0] = EMPTY      # a8 rook missed            -> occupancy false negative
    grid[2, 2] = "N"        # c6 is empty                -> occupancy false positive
    grid[3, 3] = "k"        # d5 queen called a king     -> piece confusion
    evaluator.add(sample, type(result)(result.corners, grid), image_width=1000)

    summary = evaluator.summary()
    n_pieces = int((sample.grid != EMPTY).sum())
    assert summary["boards"] == 2
    assert summary["localisation"]["success_rate"] == 1.0
    assert summary["board"]["mean_errors_per_board"] == 1.5
    assert summary["board"]["boards_no_error"] == 0.5
    assert summary["occupancy"]["confusion_true_rows"] == [
        [2 * (64 - n_pieces) - 1, 1], [1, 2 * n_pieces - 1]]
    assert summary["pieces"]["per_class_recall"]["black_queen"] == 0.5
    assert summary["pieces"]["per_class_recall"]["white_king"] == 1.0
    q, k = SYMBOL_TO_ID["q"], SYMBOL_TO_ID["k"]
    assert summary["pieces"]["confusion_true_rows"][q][k] == 1
    assert "boards with no error" in format_report(summary)


def test_failed_localisation_is_counted():
    photo, sample = make_sample(True)
    evaluator = Evaluator()
    evaluator.add(sample, recognise(photo, localiser=FixedLocaliser(None), truth=sample),
                  image_width=1000)
    evaluator.add(sample, recognise(photo, localiser=FixedLocaliser(CORNERS + 30), truth=sample),
                  image_width=1000)  # found, but 42 px off: outside the tolerance
    summary = evaluator.summary()
    assert summary["localisation"]["success_rate"] == 0.0
    assert summary["localisation"]["mean_corner_error_px"] == pytest.approx(42.43, abs=0.01)
