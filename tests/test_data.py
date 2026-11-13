"""Loader tests on a tiny fake dataset written to a temporary folder."""

import json

import cv2
import numpy as np
import pytest

from chessrec.data import RenderDataset

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def write_sample(folder, sample_id, white_turn=True):
    image = np.full((60, 90, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(folder / f"{sample_id}.png"), image)
    label = {
        "fen": FEN,
        "corners": [[10, 10], [80, 10], [80, 50], [10, 50]],
        "white_turn": white_turn,
        "lighting": {"mode": "spotlights"},
    }
    (folder / f"{sample_id}.json").write_text(json.dumps(label), encoding="utf-8")


def test_dataset_lists_and_loads(tmp_path):
    folder = tmp_path / "val"
    folder.mkdir()
    write_sample(folder, "0002", white_turn=False)
    write_sample(folder, "0001")

    dataset = RenderDataset("val", root=tmp_path)
    assert len(dataset) == 2
    assert [s.sample_id for s in dataset] == ["0001", "0002"]  # sorted

    sample = dataset[1]
    assert sample.white_turn is False
    assert sample.corners.shape == (4, 2) and sample.corners.dtype == np.float32
    assert sample.grid[7, 4] == "K"
    assert sample.meta == {"lighting": {"mode": "spotlights"}}
    assert sample.load_image().shape == (60, 90, 3)


def test_missing_split_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        RenderDataset("train", root=tmp_path)


def test_extra_split_with_jpeg_photos(tmp_path):
    """Our own photos live in an extra folder and may be JPEGs."""
    folder = tmp_path / "real"
    folder.mkdir()
    write_sample(folder, "IMG_01")
    cv2.imwrite(str(folder / "IMG_01.jpg"), np.zeros((30, 40, 3), dtype=np.uint8))
    (folder / "IMG_01.png").unlink()

    sample = RenderDataset("real", root=tmp_path)[0]
    assert sample.image_path.suffix == ".jpg"
    assert sample.load_image().shape == (30, 40, 3)
