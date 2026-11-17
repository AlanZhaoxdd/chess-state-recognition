"""Loading the rendered dataset (data/render/<split>/<id>.png + <id>.json).

Each label has at least `fen`, `corners` (4 corners of the playing area) and
`white_turn` (the camera is on the side of the player to move).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from chessrec import config
from chessrec.fen import fen_to_grid


@dataclass(frozen=True)
class BoardSample:
    """One labelled photo. The image is only read when load_image() is called."""

    sample_id: str
    image_path: Path
    fen: str
    corners: np.ndarray  # (4, 2) float32, any order
    white_turn: bool
    meta: dict = field(default_factory=dict, compare=False)  # other label keys

    @property
    def grid(self) -> np.ndarray:
        return fen_to_grid(self.fen)

    def load_image(self) -> np.ndarray:
        """Read the photo as a BGR array."""
        # cv2.imread can't open non-ASCII paths on Windows
        buffer = np.fromfile(self.image_path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image {self.image_path}")
        return image


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _find_image(label_path: Path) -> Path:
    # the dataset uses .png, our own photos may be .jpg
    for suffix in IMAGE_SUFFIXES:
        for candidate in (label_path.with_suffix(suffix),
                          label_path.with_suffix(suffix.upper())):
            if candidate.is_file():
                return candidate
    return label_path.with_suffix(".png")


def load_sample(label_path: Path) -> BoardSample:
    """Build a BoardSample from a <id>.json label file."""
    label_path = Path(label_path)
    with open(label_path, encoding="utf-8") as f:
        label = json.load(f)

    corners = np.asarray(label["corners"], dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"{label_path}: corners have shape {corners.shape}")

    known = {"fen", "corners", "white_turn"}
    return BoardSample(
        sample_id=label_path.stem,
        image_path=_find_image(label_path),
        fen=label["fen"],
        corners=corners,
        white_turn=bool(label["white_turn"]),
        meta={k: v for k, v in label.items() if k not in known},
    )


def load_square_crops(
    kind: str, split: str, max_boards: int | None = None, root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load crops written by scripts/export_squares.py into memory.

    kind is "occupancy" or "pieces". max_boards keeps only the first N boards,
    with all crops of a board kept together. Returns (images, labels, board_ids).
    """
    from chessrec.fen import OCCUPANCY_NAMES, PIECE_NAMES  # avoid import cycle

    class_names = {"occupancy": OCCUPANCY_NAMES, "pieces": PIECE_NAMES}[kind]
    folder = Path(root or config.SQUARES_DIR) / kind / split
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist. Run `python scripts/export_squares.py` first.")

    files = [(path, class_id)
             for class_id, name in enumerate(class_names)
             for path in sorted((folder / name).glob("*.png"))]
    # file names are <board id>_<square>.png
    board_of = [path.stem.rsplit("_", 1)[0] for path, _ in files]
    if max_boards is not None:
        keep = set(sorted(set(board_of))[:max_boards])
        files = [f for f, board in zip(files, board_of) if board in keep]
        board_of = [board for board in board_of if board in keep]

    images = np.stack([
        cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        for path, _ in files
    ])
    labels = np.array([class_id for _, class_id in files])
    return images, labels, np.array(board_of)


class RenderDataset:
    """All labelled boards of one split, sorted by id."""

    def __init__(self, split: str, root: Path | None = None):
        # any folder under data/render/ works, e.g. "real" for our own photos
        self.split = split
        self.folder = Path(root or config.RENDER_DIR) / split
        if not self.folder.is_dir():
            raise FileNotFoundError(
                f"{self.folder} does not exist. For the published splits "
                f"{config.SPLITS} run `python scripts/download_data.py` first."
            )
        self._labels = sorted(self.folder.glob("*.json"))
        if not self._labels:
            raise FileNotFoundError(f"No .json labels found in {self.folder}")

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> BoardSample:
        return load_sample(self._labels[index])

    def __iter__(self) -> Iterator[BoardSample]:
        return (load_sample(path) for path in self._labels)
