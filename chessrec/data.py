"""Loading the rendered chessboard dataset.

Expected layout (created by ``scripts/download_data.py``)::

    data/render/
        train/0000.png  0000.json  0001.png  0001.json ...
        val/...
        test/...

Each JSON label holds at least

* ``fen``        -- the position
* ``corners``    -- the four outer corners of the playing area, in pixels
* ``white_turn`` -- True if White is to move; the camera stands on the side
  of the player to move
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
    """One labelled photo of a chessboard.

    The image is read from disk only when :meth:`load_image` is called, so a
    whole split can be listed cheaply.
    """

    sample_id: str
    image_path: Path
    fen: str
    corners: np.ndarray  # (4, 2) float32 pixel coordinates, order not fixed
    white_turn: bool
    meta: dict = field(default_factory=dict, compare=False)  # other label keys

    @property
    def grid(self) -> np.ndarray:
        """The position as an 8x8 grid (see :mod:`chessrec.fen`)."""
        return fen_to_grid(self.fen)

    def load_image(self) -> np.ndarray:
        """Read the photo as an H x W x 3 BGR array."""
        # np.fromfile + imdecode instead of cv2.imread: imread cannot open
        # paths with non-ASCII characters on Windows.
        buffer = np.fromfile(self.image_path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image {self.image_path}")
        return image


#: Image file types looked for next to a label, in order of preference.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _find_image(label_path: Path) -> Path:
    """Path of the photo that belongs to a label file.

    The dataset uses ``<id>.png``; our own photos may be JPEGs. If no image
    exists yet the ``.png`` path is returned, and reading it fails later
    with a clear message.
    """
    for suffix in IMAGE_SUFFIXES:
        for candidate in (label_path.with_suffix(suffix),
                          label_path.with_suffix(suffix.upper())):
            if candidate.is_file():
                return candidate
    return label_path.with_suffix(".png")


def load_sample(label_path: Path) -> BoardSample:
    """Build a :class:`BoardSample` from a ``<id>.json`` label file.

    The photo is expected next to it with the same name (``<id>.png`` or
    ``<id>.jpg``).
    """
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
    """Load exported square crops into memory (for classical classifiers).

    The crops must have been written by ``scripts/export_squares.py``.

    Args:
        kind: ``"occupancy"`` or ``"pieces"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        max_boards: use only the first N boards (by sorted id). All crops of
            a board are kept or dropped together, so a subset never shares a
            board with the rest.
        root: folder holding the exported crops (default: ``SQUARES_DIR``).

    Returns:
        ``(images, labels, board_ids)``: uint8 BGR images of shape
        (N, H, W, 3), integer class ids of shape (N,) (see
        :mod:`chessrec.fen`), and the board id of every crop, shape (N,).
    """
    from chessrec.fen import OCCUPANCY_NAMES, PIECE_NAMES  # avoid a cycle at import

    class_names = {"occupancy": OCCUPANCY_NAMES, "pieces": PIECE_NAMES}[kind]
    folder = Path(root or config.SQUARES_DIR) / kind / split
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist. Run `python scripts/export_squares.py` first.")

    files = [(path, class_id)
             for class_id, name in enumerate(class_names)
             for path in sorted((folder / name).glob("*.png"))]
    # File names are "<board id>_<square>.png".
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
    """All labelled boards of one split, in a fixed (sorted) order.

    Example::

        val = RenderDataset("val")
        sample = val[0]
        image = sample.load_image()
    """

    def __init__(self, split: str, root: Path | None = None):
        # Any folder under data/render/ is a valid split, so extra test sets
        # (e.g. "real" for our own photos) work without code changes.
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
