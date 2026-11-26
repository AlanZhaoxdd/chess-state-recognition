"""CNN wrapper tests on random data (no dataset or ImageNet download needed)."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from chessrec import config  # noqa: E402
from chessrec.cnn import (TASKS, CNNClassifier, SquareCropDataset, bgr_to_tensor,  # noqa: E402
                          build_network, train_augmentation)
from chessrec.fen import PIECE_NAMES  # noqa: E402


def random_crops(task: str, n: int = 5) -> np.ndarray:
    height, width = TASKS[task][1]
    return np.random.default_rng(0).integers(0, 256, (n, height, width, 3), dtype=np.uint8)


def make_classifier(task: str, arch: str = "resnet18") -> CNNClassifier:
    network = build_network(arch, len(TASKS[task][0]), pretrained=False)
    return CNNClassifier(network, task=task, arch=arch, device="cpu")


def test_bgr_to_tensor_swaps_channels_and_scales():
    crop = np.zeros((4, 6, 3), dtype=np.uint8)
    crop[..., 0] = 255  # pure blue in BGR
    x = bgr_to_tensor(crop)
    assert x.shape == (3, 4, 6)
    assert torch.all(x[2] == 1) and torch.all(x[:2] == 0)  # blue is the last RGB channel
    assert bgr_to_tensor(crop[None]).shape == (1, 3, 4, 6)


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v3_small"])
def test_predict_returns_one_class_id_per_crop(task, arch):
    predictions = make_classifier(task, arch).predict(random_crops(task))
    assert predictions.shape == (5,)
    assert predictions.min() >= 0 and predictions.max() < len(TASKS[task][0])


def test_predict_handles_no_crops():
    assert make_classifier("pieces").predict(random_crops("pieces", 0)).shape == (0,)


def test_predict_rejects_wrong_crop_size():
    with pytest.raises(ValueError):
        make_classifier("pieces").predict(random_crops("occupancy"))


def test_save_and_load_give_the_same_predictions(tmp_path):
    classifier = make_classifier("pieces")
    crops = random_crops("pieces")
    classifier.save(tmp_path / "model.pt")
    loaded = CNNClassifier.load(tmp_path / "model.pt", device="cpu")
    assert loaded.task == "pieces" and loaded.arch == "resnet18"
    np.testing.assert_array_equal(loaded.predict(crops), classifier.predict(crops))


def test_load_missing_file_explains_how_to_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_cnn"):
        CNNClassifier.load(tmp_path / "missing.pt")


def test_dataset_uses_fen_class_order_not_alphabetical(tmp_path):
    """ImageFolder would number black_bishop as 0; we need white_pawn = 0 (PIECE_SYMBOLS)."""
    crop = np.zeros((*config.PIECE_CROP_SIZE, 3), dtype=np.uint8)
    for board, name in enumerate(("black_bishop", "white_pawn", "black_king")):
        folder = tmp_path / "pieces" / "train" / name
        folder.mkdir(parents=True)
        cv2.imencode(".png", crop)[1].tofile(folder / f"{board:04d}_e4.png")

    dataset = SquareCropDataset("pieces", "train", root=tmp_path)
    labels = {path.parent.name: label for path, label in dataset.files}
    assert labels == {name: PIECE_NAMES.index(name)
                      for name in ("black_bishop", "white_pawn", "black_king")}
    assert len(SquareCropDataset("pieces", "train", max_boards=2, root=tmp_path)) == 2

    x, label = dataset[0]
    assert x.shape == (3, *config.PIECE_CROP_SIZE) and x.dtype == torch.float32


@pytest.mark.parametrize("task", list(TASKS))
def test_augmentation_keeps_shape_and_range(task):
    x = bgr_to_tensor(random_crops(task, 1)[0])
    y = train_augmentation(task)(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1