"""Tests for the hand-crafted feature + SVM classifiers (work package B)."""

import cv2
import numpy as np
import pytest

from chessrec import classical, config
from chessrec.classical import (BlockWeights, FeatureExtractor, SvmClassifier,
                                cached_features, list_crops)
from chessrec.fen import OCCUPANCY_NAMES

rng = np.random.default_rng(0)


def stripes(vertical: bool, n: int, shape=(100, 100, 3)) -> np.ndarray:
    """Noisy crops with vertical or horizontal stripes: two easy classes."""
    crops = rng.integers(0, 40, (n,) + shape, dtype=np.uint8)
    if vertical:
        crops[:, :, ::10] += 200
    else:
        crops[:, ::10, :] += 200
    return crops


def test_feature_lengths_match_block_sizes():
    fx = FeatureExtractor(features=("hog", "lbp", "colour"), size=(64, 128),
                          colour_grid=(3, 6), colour_bins=4)
    # HOG: 15 x 7 blocks of 2 x 2 cells x 9 bins; LBP: 4 x 8 cells x 10 codes;
    # colour: 18 cells x 3 channels x 4 bins
    assert fx.block_sizes() == [3780, 320, 216]
    features = fx.transform(rng.integers(0, 255, (3, 200, 100, 3), dtype=np.uint8))
    assert features.shape == (3, 4316) and features.dtype == np.float32


def test_unknown_feature_type_is_rejected():
    with pytest.raises(ValueError):
        FeatureExtractor(features=("sift",))
    with pytest.raises(ValueError):
        FeatureExtractor(features=())


def test_colour_tells_white_from_black_but_hog_does_not():
    white = np.full((1, 100, 100, 3), 230, dtype=np.uint8)
    black = np.full((1, 100, 100, 3), 30, dtype=np.uint8)
    hog = FeatureExtractor(features=("hog",), size=(64, 64))
    colour = FeatureExtractor(features=("colour",), size=(64, 64))
    assert np.allclose(hog.transform(white), hog.transform(black))
    assert not np.allclose(colour.transform(white), colour.transform(black))


def test_block_weights_give_each_block_unit_variance():
    x = rng.normal(size=(500, 30))
    standardised = (x - x.mean(0)) / x.std(0)
    weighted = BlockWeights([20, 10]).fit(standardised).transform(standardised)
    assert weighted[:, :20].var(axis=0).sum() == pytest.approx(1)
    assert weighted[:, 20:].var(axis=0).sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        BlockWeights([20, 5]).fit(standardised)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "nystroem"])
def test_svm_separates_easy_classes(kernel, tmp_path):
    crops = np.concatenate([stripes(True, 20), stripes(False, 20)])
    labels = np.repeat([0, 1], 20)
    fx = FeatureExtractor(features=("hog", "lbp"), size=(64, 64))
    clf = SvmClassifier(fx, kernel=kernel, n_components=30)
    clf.fit_features(fx.transform(crops), labels)

    test = np.concatenate([stripes(True, 5), stripes(False, 5)])
    assert (clf.predict(test) == np.repeat([0, 1], 5)).all()
    assert clf.predict(np.zeros((0, 100, 100, 3), dtype=np.uint8)).shape == (0,)

    path = tmp_path / "model.joblib"
    clf.save(path)
    assert (SvmClassifier.load(path).predict(test) == clf.predict(test)).all()


def test_untrained_classifier_and_missing_model_fail_clearly(tmp_path):
    with pytest.raises(RuntimeError):
        SvmClassifier(FeatureExtractor()).predict_features(np.zeros((1, 5)))
    with pytest.raises(FileNotFoundError, match="train_classical"):
        SvmClassifier.load(tmp_path / "missing.joblib")


def write_crops(squares_dir, boards):
    """Fake exported occupancy crops: board b has one empty and one occupied square."""
    for b in boards:
        for name, square in zip(OCCUPANCY_NAMES, ("a1", "e4")):
            folder = squares_dir / "occupancy" / "train" / name
            folder.mkdir(parents=True, exist_ok=True)
            image = np.full((100, 100, 3), 0 if name == "empty" else 200, dtype=np.uint8)
            cv2.imwrite(str(folder / f"{b}_{square}.png"), image)


def test_list_crops_keeps_first_boards(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQUARES_DIR", tmp_path)
    write_crops(tmp_path, ["0003", "0001", "0002"])

    paths, labels, boards = list_crops("occupancy", "train")
    assert len(paths) == 6 and sorted(labels) == [0, 0, 0, 1, 1, 1]

    paths, labels, boards = list_crops("occupancy", "train", max_boards=2)
    assert sorted(set(boards)) == ["0001", "0002"] and len(paths) == 4

    with pytest.raises(FileNotFoundError, match="export_squares"):
        list_crops("occupancy", "val")


def test_cached_features_are_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SQUARES_DIR", tmp_path / "squares")
    monkeypatch.setattr(classical, "FEATURE_CACHE", tmp_path / "features")
    write_crops(tmp_path / "squares", ["0001", "0002"])
    fx = FeatureExtractor(features=("hog", "colour"), size=(64, 64))

    first, labels = cached_features(fx, "occupancy", "train", n_jobs=1)
    assert first.shape == (4, sum(fx.block_sizes()))
    assert len(list((tmp_path / "features").glob("*.npy"))) == 2  # one file per type

    # a different selection of boards must not reuse the cached file
    fewer, _ = cached_features(fx, "occupancy", "train", max_boards=1, n_jobs=1)
    assert fewer.shape[0] == 2
    second, _ = cached_features(fx, "occupancy", "train", n_jobs=1)
    assert np.array_equal(first, second)
