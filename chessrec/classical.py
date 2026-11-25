"""Hand-crafted features + SVM, for both occupancy and piece classification.

Work package B. Each crop is turned into a fixed-length feature vector made of
one or more lecture features, then classified by a support vector machine:

* ``hog``    -- histogram of oriented gradients (shape of the piece outline)
* ``lbp``    -- local binary pattern histograms on a grid of cells (texture)
* ``colour`` -- HSV colour histograms (HOG and LBP ignore absolute brightness,
  which is what separates white pieces from black ones)

Features are standardised (zero mean, unit variance) and each feature type is
then scaled so it carries the same total weight; otherwise the 3780 HOG values
would drown out the few hundred colour values. The training set is far too
large to hold as images, so :func:`iter_crop_batches` reads crops from disk in
batches and only the feature vectors are kept in memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from chessrec import config
from chessrec.fen import OCCUPANCY_NAMES, PIECE_NAMES

FEATURE_TYPES = ("hog", "lbp", "colour")

#: Size (width, height) every crop is resized to before feature extraction.
#: Piece crops keep their 1:2 aspect ratio, the classic HOG pedestrian window.
WORK_SIZES = {"occupancy": (64, 64), "pieces": (64, 128)}


@dataclass
class FeatureExtractor:
    """Turns BGR crops into feature vectors.

    Attributes:
        features: which feature types to concatenate, from :data:`FEATURE_TYPES`.
        size: (width, height) the crop is resized to first.
        hog_cell: HOG cell size in pixels.
        hog_orientations: number of HOG orientation bins (unsigned, 0-180 degrees).
        lbp_points, lbp_radius: LBP neighbourhood (P sampling points on a circle of radius R).
        lbp_cell: side of the square cells that get their own LBP histogram.
        colour_bins: bins per HSV channel for the colour histogram.
        colour_grid: (columns, rows) of cells that each get their own colour
            histogram, so the histogram also says where the colours are.
    """

    features: tuple[str, ...] = ("hog",)
    size: tuple[int, int] = (64, 128)
    hog_cell: int = 8
    hog_orientations: int = 9
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_cell: int = 16
    colour_bins: int = 16
    colour_grid: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        unknown = set(self.features) - set(FEATURE_TYPES)
        if unknown or not self.features:
            raise ValueError(f"features must be a non-empty subset of {FEATURE_TYPES}")
        self.features = tuple(self.features)

    def _hog(self, grey: np.ndarray) -> np.ndarray:
        return hog(grey, orientations=self.hog_orientations,
                   pixels_per_cell=(self.hog_cell, self.hog_cell),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

    def _lbp(self, grey: np.ndarray) -> np.ndarray:
        # "uniform" LBP has P + 2 distinct codes
        codes = local_binary_pattern(grey, self.lbp_points, self.lbp_radius, method="uniform")
        n_bins = self.lbp_points + 2
        c = self.lbp_cell
        histograms = []
        for y in range(0, grey.shape[0] - c + 1, c):
            for x in range(0, grey.shape[1] - c + 1, c):
                cell = codes[y:y + c, x:x + c]
                hist = np.bincount(cell.astype(np.int64).ravel(), minlength=n_bins)
                histograms.append(hist / cell.size)
        return np.concatenate(histograms)

    def _colour(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ranges = (180, 256, 256)  # OpenCV hue is 0-179
        cols, rows = self.colour_grid
        h, w = hsv.shape[:2]
        hists = []
        for r in range(rows):
            for c in range(cols):
                cell = hsv[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
                hists += [np.histogram(cell[..., i], bins=self.colour_bins, range=(0, top))[0]
                          / cell[..., 0].size for i, top in enumerate(ranges)]
        return np.concatenate(hists)

    def block_sizes(self) -> list[int]:
        """Length of each feature type's part of the vector, in order."""
        blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        return [len(FeatureExtractor(**{**vars(self), "features": (name,)}).transform_one(blank))
                for name in self.features]

    def transform_one(self, crop: np.ndarray) -> np.ndarray:
        """Feature vector of one BGR crop."""
        bgr = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        parts = {"hog": lambda: self._hog(grey), "lbp": lambda: self._lbp(grey),
                 "colour": lambda: self._colour(bgr)}
        return np.concatenate([parts[name]() for name in self.features]).astype(np.float32)

    def transform(self, crops: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Feature matrix, shape (N, n_features), for a batch of BGR crops."""
        return np.stack([self.transform_one(crop) for crop in crops])


class BlockWeights(BaseEstimator, TransformerMixin):
    """Divide each feature block by the square root of its length.

    Applied after standardisation, every block then has the same total
    variance, so each feature type counts equally in the SVM's distances.
    """

    def __init__(self, sizes: Sequence[int] = ()):
        self.sizes = sizes

    def fit(self, X, y=None) -> "BlockWeights":
        if sum(self.sizes) != X.shape[1]:
            raise ValueError(f"Block sizes {self.sizes} do not add up to {X.shape[1]} features")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        weights = np.concatenate([np.full(n, 1 / np.sqrt(n)) for n in self.sizes])
        return X * weights.astype(X.dtype)


@dataclass
class SvmClassifier:
    """Feature extractor + standardisation (+ optional PCA) + SVM.

    Implements ``predict(crops)`` from :mod:`chessrec.stages`, so the same
    class serves as occupancy and as piece classifier.

    Attributes:
        extractor: how crops become feature vectors.
        kernel: "linear" (LinearSVC, scales to all training crops),
            "rbf" (SVC, more flexible but quadratic in the number of samples) or
            "nystroem" (RBF kernel approximated with ``n_components`` landmark
            samples, then a linear SVM; close to "rbf" but scales linearly).
        C: SVM regularisation; larger values fit the training data more closely.
        gamma: RBF kernel width, ignored for the linear kernel. "scale" means
            1 / number of feature types, which matches scikit-learn's "scale"
            once every feature type has unit total variance.
        n_components: landmark samples for the "nystroem" kernel.
        pca: if set, project the standardised features onto this many
            principal components first (mainly to speed up the RBF kernel).
    """

    extractor: FeatureExtractor
    kernel: str = "linear"
    C: float = 1.0
    gamma: float | str = "scale"
    pca: int | None = None
    n_components: int = 2000
    model: Pipeline | None = field(default=None, repr=False)

    def _build(self) -> Pipeline:
        """Untrained scikit-learn pipeline for the chosen kernel."""
        gamma = 1 / len(self.extractor.features) if self.gamma == "scale" else self.gamma
        steps = [("scale", StandardScaler()),
                 ("blocks", BlockWeights(self.extractor.block_sizes()))]
        if self.pca:
            steps.append(("pca", PCA(n_components=self.pca, random_state=config.SEED)))
        if self.kernel == "linear":
            steps.append(("svm", LinearSVC(C=self.C, class_weight="balanced",
                                           max_iter=10_000, random_state=config.SEED)))
        elif self.kernel == "rbf":
            steps.append(("svm", SVC(C=self.C, gamma=gamma, class_weight="balanced",
                                     cache_size=1000, random_state=config.SEED)))
        elif self.kernel == "nystroem":
            steps.append(("rbf", Nystroem(gamma=gamma, n_components=self.n_components,
                                          random_state=config.SEED)))
            steps.append(("svm", LinearSVC(C=self.C, class_weight="balanced",
                                           max_iter=10_000, random_state=config.SEED)))
        else:
            raise ValueError(f"Unknown kernel {self.kernel!r}")
        return Pipeline(steps)

    def fit_features(self, features: np.ndarray, labels: np.ndarray) -> "SvmClassifier":
        """Fit on precomputed feature vectors (see :func:`extract_features`)."""
        self.model = self._build().fit(features, labels)
        return self

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Class ids for precomputed feature vectors."""
        if self.model is None:
            raise RuntimeError("Classifier is not trained yet.")
        return self.model.predict(features)

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """Class ids for a batch of BGR crops."""
        if len(crops) == 0:
            return np.zeros(0, dtype=int)
        return self.predict_features(self.extractor.transform(crops))

    def save(self, path: Path) -> None:
        """Save the whole classifier (extractor settings and fitted model)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "SvmClassifier":
        """Load a classifier saved by :meth:`save`."""
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path} not found. Train it with `python scripts/train_classical.py`.")
        return joblib.load(path)


def list_crops(kind: str, split: str, max_boards: int | None = None,
               root: Path | None = None) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Paths, class ids and board ids of the exported crops, without reading them.

    Same selection rules as :func:`chessrec.data.load_square_crops`: ``max_boards``
    keeps all crops of the first N boards. ``root`` defaults to data/squares;
    pass another folder with the same layout, e.g. jittered crops.
    """
    class_names = {"occupancy": OCCUPANCY_NAMES, "pieces": PIECE_NAMES}[kind]
    folder = Path(root or config.SQUARES_DIR) / kind / split
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} does not exist. Run `python scripts/export_squares.py` first.")
    files = [(path, class_id)
             for class_id, name in enumerate(class_names)
             for path in sorted((folder / name).glob("*.png"))]
    boards = [path.stem.rsplit("_", 1)[0] for path, _ in files]
    if max_boards is not None:
        keep = set(sorted(set(boards))[:max_boards])
        files = [f for f, b in zip(files, boards) if b in keep]
        boards = [b for b in boards if b in keep]
    return [p for p, _ in files], np.array([c for _, c in files]), np.array(boards)


def iter_crop_batches(paths: Sequence[Path], batch_size: int = 2000
                      ) -> Iterator[np.ndarray]:
    """Read crops from disk in batches, so the images never all sit in memory."""
    for start in range(0, len(paths), batch_size):
        yield np.stack([cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                        for p in paths[start:start + batch_size]])


def _features_of_paths(extractor: FeatureExtractor, paths: Sequence[Path]) -> np.ndarray:
    """Feature matrix for one chunk of crop files (run in a worker process)."""
    return np.concatenate([extractor.transform(batch) for batch in iter_crop_batches(paths)])


def extract_features(extractor: FeatureExtractor, paths: Sequence[Path],
                     n_jobs: int = 1, chunk: int = 2000) -> np.ndarray:
    """Feature matrix for the crops at ``paths``, optionally in parallel processes."""
    chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    if n_jobs == 1 or len(chunks) == 1:
        parts = [_features_of_paths(extractor, c) for c in chunks]
    else:
        parts = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_features_of_paths)(extractor, c) for c in chunks)
    return np.concatenate(parts)


#: Feature matrices are cached here, one file per feature type, so that trying
#: another combination or another C does not recompute them (git-ignored).
FEATURE_CACHE = config.DATA_DIR / "features"


def cached_features(extractor: FeatureExtractor, kind: str, split: str,
                    max_boards: int | None = None, n_jobs: int = -1,
                    root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and class ids for a split, computed once per feature type.

    Each feature type is cached separately under :data:`FEATURE_CACHE`, keyed
    by its settings and the selected crops, and the requested types are then
    concatenated in order.
    """
    paths, labels, _ = list_crops(kind, split, max_boards, root)
    listing = [str(root or "")] + [p.name for p in paths]
    selection = hashlib.md5("\n".join(listing).encode()).hexdigest()[:10]
    blocks = []
    for name in extractor.features:
        single = FeatureExtractor(**{**vars(extractor), "features": (name,)})
        settings = json.dumps(vars(single), sort_keys=True, default=list)
        key = hashlib.md5(settings.encode()).hexdigest()[:10]
        path = FEATURE_CACHE / f"{kind}_{split}_{name}_{key}_{selection}.npy"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, extract_features(single, paths, n_jobs=n_jobs))
        blocks.append(np.load(path))
    return np.hstack(blocks), labels
