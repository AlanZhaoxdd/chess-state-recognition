"""A deliberately simple baseline classifier.

"Tiny images + k-nearest-neighbours": every crop is shrunk to a few pixels,
flattened into a vector, normalised, and classified by a vote among the
most similar training crops. It uses no engineered features and no learning
beyond memorising examples, so it gives

* a working implementation of the classifier interfaces
  (:mod:`chessrec.stages`) to copy from, and
* a floor that HOG/LBP + SVM and the CNN have to beat in the report.

The same class serves both tasks; only the training data differs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class TinyImageClassifier:
    """k-NN on down-sampled crops.

    Args:
        size: (width, height) each crop is resized to before flattening.
        k: number of neighbours that vote.
    """

    def __init__(self, size: tuple[int, int] = (16, 16), k: int = 5):
        self.size = size
        self.k = k
        self._knn = KNeighborsClassifier(n_neighbors=k)

    def _features(self, crops: np.ndarray) -> np.ndarray:
        """Turn crops (N, H, W, 3) into unit-length, zero-mean vectors (N, D)."""
        tiny = np.stack([
            cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA) for crop in crops
        ]).reshape(len(crops), -1).astype(np.float32)
        # Zero mean and unit length remove overall brightness and contrast,
        # which vary with the lighting rather than with the piece.
        tiny -= tiny.mean(axis=1, keepdims=True)
        tiny /= np.linalg.norm(tiny, axis=1, keepdims=True) + 1e-8
        return tiny

    def fit(self, crops: np.ndarray, labels: np.ndarray) -> "TinyImageClassifier":
        """Memorise the training crops and their class ids."""
        self._knn.fit(self._features(crops), labels)
        return self

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """Class id for each crop (see :mod:`chessrec.stages`)."""
        return self._knn.predict(self._features(crops))

    def save(self, path: Path) -> None:
        """Store the fitted classifier on disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TinyImageClassifier":
        """Load a classifier written by :meth:`save`."""
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path} not found. Train it with `python scripts/train_baseline.py`.")
        return joblib.load(path)
