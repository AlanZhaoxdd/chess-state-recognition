"""Tiny-image + k-NN baseline, used for both occupancy and piece classification."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class TinyImageClassifier:
    """k-NN on crops shrunk to `size` (width, height) pixels."""

    def __init__(self, size: tuple[int, int] = (16, 16), k: int = 5):
        self.size = size
        self.k = k
        self._knn = KNeighborsClassifier(n_neighbors=k)

    def _features(self, crops: np.ndarray) -> np.ndarray:
        tiny = np.stack([
            cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA) for crop in crops
        ]).reshape(len(crops), -1).astype(np.float32)
        # zero mean + unit length removes brightness/contrast changes from lighting
        tiny -= tiny.mean(axis=1, keepdims=True)
        tiny /= np.linalg.norm(tiny, axis=1, keepdims=True) + 1e-8
        return tiny

    def fit(self, crops: np.ndarray, labels: np.ndarray) -> "TinyImageClassifier":
        self._knn.fit(self._features(crops), labels)
        return self

    def predict(self, crops: np.ndarray) -> np.ndarray:
        return self._knn.predict(self._features(crops))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TinyImageClassifier":
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path} not found. Train it with `python scripts/train_baseline.py`.")
        return joblib.load(path)
