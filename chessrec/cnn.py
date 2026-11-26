"""CNN occupancy and piece classifiers, fine-tuned from ImageNet weights.

The approach (separate occupancy and piece CNNs on warped square crops) follows
Wölflein & Arandjelović, "Determining Chess Game State from an Image" (J. Imaging,
2021). Backbones and ImageNet weights come from torchvision: ResNet (He et al.,
CVPR 2016) and MobileNetV3 (Howard et al., ICCV 2019).

The same code serves both tasks; only the number of classes and the crop size
differ (see TASKS). A trained network is wrapped in CNNClassifier, which has
the predict(crops) method the pipeline expects (chessrec/stages.py).

Preprocessing has to be identical in training and in the pipeline, so there is
exactly one path from crop to network input:

    uint8 BGR crop (OpenCV) --bgr_to_tensor--> float RGB in [0, 1], (N, 3, H, W)
                            --network--------> ImageNet normalisation + backbone

ImageNet normalisation is the first layer of the network itself, so it is
saved with the weights and can't be forgotten at inference time.

Training data is read with SquareCropDataset rather than torchvision's
ImageFolder for two reasons: ImageFolder loads RGB with PIL while the pipeline
passes BGR, and ImageFolder numbers classes alphabetically (black_bishop = 0),
which does not match the class ids in fen.PIECE_SYMBOLS.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from chessrec import config
from chessrec.fen import OCCUPANCY_NAMES, PIECE_NAMES

# task -> (class names in class-id order, crop size (height, width))
TASKS = {
    "occupancy": (OCCUPANCY_NAMES, config.OCCUPANCY_CROP_SIZE),
    "pieces": (PIECE_NAMES, config.PIECE_CROP_SIZE),
}

ARCHITECTURES = ("resnet18", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large")

# statistics the torchvision ImageNet weights were trained with (RGB order)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ImageNetNormalise(nn.Module):
    """Normalise RGB input in [0, 1] with the ImageNet mean and std."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def build_backbone(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """A torchvision classifier with its last layer replaced by `num_classes` outputs.

    All supported networks end in global average pooling, so they accept the
    non-square 200x100 piece crops without resizing.
    """
    from torchvision import models

    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}, choose from {ARCHITECTURES}")
    weights = "DEFAULT" if pretrained else None  # DEFAULT = best ImageNet weights
    model = getattr(models, arch)(weights=weights)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # MobileNetV3: classifier is Sequential(..., Linear)
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def build_network(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Normalisation layer followed by the backbone. Input: RGB in [0, 1]."""
    return nn.Sequential(ImageNetNormalise(), build_backbone(arch, num_classes, pretrained))


def pick_device(name: str | None = None) -> torch.device:
    """The given device, else the best available: CUDA, then Intel XPU, then CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def bgr_to_tensor(crops: np.ndarray) -> torch.Tensor:
    """uint8 BGR crops (N, H, W, 3), or one crop (H, W, 3), -> float RGB in [0, 1].

    Returns (N, 3, H, W), or (3, H, W) for a single crop.
    """
    crops = np.asarray(crops)
    if crops.shape[-1] != 3 or crops.ndim not in (3, 4):
        raise ValueError(f"Expected (N, H, W, 3) or (H, W, 3) crops, got {crops.shape}")
    rgb = np.ascontiguousarray(crops[..., ::-1])  # BGR -> RGB
    tensor = torch.from_numpy(rgb).float().div_(255)
    return tensor.movedim(-1, -3)  # channels first


def train_augmentation(task: str):
    """Random changes applied to training crops (tensors in [0, 1]).

    Colour jitter imitates the lighting changes in the renders; the small
    shift/scale covers slightly wrong board corners. Piece crops are not
    flipped: geometry.crop_piece already mirrors them so every piece leans
    the same way, and a random flip would undo that.
    """
    from torchvision.transforms import v2

    steps = [
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]
    if task == "occupancy":
        steps.insert(0, v2.RandomHorizontalFlip())
    return v2.Compose(steps)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class SquareCropDataset(Dataset):
    """Crops exported by scripts/export_squares.py, read lazily from disk.

    Class ids follow TASKS[task] (the fen.py order), not alphabetical order.
    max_boards keeps only the first N boards, with all crops of a board kept
    together, the same selection as data.load_square_crops.
    """

    def __init__(self, task: str, split: str, max_boards: int | None = None,
                 transform=None, root: Path | None = None):
        class_names, self.crop_size = TASKS[task]
        folder = Path(root or config.SQUARES_DIR) / task / split
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{folder} does not exist. Run `python scripts/export_squares.py` first.")

        files = [(path, class_id)
                 for class_id, name in enumerate(class_names)
                 for path in sorted((folder / name).glob("*.png"))]
        if max_boards is not None:
            # file names are <board id>_<square>.png
            boards = sorted({path.stem.rsplit("_", 1)[0] for path, _ in files})
            keep = set(boards[:max_boards])
            files = [(p, c) for p, c in files if p.stem.rsplit("_", 1)[0] in keep]
        if not files:
            raise FileNotFoundError(f"No crops found in {folder}")

        self.task = task
        self.class_names = class_names
        self.files = files
        self.labels = np.array([class_id for _, class_id in files])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.files[index]
        # cv2.imread can't open non-ASCII paths on Windows
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {path}")
        x = bgr_to_tensor(image)
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    def class_counts(self) -> np.ndarray:
        """Number of crops per class, in class-id order."""
        return np.bincount(self.labels, minlength=len(self.class_names))


def seed_worker(worker_id: int) -> None:
    """DataLoader worker setup: seed NumPy and stop OpenCV starting its own threads."""
    cv2.setNumThreads(0)
    np.random.seed(torch.initial_seed() % 2**32)


# --------------------------------------------------------------------------
# Pipeline wrapper
# --------------------------------------------------------------------------

class CNNClassifier:
    """A trained network with the predict(crops) interface of chessrec/stages.py."""

    FORMAT_VERSION = 1

    def __init__(self, network: nn.Module, *, task: str, arch: str,
                 device: torch.device | str | None = None, batch_size: int = 256,
                 meta: dict | None = None):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}, choose from {tuple(TASKS)}")
        self.task = task
        self.arch = arch
        self.num_classes = len(TASKS[task][0])
        self.crop_size = TASKS[task][1]
        self.batch_size = batch_size
        self.meta = meta or {}
        self.device = device if isinstance(device, torch.device) else pick_device(device)
        self.network = network.to(self.device).eval()

    def predict(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, H, W, 3) uint8 BGR from chessrec.geometry. Returns (N,) class ids."""
        crops = np.asarray(crops)
        if crops.ndim != 4 or crops.shape[1:3] != tuple(self.crop_size):
            raise ValueError(f"{self.task} crops must be (N, {self.crop_size[0]}, "
                             f"{self.crop_size[1]}, 3), got {crops.shape}")

        self.network.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                x = bgr_to_tensor(crops[start:start + self.batch_size]).to(self.device)
                predictions.append(self.network(x).argmax(dim=1).cpu())
        if not predictions:
            return np.zeros(0, dtype=np.int64)
        return torch.cat(predictions).numpy()

    def save(self, path: Path) -> None:
        """Save weights plus what is needed to rebuild the network."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": self.FORMAT_VERSION,
            "task": self.task,
            "arch": self.arch,
            "num_classes": self.num_classes,
            "state_dict": self.network.state_dict(),
            "meta": self.meta,  # plain Python types only (training args, val metrics)
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device | str | None = None) -> "CNNClassifier":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Train it with `python scripts/train_cnn.py`.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = build_network(checkpoint["arch"], checkpoint["num_classes"], pretrained=False)
        network.load_state_dict(checkpoint["state_dict"])
        return cls(network, task=checkpoint["task"], arch=checkpoint["arch"],
                   device=device, meta=checkpoint.get("meta", {}))