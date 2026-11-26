"""Train the CNN occupancy and/or piece classifier by fine-tuning an ImageNet network.

Reads data/squares/<task>/{train,val} and keeps the epoch with the best val
accuracy in experiments/models/cnn_<task>[_<variant>].pt. The per-epoch
history (loss, accuracy, per-class recall, learning rate) is written to
experiments/runs/cnn_<task>[_<variant>]_history.json for the training curves.

Examples:
  python scripts/train_cnn.py --max-boards 200 --epochs 2          # quick check on a laptop
  python scripts/train_cnn.py                                      # both tasks, all data
  python scripts/train_cnn.py --task pieces --no-pretrained --variant scratch
  python scripts/train_cnn.py --task pieces --no-augment --variant noaug

Then evaluate with e.g. `python scripts/evaluate.py --pieces cnn`.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chessrec import config
from chessrec.cnn import (ARCHITECTURES, TASKS, CNNClassifier, SquareCropDataset,
                          build_network, pick_device, seed_worker, train_augmentation)
from chessrec.registry import MODELS_DIR

RUNS_DIR = config.ROOT / "experiments" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", choices=(*TASKS, "both"), default="both")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4, help="peak learning rate (AdamW)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random initialisation instead of ImageNet weights (ablation)")
    parser.add_argument("--no-augment", action="store_true",
                        help="train without data augmentation (ablation)")
    parser.add_argument("--balanced", action="store_true",
                        help="weight the loss by inverse class frequency")
    parser.add_argument("--max-boards", type=int, default=None,
                        help="use only the first N training boards (quick runs)")
    parser.add_argument("--val-max-boards", type=int, default=None,
                        help="use only the first N validation boards")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", default=None,
                        help="cuda, xpu or cpu (default: best available)")
    parser.add_argument("--amp", action="store_true",
                        help="bfloat16 mixed precision (faster on recent GPUs)")
    parser.add_argument("--variant", default="",
                        help="suffix for the output files, e.g. 'scratch' -> cnn_pieces_scratch.pt")
    return parser.parse_args()


def warmup_cosine(total_steps: int, warmup_fraction: float = 0.1):
    """Learning-rate factor per step: linear warm-up, then cosine decay to zero
    (Loshchilov & Hutter, "SGDR", ICLR 2017).

    Works for any number of steps, including runs of only a few batches.
    """
    warmup = max(1, round(warmup_fraction * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def summarise(confusion: torch.Tensor, loss: float) -> dict:
    """Loss, accuracy and per-class recall from a [true, predicted] confusion matrix."""
    confusion = confusion.double()
    per_class = (confusion.diag() / confusion.sum(dim=1)).tolist()  # nan for absent classes
    return {
        "loss": loss,
        "accuracy": float(confusion.diag().sum() / confusion.sum()),
        "mean_class_recall": float(np.nanmean(per_class)),
        "per_class_recall": per_class,
    }


def run_epoch(network: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, num_classes: int, amp: bool,
              optimiser: torch.optim.Optimizer | None = None,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> dict:
    """One pass over `loader`. Trains if an optimiser is given, else only evaluates."""
    training = optimiser is not None
    network.train(training)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total_loss = torch.zeros((), device=device)

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc="train" if training else "val", leave=False, unit="batch"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = network(x)
                loss = criterion(logits, y)
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                scheduler.step()
            # accumulate on the device; .item() every batch would stall the GPU
            total_loss += loss.detach().float() * len(y)
            confusion += torch.bincount(y * num_classes + logits.argmax(dim=1),
                                        minlength=num_classes ** 2).view(num_classes, num_classes)

    return summarise(confusion.cpu(), float(total_loss) / len(loader.dataset))


def make_loader(dataset: SquareCropDataset, args: argparse.Namespace, device: torch.device,
                shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.SEED)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, persistent_workers=args.workers > 0,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=generator)


def train_task(task: str, args: argparse.Namespace, device: torch.device) -> None:
    config.set_seed()  # per task, so each model is reproducible on its own
    class_names = TASKS[task][0]
    num_classes = len(class_names)
    suffix = f"_{args.variant}" if args.variant else ""
    model_path = MODELS_DIR / f"cnn_{task}{suffix}.pt"
    history_path = RUNS_DIR / f"cnn_{task}{suffix}_history.json"

    transform = None if args.no_augment else train_augmentation(task)
    train_set = SquareCropDataset(task, "train", args.max_boards, transform=transform)
    val_set = SquareCropDataset(task, "val", args.val_max_boards)
    train_loader = make_loader(train_set, args, device, shuffle=True)
    val_loader = make_loader(val_set, args, device, shuffle=False)

    counts = train_set.class_counts()
    print(f"\n[{task}] {len(train_set)} train / {len(val_set)} val crops on {device}")
    for name, count in zip(class_names, counts):
        print(f"  {name:<14} {count:>8}")

    weight = None
    if args.balanced:
        # inverse frequency, so every class contributes equally to the loss
        weight = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)),
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    network = build_network(args.arch, num_classes, pretrained=not args.no_pretrained).to(device)
    # AdamW (Loshchilov & Hutter, ICLR 2019) with warm-up + cosine decay (Loshchilov & Hutter, ICLR 2017)
    optimiser = torch.optim.AdamW(network.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # stepped every batch: short linear warm-up, then cosine decay towards zero
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, warmup_cosine(args.epochs * len(train_loader)))

    setup = {**vars(args), "task": task}
    history = {"setup": setup, "class_names": list(class_names),
               "train_class_counts": counts.tolist(), "epochs": []}
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        lr = scheduler.get_last_lr()[0]
        train_metrics = run_epoch(network, train_loader, criterion, device, num_classes,
                                  args.amp, optimiser, scheduler)
        val_metrics = run_epoch(network, val_loader, criterion, device, num_classes, args.amp)
        seconds = time.perf_counter() - start

        improved = val_metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = val_metrics["accuracy"]
            CNNClassifier(network, task=task, arch=args.arch, device=device,
                          meta={"setup": setup, "epoch": epoch, "val": val_metrics}
                          ).save(model_path)

        history["epochs"].append({"epoch": epoch, "lr_start": lr, "seconds": seconds,
                                  "train": train_metrics, "val": val_metrics,
                                  "saved": improved})
        # written every epoch so a long run can be inspected (or stopped) halfway
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(f"[{task}] epoch {epoch}/{args.epochs}  "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.2%}  |  "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.2%} "
              f"mean recall {val_metrics['mean_class_recall']:.2%}  "
              f"({seconds:.0f} s){'  -> saved' if improved else ''}")

    print(f"[{task}] best val accuracy {best_accuracy:.2%} -> {model_path}")
    print(f"[{task}] history -> {history_path}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    tasks = tuple(TASKS) if args.task == "both" else (args.task,)
    for task in tasks:
        train_task(task, args, device)


if __name__ == "__main__":  # required on Windows, where DataLoader workers re-import this file
    main()