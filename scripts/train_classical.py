"""Train the hand-crafted feature + SVM classifiers (work package B).

Extracts the chosen features from the exported crops, fits one SVM per value
of C, keeps the one with the best mean per-class recall on the validation
crops and saves it as experiments/models/<name>_<task>.joblib. The sweep is
logged to experiments/runs/<name>_<task>_train.json.

The validation numbers printed here are a quick check for tuning. Numbers for
the report come from scripts/evaluate.py, e.g.

    python scripts/evaluate.py --pieces hog_svm

Examples::

    python scripts/train_classical.py --task pieces --features hog --C 0.01 0.1 1
    python scripts/train_classical.py --task occupancy --features hog lbp --max-boards 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from chessrec import config
from chessrec.classical import (FEATURE_TYPES, WORK_SIZES, FeatureExtractor,
                                SvmClassifier, cached_features)
from chessrec.registry import MODELS_DIR

RUNS_DIR = config.ROOT / "experiments" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", choices=("occupancy", "pieces", "both"), default="both")
    parser.add_argument("--name", default="hog_svm",
                        help="model file name; hog_svm and hog_svm_jitter are registered "
                             "for evaluate.py (default: hog_svm)")
    parser.add_argument("--features", nargs="+", choices=FEATURE_TYPES, default=["hog"])
    parser.add_argument("--kernel", choices=("linear", "rbf", "nystroem"), default="linear")
    parser.add_argument("--n-components", type=int, default=2000,
                        help="landmark samples for --kernel nystroem (default: 2000)")
    parser.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0],
                        help="values of C to try; the best on val is kept")
    parser.add_argument("--gamma", default="scale", help="RBF width (default: scale)")
    parser.add_argument("--pca", type=int, default=None, help="PCA components before the SVM")
    parser.add_argument("--hog-cell", type=int, default=8)
    parser.add_argument("--hog-orientations", type=int, default=9)
    parser.add_argument("--lbp-points", type=int, default=8)
    parser.add_argument("--lbp-radius", type=int, default=1)
    parser.add_argument("--lbp-cell", type=int, default=16)
    parser.add_argument("--colour-bins", type=int, default=4)
    parser.add_argument("--colour-grid", nargs=2, type=int, default=[3, 6], metavar=("COLS", "ROWS"),
                        help="grid of cells with their own colour histogram (default: 3 6)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--extra-squares-dir", type=Path, default=None,
                        help="also train on the crops in this folder, e.g. "
                             "data/squares_jitter8 from export_jittered_squares.py")
    parser.add_argument("--max-boards", type=int, default=None,
                        help="train on the first N boards only (default: all)")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="processes for feature extraction (default: all cores)")
    return parser.parse_args()


def train_task(task: str, args: argparse.Namespace) -> dict:
    """Fit and select the SVM for one task, save it and return the log."""
    gamma = args.gamma if args.gamma == "scale" else float(args.gamma)
    extractor = FeatureExtractor(
        features=tuple(args.features), size=WORK_SIZES[task],
        hog_cell=args.hog_cell, hog_orientations=args.hog_orientations,
        lbp_points=args.lbp_points, lbp_radius=args.lbp_radius,
        lbp_cell=args.lbp_cell, colour_bins=args.colour_bins,
        colour_grid=tuple(args.colour_grid))

    start = time.perf_counter()
    x_train, y_train = cached_features(extractor, task, args.train_split,
                                       args.max_boards, n_jobs=args.jobs)
    if args.extra_squares_dir is not None:
        x_extra, y_extra = cached_features(extractor, task, args.train_split, args.max_boards,
                                           n_jobs=args.jobs, root=args.extra_squares_dir)
        x_train, y_train = np.vstack([x_train, x_extra]), np.concatenate([y_train, y_extra])
    x_val, y_val = cached_features(extractor, task, args.val_split, n_jobs=args.jobs)
    print(f"[{task}] {len(y_train)} train crops, {len(y_val)} val crops, "
          f"{x_train.shape[1]} features ({time.perf_counter() - start:.0f}s)")

    sweep, best = [], None
    for C in args.C:
        start = time.perf_counter()
        clf = SvmClassifier(extractor, kernel=args.kernel, C=C, gamma=gamma,
                            pca=args.pca, n_components=args.n_components
                            ).fit_features(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        predicted = clf.predict_features(x_val)
        row = {"C": C,
               "val_accuracy": float(accuracy_score(y_val, predicted)),
               "val_mean_class_recall": float(balanced_accuracy_score(y_val, predicted)),
               "fit_seconds": round(fit_seconds, 1)}
        sweep.append(row)
        print(f"[{task}] C={C:<8g} val acc {row['val_accuracy']:.2%}  "
              f"mean class recall {row['val_mean_class_recall']:.2%}  ({fit_seconds:.0f}s)")
        if best is None or row["val_mean_class_recall"] > best[0]["val_mean_class_recall"]:
            best = (row, clf)

    path = MODELS_DIR / f"{args.name}_{task}.joblib"
    best[1].save(path)
    print(f"[{task}] kept C={best[0]['C']:g} -> {path}")
    return {"task": task, "name": args.name, "extractor": vars(extractor),
            "kernel": args.kernel, "gamma": args.gamma, "pca": args.pca,
            "n_components": args.n_components,
            "train_split": args.train_split, "max_boards": args.max_boards,
            "extra_squares_dir": str(args.extra_squares_dir) if args.extra_squares_dir else None,
            "n_train": len(y_train), "n_features": int(x_train.shape[1]),
            "sweep": sweep, "best_C": best[0]["C"]}


def main() -> None:
    args = parse_args()
    config.set_seed()
    tasks = ("occupancy", "pieces") if args.task == "both" else (args.task,)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        log = train_task(task, args)
        log_path = RUNS_DIR / f"{args.name}_{task}_train.json"
        log_path.write_text(json.dumps(log, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
