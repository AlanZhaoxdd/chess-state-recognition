"""Figures and a results table for the CNN part of the report.

Reads the per-epoch histories written by scripts/train_cnn.py and the
summaries written by scripts/evaluate.py from experiments/results/cnn_results/
(copy them there from experiments/runs/ first), and writes the following to the
same folder (PDF for LaTeX, PNG for a quick look):

  cnn_val_error.pdf        validation error per epoch, main model vs ablations
  cnn_train_loss.pdf       training loss per epoch, main model vs ablations
  cnn_confusion.pdf        piece confusion matrix from one evaluate.py run
  results_table_<split>.tex  occupancy / piece / whole-board metrics per method

Errors are plotted on a log scale because every model is above 99% accuracy;
on a linear scale the curves would sit on top of each other.

Usage:
  python scripts/plot_cnn.py
  python scripts/plot_cnn.py --split test --methods tiny hog_svm cnn
  python scripts/plot_cnn.py --runs-dir experiments/runs   # quick look before copying
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window; works without a display
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from chessrec import config  # noqa: E402
from chessrec.fen import PIECE_SYMBOLS  # noqa: E402

# committed results that the report cites: input JSON files and output figures
RESULTS_DIR = config.ROOT / "experiments" / "results" / "cnn_results"
TASKS = ("occupancy", "pieces")

# variant (train_cnn.py --variant) -> legend label; colours are assigned in this order
VARIANT_LABELS = {"": "ImageNet + aug. (main)", "scratch": "From scratch",
                  "noaug": "No augmentation"}
# categorical colours, fixed order, plus line style and marker so that the
# series are told apart without colour too (print, colour-blind readers)
SERIES_STYLES = [("#257de0", "-", "o"), ("#f47230", "--", "s"), ("#00b894", ":", "^")]

# evaluate.py method name -> row label in the results table
METHOD_LABELS = {"tiny": "Tiny image + $k$-NN", "hog_svm": "HOG/LBP + SVM",
                 "cnn": "CNN (ResNet-18)", "cnn_scratch": "CNN, from scratch",
                 "cnn_noaug": "CNN, no augmentation"}

TEXT, TEXT_SECONDARY, GRID = "#191919", "#666666", "#e6e6e6"
COLUMN_WIDTH = 3.3  # inches, one column of the CVPR template


def set_style() -> None:
    """Small serif text to match the CVPR template, recessive axes and grid."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT_SECONDARY, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_dir / f"{name}{suffix}")
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf (+ .png)")


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_histories(runs_dir: Path, variants: list[str]) -> dict[str, dict[str, dict]]:
    """{task: {variant: history}} for the history files that exist."""
    histories = {}
    for task in TASKS:
        for variant in variants:
            suffix = f"_{variant}" if variant else ""
            history = load_json(runs_dir / f"cnn_{task}{suffix}_history.json")
            if history is not None:
                histories.setdefault(task, {})[variant] = history
    return histories


def val_crop_count(task: str) -> int | None:
    """Number of validation crops, used to place '0 errors' on the log axis."""
    folder = config.SQUARES_DIR / task / "val"
    return sum(1 for _ in folder.rglob("*.png")) if folder.is_dir() else None


def plot_curves(histories: dict, variants: list[str], metric: str, out_dir: Path) -> None:
    """One panel per task, one line per variant. metric: 'val_error' or 'train_loss'."""
    tasks = [t for t in TASKS if t in histories]
    fig, axes = plt.subplots(1, len(tasks), figsize=(COLUMN_WIDTH * len(tasks) / 1.5, 1.9),
                             squeeze=False, constrained_layout=True)

    for ax, task in zip(axes[0], tasks):
        floor = None
        if metric == "val_error":
            # a perfect epoch has no point on a log axis, so draw it on a
            # "0 errors" line at half a crop, below every real error
            n = val_crop_count(task)
            errors = [100 * (1 - e["val"]["accuracy"])
                      for h in histories[task].values() for e in h["epochs"]]
            floor = min([100 * 0.5 / n if n else 1e-3]
                        + [e / 2 for e in errors if e > 0])
            ax.axhline(floor, color=TEXT_SECONDARY, linewidth=0.6, linestyle=(0, (1, 2)))
            ax.annotate("0 errors", (0, floor), xycoords=("axes fraction", "data"),
                        xytext=(2, 2), textcoords="offset points", ha="left",
                        va="bottom", fontsize=6, color=TEXT_SECONDARY)

        for i, variant in enumerate(v for v in variants if v in histories[task]):
            colour, linestyle, marker = SERIES_STYLES[i % len(SERIES_STYLES)]
            epochs = histories[task][variant]["epochs"]
            x = [e["epoch"] for e in epochs]
            if metric == "val_error":
                y = [max(100 * (1 - e["val"]["accuracy"]), floor) for e in epochs]
            else:
                y = [e["train"]["loss"] for e in epochs]
            ax.plot(x, y, color=colour, linestyle=linestyle, linewidth=1.5, marker=marker,
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    label=VARIANT_LABELS.get(variant, variant))

        ax.set_yscale("log")
        # plain numbers (0.1, 1, 10) rather than powers of ten
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(task.capitalize())
        ax.set_xlabel("Epoch")
        longest = max(len(h["epochs"]) for h in histories[task].values())
        ax.set_xticks(range(1, longest + 1))
        ax.set_ylabel("Val error (%)" if metric == "val_error" else "Training loss")
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(handles) > 1:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(handles),
                   frameon=False)
    save(fig, f"cnn_{metric}", out_dir)


def plot_confusion(summary: dict, name: str, out_dir: Path) -> None:
    """Piece confusion matrix: shade = row-normalised (recall), numbers = counts."""
    counts = np.array(summary["pieces"]["confusion_true_rows"])
    rows = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, rows, out=np.zeros_like(counts, dtype=float), where=rows > 0)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.92), constrained_layout=True)
    # one-hue sequential ramp, white (none) to dark blue (all of the row)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"])
    image = ax.imshow(fraction, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (t, p), count in np.ndenumerate(counts):
        if count:  # leave zero cells blank so the few mistakes stand out
            ax.text(p, t, str(count), ha="center", va="center", fontsize=5.5,
                    color="white" if fraction[t, p] > 0.55 else TEXT,
                    fontweight="bold" if t != p else "normal")
    # outline off-diagonal mistakes, which are too pale to see from shade alone
    for t, p in zip(*np.nonzero(counts)):
        if t != p:
            ax.add_patch(plt.Rectangle((p - 0.5, t - 0.5), 1, 1, fill=False,
                                       edgecolor="#ec4647", linewidth=1.0))

    ticks = range(len(PIECE_SYMBOLS))
    ax.set_xticks(ticks, PIECE_SYMBOLS)
    ax.set_yticks(ticks, PIECE_SYMBOLS)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Fraction of true class")
    colorbar.outline.set_visible(False)
    save(fig, "cnn_confusion", out_dir)
    print(f"  confusion matrix from {name}.json "
          f"({counts.sum() - np.trace(counts)} mistake(s) in {counts.sum()} crops)")


def results_table(runs_dir: Path, split: str, methods: list[str], out_dir: Path) -> None:
    """One row per method, from its single-stage and full-pipeline evaluate.py runs.

    Occupancy comes from <split>_oracle_<m>_oracle.json, pieces from
    <split>_oracle_oracle_<m>.json and whole-board numbers from
    <split>_oracle_<m>_<m>.json (oracle localisation throughout).
    """
    header = ["Method", "Occ. acc.", "Piece acc.", "Piece mean rec.",
              "Square acc.", "Boards, no error"]
    rows = []
    for method in methods:
        occupancy = load_json(runs_dir / f"{split}_oracle_{method}_oracle.json")
        pieces = load_json(runs_dir / f"{split}_oracle_oracle_{method}.json")
        full = load_json(runs_dir / f"{split}_oracle_{method}_{method}.json")
        if not (occupancy or pieces or full):
            continue

        def pct(summary, *keys):
            if summary is None:
                return "--"
            for key in keys:
                summary = summary[key]
            return f"{100 * summary:.2f}"

        rows.append([METHOD_LABELS.get(method, method),
                     pct(occupancy, "occupancy", "accuracy"),
                     pct(pieces, "pieces", "accuracy"),
                     pct(pieces, "pieces", "mean_class_recall"),
                     pct(full, "board", "square_accuracy"),
                     pct(full, "board", "boards_no_error")])
    if not rows:
        print(f"no evaluate.py results for split '{split}' in {runs_dir}; skipped the table")
        return

    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    print(f"\nResults on {split} (%, oracle localisation)")
    for row in [header, *rows]:
        print("  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    lines = [
        r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_table_{split}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} (needs \\usepackage{{booktabs}})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_DIR,
                        help="folder with the history and evaluate.py JSON files "
                             "(default: experiments/results/cnn_results)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                        help="where figures and the table go (default: same folder)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_LABELS),
                        help="train_cnn.py variants to plot; '' is the main model")
    parser.add_argument("--split", default="val", help="split for the table and confusion matrix")
    parser.add_argument("--methods", nargs="+",
                        default=["tiny", "hog_svm", "cnn", "cnn_scratch", "cnn_noaug"],
                        help="evaluate.py method names for the results table")
    parser.add_argument("--confusion", default=None,
                        help="evaluate.py result to draw (default: <split>_oracle_oracle_cnn)")
    args = parser.parse_args()

    set_style()
    histories = load_histories(args.runs_dir, args.variants)
    if histories:
        plot_curves(histories, args.variants, "val_error", args.out_dir)
        plot_curves(histories, args.variants, "train_loss", args.out_dir)
    else:
        print(f"no training histories in {args.runs_dir}; skipped the curves")

    name = args.confusion or f"{args.split}_oracle_oracle_cnn"
    summary = load_json(args.runs_dir / f"{name}.json")
    if summary is not None:
        plot_confusion(summary, name, args.out_dir)
    else:
        print(f"{name}.json not found; skipped the confusion matrix")

    results_table(args.runs_dir, args.split, args.methods, args.out_dir)


if __name__ == "__main__":
    main()