"""Figures for the work package B part of the report.

Reads the result files in experiments/results/ and writes, next to them:

* classical_confusion_test.pdf/.png -- row-normalised confusion matrix of the
  piece SVM on test (oracle localisation and occupancy, so only the piece
  classifier's own errors show)
* classical_ablation_pieces.pdf/.png -- mean per-class recall on val for each
  ablation configuration, with the tiny-image baseline for reference

Usage::

    python scripts/plot_classical.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chessrec import config  # noqa: E402
from chessrec.fen import PIECE_SYMBOLS  # noqa: E402

RESULTS = config.ROOT / "experiments" / "results"
BLUE, GREY, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
CHOSEN = "HOG + LBP + colour, HOG cell 12"

plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in (".pdf", ".png"):
        fig.savefig(RESULTS / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS / stem}.pdf and .png")


def confusion_figure() -> None:
    counts = np.array(load("test_oracle_oracle_hog_svm.json")["pieces"]["confusion_true_rows"])
    rates = counts / counts.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=100)
    for i in range(12):
        for j in range(12):
            if rates[i, j] >= 1:  # label only cells that matter
                ax.text(j, i, f"{rates[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if rates[i, j] > 60 else INK)
    ax.set_xticks(range(12), PIECE_SYMBOLS)
    ax.set_yticks(range(12), PIECE_SYMBOLS)
    ax.set_xlabel("Predicted piece")
    ax.set_ylabel("True piece")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # white/black boundary
    ax.axhline(5.5, color="white", lw=1.5)
    ax.axvline(5.5, color="white", lw=1.5)
    save(fig, "classical_confusion_test")


def ablation_figure() -> None:
    rows = [(r["label"], r["mean_class_recall"] * 100)
            for r in load("classical_ablation_pieces.json")["results"]]
    rows.append(("Tiny image + k-NN (baseline)",
                 load("val_oracle_oracle_tiny.json")["pieces"]["mean_class_recall"] * 100))
    labels, values = zip(*rows)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    y = np.arange(len(rows))[::-1]
    colours = [GREY if "baseline" in name else BLUE for name in labels]
    ax.barh(y, values, height=0.6, color=colours)
    for yi, value in zip(y, values):
        ax.text(value + 0.8, yi, f"{value:.1f}", va="center", color=INK)
    ax.set_yticks(y, labels)
    for tick, name in zip(ax.get_yticklabels(), labels):
        if name == CHOSEN:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean per-class recall on val (%)")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, "classical_ablation_pieces")


if __name__ == "__main__":
    confusion_figure()
    ablation_figure()
