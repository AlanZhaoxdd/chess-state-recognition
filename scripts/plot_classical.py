"""Figures for the work package B part of the report.

Reads the result files in experiments/results/ and writes, next to them:

* classical_confusion_test.pdf/.png -- row-normalised confusion matrix of the
  final piece SVM on test (oracle localisation and occupancy, so only the
  piece classifier's own errors show)
* classical_ablation_pieces.pdf/.png -- mean per-class recall on val for each
  ablation configuration, with the tiny-image baseline for reference
* classical_robustness_val.pdf/.png -- piece accuracy on val as the board
  corners get noisier, for the plain and the jitter-trained SVM, with the
  accuracy reached on package A's real corners marked
* with --failures: classical_failures_test.pdf/.png -- example test crops
  for the most common mistakes (needs the trained hog_svm model and the crops)

Usage::

    python scripts/plot_classical.py
    python scripts/plot_classical.py --failures
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chessrec import config, registry  # noqa: E402
from chessrec.classical import iter_crop_batches, list_crops  # noqa: E402
from chessrec.fen import PIECE_NAMES, PIECE_SYMBOLS  # noqa: E402

RESULTS = config.ROOT / "experiments" / "results"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
FINAL = "hog_svm_jitter"  # the model package B hands over
CHOSEN = "Final: + jittered corners"

plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def load(name: str) -> dict:
    """Read one result file from experiments/results/."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    """Write the figure as PDF (for LaTeX) and PNG next to the results."""
    for suffix in (".pdf", ".png"):
        fig.savefig(RESULTS / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS / stem}.pdf and .png")


def confusion_figure() -> None:
    """Row-normalised confusion matrix of the final piece model on test."""
    counts = np.array(load(f"test_oracle_oracle_{FINAL}.json")["pieces"]["confusion_true_rows"])
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
    """Mean per-class recall on val for every ablation configuration."""
    rows = [(r["label"], r["mean_class_recall"] * 100)
            for r in load("classical_ablation_pieces.json")["results"]]
    final = load(f"{FINAL}_pieces_train.json")["sweep"][0]["val_mean_class_recall"]
    rows.append((CHOSEN, final * 100))
    rows.append(("Tiny image + k-NN (baseline)",
                 load("val_oracle_oracle_tiny.json")["pieces"]["mean_class_recall"] * 100))
    labels, values = zip(*rows)

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
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


def robustness_figure() -> None:
    """Piece accuracy on val against corner noise, for both SVMs."""
    models = [("hog_svm_jitter", "Trained with jittered corners", BLUE),
              ("hog_svm", "Trained on exact corners", ORANGE)]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, label, colour in models:
        rows = load(f"robustness_val_{name}_{name}.json")["results"]
        x = [r["mean_corner_error_px"] for r in rows]
        y = [r["pieces_accuracy"] * 100 for r in rows]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=4, label=label)
        ax.text(x[-1] + 0.4, y[-1], f"{y[-1]:.1f}", va="center", color=INK, fontsize=6.5)
        real = load(f"integration_val_hough_{name}_{name}.json")
        ax.plot(real["localisation"]["mean_corner_error_px"], real["pieces"]["accuracy"] * 100,
                marker="D", ms=5, color=colour, markeredgecolor="white", markeredgewidth=0.8,
                linestyle="none")
    ax.plot([], [], marker="D", ms=5, color=MUTED, linestyle="none",
            label="With package A's corners")
    ax.set_xlabel("Mean corner error (px)")
    ax.set_ylabel("Piece accuracy on val (%)")
    ax.set_xlim(-0.5, 17.5)
    ax.yaxis.grid(True, color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "classical_robustness_val")


def top_mistakes(n: int = 6) -> list[tuple[str, str]]:
    """The n most frequent (true, predicted) piece confusions of the final model on test."""
    counts = np.array(load(f"test_oracle_oracle_{FINAL}.json")["pieces"]["confusion_true_rows"])
    np.fill_diagonal(counts, 0)
    order = np.argsort(-counts, axis=None, kind="stable")[:n]
    return [(PIECE_NAMES[i], PIECE_NAMES[j]) for i, j in zip(*np.unravel_index(order, counts.shape))]


def short(name: str) -> str:
    """white_queen -> W queen"""
    colour, piece = name.split("_")
    return f"{colour[0].upper()} {piece}"


def failure_figure(per_mistake: int = 2) -> None:
    """Example test crops for the most frequent mistakes of the final piece model."""
    mistakes = top_mistakes()
    classifier = registry.PIECE_CLASSIFIERS[FINAL]()
    paths, labels, _ = list_crops("pieces", "test")
    wanted = np.isin(labels, [PIECE_NAMES.index(t) for t, _ in mistakes])
    paths = [p for p, keep in zip(paths, wanted) if keep]
    labels = labels[wanted]
    predicted = np.concatenate([classifier.predict(batch) for batch in iter_crop_batches(paths, 500)])

    fig, axes = plt.subplots(per_mistake, len(mistakes), figsize=(6.9, 3.4 * per_mistake / 2))
    for col, (true, wrong) in enumerate(mistakes):
        hits = np.flatnonzero((labels == PIECE_NAMES.index(true))
                              & (predicted == PIECE_NAMES.index(wrong)))[:per_mistake]
        for row in range(per_mistake):
            ax = axes[row, col]
            ax.axis("off")
            if row < len(hits):
                crop = next(iter_crop_batches([paths[hits[row]]]))[0]
                ax.imshow(crop[..., ::-1])  # BGR -> RGB
        axes[0, col].set_title(f"{short(true)}\n→ {short(wrong)}", fontsize=7, color=INK)
    save(fig, "classical_failures_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--failures", action="store_true",
                        help="also draw example mistakes (needs the model and the test crops)")
    args = parser.parse_args()
    confusion_figure()
    ablation_figure()
    robustness_figure()
    if args.failures:
        failure_figure()
