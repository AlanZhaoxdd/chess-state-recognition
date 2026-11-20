"""Ablations for work package B: which features, kernel and HOG cell size work best.

Every configuration is trained on the train crops; C is picked on val crops,
and the picked model is then run through the shared pipeline and evaluator on
val (oracle for the other stages), so its numbers match scripts/evaluate.py.
Results go to experiments/results/classical_ablation_<task>.{json,md}.

Features are cached in data/features/, so re-running is quick. The test split
is never touched here.

Usage::

    python scripts/classical_ablation.py --task pieces
    python scripts/classical_ablation.py --task occupancy --max-boards 1000
"""

from __future__ import annotations

import argparse
import json
import time

from tqdm import tqdm

from chessrec import config
from chessrec.classical import WORK_SIZES, FeatureExtractor, SvmClassifier, cached_features
from chessrec.data import RenderDataset
from chessrec.evaluation import Evaluator
from chessrec.pipeline import recognise

RESULTS_DIR = config.ROOT / "experiments" / "results"
COLOUR = {"colour_grid": (3, 6), "colour_bins": 4}

# (label, features, kernel, C values, extra extractor settings). C = 10 was the
# best of 1 and 10 for the full model on pieces, so the ablations reuse it.
CONFIGS = [
    ("HOG", ("hog",), "nystroem", (10,), {}),
    ("LBP", ("lbp",), "nystroem", (10,), {}),
    ("Colour", ("colour",), "nystroem", (10,), COLOUR),
    ("HOG + LBP", ("hog", "lbp"), "nystroem", (10,), {}),
    ("HOG + colour", ("hog", "colour"), "nystroem", (10,), COLOUR),
    ("HOG + LBP + colour", ("hog", "lbp", "colour"), "nystroem", (10,), COLOUR),
    ("HOG + LBP + colour, linear SVM", ("hog", "lbp", "colour"), "linear", (1, 10), COLOUR),
    ("HOG + LBP + colour, HOG cell 6", ("hog", "lbp", "colour"), "nystroem", (10,),
     {**COLOUR, "hog_cell": 6}),
    ("HOG + LBP + colour, HOG cell 12", ("hog", "lbp", "colour"), "nystroem", (10,),
     {**COLOUR, "hog_cell": 12}),
]


def evaluate_on_val(task: str, classifier: SvmClassifier, samples: list) -> dict:
    """Run the pipeline on val with this classifier and oracles elsewhere."""
    stage = {"occupancy": classifier} if task == "occupancy" else {"pieces": classifier}
    evaluator = Evaluator()
    for sample in tqdm(samples, desc="evaluate", unit="board", leave=False):
        image = sample.load_image()
        evaluator.add(sample, recognise(image, truth=sample, **stage), image_width=image.shape[1])
    summary = evaluator.summary()
    if task == "occupancy":
        occ = summary["occupancy"]
        return {"accuracy": occ["accuracy"], "precision": occ["precision"],
                "recall": occ["recall"], "confusion": occ["confusion_true_rows"]}
    pcs = summary["pieces"]
    return {"accuracy": pcs["accuracy"], "mean_class_recall": pcs["mean_class_recall"],
            "per_class_recall": pcs["per_class_recall"],
            "confusion": pcs["confusion_true_rows"],
            "boards_no_error": summary["board"]["boards_no_error"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", choices=("occupancy", "pieces"), required=True)
    parser.add_argument("--max-boards", type=int, default=None,
                        help="train on the first N train boards only (default: all)")
    parser.add_argument("--n-components", type=int, default=2000)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--only", nargs="+", type=int, default=None,
                        help="run only these config numbers (1-based)")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    config.set_seed()
    samples = list(RenderDataset("val"))
    results = []
    for number, (label, features, kernel, Cs, extra) in enumerate(CONFIGS, 1):
        if args.only and number not in args.only:
            continue
        print(f"{number}. {label} ...", flush=True)
        extractor = FeatureExtractor(features=features, size=WORK_SIZES[args.task], **extra)
        start = time.perf_counter()
        x_train, y_train = cached_features(extractor, args.task, args.train_split,
                                           args.max_boards, args.jobs)
        x_val, y_val = cached_features(extractor, args.task, "val", n_jobs=args.jobs)
        best = None
        for C in Cs:
            clf = SvmClassifier(extractor, kernel=kernel, C=C,
                                n_components=args.n_components).fit_features(x_train, y_train)
            score = (clf.predict_features(x_val) == y_val).mean()
            if best is None or score > best[0]:
                best = (score, C, clf)
        metrics = evaluate_on_val(args.task, best[2], samples)
        seconds = time.perf_counter() - start
        results.append({"config": number, "label": label, "features": features,
                        "kernel": kernel, "C": best[1], "extractor": vars(extractor),
                        "n_train": len(y_train), "n_features": int(x_train.shape[1]),
                        "seconds": round(seconds), **metrics})
        key = "mean_class_recall" if args.task == "pieces" else "recall"
        print(f"   C={best[1]:<4g} accuracy {metrics['accuracy']:.2%}  "
              f"{key.replace('_', ' ')} {metrics[key]:.2%}  ({seconds:.0f}s)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"classical_ablation_{args.task}"
    setup = {"task": args.task, "train_split": args.train_split, "max_boards": args.max_boards,
             "n_components": args.n_components, "split": "val"}
    stem.with_suffix(".json").write_text(
        json.dumps({"setup": setup, "results": results}, indent=2, default=list), encoding="utf-8")

    second = ("Mean class recall", "mean_class_recall") if args.task == "pieces" \
        else ("Recall (occupied)", "recall")
    lines = [f"| # | Configuration | Kernel | C | Features | Accuracy | {second[0]} |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['label']} | {r['kernel']} | {r['C']:g} | {r['n_features']} "
              f"| {r['accuracy']:.2%} | {r[second[1]]:.2%} |" for r in results]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {stem}.json and .md")


if __name__ == "__main__":
    main()
