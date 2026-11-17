"""Run the pipeline on a split and print/save all metrics.

Pick each stage by name from chessrec/registry.py; "oracle" uses the label.
Results are saved to experiments/runs/<name>.json. Tune on val, use test
only for the final numbers.
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from chessrec import config, registry
from chessrec.data import RenderDataset
from chessrec.evaluation import Evaluator, format_report
from chessrec.pipeline import recognise

RUNS_DIR = config.ROOT / "experiments" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="val",
                        help="folder under data/render/: train, val, test, or an "
                             "extra set such as real (default: val)")
    parser.add_argument("--localiser", choices=registry.LOCALISERS, default="oracle")
    parser.add_argument("--occupancy", choices=registry.OCCUPANCY_CLASSIFIERS, default="oracle")
    parser.add_argument("--pieces", choices=registry.PIECE_CLASSIFIERS, default="oracle")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N boards")
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N boards")
    parser.add_argument("--name", default=None,
                        help="name of the saved result (default: built from the options)")
    args = parser.parse_args()

    config.set_seed()
    localiser = registry.LOCALISERS[args.localiser]()
    occupancy = registry.OCCUPANCY_CLASSIFIERS[args.occupancy]()
    pieces = registry.PIECE_CLASSIFIERS[args.pieces]()

    samples = list(RenderDataset(args.split))[args.skip:]
    samples = samples[:args.limit]

    evaluator = Evaluator()
    for sample in tqdm(samples, desc="evaluate", unit="board"):
        image = sample.load_image()
        result = recognise(image, localiser=localiser, occupancy=occupancy,
                           pieces=pieces, truth=sample)
        evaluator.add(sample, result, image_width=image.shape[1])

    summary = evaluator.summary()
    summary["setup"] = {key: getattr(args, key) for key in
                        ("split", "localiser", "occupancy", "pieces", "skip", "limit")}
    print(f"\nsplit={args.split}  localiser={args.localiser}  "
          f"occupancy={args.occupancy}  pieces={args.pieces}\n")
    print(format_report(summary))

    name = args.name or f"{args.split}_{args.localiser}_{args.occupancy}_{args.pieces}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
