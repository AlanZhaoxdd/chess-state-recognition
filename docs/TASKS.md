# Work packages

The shared framework (data loading, cropping, pipeline, metrics) is in place. What remains is four work packages that can proceed in parallel, because each one plugs into a fixed interface and is evaluated with oracle answers for the other stages.

Read the [README](../README.md) first: setup, data, conventions and how to register an implementation are described there and not repeated here.

## Timeline

| Dates | Milestone |
|---|---|
| by 29 Nov | Each package has a first result on `val` |
| 30 Nov – 3 Dec | Full pipeline with real stages, ablations |
| 4 – 8 Dec | Report writing, code clean-up, `code.zip` |
| **9 Dec, 16:00** | Submission |

## A. Board localisation

Find the four outer corners of the 8×8 playing area in a photo using traditional CV.

- **Deliver:** `chessrec/localisation.py` with a class that has `localise(image)`, returning a (4, 2) array of corners in any order, or `None` if no board is found. Register it as `hough` in `LOCALISERS`.
- **Suggested method:** greyscale → Gaussian blur → Canny edges → Hough lines → split lines into two direction clusters → intersections → RANSAC to fit the 8×8 grid homography → corners.
- **Evaluate:** `python scripts/evaluate.py --localiser hough`. Look at the localisation block (success rate, mean corner error) and at the whole-board block, which shows what localisation errors cost when the classifiers are perfect.
- **For the report:** success rate on `val`/`test`, an ablation (e.g. with/without RANSAC, Canny thresholds), and two or three failure cases as figures.

## B. Hand-crafted features + SVM

Classify occupancy and piece type from the square crops with lecture features.

- **Deliver:** `chessrec/classical.py` (feature extraction + classifier with `predict(crops)`) and `scripts/train_classical.py`. Register as `hog_svm` in both `OCCUPANCY_CLASSIFIERS` and `PIECE_CLASSIFIERS`.
- **Suggested method:** HOG and/or LBP features (scikit-image) → standardise → SVM (scikit-learn). `chessrec/baselines.py` and `scripts/train_baseline.py` show the whole pattern in about 100 lines.
- **Evaluate:** `python scripts/evaluate.py --occupancy hog_svm` and `--pieces hog_svm` separately, then both together.
- **For the report:** HOG vs LBP vs both, effect of one or two parameters (e.g. HOG cell size), per-class recall and the confusion matrix, comparison against the tiny-image baseline and the CNN.

## C. CNN classifiers

The same two classification tasks with a deep network, for comparison with B.

- **Deliver:** `chessrec/cnn.py` (model wrapper with `predict(crops)`) and `scripts/train_cnn.py`. Register as `cnn` in both classifier dictionaries; import PyTorch inside the registry function.
- **Suggested method:** transfer learning from an ImageNet model (ResNet-18 or MobileNetV3). `data/squares/` is laid out for `torchvision.datasets.ImageFolder`. The pipeline passes crops as BGR (OpenCV) while `ImageFolder` loads RGB, so convert inside `predict`.
- **Evaluate:** as in B, with `cnn` as the name.
- **For the report:** accuracy and per-class recall against B, training curves, an ablation (e.g. with/without pre-training or augmentation), remaining confusions.

## D. Experiments and report

Run the final experiments and own the report.

- **Deliver:**
  - The data visualisation for the report (class distribution, camera angles and lighting, example crops).
  - The report in `report/` using the CVPR LaTeX template (4 pages + 1 page of references, no appendix).
- **For the report:** the final results table (from the JSON files in `experiments/results/`), failure cases, which lecture techniques are used (copy the table from the README), and the individual contribution statement.

## Rules that keep results comparable

- Tune on `val` only. Run `test` once per final system, for the numbers in the report.
- Always produce numbers with `scripts/evaluate.py`; do not re-implement metrics.
- Copy the result files that the report cites from `experiments/runs/` (git-ignored) to `experiments/results/` and commit them.
- Never commit data or trained models.
