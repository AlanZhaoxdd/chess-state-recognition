# Work packages

The shared framework (data loading, cropping, pipeline, metrics) is in place. What remains is four work packages that can proceed in parallel, because each one plugs into a fixed interface and is evaluated with oracle answers for the other stages.

Read the [README](../README.md) first: setup, data, conventions and how to register an implementation are described there and not repeated here.

## Timeline

| Dates | Milestone |
|---|---|
| by 29 Nov | Each package has a first result on `val` |
| 30 Nov – 3 Dec | Full pipeline with real stages, ablations, real photos |
| 4 – 8 Dec | Report writing, code clean-up, `code.zip` |
| **9 Dec, 16:00** | Submission |

## A. Board localisation

Find the four outer corners of the 8×8 playing area in a photo using traditional CV.

- **Deliver:** `chessrec/localisation.py` with a class that has `localise(image)`, returning a (4, 2) array of corners in any order, or `None` if no board is found. Register it as `hough` in `LOCALISERS`.
- **Suggested method:** greyscale → Gaussian blur → Canny edges → Hough lines → split lines into two direction clusters → intersections → RANSAC to fit the 8×8 grid homography → corners.
- **Evaluate:** `python scripts/evaluate.py --localiser hough`. Look at the localisation block (success rate, mean corner error) and at the whole-board block, which shows what localisation errors cost when the classifiers are perfect.
- **For the report:** success rate on `val`/`test`, an ablation (e.g. with/without RANSAC, Canny thresholds), and two or three failure cases as figures.
- **Notes:**
  - The label corners are the corners of the playing area, not of the wooden border.
  - A board counts as localised if every corner is within 1% of the image width (12 px) of the truth; see `CORNER_TOLERANCE` in `chessrec/evaluation.py`.
  - The label also has a bounding box per piece (`sample.meta["pieces"]`), handy for debugging overlays.

## B. Hand-crafted features + SVM

Classify occupancy and piece type from the square crops with lecture features.

- **Deliver:** `chessrec/classical.py` (feature extraction + classifier with `predict(crops)`) and `scripts/train_classical.py`. Register as `hog_svm` in both `OCCUPANCY_CLASSIFIERS` and `PIECE_CLASSIFIERS`.
- **Suggested method:** HOG and/or LBP features (scikit-image) → standardise → SVM (scikit-learn). `chessrec/baselines.py` and `scripts/train_baseline.py` show the whole pattern in about 100 lines.
- **Evaluate:** `python scripts/evaluate.py --occupancy hog_svm` and `--pieces hog_svm` separately, then both together.
- **For the report:** HOG vs LBP vs both, effect of one or two parameters (e.g. HOG cell size), per-class recall and the confusion matrix, comparison against the tiny-image baseline and the CNN.
- **Notes:**
  - Load crops with `chessrec.data.load_square_crops`. The full training set does not fit in memory as raw images (about 280,000 occupancy crops); use `max_boards` and compute features in batches.
  - Classes are imbalanced (pawns are about nine times as common as queens; two thirds of squares are empty). Use `class_weight="balanced"` and report per-class recall.

## C. CNN classifiers

The same two classification tasks with a deep network, for comparison with B.

- **Deliver:** `chessrec/cnn.py` (model wrapper with `predict(crops)`) and `scripts/train_cnn.py`. Register as `cnn` in both classifier dictionaries; import PyTorch inside the registry function.
- **Suggested method:** transfer learning from an ImageNet model (ResNet-18 or MobileNetV3). `data/squares/` is laid out for `torchvision.datasets.ImageFolder`.
- **Evaluate:** as in B, with `cnn` as the name.
- **For the report:** accuracy and per-class recall against B, training curves, an ablation (e.g. with/without pre-training or augmentation), remaining confusions.
- **Notes:**
  - **Channel order.** The pipeline passes crops as BGR `uint8` arrays (OpenCV), while `ImageFolder` loads RGB. Convert inside `predict`, otherwise the model sees swapped colours at test time and accuracy drops for no visible reason.
  - Piece crops from the left half of the board are already mirrored by `crop_piece`; do not add horizontal-flip augmentation for pieces on top of that without checking it helps.
  - Record the seed (`config.set_seed()`), epochs and learning rate so the result can be reproduced.

## D. Real photos, experiments and report

Test how well a system trained on rendered images works on real photos, and own the report.

- **Deliver:**
  - A small real test set in `data/render/real/`: each photo with a label `<name>.json` holding `fen`, `corners` (four [x, y] points of the playing area) and `white_turn`. JPEG photos are fine. It then works with `python scripts/evaluate.py --split real ...`.
  - The data visualisation for the report (class distribution, camera angles and lighting, example crops).
  - The report in `report/` using the CVPR LaTeX template (4 pages + 1 page of references, no appendix).
- **For the report:** the synthetic-to-real gap for each stage, the final results table (from the JSON files in `experiments/results/`), failure cases, and the individual contribution statement.
- **Notes:**
  - Take photos from one player's side at an ordinary sitting angle; note which side for `white_turn`.
  - The report must state explicitly which lecture techniques are used; keep the table in the README current and copy it across.

## Rules that keep results comparable

- Tune on `val` only. Run `test` once per final system, for the numbers in the report.
- Always produce numbers with `scripts/evaluate.py`; do not re-implement metrics.
- Copy the result files that the report cites from `experiments/runs/` (git-ignored) to `experiments/results/` and commit them.
- Never commit data or trained models.
