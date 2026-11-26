# Chess Game State Recognition from a Single Image

## Goal

Given one image of a chessboard, find out which piece is on each of the 64 squares and output the position as FEN. We use traditional CV to find the board, hand-crafted features + SVM to classify the pieces, and a CNN for the same task so we can compare the two.

## Pipeline

1. Board localisation: find the board and warp it to a top-down view
2. Square extraction: cut the board into 64 squares
3. Occupancy classification: empty or occupied
4. Piece classification: type and colour of each piece
5. Output: combine into the board position (FEN)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # so `import chessrec` works everywhere
pytest
```

Install PyTorch separately for your GPU, see the comment in `requirements.txt`.

## Data

We use the synthetic dataset from Wölflein & Arandjelović, *Determining Chess Game State from an Image* (J. Imaging, 2021). It has about 5,000 rendered boards with different camera angles and lighting. Each label has the FEN, the four board corners, which side the camera is on, and a bounding box per piece. We keep their train/val/test split.

```bash
python scripts/download_data.py            # val + test, about 0.75 GB
python scripts/download_data.py --splits train val test   # everything, about 7.5 GB
python scripts/export_squares.py           # cut boards into per-square crops
```

The training set is a two-part zip, so you need 7-Zip (or Info-ZIP) on the PATH to extract it.

`data/` is not committed. After running the scripts you get:

```
data/render/<split>/<id>.png + <id>.json        photos and labels
data/squares/occupancy/<split>/<empty|occupied>/<id>_<square>.png     100x100
data/squares/pieces/<split>/<white_pawn|...|black_king>/<id>_<square>.png   200x100 (HxW)
```

The crops use the ground-truth corners, so the classifiers can be trained before localisation works. Folder name = class, so `ImageFolder` can load them directly.

## Repository layout

```
chessrec/        shared code (import it, don't copy it)
  config.py      paths, crop sizes, seed
  fen.py         class names, FEN <-> 8x8 grid
  data.py        RenderDataset / BoardSample
  geometry.py    perspective warp and cropping
  stages.py      interfaces for the three stages
  pipeline.py    recognise(): photo -> position
  evaluation.py  metrics
  registry.py    name -> implementation, used by the scripts
  baselines.py   tiny-image k-NN baseline
scripts/         download, export, train, evaluate, predict
tests/           run `pytest` before pushing
docs/TASKS.md    the four work packages
experiments/     models/ and runs/ (git-ignored), results/ (committed)
report/          LaTeX report (CVPR template)
data/            datasets (git-ignored)
```

## Conventions

- Grid: 8x8 array of FEN symbols, `"."` for empty. `grid[0, 0]` is a8, `grid[7, 0]` is a1.
- Class ids: index in `PIECE_SYMBOLS` (`P N B R Q K p n b r q k`). Occupancy: 0 = empty, 1 = occupied.
- Camera side: we assume the camera is on the side of the player to move (`white_turn`). `geometry.to_image_view` converts between the chess grid and the warped image.
- Crops: always make them with `chessrec.geometry` so training and the pipeline use the same input.
- Call `config.set_seed()` in every training/evaluation script. Tune on `val`, use `test` only for final numbers.

Example:

```python
from chessrec import geometry
from chessrec.data import RenderDataset

sample = RenderDataset("val")[0]
warped = geometry.warp_board(sample.load_image(), sample.corners)
occupancy_crops, piece_crops = geometry.crop_all(warped)   # indexed [row, col]
labels = geometry.to_image_view(sample.grid, sample.white_turn)
```

## Running and evaluating

```bash
python scripts/evaluate.py                                  # all oracle, should be 100%
python scripts/train_baseline.py                            # train the tiny-image baseline
python scripts/evaluate.py --occupancy tiny --pieces tiny   # baseline classifiers
python scripts/predict.py photo.jpg --localiser ... --occupancy ... --pieces ...   # one photo -> FEN
```

Each stage is picked by name. `oracle` takes the answer from the label, so you can test one stage at a time:

| Localiser | Occupancy | Pieces | Measures |
|---|---|---|---|
| oracle | *yours* | oracle | occupancy classifier |
| oracle | oracle | *yours* | piece classifier |
| *yours* | oracle | oracle | cost of localisation errors |
| *yours* | *yours* | *yours* | full system |

Results are printed and saved to `experiments/runs/<name>.json`.

### Adding your implementation

1. Write a class with `localise(image)` or `predict(crops)` as in `chessrec/stages.py`. See `chessrec/baselines.py` for an example.
2. Save trained models in `experiments/models/`.
3. Register it in `chessrec/registry.py`.
4. Run `scripts/evaluate.py` with your name for that stage and `oracle` for the rest.

## Lecture techniques used

The report has to say which lecture techniques we use, so update this table when something is done.

| Stage | Technique | Kind | Status |
|---|---|---|---|
| Board localisation | Gaussian filtering, Canny edges, Hough lines | Traditional CV | planned |
| Board localisation | RANSAC, homography | Traditional CV | planned |
| Square extraction | Perspective warp | Traditional CV | done (`geometry.py`) |
| Occupancy / pieces | Tiny images + k-NN (baseline) | Feature extraction | done (`baselines.py`) |
| Occupancy / pieces | HSV colour histograms on a grid of cells, gradient orientation histograms (HOG) | Feature extraction | done (`classical.py`) |
| Occupancy / pieces | SVM with RBF kernel, one-vs-rest for many classes | Classification (lectures) | done (`classical.py`) |
| Occupancy / pieces | LBP texture histograms, Nyström kernel approximation, equal weight per feature type, class-balanced training, corner-jitter augmentation | Self-study (not from lectures) | done (`classical.py`) |
| Occupancy / pieces | CNN (transfer learning) | Deep learning | planned |

## Working together

- What each package needs to do is in [docs/TASKS.md](docs/TASKS.md).
- One branch per package (`localisation`, `classical`, `cnn`, `report`), merge into `main` with a PR after `pytest` passes.
- Ask the group before changing `config.py`, `fen.py`, `geometry.py`, `pipeline.py` or `evaluation.py`, e.g. changing the crop size breaks every trained model.
- Copy the result files used in the report from `experiments/runs/` to `experiments/results/` and commit them.
- Code quality is marked, so add docstrings and keep scripts runnable with one command.

## Team

| Member | ECS ID | Main responsibility |
|---|---|---|
| | | Framework and integration |
| | | A. Board localisation |
| Zifan Yang | zy1d22 | B. Hand-crafted features + SVM |
| | | C. CNN classifiers |
| | | D. Experiments and report |
