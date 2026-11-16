# Chess Game State Recognition from a Single Image

## Goal

We are building a system that takes a single photo of a real chessboard, taken from an ordinary angle such as from a phone, and works out the full game state: which piece, if any, sits on each of the 64 squares. The output is the board position, written in FEN, the standard text notation for chess positions, so it can be loaded straight into a chess engine or an online board.

The problem is harder than it looks. The board appears at an angle and in perspective, lighting and backgrounds vary, pieces occlude each other, and some pieces (e.g. bishops and pawns, or kings and queens) look similar. We split the task into smaller steps (see Pipeline below) so each step can be built and tested on its own.

The coursework requires both traditional computer vision techniques and feature extraction techniques from the lectures, so the project is designed to use both:

- **Traditional CV** to find the board and correct the perspective (e.g. edge detection, line detection, geometric transforms).
- **Hand-crafted features + a classic classifier** (e.g. HOG or LBP with an SVM) to recognise the pieces.
- **A deep learning model** (a CNN) for the same recognition task, so we can compare it with the hand-crafted approach.

The final deliverables are the working code and a 4-page CVPR-style report that explains the method, evaluates each step, and compares the traditional and deep learning approaches.

## Pipeline

1. **Board localisation** — find the chessboard in the photo and warp it to a top-down view.
2. **Square extraction** — split the board into its 64 squares.
3. **Occupancy classification** — decide whether each square is empty or has a piece on it.
4. **Piece classification** — identify the type and colour of the piece on each occupied square.
5. **Output** — combine the results into the full board position.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # makes `import chessrec` work everywhere
pytest                          # should report all tests passing
```

PyTorch is installed separately so that you get the build for your GPU; see the comment in `requirements.txt`.

## Data

We use the synthetic dataset published with Wölflein & Arandjelović, *Determining Chess Game State from an Image* (J. Imaging, 2021): about 5,000 rendered photos of chess positions with varied camera angles and lighting. Every photo has a label with the position (FEN), the four board corners, which side the camera is on, and a bounding box for each piece. We keep the published train/val/test split.

```bash
python scripts/download_data.py            # val + test, about 0.75 GB
python scripts/download_data.py --splits train val test   # everything, about 7.5 GB
python scripts/export_squares.py           # cut boards into per-square crops
```

The training set is a two-part ZIP archive. Extracting it needs [7-Zip](https://www.7-zip.org) (or Info-ZIP) on the PATH; the script tells you what to do if it cannot find one.

Nothing under `data/` is committed. After the two scripts you have:

```
data/render/<split>/<id>.png + <id>.json        the photos and labels
data/squares/occupancy/<split>/<empty|occupied>/<id>_<square>.png     100x100
data/squares/pieces/<split>/<white_pawn|...|black_king>/<id>_<square>.png   200x100 (HxW)
```

The crops are cut using the ground-truth corners, so the classifiers can be built and evaluated before board localisation is finished. The class is the folder name, so `torchvision.datasets.ImageFolder` reads them directly.

## Repository layout

```
chessrec/        shared library (import this; do not copy code out of it)
  config.py      paths, crop sizes, random seed
  fen.py         class names, FEN <-> 8x8 grid
  data.py        RenderDataset / BoardSample: photos and labels
  geometry.py    perspective warp and per-square cropping
  stages.py      the three interfaces a stage implements
  pipeline.py    recognise(): photo -> position, with oracle stages
  evaluation.py  all metrics (use these, so numbers are comparable)
  registry.py    name -> implementation, used by the scripts
  baselines.py   tiny-image k-NN baseline classifier
scripts/         command-line entry points (download, export, train, evaluate, predict)
tests/           unit tests; run `pytest` before pushing
docs/TASKS.md    the four work packages
experiments/     models/ and runs/ (git-ignored), results/ (committed)
report/          LaTeX report (CVPR template)
data/            datasets (git-ignored)
```

## Conventions everyone relies on

- **Grid.** A position is an 8×8 array of FEN symbols with `"."` for empty; `grid[0, 0]` is a8 and `grid[7, 0]` is a1 (`chessrec/fen.py`).
- **Class ids.** Pieces are numbered by their index in `PIECE_SYMBOLS` (`P N B R Q K p n b r q k`); occupancy is `0 = empty`, `1 = occupied`.
- **Camera side.** One photo cannot tell which corner is a1. Like the dataset, we assume the camera is on the side of the player to move (`white_turn` in the label). `geometry.to_image_view` converts between the chess grid and what the warped image shows.
- **Crops.** Always produce crops with `chessrec.geometry` so that training data and the final pipeline see identical inputs.
- **Reproducibility.** Call `config.set_seed()` at the start of every training or evaluation script, and tune only on `val`; `test` is for the final numbers in the report.

Minimal example:

```python
from chessrec import geometry
from chessrec.data import RenderDataset

sample = RenderDataset("val")[0]
warped = geometry.warp_board(sample.load_image(), sample.corners)
occupancy_crops, piece_crops = geometry.crop_all(warped)   # indexed [row, col]
labels = geometry.to_image_view(sample.grid, sample.white_turn)
```

## Running and evaluating the pipeline

```bash
python scripts/evaluate.py                                  # every stage oracle: must print 100%
python scripts/train_baseline.py                            # fit the tiny-image k-NN baseline
python scripts/evaluate.py --occupancy tiny --pieces tiny   # baseline classifiers, oracle corners
python scripts/predict.py photo.jpg --localiser ... --occupancy ... --pieces ...   # one photo -> FEN
```

Each stage is chosen by name. `oracle` reads that stage's answer from the labels, so the other stages can be measured in isolation:

| Localiser | Occupancy | Pieces | What it measures |
|---|---|---|---|
| oracle | *yours* | oracle | the occupancy classifier alone |
| oracle | oracle | *yours* | the piece classifier alone |
| *yours* | oracle | oracle | how much localisation errors cost |
| *yours* | *yours* | *yours* | the complete system |

`evaluate.py` prints localisation, occupancy, piece and whole-board metrics (defined in `chessrec/evaluation.py`) and saves them to `experiments/runs/<name>.json`.

### Adding your implementation

1. Write a class with the method from `chessrec/stages.py`: `localise(image)` returning four corners (or `None`), or `predict(crops)` returning class ids. `chessrec/baselines.py` is a complete example of a classifier.
2. Save trained models under `experiments/models/` (git-ignored).
3. Register a name for it in `chessrec/registry.py`; it then appears as an option in every script.
4. Run `python scripts/evaluate.py` with your name for that stage and `oracle` for the others.

## Lecture techniques used

The coursework requires traditional CV and feature extraction techniques from the lectures, stated clearly in the report. Keep this table up to date as modules land.

| Stage | Technique | Kind | Status |
|---|---|---|---|
| Board localisation | Gaussian filtering, Canny edges, Hough lines | Traditional CV | planned |
| Board localisation | RANSAC, homography | Traditional CV | planned |
| Square extraction | Perspective (projective) warp | Traditional CV | done (`geometry.py`) |
| Occupancy / piece classification | Tiny images + k-nearest-neighbours (baseline) | Feature extraction | done (`baselines.py`) |
| Occupancy / piece classification | HOG, LBP + SVM | Feature extraction | planned |
| Occupancy / piece classification | CNN (transfer learning) | Deep learning | planned |

## Working together

- **What to build:** each work package, with its deliverable, how to evaluate it and what the report needs from it, is described in [docs/TASKS.md](docs/TASKS.md).
- **Branches:** work on a branch per package (`localisation`, `classical`, `cnn`, `report`) and merge into `main` through a pull request once `pytest` passes.
- **Shared code:** `chessrec/config.py`, `fen.py`, `geometry.py`, `pipeline.py` and `evaluation.py` define what everyone's results depend on. Agree in the group before changing them; a changed crop size invalidates every trained model.
- **Results:** `scripts/evaluate.py` writes to `experiments/runs/` (git-ignored). Copy the files the report cites to `experiments/results/` and commit those.
- **Code quality is marked:** give every module and function a docstring, and keep scripts runnable with one command.

## Team

| Member | ECS ID | Main responsibility |
|---|---|---|
| | | Framework (data, pipeline, evaluation) and integration |
| | | A. Board localisation |
| | | B. Hand-crafted features + SVM |
| | | C. CNN classifiers |
| | | D. Real photos, experiments and report |

The framework owner also takes one of A–D; fill in names and IDs once the split is agreed.
