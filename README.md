# Chess Game State Recognition from a Single Image

COMP3204 Computer Vision — Coursework 2 (group of 4).
Due **Wednesday 9 December, 16:00**. Deliverables: `report.pdf` (CVPR format, 4 pages + 1 page references, no appendix) and `code.zip`.

## Goal

Given a photo of a chessboard, output the board position (e.g. as a FEN string).
The pipeline combines traditional computer vision (board localisation) with feature-based and deep-learning classifiers (square / piece recognition), so we can compare them directly.

## Pipeline

1. **Board localisation (traditional CV)**
   grayscale → Gaussian blur → Canny edges → Hough lines → line clustering → grid intersections → RANSAC homography → perspective warp to a top-down board
2. **Square extraction** — crop the 64 squares (with some margin for tall pieces)
3. **Occupancy classification** — empty vs occupied
4. **Piece classification** — 12 classes (6 piece types × 2 colours)
   - Baseline: HOG / LBP features + SVM
   - Deep learning: CNN with transfer learning (e.g. ResNet / MobileNet)
5. **Output** — assemble the 8×8 result into FEN

## Coursework requirements checklist

- [ ] Traditional CV techniques from lectures (filtering, Canny, Hough, homography, ...)
- [ ] Feature extraction techniques from lectures (HOG, LBP, Harris, SIFT, ...)
- [ ] Data description + visualisation
- [ ] Validation / experiments (traditional vs deep, ablations, failure cases)
- [ ] Discussion of results
- [ ] Individual contribution statement

## Data

- Synthetic rendered dataset from [chesscog](https://github.com/georg-wolflein/chesscog) (Wölflein & Arandjelović, *Determining Chess Game State from an Image*, J. Imaging 2021)
- Our own real photos as an extra test set (synthetic → real gap)

Raw data is **not** committed; put it under `data/` (git-ignored).

## Repository layout

```
data/        datasets (git-ignored)
notebooks/   exploration and visualisation
src/         pipeline code
experiments/ configs and results
report/      LaTeX report (CVPR template)
```

## Team

| Member | ECS ID | Main responsibility |
|---|---|---|
| | | Board localisation |
| | | Traditional features + SVM |
| | | CNN classifiers |
| | | Data, evaluation, report |

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```
