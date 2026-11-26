| # | Configuration | Kernel | C | Features | Accuracy | Recall (occupied) |
|---|---|---|---|---|---|---|
| 1 | HOG | nystroem | 10 | 1764 | 98.87% | 99.22% |
| 2 | LBP | nystroem | 10 | 160 | 96.55% | 96.88% |
| 3 | Colour | nystroem | 10 | 216 | 96.58% | 97.10% |
| 4 | HOG + LBP | nystroem | 10 | 1924 | 99.05% | 99.31% |
| 5 | HOG + colour | nystroem | 10 | 1980 | 99.13% | 99.31% |
| 6 | HOG + LBP + colour | nystroem | 10 | 2140 | 99.22% | 99.35% |
| 7 | HOG + LBP + colour, linear SVM | linear | 10 | 2140 | 99.00% | 99.00% |
| 8 | HOG + LBP + colour, HOG cell 6 | nystroem | 10 | 3292 | 99.10% | 99.44% |
| 9 | HOG + LBP + colour, HOG cell 12 | nystroem | 10 | 952 | 99.12% | 99.41% |
