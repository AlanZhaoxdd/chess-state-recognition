| # | Configuration | Kernel | C | Features | Accuracy | Mean class recall |
|---|---|---|---|---|---|---|
| 1 | HOG | nystroem | 10 | 3780 | 85.00% | 79.35% |
| 2 | LBP | nystroem | 10 | 320 | 78.73% | 73.15% |
| 3 | Colour | nystroem | 10 | 216 | 80.51% | 73.81% |
| 4 | HOG + LBP | nystroem | 10 | 4100 | 87.18% | 82.55% |
| 5 | HOG + colour | nystroem | 10 | 3996 | 89.80% | 84.67% |
| 6 | HOG + LBP + colour | nystroem | 10 | 4316 | 91.14% | 86.75% |
| 7 | HOG + LBP + colour, linear SVM | linear | 10 | 4316 | 88.77% | 83.17% |
| 8 | HOG + LBP + colour, HOG cell 6 | nystroem | 10 | 7016 | 89.96% | 85.15% |
| 9 | HOG + LBP + colour, HOG cell 12 | nystroem | 10 | 1832 | 91.39% | 87.12% |
