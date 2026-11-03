# Chess Game State Recognition from a Single Image

COMP3204 Computer Vision — Coursework 2 (group of 4).

## Goal

Given a photo of a chessboard, output the board position (e.g. as a FEN string).

## Pipeline

1. **Board localisation** — find the chessboard in the photo and warp it to a top-down view.
2. **Square extraction** — split the board into its 64 squares.
3. **Occupancy classification** — decide whether each square is empty or has a piece on it.
4. **Piece classification** — identify the type and colour of the piece on each occupied square.
5. **Output** — combine the results into the full board position.

## Team

| Member | ECS ID | Main responsibility |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
