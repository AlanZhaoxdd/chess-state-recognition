# Chess Game State Recognition from a Single Image

COMP3204 Computer Vision — Coursework 2 (group of 4).

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

## Team

| Member | ECS ID | Main responsibility |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
