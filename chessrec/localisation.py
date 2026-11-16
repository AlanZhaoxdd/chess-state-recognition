
"""Chessboard localisation inspired by Wölflein & Arandjelović (2021).

Paper: Determining Chess Game State from an Image.
DOI: https://doi.org/10.3390/jimaging7060094

Methods:
    1. Canny and Hough line detection
    2. Agglomerative clustering of line directions
    3. DBSCAN clustering of duplicate lines
    4. RANSAC fitting of a regular chessboard grid
    5. Checkerboard verification and border refinement

This is an independent educational implementation.

The localise() method returns the four corners of the
8x8 playing area, not the wooden outer frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


_GRID = np.array(
    [[0., 0.], [8., 0.], [8., 8.], [0., 8.]],
    dtype=np.float32,
)


class ChessboardLocaliser:
    """Locate the 8x8 playing area using classical CV."""

    def __init__(
        self,
        max_side: int = 1100,
        ransac_trials: int = 1600,
        min_inliers: int = 20,
        seed: int = 3204,
    ):
        self.max_side = max_side
        self.ransac_trials = ransac_trials
        self.min_inliers = min_inliers
        self.seed = seed

    @staticmethod
    def _order(pts):
        """Order corners as TL, TR, BR, BL."""
        q = np.asarray(pts, np.float32).reshape(4, 2)
        by_y = q[np.argsort(q[:, 1], kind="stable")]

        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bottom = by_y[2:][np.argsort(by_y[2:, 0])]

        return np.array(
            [top[0], top[1], bottom[1], bottom[0]],
            dtype=np.float32,
        )

    @classmethod
    def _valid(cls, quad, shape):
        """Check whether a quadrilateral is geometrically valid."""
        h, w = shape[:2]

        if quad is None or not np.isfinite(quad).all():
            return False

        q = cls._order(quad)

        if not cv2.isContourConvex(q.reshape(4, 1, 2)):
            return False

        area = abs(cv2.contourArea(q))

        if not 0.028 * h * w <= area <= 0.95 * h * w:
            return False

        if (
            q[:, 0].min() < -0.07 * w
            or q[:, 0].max() > 1.07 * w
            or q[:, 1].min() < -0.07 * h
            or q[:, 1].max() > 1.07 * h
        ):
            return False

        lengths = np.linalg.norm(
            q - np.roll(q, -1, axis=0), axis=1
        )

        return bool(lengths.min() > 0.07 * min(w, h))

    @staticmethod
    def _sample(gray, xy):
        """Read grayscale intensities at subpixel locations."""
        pts = np.asarray(xy, np.float32).reshape(-1, 2)

        return cv2.remap(
            gray,
            pts[:, 0].reshape(1, -1),
            pts[:, 1].reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=127,
        ).reshape(-1).astype(np.float32)

    @classmethod
    def _board_score(cls, gray, corners, grad=None):
        """Measure checkerboard contrast and grid alignment."""
        if not cls._valid(corners, gray.shape):
            return -1e6

        q = cls._order(corners)

        H = cv2.getPerspectiveTransform(_GRID, q)

        # Sample four positions inside every square.
        samples = np.array(
            [
                (x + a, y + b)
                for y in range(8)
                for x in range(8)
                for a, b in (
                    (.28, .28),
                    (.72, .28),
                    (.28, .72),
                    (.72, .72),
                )
            ],
            dtype=np.float32,
        )

        xy = cv2.perspectiveTransform(
            samples.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        cells = np.median(
            cls._sample(gray, xy).reshape(8, 8, 4),
            axis=2,
        )

        parity = 1 - 2 * (
            np.indices((8, 8)).sum(axis=0) % 2
        )

        horizontal = (
            cells[:, :-1] - cells[:, 1:]
        ) * parity[:, :-1]

        vertical = (
            cells[:-1, :] - cells[1:, :]
        ) * parity[:-1, :]

        diffs = np.r_[
            horizontal.ravel(),
            vertical.ravel(),
        ]

        median = np.median(diffs)
        contrast = float(abs(median))

        consistency = float(
            np.mean(np.sign(diffs) == np.sign(median))
        )

        checker = contrast * max(
            0.0, (consistency - .48) / .52
        )

        # Evaluate internal checkerboard intersections.
        cross = np.array(
            [
                (x + a, y + b)
                for y in range(1, 8)
                for x in range(1, 8)
                for a, b in (
                    (-.22, -.22),
                    (.22, -.22),
                    (-.22, .22),
                    (.22, .22),
                )
            ],
            dtype=np.float32,
        )

        xy = cv2.perspectiveTransform(
            cross.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        z = cls._sample(gray, xy).reshape(-1, 4)

        junction = float(
            np.median(
                np.abs(z[:, 0] + z[:, 3] - z[:, 1] - z[:, 2])
            )
        )

        edge = 0.0

        if grad is not None:
            linepts = np.array(
                [
                    (x, y + f)
                    for x in range(1, 8)
                    for y in range(8)
                    for f in (.2, .5, .8)
                ]
                +
                [
                    (x + f, y)
                    for y in range(1, 8)
                    for x in range(8)
                    for f in (.2, .5, .8)
                ],
                dtype=np.float32,
            )

            xy = cv2.perspectiveTransform(
                linepts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)

            edge = float(
                np.median(cls._sample(grad, xy))
            )

        return (
            .85 * checker
            + .35 * junction
            + .04 * edge
        )

    @staticmethod
    def _line_intersection(line1, line2):
        """Find intersection of two homogeneous lines."""
        p = np.cross(line1, line2)

        if abs(p[2]) < 1e-7:
            return None

        return (p[:2] / p[2]).astype(np.float32)

    def _line_families(self, gray):
        """Find two sets of grid lines with clustering."""
        h, w = gray.shape

        blurred = cv2.GaussianBlur(
            gray, (5, 5), 0
        )

        edges = cv2.Canny(
            blurred, 50, 150
        )

        segments = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=55,
            minLineLength=max(
                24, int(.10 * min(h, w))
            ),
            maxLineGap=max(
                9, int(.022 * min(h, w))
            ),
        )

        if segments is None:
            return []

        segments = np.asarray(
            segments
        ).reshape(-1, 4)

        info = []

        for x1, y1, x2, y2 in segments:
            dx = float(x2 - x1)
            dy = float(y2 - y1)

            length = float(np.hypot(dx, dy))

            if length < .10 * min(h, w):
                continue

            angle = np.arctan2(dy, dx) % np.pi

            line = np.array(
                [dy, -dx, dx * y1 - dy * x1],
                dtype=np.float64,
            ) / length

            info.append((angle, line, length))

        if len(info) < 6:
            return []

        # Keep stronger Hough lines for clustering.
        info = sorted(
            info, key=lambda r: -r[2]
        )[:240]

        angles = np.array([item[0] for item in info])

        distance = np.abs(
            angles[:, None] - angles[None, :]
        )

        distance = np.minimum(
            distance, np.pi - distance
        )

        # Agglomerative clustering separates line orientations.
        clusters = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)

        result = []

        center = np.array(
            [w / 2, h / 2],
            dtype=np.float64,
        )

        for cid in (0, 1):
            group = [
                info[i]
                for i in range(len(info))
                if clusters[i] == cid
            ]

            if len(group) < 2:
                return []

            aa = np.array([v[0] for v in group])
            weights = np.array([v[2] for v in group])

            direction = (
                .5 * np.arctan2(
                    np.sum(weights * np.sin(2 * aa)),
                    np.sum(weights * np.cos(2 * aa)),
                )
            ) % np.pi

            normal = np.array([
                -np.sin(direction),
                np.cos(direction),
            ])

            filtered = []

            for angle, line, length in group:
                diff = abs(angle - direction)
                diff = min(diff, np.pi - diff)

                if diff > np.deg2rad(27):
                    continue

                l = line.copy()

                if l[:2] @ normal < 0:
                    l = -l

                denominator = l[:2] @ normal

                if abs(denominator) < .45:
                    continue

                t = -(
                    l[:2] @ center + l[2]
                ) / denominator

                if abs(t) > 2 * max(h, w):
                    continue

                filtered.append((t, l, length))

            if len(filtered) < 2:
                return []

            positions = np.array(
                [p[0] for p in filtered]
            ).reshape(-1, 1)

            # Merge near-duplicate lines using DBSCAN.
            ids = DBSCAN(
                eps=max(5., .012 * min(h, w)),
                min_samples=1,
            ).fit_predict(positions)

            representatives = []

            for label in np.unique(ids):
                matches = [
                    filtered[i]
                    for i in np.where(ids == label)[0]
                ]

                weights = np.array(
                    [r[2] for r in matches]
                )

                line = np.average(
                    np.stack([r[1] for r in matches]),
                    axis=0,
                    weights=weights,
                )

                norm = np.linalg.norm(line[:2])

                if norm < 1e-9:
                    continue

                line /= norm

                strength = float(np.sum(weights))

                representatives.append((
                    float(np.mean([r[0] for r in matches])),
                    line,
                    strength,
                ))

            representatives = sorted(
                representatives,
                key=lambda z: -z[2],
            )[:13]

            representatives.sort(
                key=lambda z: z[0]
            )

            result.append([
                r[1] for r in representatives
            ])

        if min(map(len, result)) < 3:
            return []

        return result

    def _contour_seeds(self, gray):
        """Generate extra quadrilateral candidates from contours."""
        h, w = gray.shape

        edges = cv2.Canny(
            gray, 55, 145
        )

        masks = [edges]

        for fraction in (.008, .018):
            width = max(
                3,
                int(min(h, w) * fraction) | 1,
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (width, width),
            )

            masks.append(
                cv2.morphologyEx(
                    edges, cv2.MORPH_CLOSE, kernel
                )
            )

            masks.append(
                cv2.dilate(edges, kernel)
            )

        quads = []

        for mask in masks:
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contours = sorted(
                contours,
                key=cv2.contourArea,
                reverse=True,
            )[:80]

            for contour in contours:
                if cv2.contourArea(contour) < .04 * h * w:
                    continue

                perimeter = cv2.arcLength(
                    contour, True
                )

                for epsilon in (.015, .03, .05):
                    poly = cv2.approxPolyDP(
                        contour,
                        epsilon * perimeter,
                        True,
                    )

                    if len(poly) != 4:
                        continue

                    q = self._order(
                        poly.reshape(4, 2)
                    )

                    if self._valid(q, gray.shape):
                        at_edge = (
                            (q[:, 0] < .02 * w)
                            | (q[:, 0] > .98 * w)
                            | (q[:, 1] < .02 * h)
                            | (q[:, 1] > .98 * h)
                        )

                        if at_edge.sum() < 2:
                            quads.append(q)

                    break

        return quads

    def _adjust_candidate(self, gray, corners, grad):
        """Optimise approximate boundaries against the grid pattern."""
        seed = self._order(corners)

        transform = cv2.getPerspectiveTransform(
            _GRID, seed
        )

        def reproject(bounds):
            left, right, top, bottom = bounds

            src = np.array(
                [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
                dtype=np.float32,
            )

            q = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2),
                transform,
            ).reshape(4, 2)

            return self._order(q)

        best = seed
        score = self._board_score(
            gray, seed, grad
        )

        bounds = np.array([
            0., 8., 0., 8.
        ])

        # Coarse position search.
        for shift in (-.9, -.5, 0., .5, .9):
            for shift2 in (-.9, -.5, 0., .5, .9):
                trial_bounds = np.array([
                    shift,
                    8 + shift,
                    shift2,
                    8 + shift2,
                ])

                trial = reproject(trial_bounds)

                value = self._board_score(
                    gray, trial, grad
                )

                if value > score:
                    best = trial
                    score = value
                    bounds = trial_bounds

        # Fine adjustment of individual borders.
        for step in (.55, .28, .14, .07, .035):
            for _ in range(3):
                improved = False

                for i in range(4):
                    for sign in (-1, 1):
                        candidate_bounds = bounds.copy()
                        candidate_bounds[i] += sign * step

                        width = candidate_bounds[1] - candidate_bounds[0]
                        height = candidate_bounds[3] - candidate_bounds[2]

                        if not (
                            6 < width < 10
                            and 6 < height < 10
                        ):
                            continue

                        if max(np.abs(candidate_bounds)) > 10.5:
                            continue

                        trial = reproject(candidate_bounds)

                        value = self._board_score(
                            gray, trial, grad
                        )

                        if value > score:
                            best = trial
                            score = value
                            bounds = candidate_bounds
                            improved = True

                if not improved:
                    break

        return best, score

    def _find_grid(self, gray, families, grad):
        """Fit a regular chessboard lattice using RANSAC."""
        fam1, fam2 = families

        points = np.zeros(
            (len(fam1), len(fam2), 2),
            dtype=np.float32,
        )

        valid = np.zeros(
            points.shape[:2],
            dtype=bool,
        )

        # Compute all intersections of the two line families.
        for i, a in enumerate(fam1):
            for j, b in enumerate(fam2):
                p = self._line_intersection(a, b)

                if p is not None and np.isfinite(p).all():
                    points[i, j] = p
                    valid[i, j] = True

        allpoints = points[valid]

        if len(allpoints) < self.min_inliers:
            return None

        rng = np.random.default_rng(self.seed)

        proposals = []
        seen = set()

        # RANSAC hypotheses built from four intersections.
        for _ in range(self.ransac_trials):
            ai, bi = sorted(
                rng.choice(len(fam1), 2, replace=False)
            )

            aj, bj = sorted(
                rng.choice(len(fam2), 2, replace=False)
            )

            if not (
                valid[ai, aj]
                and valid[ai, bj]
                and valid[bi, aj]
                and valid[bi, bj]
            ):
                continue

            da = bi - ai
            db = bj - aj

            if da > 5 or db > 5:
                continue

            key = (ai, bi, aj, bj)

            if key in seen:
                continue

            seen.add(key)

            src = np.array(
                [
                    points[ai, aj],
                    points[ai, bj],
                    points[bi, bj],
                    points[bi, aj],
                ],
                dtype=np.float32,
            )

            if abs(cv2.contourArea(src)) < .001 * gray.size:
                continue

            # Test different possible numbers of grid intervals.
            scale_options = (
                (db, da),
                (max(1, db - 1), max(1, da - 1)),
                (min(5, db + 1), min(5, da + 1)),
            )

            for sx, sy in scale_options:
                dst = np.array(
                    [
                        [0, 0],
                        [sx, 0],
                        [sx, sy],
                        [0, sy],
                    ],
                    dtype=np.float32,
                )

                H = cv2.getPerspectiveTransform(
                    src, dst
                )

                warped = cv2.perspectiveTransform(
                    allpoints.reshape(-1, 1, 2),
                    H,
                ).reshape(-1, 2)

                if not np.isfinite(warped).all():
                    continue

                rounded = np.round(warped)

                distance = np.linalg.norm(
                    warped - rounded,
                    axis=1,
                )

                # Inliers should lie near integer grid coordinates.
                mask = (
                    (distance < .18)
                    & (np.max(np.abs(warped), axis=1) < 25)
                )

                n = int(mask.sum())

                if n < self.min_inliers:
                    continue

                rounded = rounded[mask]

                xs = np.unique(rounded[:, 0])
                ys = np.unique(rounded[:, 1])

                if len(xs) < 4 or len(ys) < 4:
                    continue

                if np.ptp(xs) > 11 or np.ptp(ys) > 11:
                    continue

                score = (
                    n
                    + .6 * min(len(xs), 9)
                    * min(len(ys), 9)
                )

                proposals.append((
                    score, H, rounded
                ))

        if not proposals:
            return None

        proposals.sort(
            key=lambda item: -item[0]
        )

        best = None
        bestscore = -1e6

        # Evaluate the strongest homography candidates.
        for _, H, rounded in proposals[:18]:
            try:
                inverse = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue

            minimum = np.floor(
                np.min(rounded, axis=0)
            ).astype(int)

            maximum = np.ceil(
                np.max(rounded, axis=0)
            ).astype(int)

            # Search for missing outermost grid lines.
            xstarts = range(
                max(-15, int(maximum[0]) - 9),
                min(15, int(minimum[0]) + 1) + 1,
            )

            ystarts = range(
                max(-15, int(maximum[1]) - 9),
                min(15, int(minimum[1]) + 1) + 1,
            )

            for x0 in xstarts:
                for y0 in ystarts:
                    grid = _GRID + np.array(
                        [x0, y0],
                        dtype=np.float32,
                    )

                    q = cv2.perspectiveTransform(
                        grid.reshape(-1, 1, 2),
                        inverse,
                    ).reshape(4, 2)

                    q = self._order(q)

                    if not self._valid(q, gray.shape):
                        continue

                    boardscore = self._board_score(
                        gray, q, grad
                    )

                    in_board = (
                        (rounded[:, 0] >= x0 - .25)
                        & (rounded[:, 0] <= x0 + 8.25)
                        & (rounded[:, 1] >= y0 - .25)
                        & (rounded[:, 1] <= y0 + 8.25)
                    ).sum()

                    total = boardscore + .30 * in_board

                    if total > bestscore:
                        bestscore = total
                        best = q

        if best is None:
            return None

        return best, bestscore

    def localise(
        self,
        image: np.ndarray,
    ) -> np.ndarray | None:
        """Detect the four corners of the 8x8 playing area.

        Args:
            image: OpenCV BGR image (H, W, 3).

        Returns:
            NumPy float32 array of shape (4, 2),
            ordered TL, TR, BR, BL.

            Returns None if no reliable board is found.
        """
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) < 80
        ):
            return None

        h, w = image.shape[:2]

        factor = min(
            1.0,
            self.max_side / max(h, w),
        )

        if factor < 1:
            work = cv2.resize(
                image,
                None,
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = image

        # Step 1: Grayscale and Gaussian filtering.
        gray = cv2.cvtColor(
            work, cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray, (5, 5), 0
        )

        # Step 2: Try detecting the 7x7 internal intersections.
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                success, internal = cv2.findChessboardCornersSB(
                    gray, (7, 7)
                )

                if success:
                    known = np.array(
                        [
                            (x, y)
                            for y in range(1, 8)
                            for x in range(1, 8)
                        ],
                        dtype=np.float32,
                    )

                    H, inliers = cv2.findHomography(
                        known,
                        internal.reshape(-1, 2),
                        cv2.RANSAC,
                        3.0,
                    )

                    if (
                        H is not None
                        and int(inliers.sum()) >= 38
                    ):
                        corners = cv2.perspectiveTransform(
                            _GRID.reshape(-1, 1, 2),
                            H,
                        ).reshape(4, 2)

                        corners = self._order(corners)

                        if self._valid(corners, gray.shape):
                            return (
                                corners / factor
                            ).astype(np.float32)

            except cv2.error:
                pass

        # Step 3: Canny, Hough and line clustering.
        families = self._line_families(gray)

        # Step 4: Calculate image gradients.
        gx = cv2.Sobel(
            gray, cv2.CV_32F, 1, 0, ksize=3
        )

        gy = cv2.Sobel(
            gray, cv2.CV_32F, 0, 1, ksize=3
        )

        gradient = cv2.magnitude(gx, gy)

        proposals = []

        # Step 5: RANSAC grid fitting.
        if families:
            solution = self._find_grid(
                gray, families, gradient
            )

            if solution is not None:
                proposals.append(solution)

        # Step 6: Contour candidates as an additional fallback.
        contours = self._contour_seeds(gray)

        contours.sort(
            key=lambda q: -self._board_score(
                gray, q, gradient
            )
        )

        for candidate in contours[:6]:
            proposals.append(
                self._adjust_candidate(
                    gray, candidate, gradient
                )
            )

        if not proposals:
            return None

        # Step 7: Select the best candidate.
        corners, score = max(
            proposals,
            key=lambda item: item[1],
        )

        if score < 36:
            return None

        # Step 8: Return coordinates in the original image.
        return (
            self._order(corners) / factor
        ).astype(np.float32)
