"""Board detection module for 5x5 Connect Four grid with blue board and orange/yellow balls."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


logger = logging.getLogger(__name__)

# Board configuration
BOARD_ROWS = 5
BOARD_COLS = 5

# HSV thresholds (OpenCV: H=0-179, S/V=0-255)
# The original thresholds were too strict for typical indoor lighting, resulting
# in very few ball pixels passing the mask.
BLUE_HSV_LO = np.array([100, 100, 50])
BLUE_HSV_HI = np.array([130, 255, 255])

ORANGE_HSV_LO = np.array([5, 70, 80])
ORANGE_HSV_HI = np.array([25, 255, 255])

YELLOW_HSV_LO = np.array([18, 40, 80])
YELLOW_HSV_HI = np.array([45, 255, 255])

# Often the empty holes show a red background; useful as an "empty" signal.
RED1_HSV_LO = np.array([0, 70, 50])
RED1_HSV_HI = np.array([4, 255, 255])
RED2_HSV_LO = np.array([165, 70, 50])
RED2_HSV_HI = np.array([179, 255, 255])

DEFAULT_WARP_SIZE = (400, 400)


@dataclass(frozen=True)
class HSVConfig:
    """HSV thresholds for board + ball detection (OpenCV HSV ranges)."""

    blue_lo: tuple[int, int, int] = (100, 100, 50)
    blue_hi: tuple[int, int, int] = (130, 255, 255)

    orange_lo: tuple[int, int, int] = (5, 70, 80)
    orange_hi: tuple[int, int, int] = (25, 255, 255)

    yellow_lo: tuple[int, int, int] = (18, 40, 80)
    yellow_hi: tuple[int, int, int] = (45, 255, 255)

    red1_lo: tuple[int, int, int] = (0, 70, 50)
    red1_hi: tuple[int, int, int] = (4, 255, 255)
    red2_lo: tuple[int, int, int] = (165, 70, 50)
    red2_hi: tuple[int, int, int] = (179, 255, 255)

    # Optional additional "empty/background" range to suppress false positives from
    # the material behind the holes (wood, wall, etc). When unset, only the red
    # ranges are used as an empty signal.
    empty_lo: tuple[int, int, int] | None = None
    empty_hi: tuple[int, int, int] | None = None


def _hsv_arrays(config: HSVConfig | None) -> dict[str, np.ndarray]:
    if config is None:
        return {
            "blue_lo": BLUE_HSV_LO,
            "blue_hi": BLUE_HSV_HI,
            "orange_lo": ORANGE_HSV_LO,
            "orange_hi": ORANGE_HSV_HI,
            "yellow_lo": YELLOW_HSV_LO,
            "yellow_hi": YELLOW_HSV_HI,
            "red1_lo": RED1_HSV_LO,
            "red1_hi": RED1_HSV_HI,
            "red2_lo": RED2_HSV_LO,
            "red2_hi": RED2_HSV_HI,
        }

    out = {
        "blue_lo": np.array(config.blue_lo, dtype=np.uint8),
        "blue_hi": np.array(config.blue_hi, dtype=np.uint8),
        "orange_lo": np.array(config.orange_lo, dtype=np.uint8),
        "orange_hi": np.array(config.orange_hi, dtype=np.uint8),
        "yellow_lo": np.array(config.yellow_lo, dtype=np.uint8),
        "yellow_hi": np.array(config.yellow_hi, dtype=np.uint8),
        "red1_lo": np.array(config.red1_lo, dtype=np.uint8),
        "red1_hi": np.array(config.red1_hi, dtype=np.uint8),
        "red2_lo": np.array(config.red2_lo, dtype=np.uint8),
        "red2_hi": np.array(config.red2_hi, dtype=np.uint8),
    }
    if config.empty_lo is not None and config.empty_hi is not None:
        out["empty_lo"] = np.array(config.empty_lo, dtype=np.uint8)
        out["empty_hi"] = np.array(config.empty_hi, dtype=np.uint8)
    return out


@dataclass
class DetectionResult:
    """Result of board detection."""
    board_detected: bool = False
    balls: dict = field(default_factory=lambda: {"orange": [], "yellow": []})
    board_corners: list = field(default_factory=list)
    grid: list = field(default_factory=lambda: [[None] * BOARD_COLS for _ in range(BOARD_ROWS)])
    annotated_image: np.ndarray | None = None
    warped_image: np.ndarray | None = None
    warped_annotated_image: np.ndarray | None = None
    cell_debug: list[list[dict]] | None = None
    error: str | None = None


def detect_blue_board(
    hsv: np.ndarray,
    debug: bool = False,
    config: HSVConfig | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Detect blue board and return contour + mask.
    
    Args:
        hsv: Image in HSV color space
        debug: If True, save debug masks to outputs/stream/
        
    Returns:
        Tuple of (largest_contour, blue_mask). Contour is None if not found.
    """
    hsv_cfg = _hsv_arrays(config)
    blue_mask = cv2.inRange(hsv, hsv_cfg["blue_lo"], hsv_cfg["blue_hi"])

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    if debug:
        os.makedirs("outputs/stream", exist_ok=True)
        cv2.imwrite("outputs/stream/debug_blue_mask.jpg", blue_mask)
        logger.debug("Saved blue mask to outputs/stream/debug_blue_mask.jpg")

    # Find largest contour (the board)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, blue_mask

    largest = max(contours, key=cv2.contourArea)
    return largest, blue_mask


def detect_balls(
    hsv: np.ndarray,
    mask_region: np.ndarray | None = None,
    debug: bool = False,
    config: HSVConfig | None = None,
) -> dict:
    """Detect orange and yellow balls.
    
    Note: Blue board has holes - we detect orange and yellow balls only.
    
    Args:
        hsv: Image in HSV color space
        mask_region: Optional mask to limit detection area (e.g., board region)
        debug: If True, save debug masks
        
    Returns:
        Dict with "orange" and "yellow" lists of ball dicts {"x", "y", "radius"}
    """
    balls = {"orange": [], "yellow": []}

    hsv_cfg = _hsv_arrays(config)
    orange_mask = cv2.inRange(hsv, hsv_cfg["orange_lo"], hsv_cfg["orange_hi"])
    yellow_mask = cv2.inRange(hsv, hsv_cfg["yellow_lo"], hsv_cfg["yellow_hi"])

    # Apply region mask if provided
    if mask_region is not None:
        orange_mask = cv2.bitwise_and(orange_mask, mask_region)
        yellow_mask = cv2.bitwise_and(yellow_mask, mask_region)

    # Clean up masks
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    if debug:
        os.makedirs("outputs/stream", exist_ok=True)
        cv2.imwrite("outputs/stream/debug_orange_mask.jpg", orange_mask)
        cv2.imwrite("outputs/stream/debug_yellow_mask.jpg", yellow_mask)
        logger.debug("Saved ball masks to outputs/stream/")

    # Find ball contours
    for color, mask in [("orange", orange_mask), ("yellow", yellow_mask)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 60:  # More tolerant for partial occlusion
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Balls can be partially occluded by the board holes; relax circularity.
                    if radius > 6 and circularity > 0.3:
                        balls[color].append({"x": int(x), "y": int(y), "radius": int(radius)})

    return balls


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)



def get_board_corners(board_contour: np.ndarray) -> np.ndarray | None:
    """Extract 4 corner points from board contour.
    
    Returns:
        Array of 4 corner points [top-left, top-right, bottom-right, bottom-left] or None
    """
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    approx = cv2.approxPolyDP(board_contour, epsilon, True)

    if len(approx) != 4:
        rect = cv2.minAreaRect(board_contour)
        approx = cv2.boxPoints(rect)

    pts = approx.reshape(4, 2).astype(np.float32)
    return _order_points(pts)


def get_perspective_transform(
    board_contour: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_WARP_SIZE,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Get perspective transform matrix from board contour to square view.
    
    Args:
        board_contour: Board contour points
        target_size: Output size (width, height)
        
    Returns:
        Tuple of (transform_matrix, source_corners)
    """
    corners = get_board_corners(board_contour)
    if corners is None:
        return None, None

    dst_pts = np.array([
        [0, 0],
        [target_size[0] - 1, 0],
        [target_size[0] - 1, target_size[1] - 1],
        [0, target_size[1] - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, dst_pts)
    return matrix, corners


def get_perspective_transform_from_corners(
    corners: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_WARP_SIZE,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Get perspective transform matrix from 4 board corners to square view."""
    if corners is None or len(corners) != 4:
        return None, None

    ordered = _order_points(np.asarray(corners, dtype=np.float32))
    dst_pts = np.array(
        [
            [0, 0],
            [target_size[0] - 1, 0],
            [target_size[0] - 1, target_size[1] - 1],
            [0, target_size[1] - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
    return matrix, ordered


def balls_to_grid(
    balls: dict,
    corners: np.ndarray | None,
    target_size: tuple[int, int] = DEFAULT_WARP_SIZE,
) -> list[list[str | None]]:
    """Convert ball positions to 5x5 grid coordinates.
    
    Args:
        balls: Dict with "orange" and "yellow" ball lists
        corners: 4 corner points of the board
        
    Returns:
        5x5 grid with "orange", "yellow", or None values
    """
    grid = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

    if corners is None or len(corners) != 4:
        return grid

    dst_pts = np.array(
        [
            [0, 0],
            [target_size[0] - 1, 0],
            [target_size[0] - 1, target_size[1] - 1],
            [0, target_size[1] - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)

    cell_width = target_size[0] / BOARD_COLS
    cell_height = target_size[1] / BOARD_ROWS

    for color in ["orange", "yellow"]:
        points = balls.get(color, [])
        if not points:
            continue

        src = np.array([[[b["x"], b["y"]]] for b in points], dtype=np.float32)
        warped = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
        for x, y in warped:
            col = int(x / cell_width)
            row = int(y / cell_height)
            col = max(0, min(BOARD_COLS - 1, col))
            row = max(0, min(BOARD_ROWS - 1, row))
            grid[row][col] = color

    return grid


def infer_grid_from_warped(
    warped_bgr: np.ndarray,
    rows: int = BOARD_ROWS,
    cols: int = BOARD_COLS,
    config: HSVConfig | None = None,
    min_cell_frac: float = 0.11,
    cell_radius_frac: float = 0.33,
) -> tuple[list[list[str | None]], dict[str, list[tuple[int, int, int]]], list[list[dict]]]:
    """Infer grid state directly from a top-down warped image.

    Instead of relying on contour/circularity detection (which is brittle with
    partial occlusions and lighting), sample the expected hole locations and
    classify the dominant ball color inside each cell.
    """
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    hsv_cfg = _hsv_arrays(config)
    height, width = hsv.shape[:2]
    cell_w = width / cols
    cell_h = height / rows
    radius = max(3, int(cell_radius_frac * min(cell_w, cell_h)))

    grid: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    cell_debug: list[list[dict]] = [[{} for _ in range(cols)] for _ in range(rows)]
    centers: dict[str, list[tuple[int, int, int]]] = {"orange": [], "yellow": []}

    for row in range(rows):
        for col in range(cols):
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)

            x0 = max(0, cx - radius)
            y0 = max(0, cy - radius)
            x1 = min(width, cx + radius)
            y1 = min(height, cy + radius)

            roi = hsv[y0:y1, x0:x1]
            circle = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(circle, (cx - x0, cy - y0), radius, 255, -1)

            orange = cv2.inRange(roi, hsv_cfg["orange_lo"], hsv_cfg["orange_hi"])
            yellow = cv2.inRange(roi, hsv_cfg["yellow_lo"], hsv_cfg["yellow_hi"])
            red1 = cv2.inRange(roi, hsv_cfg["red1_lo"], hsv_cfg["red1_hi"])
            red2 = cv2.inRange(roi, hsv_cfg["red2_lo"], hsv_cfg["red2_hi"])
            red = cv2.bitwise_or(red1, red2)
            empty_custom = None
            if "empty_lo" in hsv_cfg and "empty_hi" in hsv_cfg:
                empty_custom = cv2.inRange(roi, hsv_cfg["empty_lo"], hsv_cfg["empty_hi"])

            orange = cv2.bitwise_and(orange, circle)
            yellow = cv2.bitwise_and(yellow, circle)
            red = cv2.bitwise_and(red, circle)
            if empty_custom is not None:
                empty_custom = cv2.bitwise_and(empty_custom, circle)

            circle_area = int(cv2.countNonZero(circle))
            if circle_area <= 0:
                continue

            orange_n = int(cv2.countNonZero(orange))
            yellow_n = int(cv2.countNonZero(yellow))
            # If "empty" (red background) overlaps with orange/yellow ranges under current lighting,
            # don't let those pixels vote for empty.
            ball_union = cv2.bitwise_or(orange, yellow)
            empty = red
            if empty_custom is not None:
                empty = cv2.bitwise_or(empty, empty_custom)
            empty = cv2.bitwise_and(empty, cv2.bitwise_not(ball_union))
            empty_n = int(cv2.countNonZero(empty))

            # Require a minimum fraction of pixels to be classified as ball.
            min_pixels = int(circle_area * min_cell_frac)

            decision = None
            decision_reason = "none"
            mean_h: float | None = None
            boundary_h: int | None = None
            g_ratio: float | None = None

            # Empty holes often show red background which can also overlap with "orange".
            # Prefer empty when it ties or wins.
            if empty_n > max(min_pixels, orange_n, yellow_n):
                decision = None
                decision_reason = "empty>orange,yellow"
            # If both orange+yellow are present, resolve by mean hue using a boundary derived
            # from the configured HSV ranges (more stable than picking the larger mask).
            elif orange_n > min_pixels and yellow_n > min_pixels and max(orange_n, yellow_n) > empty_n:
                union = cv2.bitwise_or(orange, yellow)
                if cv2.countNonZero(union) > 0:
                    hue = roi[:, :, 0]
                    # Median is less sensitive to glare/shadows than mean.
                    mean_h = float(np.median(hue[union > 0]))
                    orange_hi_h = int(hsv_cfg["orange_hi"][0])
                    yellow_lo_h = int(hsv_cfg["yellow_lo"][0])
                    boundary_h = int(round((orange_hi_h + yellow_lo_h) / 2.0))

                    # Extra safety: confirm overlap decisions using relative RGB.
                    # Yellow tends to have higher G/R than orange under most lighting.
                    try:
                        roi_bgr = warped_bgr[y0:y1, x0:x1]
                        bgr_vals = roi_bgr[union > 0].reshape(-1, 3).astype(np.float32)
                        if bgr_vals.size:
                            b, g, r = (float(x) for x in bgr_vals.mean(axis=0))
                            g_ratio = float(g / (r + 1e-6))
                    except Exception:
                        g_ratio = None

                    tentative = "yellow" if mean_h >= boundary_h else "orange"
                    ratio_thr = 0.80
                    ratio_band = 0.03
                    if g_ratio is None:
                        decision = tentative
                    elif tentative == "yellow" and g_ratio >= (ratio_thr - ratio_band):
                        decision = "yellow"
                    elif tentative == "orange" and g_ratio <= (ratio_thr + ratio_band):
                        decision = "orange"
                    elif g_ratio >= (ratio_thr + 0.05):
                        decision = "yellow"
                    elif g_ratio <= (ratio_thr - 0.05):
                        decision = "orange"
                    else:
                        decision = tentative

                    if g_ratio is None:
                        decision_reason = f"mean_h={mean_h:.1f},boundary={boundary_h}"
                    else:
                        decision_reason = f"mean_h={mean_h:.1f},boundary={boundary_h},g_ratio={g_ratio:.2f}"
            elif yellow_n > max(min_pixels, orange_n) and yellow_n > empty_n:
                decision = "yellow"
                decision_reason = "yellow>orange,empty"
            elif orange_n > max(min_pixels, yellow_n) and orange_n > empty_n:
                decision = "orange"
                decision_reason = "orange>yellow,empty"

            if decision is not None:
                grid[row][col] = decision
                centers[decision].append((cx, cy, radius))

            cell_debug[row][col] = {
                "orange_px": orange_n,
                "yellow_px": yellow_n,
                "empty_px": empty_n,
                "circle_px": circle_area,
                "min_px": min_pixels,
                "decision": decision,
                "reason": decision_reason,
                "mean_h": mean_h,
                "boundary_h": boundary_h,
                "g_ratio": g_ratio,
            }

    return grid, centers, cell_debug


def infer_grid_from_warped_robust(
    warped_bgr: np.ndarray,
    rows: int = BOARD_ROWS,
    cols: int = BOARD_COLS,
    config: HSVConfig | None = None,
    *,
    cell_radius_frac: float = 0.33,
    saturation_threshold: float = 30.0,
    warm_r_over_b: float = 20.0,
    yellow_g_ratio: float = 0.80,
    hough_param1: float = 50.0,
    hough_param2: float = 30.0,
) -> tuple[list[list[str | None]], dict[str, list[tuple[int, int, int]]], list[list[dict]]]:
    """Infer grid using Hough circles + relative color (no HSV thresholds).

    This is useful when HSV thresholds are hard to maintain across lighting
    changes. It is intentionally simple and meant as an optional alternative to
    `infer_grid_from_warped()`.
    """
    height, width = warped_bgr.shape[:2]
    cell_w = width / cols
    cell_h = height / rows
    expected_radius = max(3, int(cell_radius_frac * min(cell_w, cell_h)))
    # Wider range: warped views and boards vary; Hough is sensitive to radius bounds.
    min_radius = max(2, int(expected_radius * 0.55))
    max_radius = max(min_radius + 1, int(expected_radius * 1.70))
    min_dist = float(min(cell_w, cell_h) * 0.65)

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=min_dist,
        param1=float(hough_param1),
        param2=float(hough_param2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        # Fallback: a slightly different blur + looser accumulator threshold.
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=min_dist,
            param1=float(hough_param1),
            param2=max(5.0, float(hough_param2) * 0.7),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )

    grid: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    cell_debug: list[list[dict]] = [[{} for _ in range(cols)] for _ in range(rows)]
    centers: dict[str, list[tuple[int, int, int]]] = {"orange": [], "yellow": []}

    if circles is None:
        return grid, centers, cell_debug

    circles = np.uint16(np.around(circles))

    for cx, cy, radius in circles[0, :]:
        col_idx = int(cx // cell_w)
        row_idx = int(cy // cell_h)
        if not (0 <= col_idx < cols and 0 <= row_idx < rows):
            continue

        # Prevent double-filling the same cell.
        if grid[row_idx][col_idx] is not None:
            continue

        hsv_cfg = _hsv_arrays(config)

        def classify_patch(px: int, py: int) -> tuple[str | None, float, dict]:
            half = max(2, int(radius * 0.20))
            y_low, y_high = max(0, py - half), min(height, py + half)
            x_low, x_high = max(0, px - half), min(width, px + half)
            patch = warped_bgr[y_low:y_high, x_low:x_high]
            if patch.size == 0:
                return None, 0.0, {"reason": "empty_patch"}

            avg_bgr = patch.reshape(-1, 3).mean(axis=0)
            b, g, r = float(avg_bgr[0]), float(avg_bgr[1]), float(avg_bgr[2])
            sat = max(r, g, b) - min(r, g, b)

            hsv_avg = cv2.cvtColor(
                np.uint8([[[int(b), int(g), int(r)]]]),
                cv2.COLOR_BGR2HSV,
            )[0, 0]
            h, s, v = int(hsv_avg[0]), int(hsv_avg[1]), int(hsv_avg[2])

            def in_range(lo: np.ndarray, hi: np.ndarray) -> bool:
                return (int(lo[0]) <= h <= int(hi[0])) and (int(lo[1]) <= s <= int(hi[1])) and (int(lo[2]) <= v <= int(hi[2]))

            is_empty_hsv = in_range(hsv_cfg["red1_lo"], hsv_cfg["red1_hi"]) or in_range(hsv_cfg["red2_lo"], hsv_cfg["red2_hi"])
            if "empty_lo" in hsv_cfg and "empty_hi" in hsv_cfg:
                is_empty_hsv = is_empty_hsv or in_range(hsv_cfg["empty_lo"], hsv_cfg["empty_hi"])

            if is_empty_hsv:
                return None, 0.0, {"reason": "empty_hsv", "avg_bgr": [b, g, r], "avg_hsv": [h, s, v], "saturation": sat}
            if sat < saturation_threshold:
                return None, 0.0, {"reason": "low_saturation", "avg_bgr": [b, g, r], "avg_hsv": [h, s, v], "saturation": sat}
            if r <= b + warm_r_over_b:
                return None, 0.0, {"reason": "not_warm", "avg_bgr": [b, g, r], "avg_hsv": [h, s, v], "saturation": sat}

            decision = "yellow" if g >= r * yellow_g_ratio else "orange"
            # Heuristic confidence: prioritize high saturation and high "warmth" (R-B),
            # and include some margin away from the orange/yellow boundary.
            g_ratio = (g / r) if r > 1e-6 else 0.0
            g_margin = (g_ratio - yellow_g_ratio) if decision == "yellow" else (yellow_g_ratio - g_ratio)
            score = max(0.0, (sat - saturation_threshold) / 255.0) + max(0.0, ((r - b) - warm_r_over_b) / 255.0)
            score *= (0.5 + 0.5 * max(0.0, min(1.0, g_margin / 0.20)))
            return decision, float(score), {
                "reason": "classified",
                "avg_bgr": [b, g, r],
                "avg_hsv": [h, s, v],
                "saturation": sat,
                "g_ratio": g_ratio,
                "g_margin": g_margin,
            }

        # Important: balls often sit *below* the hole center in a frontal view.
        # Evaluate a few sample points and pick the most confident.
        candidates = [
            ("center", int(cx), int(cy)),
            ("lower", int(cx), int(round(cy + 0.35 * radius))),
            ("lower2", int(cx), int(round(cy + 0.55 * radius))),
        ]

        best_decision: str | None = None
        best_score = 0.0
        best_meta: dict = {}
        best_name = "none"
        for name, px, py in candidates:
            py = max(0, min(height - 1, int(py)))
            px = max(0, min(width - 1, int(px)))
            dec, score, meta = classify_patch(px, py)
            meta["sample_point"] = name
            meta["sample_xy"] = [int(px), int(py)]
            if score > best_score and dec is not None:
                best_decision = dec
                best_score = score
                best_meta = meta
                best_name = name

        decision = best_decision
        reason = best_meta.get("reason", "none") if isinstance(best_meta, dict) else "none"
        b, g, r = (0.0, 0.0, 0.0)
        h, s, v = (0, 0, 0)
        sat = 0.0
        if isinstance(best_meta, dict):
            try:
                b, g, r = (float(x) for x in (best_meta.get("avg_bgr") or [0.0, 0.0, 0.0]))
            except Exception:
                pass
            try:
                h, s, v = (int(x) for x in (best_meta.get("avg_hsv") or [0, 0, 0]))
            except Exception:
                pass
            try:
                sat = float(best_meta.get("saturation", 0.0))
            except Exception:
                pass

        cell_debug[row_idx][col_idx] = {
            "method": "hough_relative",
            "cx": int(cx),
            "cy": int(cy),
            "radius": int(radius),
            "sample_point": best_name,
            "sample_score": float(best_score),
            "avg_hsv": [int(h), int(s), int(v)],
            "avg_bgr": [float(b), float(g), float(r)],
            "saturation": float(sat),
            "decision": decision,
            "reason": reason,
        }

        if decision is not None:
            grid[row_idx][col_idx] = decision
            centers[decision].append((int(cx), int(cy), int(radius)))

    return grid, centers, cell_debug


def infer_grid_from_warped_hybrid(
    warped_bgr: np.ndarray,
    rows: int = BOARD_ROWS,
    cols: int = BOARD_COLS,
    config: HSVConfig | None = None,
    *,
    min_cell_frac: float = 0.11,
    cell_radius_frac: float = 0.33,
    # Margin threshold (0..1) for trusting the HSV sampler over Hough when they disagree.
    sample_margin_threshold: float = 0.08,
) -> tuple[list[list[str | None]], dict[str, list[tuple[int, int, int]]], list[list[dict]]]:
    """Hybrid inference: per-cell HSV sampling + Hough/relative fallback.

    - Uses `infer_grid_from_warped()` as the primary method (deterministic 25 cells).
    - Runs `infer_grid_from_warped_robust()` and uses it as a fallback when the
      sampler is uncertain or when it returns empty/None but Hough is confident.
    """
    sample_grid, _, sample_debug = infer_grid_from_warped(
        warped_bgr,
        rows=rows,
        cols=cols,
        config=config,
        min_cell_frac=min_cell_frac,
        cell_radius_frac=cell_radius_frac,
    )
    hough_grid, _, hough_debug = infer_grid_from_warped_robust(
        warped_bgr,
        rows=rows,
        cols=cols,
        config=config,
        cell_radius_frac=cell_radius_frac,
    )

    height, width = warped_bgr.shape[:2]
    cell_w = width / cols
    cell_h = height / rows
    default_radius = max(3, int(cell_radius_frac * min(cell_w, cell_h)))

    final_grid: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    centers: dict[str, list[tuple[int, int, int]]] = {"orange": [], "yellow": []}

    def _sample_margin(cell: dict, decision: str | None) -> float:
        circle_px = max(1, int(cell.get("circle_px", 1)))
        orange_px = int(cell.get("orange_px", 0))
        yellow_px = int(cell.get("yellow_px", 0))
        empty_px = int(cell.get("empty_px", cell.get("red_px", 0)))
        if decision == "orange":
            chosen = orange_px
            other = max(yellow_px, empty_px)
        elif decision == "yellow":
            chosen = yellow_px
            other = max(orange_px, empty_px)
        else:
            chosen = empty_px
            other = max(orange_px, yellow_px)
        return max(0.0, float(chosen - other) / float(circle_px))

    def _hough_score(cell: dict, decision: str | None) -> float:
        if decision not in {"orange", "yellow"}:
            return 0.0
        try:
            b, g, r = (float(x) for x in (cell.get("avg_bgr") or [0.0, 0.0, 0.0]))
            sat = float(cell.get("saturation", 0.0))
        except Exception:
            return 0.0
        if r <= 1e-6:
            return 0.0
        g_ratio = g / r
        # Keep in sync with defaults in `infer_grid_from_warped_robust`.
        sat_thr = 30.0
        warm_thr = 20.0
        ratio_thr = 0.80
        margin_g = (g_ratio - ratio_thr) if decision == "yellow" else (ratio_thr - g_ratio)
        margin_sat = (sat - sat_thr) / 255.0
        margin_warm = ((r - b) - warm_thr) / 255.0
        return max(0.0, float(min(margin_g, margin_sat, margin_warm)))

    for row in range(rows):
        for col in range(cols):
            s_dec = sample_grid[row][col]
            h_dec = hough_grid[row][col]
            s_cell = sample_debug[row][col] if sample_debug else {}
            h_cell = hough_debug[row][col] if hough_debug else {}

            decision: str | None
            source = "sample_hsv"

            if s_dec == h_dec:
                decision = s_dec
                source = "agree"
            elif s_dec is None and h_dec in {"orange", "yellow"}:
                # Only override empty when the sampler isn't strongly confident it's empty.
                if _sample_margin(s_cell, None) >= sample_margin_threshold:
                    decision = None
                    source = "sample_empty"
                else:
                    decision = h_dec
                    source = "hough"
            elif s_dec in {"orange", "yellow"} and h_dec is None:
                decision = s_dec
                source = "sample_only"
            elif s_dec in {"orange", "yellow"} and h_dec in {"orange", "yellow"} and s_dec != h_dec:
                s_margin = _sample_margin(s_cell, s_dec)
                h_score = _hough_score(h_cell, h_dec)
                if s_margin >= max(sample_margin_threshold, h_score):
                    decision = s_dec
                    source = "sample_strong"
                else:
                    decision = h_dec
                    source = "hough_override"
            else:
                decision = None
                source = "none"

            final_grid[row][col] = decision

            # Keep the calibrator debug table working by retaining sampler counts
            # at the top level, but annotate with the merged decision.
            if isinstance(s_cell, dict):
                s_cell["sample_decision"] = s_dec
                s_cell["hough_decision"] = h_dec
                s_cell["hough_reason"] = h_cell.get("reason") if isinstance(h_cell, dict) else None
                s_cell["decision_source"] = source
                s_cell["decision"] = decision

            if decision in {"orange", "yellow"}:
                if isinstance(h_cell, dict) and source.startswith("hough"):
                    cx = int(h_cell.get("cx", int((col + 0.5) * cell_w)))
                    cy = int(h_cell.get("cy", int((row + 0.5) * cell_h)))
                    rad = int(h_cell.get("radius", default_radius))
                else:
                    cx = int((col + 0.5) * cell_w)
                    cy = int((row + 0.5) * cell_h)
                    rad = default_radius
                centers[decision].append((cx, cy, rad))

    return final_grid, centers, sample_debug


def grid_to_ascii(grid: list[list[str | None]]) -> str:
    """Convert grid to ASCII representation with emojis."""
    lines = []
    lines.append("  " + " ".join(str(i) for i in range(len(grid[0]))))
    lines.append("  " + "-" * (len(grid[0]) * 2 - 1))

    for i, row in enumerate(grid):
        cells = []
        for cell in row:
            if cell == "orange":
                cells.append("🟠")
            elif cell == "yellow":
                cells.append("🟡")
            else:
                cells.append("⚪")
        lines.append(f"{i}|" + "".join(cells) + "|")

    lines.append("  " + "-" * (len(grid[0]) * 2 - 1))
    return "\n".join(lines)


def run_detection(
    image_path: str,
    debug: bool = False,
    config: HSVConfig | None = None,
    manual_corners: list[tuple[int, int]] | None = None,
    min_cell_frac: float = 0.11,
    cell_radius_frac: float = 0.33,
    grid_method: str | None = None,
) -> DetectionResult:
    """Run full detection pipeline on an image file.
    
    Args:
        image_path: Path to input image
        debug: If True, save debug masks
        config: Optional HSV thresholds override
        manual_corners: Optional 4 corners (tl,tr,br,bl) in image pixels
        min_cell_frac: Minimum fraction of pixels in a cell ROI to classify a ball
        cell_radius_frac: Fraction of cell size used for the ROI radius
        
    Returns:
        DetectionResult with all detection data
    """
    result = DetectionResult()

    try:
        frame = cv2.imread(image_path)
        if frame is None:
            result.error = f"Could not load image: {image_path}"
            return result

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()

        board_contour = None
        corners = None
        matrix = None
        board_mask = None

        if manual_corners is not None and len(manual_corners) == 4:
            corners = _order_points(np.array(manual_corners, dtype=np.float32))
            result.board_detected = True
            result.board_corners = corners.tolist()

            pts = corners.astype(np.int32)
            cv2.polylines(output, [pts], True, (0, 255, 0), 2)
            board_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(board_mask, [pts], 255)

            matrix, _ = get_perspective_transform_from_corners(corners, target_size=DEFAULT_WARP_SIZE)
        else:
            # Detect blue board (auto)
            board_contour, _ = detect_blue_board(hsv, debug=debug, config=config)

            if board_contour is not None and cv2.contourArea(board_contour) > 1000:
                result.board_detected = True
                corners = get_board_corners(board_contour)
                result.board_corners = corners.tolist() if corners is not None else []

                # Draw board outline
                cv2.drawContours(output, [board_contour], -1, (0, 255, 0), 2)

                # Create board region mask (used for fallback contour-based detection)
                board_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(board_mask, [board_contour], -1, 255, -1)

                matrix, _ = get_perspective_transform(board_contour, target_size=DEFAULT_WARP_SIZE)

        if result.board_detected and matrix is not None:
            result.warped_image = cv2.warpPerspective(frame, matrix, DEFAULT_WARP_SIZE)
            method = (grid_method or os.environ.get("MISSION2_GRID_METHOD", "sample_hsv")).strip().lower()
            if method in {"hybrid"}:
                result.grid, warped_centers, result.cell_debug = infer_grid_from_warped_hybrid(
                    result.warped_image,
                    config=config,
                    min_cell_frac=min_cell_frac,
                    cell_radius_frac=cell_radius_frac,
                )
            elif method in {"hough", "hough_relative", "robust"}:
                result.grid, warped_centers, result.cell_debug = infer_grid_from_warped_robust(
                    result.warped_image,
                    config=config,
                    cell_radius_frac=cell_radius_frac,
                )
            else:
                result.grid, warped_centers, result.cell_debug = infer_grid_from_warped(
                    result.warped_image,
                    config=config,
                    min_cell_frac=min_cell_frac,
                    cell_radius_frac=cell_radius_frac,
                )

            # Annotate the warped view (grid + detected cells)
            warped_overlay = result.warped_image.copy()
            cell_w = DEFAULT_WARP_SIZE[0] / BOARD_COLS
            cell_h = DEFAULT_WARP_SIZE[1] / BOARD_ROWS
            for c in range(1, BOARD_COLS):
                x = int(round(c * cell_w))
                cv2.line(warped_overlay, (x, 0), (x, DEFAULT_WARP_SIZE[1] - 1), (255, 255, 255), 1)
            for r in range(1, BOARD_ROWS):
                y = int(round(r * cell_h))
                cv2.line(warped_overlay, (0, y), (DEFAULT_WARP_SIZE[0] - 1, y), (255, 255, 255), 1)
            for cx, cy, rad in warped_centers["orange"]:
                cv2.circle(warped_overlay, (cx, cy), rad, (0, 165, 255), 2)
            for cx, cy, rad in warped_centers["yellow"]:
                cv2.circle(warped_overlay, (cx, cy), rad, (0, 255, 255), 2)
            result.warped_annotated_image = warped_overlay

            if debug:
                os.makedirs("outputs/stream", exist_ok=True)
                cv2.imwrite("outputs/stream/debug_warped.jpg", result.warped_image)
                cv2.imwrite("outputs/stream/debug_warped_annotated.jpg", warped_overlay)
                if result.cell_debug is not None:
                    with open("outputs/stream/debug_cell_stats.json", "w", encoding="utf-8") as f:
                        json.dump(result.cell_debug, f, indent=2)

            # Back-project cell centers to original image for overlay + ball counts.
            result.balls = {"orange": [], "yellow": []}
            try:
                inv = np.linalg.inv(matrix)
                for color, bgr in [("orange", (0, 165, 255)), ("yellow", (0, 255, 255))]:
                    for cx, cy, rad in warped_centers[color]:
                        src = np.array([[[cx, cy]]], dtype=np.float32)
                        src_r = np.array([[[cx + rad, cy]]], dtype=np.float32)
                        dst = cv2.perspectiveTransform(src, inv)[0, 0]
                        dst_r = cv2.perspectiveTransform(src_r, inv)[0, 0]
                        x, y = int(dst[0]), int(dst[1])
                        radius = int(math.hypot(dst_r[0] - dst[0], dst_r[1] - dst[1]))
                        radius = max(2, radius)
                        result.balls[color].append({"x": x, "y": y, "radius": radius})
                        cv2.circle(output, (x, y), radius, bgr, 2)
                        cv2.circle(output, (x, y), 3, bgr, -1)
            except Exception as e:
                logger.warning(f"Back-projecting markers failed: {e}")
        elif result.board_detected and board_mask is not None:
            # Fallback: contour-based ball detection inside the board region
            result.balls = detect_balls(hsv, board_mask, debug=debug, config=config)
            result.grid = balls_to_grid(result.balls, corners)

            for ball in result.balls["orange"]:
                cv2.circle(output, (ball["x"], ball["y"]), ball["radius"], (0, 165, 255), 2)
                cv2.circle(output, (ball["x"], ball["y"]), 3, (0, 165, 255), -1)
            for ball in result.balls["yellow"]:
                cv2.circle(output, (ball["x"], ball["y"]), ball["radius"], (0, 255, 255), 2)
                cv2.circle(output, (ball["x"], ball["y"]), 3, (0, 255, 255), -1)
        else:
            # No board - detect balls in full frame
            result.balls = detect_balls(hsv, debug=debug, config=config)
            result.grid = balls_to_grid(result.balls, corners)

            for ball in result.balls["orange"]:
                cv2.circle(output, (ball["x"], ball["y"]), ball["radius"], (0, 165, 255), 2)
                cv2.circle(output, (ball["x"], ball["y"]), 3, (0, 165, 255), -1)
            for ball in result.balls["yellow"]:
                cv2.circle(output, (ball["x"], ball["y"]), ball["radius"], (0, 255, 255), 2)
                cv2.circle(output, (ball["x"], ball["y"]), 3, (0, 255, 255), -1)

        # If we didn't already create a warped view but the board was detected, create it.
        if result.warped_image is None and result.board_detected:
            try:
                if matrix is None:
                    if corners is not None and len(corners) == 4:
                        matrix, _ = get_perspective_transform_from_corners(corners, target_size=DEFAULT_WARP_SIZE)
                    elif board_contour is not None:
                        matrix, _ = get_perspective_transform(board_contour, target_size=DEFAULT_WARP_SIZE)
                if matrix is not None:
                    result.warped_image = cv2.warpPerspective(frame, matrix, DEFAULT_WARP_SIZE)
            except Exception as e:
                logger.warning(f"Perspective transform failed: {e}")

        # Ensure grid is always populated (fallback for any unexpected path)
        if not result.grid:
            result.grid = balls_to_grid(result.balls, corners)

        result.annotated_image = output

    except Exception as e:
        result.error = str(e)
        logger.exception("Detection failed")

    return result
