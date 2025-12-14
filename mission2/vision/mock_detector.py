"""Mock vision detector for testing without camera hardware.

Provides an editable board state for testing game logic and AI.
"""

import numpy as np

from .board_detector import BOARD_COLS, BOARD_ROWS, DetectionResult
from .interface import VisionInterface


class MockVisionDetector(VisionInterface):
    """Mock vision detector that returns an editable board state.
    
    Allows manual cell editing for testing without camera.
    """

    def __init__(self):
        """Initialize mock detector with empty board."""
        self._grid: list[list[str | None]] = [
            [None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]
        self._connected = False
        self._last_detection: DetectionResult | None = None

    def connect(self) -> bool:
        """Connect (always succeeds for mock)."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def detect_board(self) -> DetectionResult:
        """Return current mock grid state as detection result."""
        orange_balls = []
        yellow_balls = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                cell = self._grid[row][col]
                x = int((col + 0.5) * 80)
                y = int((row + 0.5) * 80)
                if cell == "orange":
                    orange_balls.append({"x": x, "y": y, "radius": 30})
                elif cell == "yellow":
                    yellow_balls.append({"x": x, "y": y, "radius": 30})

        result = DetectionResult(
            board_detected=True,
            balls={"orange": orange_balls, "yellow": yellow_balls},
            board_corners=[(0, 0), (400, 0), (400, 400), (0, 400)],
            grid=[row[:] for row in self._grid],
            annotated_image=self._generate_board_image(),
            warped_image=self._generate_board_image(),
        )
        self._last_detection = result
        return result

    def get_grid(self) -> list[list[str | None]]:
        """Get current grid state."""
        return [row[:] for row in self._grid]

    def get_last_frame(self) -> np.ndarray | None:
        """Get mock frame (generated image)."""
        return self._generate_board_image()

    def get_last_detection(self) -> DetectionResult | None:
        """Get last detection result."""
        return self._last_detection

    # --- Mock-specific methods ---

    def set_cell(self, row: int, col: int, value: str | None) -> None:
        """Set a cell value manually.
        
        Args:
            row: Row index (0-4)
            col: Column index (0-4)
            value: "orange", "yellow", or None
        """
        if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
            self._grid[row][col] = value

    def toggle_cell(self, row: int, col: int) -> str | None:
        """Toggle cell through None -> orange -> yellow -> None.
        
        Returns:
            New cell value
        """
        if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
            current = self._grid[row][col]
            if current is None:
                self._grid[row][col] = "orange"
            elif current == "orange":
                self._grid[row][col] = "yellow"
            else:
                self._grid[row][col] = None
            return self._grid[row][col]
        return None

    def clear_board(self) -> None:
        """Clear all cells to empty."""
        self._grid = [
            [None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]

    def set_grid(self, grid: list[list[str | None]]) -> None:
        """Set entire grid state.
        
        Args:
            grid: 5x5 grid with "orange", "yellow", or None values
        """
        for row in range(min(BOARD_ROWS, len(grid))):
            for col in range(min(BOARD_COLS, len(grid[row]))):
                self._grid[row][col] = grid[row][col]

    def _generate_board_image(self) -> np.ndarray:
        """Generate a simple mock board image."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:, :] = (180, 100, 50)  # Blue-ish BGR

        cell_size = 80

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                cx = int((col + 0.5) * cell_size)
                cy = int((row + 0.5) * cell_size)
                cell = self._grid[row][col]

                if cell == "orange":
                    _draw_circle(img, cx, cy, 30, (0, 128, 255))
                elif cell == "yellow":
                    _draw_circle(img, cx, cy, 30, (0, 255, 255))
                else:
                    _draw_circle(img, cx, cy, 30, (40, 40, 40))

        return img


def _draw_circle(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple) -> None:
    """Draw a filled circle on image."""
    try:
        import cv2
        cv2.circle(img, (cx, cy), radius, color, -1)
    except ImportError:
        # Fallback without cv2
        y1 = max(0, cy - radius)
        y2 = min(img.shape[0], cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(img.shape[1], cx + radius)
        img[y1:y2, x1:x2] = color
