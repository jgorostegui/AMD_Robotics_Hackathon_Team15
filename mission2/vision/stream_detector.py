"""Vision detector using ZMQ camera stream.

Connects to camera stream and performs board detection using board_detector.
"""

import numpy as np

from ..core.config import get_settings
from ..streaming.receiver import StreamReceiver
from .board_detector import DetectionResult, run_detection
from .calibration import Calibration, load_calibration_file
from .interface import VisionInterface


class StreamVisionDetector(VisionInterface):
    """Vision detector using ZMQ camera stream + board_detector.
    
    Wraps StreamReceiver and board_detector for hardware-based detection.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        calibration_path: str | None = None,
    ):
        """Initialize stream detector.
        
        Args:
            host: ZMQ stream host (defaults to settings)
            port: ZMQ stream port (defaults to settings)
            calibration_path: Path to calibration JSON (defaults to settings)
        """
        settings = get_settings()
        self._host = host or settings.stream.host
        self._port = port or settings.stream.port
        self._calibration_path = calibration_path or settings.vision.calibration_path
        self._output_dir = settings.stream.output_dir
        self._frame_path = settings.stream.frame_path
        self._timeout_ms = settings.stream.capture_timeout_ms

        self._receiver: StreamReceiver | None = None
        self._connected = False
        self._last_frame: np.ndarray | None = None
        self._last_detection: DetectionResult | None = None
        self._calibration: Calibration | None = None

    def connect(self) -> bool:
        """Connect to ZMQ stream."""
        try:
            self._receiver = StreamReceiver(
                host=self._host,
                port=self._port,
                output_dir=self._output_dir,
            )
            self._calibration = load_calibration_file(self._calibration_path)
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from ZMQ stream."""
        self._receiver = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._receiver is not None

    def detect_board(self) -> DetectionResult:
        """Capture frame and detect board.
        
        Returns:
            DetectionResult with grid, balls, and images
        """
        if not self.is_connected() or self._receiver is None:
            return DetectionResult(
                board_detected=False,
                error="Not connected to camera stream",
            )

        # capture_one_shot saves frame to self._frame_path automatically
        frame = self._receiver.capture_one_shot(timeout_ms=self._timeout_ms)
        if frame is None:
            error_msg = self._receiver.last_error or "Failed to capture frame (timeout)"
            return DetectionResult(
                board_detected=False,
                error=f"Capture failed: {error_msg}",
            )

        self._last_frame = frame

        config = self._calibration.to_hsv_config() if self._calibration else None
        
        manual_corners = None
        if self._calibration and self._calibration.corners:
            manual_corners = self._calibration.corners

        try:
            result = run_detection(
                self._frame_path,
                debug=False,
                config=config,
                manual_corners=manual_corners,
                min_cell_frac=self._calibration.min_cell_frac if self._calibration else 0.11,
                cell_radius_frac=self._calibration.cell_radius_frac if self._calibration else 0.33,
            )
            self._last_detection = result
            return result
        except Exception as e:
            import traceback
            return DetectionResult(
                board_detected=False,
                error=f"Detection error: {e}\n{traceback.format_exc()}",
            )

    def get_grid(self) -> list[list[str | None]]:
        """Get last detected grid."""
        if self._last_detection:
            return self._last_detection.grid
        return [[None] * 5 for _ in range(5)]

    def get_last_frame(self) -> np.ndarray | None:
        """Get last captured frame."""
        return self._last_frame

    def get_last_detection(self) -> DetectionResult | None:
        """Get last detection result."""
        return self._last_detection

    # --- Stream-specific methods ---

    def update_connection(self, host: str, port: int) -> None:
        """Update stream connection settings.
        
        Args:
            host: New ZMQ host
            port: New ZMQ port
        """
        self._host = host
        self._port = port
        if self._receiver:
            self._receiver.update_settings(host, port)

    def reload_calibration(self) -> bool:
        """Reload calibration from file.
        
        Returns:
            True if calibration loaded successfully
        """
        self._calibration = load_calibration_file(self._calibration_path)
        return self._calibration is not None

    def get_calibration(self) -> Calibration | None:
        """Get current calibration."""
        return self._calibration
