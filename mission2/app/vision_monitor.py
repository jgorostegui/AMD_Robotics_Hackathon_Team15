"""Background vision monitor to decouple detection from UI rendering.

Uses a daemon thread to run heavy OpenCV detection continuously,
while the UI simply peeks at the latest result (non-blocking).
"""

import threading
import time

import cv2
import numpy as np
import streamlit as st

from mission2.core.config import get_settings
from mission2.streaming.receiver import StreamReceiver
from mission2.vision.board_detector import DetectionResult, run_detection
from mission2.vision.calibration import Calibration, load_calibration_file


class VisionMonitor:
    """Background vision monitor that runs detection in a separate thread."""

    def __init__(self, host: str | None = None, port: int | None = None):
        """Initialize the vision monitor.
        
        Args:
            host: ZMQ stream host (defaults to settings)
            port: ZMQ stream port (defaults to settings)
        """
        self.settings = get_settings()
        self._host = host or self.settings.stream.host
        self._port = port or self.settings.stream.port
        self._output_dir = self.settings.stream.output_dir
        self._frame_path = self.settings.stream.frame_path
        self._calibration_path = self.settings.vision.calibration_path
        
        # Load calibration
        self.calibration: Calibration | None = load_calibration_file(
            self._calibration_path
        )
        
        # Thread-safe shared state
        self._lock = threading.Lock()
        self._latest_result: DetectionResult | None = None
        self._latest_frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._last_update = time.time()
        self._error: str | None = None
        self._capture_count = 0
        
        # Configuration
        self._interval = 0.5  # seconds between captures
        self._enabled = False  # whether to actively capture

    def start(self) -> None:
        """Start the background detection loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background detection loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def enable(self) -> None:
        """Enable active capture and ensure thread is running."""
        self._enabled = True
        # Auto-start thread if not running
        if not self._running:
            self.start()

    def disable(self) -> None:
        """Disable active capture (thread keeps running but idle)."""
        self._enabled = False
        # Clear error when disabling
        with self._lock:
            self._error = None

    def is_enabled(self) -> bool:
        """Check if capture is enabled."""
        return self._enabled

    def set_interval(self, seconds: float) -> None:
        """Set capture interval."""
        self._interval = max(0.1, seconds)

    def update_connection(self, host: str, port: int) -> None:
        """Update stream connection settings."""
        self._host = host
        self._port = port
        # Note: receiver is created fresh each capture, so just update the stored values

    def reload_calibration(self) -> None:
        """Reload calibration from file."""
        self.calibration = load_calibration_file(self.settings.vision.calibration_path)

    def _loop(self) -> None:
        """Background processing loop."""
        while self._running:
            if not self._enabled:
                time.sleep(0.1)
                continue
            
            try:
                # Create fresh receiver for each capture (Streamlit-safe)
                receiver = StreamReceiver(
                    host=self._host,
                    port=self._port,
                    output_dir=self._output_dir,
                )
                
                # Capture frame (uses one-shot for Streamlit compatibility)
                frame = receiver.capture_one_shot(
                    timeout_ms=self.settings.stream.capture_timeout_ms
                )
                
                if frame is not None:
                    # Run detection
                    config = self.calibration.to_hsv_config() if self.calibration else None
                    manual_corners = self.calibration.corners if self.calibration else None
                    
                    result = run_detection(
                        self._frame_path,
                        debug=False,
                        config=config,
                        manual_corners=manual_corners,
                        min_cell_frac=self.calibration.min_cell_frac if self.calibration else 0.11,
                        cell_radius_frac=self.calibration.cell_radius_frac if self.calibration else 0.33,
                    )
                    
                    # Update shared state
                    now = time.time()
                    with self._lock:
                        self._latest_result = result
                        self._latest_frame = frame
                        self._fps = 1.0 / max(0.001, now - self._last_update)
                        self._last_update = now
                        self._error = None
                        self._capture_count += 1
                else:
                    with self._lock:
                        self._error = receiver.last_error or "Capture timeout"
                
            except Exception as e:
                import traceback
                with self._lock:
                    self._error = f"{e}\n{traceback.format_exc()}"
            
            # Wait for next capture
            time.sleep(self._interval)

    def get_latest(self) -> tuple[DetectionResult | None, np.ndarray | None, float]:
        """Non-blocking read of latest detection result.
        
        Returns:
            Tuple of (result, frame, fps)
        """
        with self._lock:
            return self._latest_result, self._latest_frame, self._fps

    def get_error(self) -> str | None:
        """Get last error message."""
        with self._lock:
            return self._error

    def get_grid(self) -> list[list[str | None]]:
        """Get latest detected grid."""
        with self._lock:
            if self._latest_result:
                # Return a copy to avoid race conditions
                return [row[:] for row in self._latest_result.grid]
            return [[None] * 5 for _ in range(5)]

    def get_capture_count(self) -> int:
        """Get total number of successful captures."""
        with self._lock:
            return self._capture_count

    def is_running(self) -> bool:
        """Check if background thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()


@st.cache_resource
def get_vision_monitor() -> VisionMonitor:
    """Get singleton VisionMonitor instance.
    
    Uses st.cache_resource to ensure only one monitor exists
    across all Streamlit reruns.
    """
    monitor = VisionMonitor()
    monitor.start()
    return monitor
