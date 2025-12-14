"""Vision module for board detection."""

from .board_detector import (
    DetectionResult,
    HSVConfig,
    detect_balls,
    detect_blue_board,
    get_perspective_transform,
    get_perspective_transform_from_corners,
    grid_to_ascii,
    run_detection,
)
from .calibration import Calibration, load_calibration_file, save_calibration_file
from .interface import VisionInterface
from .mock_detector import MockVisionDetector
from .stream_detector import StreamVisionDetector


__all__ = [
    "Calibration",
    "DetectionResult",
    "HSVConfig",
    "MockVisionDetector",
    "StreamVisionDetector",
    "VisionInterface",
    "detect_balls",
    "detect_blue_board",
    "get_perspective_transform",
    "get_perspective_transform_from_corners",
    "grid_to_ascii",
    "load_calibration_file",
    "run_detection",
    "save_calibration_file",
]
