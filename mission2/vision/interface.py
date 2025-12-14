"""Abstract interface for vision/board detection."""

from abc import ABC, abstractmethod

import numpy as np

from .board_detector import DetectionResult


class VisionInterface(ABC):
    """Abstract interface for board vision detection.
    
    Implementations can be mock (for testing) or real (ZMQ + board_detector).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to vision source.
        
        Returns:
            True if connection succeeded
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from vision source."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if vision source is connected."""
        pass

    @abstractmethod
    def detect_board(self) -> DetectionResult:
        """Capture frame and detect board state.
        
        Returns:
            DetectionResult with grid, balls, and images
        """
        pass

    @abstractmethod
    def get_grid(self) -> list[list[str | None]]:
        """Get the last detected grid state.
        
        Returns:
            5x5 grid with "orange", "yellow", or None values
        """
        pass

    @abstractmethod
    def get_last_frame(self) -> np.ndarray | None:
        """Get the last captured frame (BGR).
        
        Returns:
            BGR image or None if no frame captured
        """
        pass

    @abstractmethod
    def get_last_detection(self) -> DetectionResult | None:
        """Get the last detection result.
        
        Returns:
            DetectionResult or None if no detection performed
        """
        pass
