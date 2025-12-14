"""HTTP frame receiver - Simple and reliable for Streamlit.

Two modes:
1. Single frame fetch via /frame.jpg (stateless, best for Streamlit)
2. OpenCV VideoCapture on MJPEG stream (for continuous capture)
"""

import os
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import cv2
import numpy as np


class HTTPFrameReceiver:
    """Fetch camera frames via HTTP - dead simple, Streamlit-friendly.
    
    Usage (single frame - recommended for Streamlit):
        receiver = HTTPFrameReceiver("192.168.1.2", 8080)
        frame = receiver.capture_frame()
    
    Usage (OpenCV stream - for continuous capture):
        receiver = HTTPFrameReceiver("192.168.1.2", 8080)
        cap = receiver.open_stream()
        ret, frame = cap.read()
        cap.release()
    """
    
    def __init__(
        self,
        host: str = "192.168.1.2",
        port: int = 8080,
        output_dir: str = "outputs/stream",
    ):
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.last_error: str = ""
        os.makedirs(output_dir, exist_ok=True)
    
    @property
    def frame_url(self) -> str:
        """URL for single JPEG frame."""
        return f"http://{self.host}:{self.port}/frame.jpg"
    
    @property
    def stream_url(self) -> str:
        """URL for MJPEG stream (works with cv2.VideoCapture)."""
        return f"http://{self.host}:{self.port}/stream"
    
    @property
    def health_url(self) -> str:
        """URL for health check."""
        return f"http://{self.host}:{self.port}/health"
    
    def update_settings(self, host: str, port: int) -> None:
        """Update connection settings."""
        self.host = host
        self.port = port
    
    def is_available(self, timeout: float = 2.0) -> bool:
        """Check if camera server is reachable."""
        try:
            req = Request(self.health_url)
            with urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def open_stream(self) -> cv2.VideoCapture:
        """Open MJPEG stream as OpenCV VideoCapture.
        
        Returns:
            cv2.VideoCapture object - caller must release() when done
            
        Example:
            cap = receiver.open_stream()
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
        """
        return cv2.VideoCapture(self.stream_url)
    
    def capture_frame(self, timeout: float = 5.0, save: bool = True) -> np.ndarray | None:
        """Fetch a single frame from the HTTP server.
        
        This is stateless - each call is independent. Best for Streamlit.
        
        Args:
            timeout: Request timeout in seconds
            save: Whether to save frame to output_dir/latest.jpg
            
        Returns:
            BGR numpy array or None on error
        """
        try:
            self.last_error = ""
            req = Request(self.frame_url)
            
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    self.last_error = f"HTTP {resp.status}"
                    return None
                
                data = resp.read()
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.last_error = "Failed to decode JPEG"
                    return None
                
                if save:
                    self._save_frame(frame)
                
                return frame
                
        except HTTPError as e:
            self.last_error = f"HTTP {e.code}: {e.reason}"
            return None
        except URLError as e:
            self.last_error = f"Connection failed: {e.reason}"
            return None
        except TimeoutError:
            self.last_error = f"Timeout after {timeout}s"
            return None
        except Exception as e:
            self.last_error = str(e)
            return None
    
    def capture_frame_cv(self, save: bool = True) -> np.ndarray | None:
        """Fetch frame using OpenCV VideoCapture (alternative method).
        
        Opens stream, grabs one frame, closes. Slightly slower but more robust.
        
        Returns:
            BGR numpy array or None on error
        """
        cap = None
        try:
            self.last_error = ""
            cap = cv2.VideoCapture(self.stream_url)
            
            if not cap.isOpened():
                self.last_error = "Failed to open stream"
                return None
            
            ret, frame = cap.read()
            if not ret or frame is None:
                self.last_error = "Failed to read frame"
                return None
            
            if save:
                self._save_frame(frame)
            
            return frame
            
        except Exception as e:
            self.last_error = str(e)
            return None
        finally:
            if cap is not None:
                cap.release()
    
    def _save_frame(self, frame: np.ndarray) -> None:
        """Save frame to output directory with proper atomic write."""
        import time
        
        final_path = os.path.join(self.output_dir, "latest.jpg")
        tmp_path = os.path.join(self.output_dir, f".tmp_{time.time_ns()}.jpg")
        
        # Write to temp file with high quality
        cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Ensure write is complete
        os.sync() if hasattr(os, 'sync') else None
        
        # Atomic rename
        os.replace(tmp_path, final_path)
