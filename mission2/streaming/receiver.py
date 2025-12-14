"""Stream receiver module for ZMQ camera streaming."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import zmq


logger = logging.getLogger(__name__)


class StreamReceiver:
    """Receives camera frames via ZMQ and saves to disk."""

    def __init__(self, host: str = "192.168.1.2", port: int = 5555, output_dir: str = "outputs/stream"):
        """Initialize receiver with connection settings.
        
        Args:
            host: IP address of the emitter
            port: ZMQ port number
            output_dir: Directory to save frames
        """
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._connected = False
        self.last_error: str = ""

    def connect(self, *, validate_timeout_ms: int | None = None) -> bool:
        """Establish ZMQ connection.
        
        Returns:
            True if connection succeeded, False otherwise
        """
        if self._connected:
            self.disconnect()
        try:
            self.last_error = ""
            os.makedirs(self.output_dir, exist_ok=True)
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            # Optional: keep only the most recent message in the socket buffer.
            # Disabled by default because some libzmq builds behave oddly with CONFLATE.
            use_conflate = os.environ.get("MISSION2_ZMQ_CONFLATE", "").strip().lower() in {"1", "true", "yes"}
            if use_conflate:
                try:
                    self._socket.setsockopt(zmq.RCVHWM, 1)
                    self._socket.setsockopt(zmq.CONFLATE, 1)
                except Exception:
                    pass
            self._socket.connect(f"tcp://{self.host}:{self.port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe AFTER connect
            self._socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
            self._connected = True
            logger.info(f"Connected to tcp://{self.host}:{self.port}")

            if validate_timeout_ms is not None and int(validate_timeout_ms) > 0:
                frame = self.receive_frame(timeout_ms=int(validate_timeout_ms))
                if frame is None:
                    self.last_error = f"No frames within {int(validate_timeout_ms)}ms"
                    logger.warning("Connected but %s", self.last_error)
                    self.disconnect()
                    return False
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False


    def disconnect(self) -> None:
        """Close ZMQ connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        self._connected = False
        logger.info("Disconnected")

    def receive_frame(self, timeout_ms: int = 1000, *, drain: bool = True, max_drain: int = 256) -> np.ndarray | None:
        """Receive single frame.
        
        Args:
            timeout_ms: Timeout in milliseconds
            drain: When True, drain queued frames and return the newest one.
            max_drain: Max extra frames to drain (safety cap).
            
        Returns:
            Frame as numpy array, or None on timeout/error
        """
        if not self._connected or not self._socket:
            self.last_error = "Not connected"
            return None

        socket = self._socket
        old_timeout: int | None = None
        try:
            old_timeout = int(socket.getsockopt(zmq.RCVTIMEO))
            socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))

            # First blocking receive (up to timeout).
            data = socket.recv()

            # Drain any backlog so "capture frame" reflects "now", not "oldest queued".
            if drain:
                for _ in range(max(0, int(max_drain))):
                    try:
                        data = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break

            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Save latest frame
                final_path = os.path.join(self.output_dir, "latest.jpg")
                tmp_path = os.path.join(self.output_dir, ".latest_tmp.jpg")
                cv2.imwrite(tmp_path, frame)
                os.replace(tmp_path, final_path)

            self.last_error = ""
            return frame
        except zmq.Again:
            self.last_error = f"Timeout after {int(timeout_ms)}ms"
            return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Receive failed: {e}")
            return None
        finally:
            if old_timeout is not None:
                try:
                    socket.setsockopt(zmq.RCVTIMEO, int(old_timeout))
                except Exception:
                    pass

    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected

    def update_settings(self, host: str, port: int) -> None:
        """Update connection settings (requires reconnect)."""
        self.host = host
        self.port = port

    def capture_one_shot(self, timeout_ms: int = 5000) -> np.ndarray | None:
        """Connect, grab one frame, disconnect. Reliable for Streamlit.

        This avoids issues with ZMQ sockets not surviving across Streamlit reruns.
        """
        context = None
        socket = None
        try:
            self.last_error = ""
            os.makedirs(self.output_dir, exist_ok=True)

            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{self.host}:{self.port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))

            # Receive frame
            data = socket.recv()

            # Drain backlog to get latest
            for _ in range(256):
                try:
                    data = socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            np_arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                final_path = os.path.join(self.output_dir, "latest.jpg")
                tmp_path = os.path.join(self.output_dir, ".latest_tmp.jpg")
                cv2.imwrite(tmp_path, frame)
                os.replace(tmp_path, final_path)

            return frame
        except zmq.Again:
            self.last_error = f"Timeout after {int(timeout_ms)}ms"
            return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"One-shot capture failed: {e}")
            return None
        finally:
            if socket:
                socket.close()
            if context:
                context.term()
