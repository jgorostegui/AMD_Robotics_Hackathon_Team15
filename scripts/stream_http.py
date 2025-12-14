#!/usr/bin/env python3
"""HTTP Camera Stream Server - Run on Computer 2 (192.168.1.2)

Simple HTTP server that serves camera frames as:
- /frame.jpg - Single JPEG frame (for polling)
- /stream - MJPEG stream (for continuous viewing)

Usage:
    python scripts/stream_http.py --camera 0 --port 8080
"""

import argparse
import io
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2


class CameraServer:
    """Thread-safe camera frame server."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self._thread = None
    
    def start(self) -> bool:
        """Start camera capture thread."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def _capture_loop(self):
        """Continuous capture loop."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
            time.sleep(1.0 / self.fps)
    
    def get_frame_jpeg(self, quality: int = 85) -> bytes | None:
        """Get current frame as JPEG bytes."""
        with self.frame_lock:
            if self.frame is None:
                return None
            _, jpeg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return jpeg.tobytes()


# Global camera instance
camera: CameraServer | None = None


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for camera streams."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        if self.path == '/frame.jpg' or self.path == '/frame':
            self._serve_single_frame()
        elif self.path == '/stream':
            self._serve_mjpeg_stream()
        elif self.path == '/health':
            self._serve_health()
        elif self.path == '/':
            self._serve_index()
        else:
            self.send_error(404)
    
    def _serve_single_frame(self):
        """Serve a single JPEG frame."""
        jpeg = camera.get_frame_jpeg() if camera else None
        if jpeg is None:
            self.send_error(503, "No frame available")
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(jpeg))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(jpeg)
    
    def _serve_mjpeg_stream(self):
        """Serve continuous MJPEG stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            while True:
                jpeg = camera.get_frame_jpeg() if camera else None
                if jpeg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                time.sleep(1.0 / 15)  # ~15 FPS for MJPEG
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def _serve_health(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        has_frame = camera is not None and camera.frame is not None
        self.wfile.write(f'{{"ok": {str(has_frame).lower()}}}'.encode())
    
    def _serve_index(self):
        """Serve simple HTML page with stream preview."""
        html = """<!DOCTYPE html>
<html>
<head><title>Camera Stream</title></head>
<body style="background:#111;color:#fff;font-family:sans-serif;text-align:center;padding:20px">
    <h1>Camera Stream</h1>
    <p>Endpoints:</p>
    <ul style="list-style:none">
        <li><a href="/frame.jpg" style="color:#0af">/frame.jpg</a> - Single frame</li>
        <li><a href="/stream" style="color:#0af">/stream</a> - MJPEG stream</li>
        <li><a href="/health" style="color:#0af">/health</a> - Health check</li>
    </ul>
    <h2>Live Preview</h2>
    <img src="/stream" style="max-width:100%;border:2px solid #333">
</body>
</html>"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    global camera
    
    parser = argparse.ArgumentParser(description="HTTP Camera Stream Server")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    args = parser.parse_args()
    
    print(f"[HTTP] Starting camera {args.camera}...")
    camera = CameraServer(args.camera, args.width, args.height, args.fps)
    
    if not camera.start():
        print(f"[ERROR] Could not open camera {args.camera}")
        return 1
    
    print(f"[HTTP] Camera started, serving on http://0.0.0.0:{args.port}")
    print(f"[HTTP] Endpoints:")
    print(f"       /frame.jpg - Single JPEG frame")
    print(f"       /stream    - MJPEG stream")
    print(f"       /health    - Health check")
    print("[HTTP] Press Ctrl+C to stop")
    
    server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[HTTP] Shutting down...")
    finally:
        camera.stop()
        server.shutdown()
    
    return 0


if __name__ == "__main__":
    exit(main())
