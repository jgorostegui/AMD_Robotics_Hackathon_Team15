#!/usr/bin/env python3
"""
Camera Stream Emitter - Run on Computer 2 (192.168.1.2)
Captures frames from USB camera and publishes via ZMQ PUB socket.

Usage:
    python scripts/stream_emit.py [--camera 0] [--port 5555] [--quality 80] [--fps 1]
"""

import argparse
import time

import cv2
import zmq


def main():
    parser = argparse.ArgumentParser(description="Stream camera via ZMQ")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100)")
    parser.add_argument("--fps", type=float, default=1.0, help="Target FPS")
    args = parser.parse_args()

    frame_interval = 1.0 / args.fps

    # Initialize ZMQ publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[EMIT] Bound to tcp://0.0.0.0:{args.port}")

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        return 1

    print(f"[EMIT] Camera {args.camera} opened, streaming at {args.fps} FPS...")
    print("[EMIT] Press Ctrl+C to stop")

    frame_count = 0

    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            # Publish frame
            socket.send(buffer.tobytes())
            frame_count += 1
            print(f"[EMIT] Sent frame {frame_count}")

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[EMIT] Stopping...")
    finally:
        cap.release()
        socket.close()
        context.term()
        print("[EMIT] Cleanup complete")

    return 0


if __name__ == "__main__":
    exit(main())
