#!/usr/bin/env python3
"""
Camera Stream Receiver - Run on Computer 1
Receives frames from ZMQ PUB socket and saves them.

Usage:
    python scripts/stream_receive.py [--host 192.168.1.2] [--port 5555] [--output outputs/stream]
"""

import argparse
import os
import time

import cv2
import numpy as np
import zmq


def main():
    default_host = os.getenv("MISSION2_EMITTER_HOST", "192.168.1.2").strip() or "192.168.1.2"
    try:
        default_port = int(os.getenv("MISSION2_EMITTER_PORT", "5555"))
    except ValueError:
        default_port = 5555

    parser = argparse.ArgumentParser(description="Receive camera stream via ZMQ")
    parser.add_argument("--host", type=str, default=default_host, help="Emitter IP address")
    parser.add_argument("--port", type=int, default=default_port, help="ZMQ port")
    parser.add_argument("--output", type=str, default="outputs/stream", help="Output directory for frames")
    parser.add_argument("--save-interval", type=float, default=2.0, help="Save one image every N seconds")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize ZMQ subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
    
    print(f"[RECV] Connected to tcp://{args.host}:{args.port}")
    print(f"[RECV] Saving frames to: {args.output} (every {args.save_interval}s)")
    print("[RECV] Press Ctrl+C to stop")

    frame_count = 0
    saved_count = 0
    last_save_time = 0

    try:
        while True:
            try:
                data = socket.recv()
                
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("[WARN] Failed to decode frame")
                    continue

                frame_count += 1
                
                # Always save latest frame
                cv2.imwrite(os.path.join(args.output, "latest.jpg"), frame)
                
                # Save numbered frame every N seconds
                now = time.time()
                if now - last_save_time >= args.save_interval:
                    filename = os.path.join(args.output, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    last_save_time = now
                    print(f"[RECV] Saved {filename} (total received: {frame_count})")
                    
            except zmq.Again:
                print("[WARN] No data received (timeout), retrying...")
                continue

    except KeyboardInterrupt:
        print("\n[RECV] Stopping...")
    finally:
        socket.close()
        context.term()
        print(f"[RECV] Done. Saved {saved_count} frames to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
