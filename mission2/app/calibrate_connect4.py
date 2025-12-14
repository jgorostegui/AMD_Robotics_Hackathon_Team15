"""Streamlit calibrator for Mission 2 (Connect4) board detection."""

import os
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st

from mission2.core.config import get_settings
from mission2.streaming.http_receiver import HTTPFrameReceiver
from mission2.vision.board_detector import (
    get_perspective_transform_from_corners,
    grid_to_ascii,
    run_detection,
)
from mission2.vision.calibration import Calibration, load_calibration_file, save_calibration_file


# Load settings once
settings = get_settings()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = (hex_color or "").lstrip("#")
    if len(value) != 6:
        return 0, 0, 0
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _format_range(name: str, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> str:
    return f"{name}: H[{lo[0]}..{hi[0]}]  S[{lo[1]}..{hi[1]}]  V[{lo[2]}..{hi[2]}]"


def _in_hsv_range(hsv: tuple[int, int, int], lo: tuple[int, int, int], hi: tuple[int, int, int]) -> bool:
    return all(int(lo[i]) <= int(hsv[i]) <= int(hi[i]) for i in range(3))


def _probe_cell(
    warped_bgr: np.ndarray,
    row: int,
    col: int,
    cfg: Any,
    min_cell_frac: float,
    cell_radius_frac: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return debug stats + overlay image for a single warped cell."""
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]

    cell_w = width / 5
    cell_h = height / 5
    cx = int((col + 0.5) * cell_w)
    cy = int((row + 0.5) * cell_h)
    radius = max(3, int(cell_radius_frac * min(cell_w, cell_h)))

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius)
    y1 = min(height, cy + radius)

    roi_hsv = hsv[y0:y1, x0:x1]
    circle = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    cv2.circle(circle, (cx - x0, cy - y0), radius, 255, -1)
    circle_area = int(cv2.countNonZero(circle))
    min_px = int(circle_area * float(min_cell_frac))

    def masked_count(lo: tuple[int, int, int], hi: tuple[int, int, int]) -> tuple[int, np.ndarray]:
        mask = cv2.inRange(roi_hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = cv2.bitwise_and(mask, circle)
        return int(cv2.countNonZero(mask)), mask

    orange_n, orange_m = masked_count(cfg.orange_lo, cfg.orange_hi)
    yellow_n, yellow_m = masked_count(cfg.yellow_lo, cfg.yellow_hi)
    red1_n, red1_m = masked_count(cfg.red1_lo, cfg.red1_hi)
    red2_n, red2_m = masked_count(cfg.red2_lo, cfg.red2_hi)
    red_m = cv2.bitwise_or(red1_m, red2_m)
    red_n = int(cv2.countNonZero(red_m))

    empty_custom_n = 0
    empty_m = red_m
    if getattr(cfg, "empty_lo", None) is not None and getattr(cfg, "empty_hi", None) is not None:
        empty_custom_n, empty_custom_m = masked_count(cfg.empty_lo, cfg.empty_hi)
        empty_m = cv2.bitwise_or(empty_m, empty_custom_m)
    empty_n = int(cv2.countNonZero(empty_m))

    decision = None
    reason = "none"
    if empty_n > max(min_px, orange_n, yellow_n):
        decision = None
        reason = "empty>orange,yellow"
    elif yellow_n > max(min_px, orange_n) and yellow_n > empty_n:
        decision = "yellow"
        reason = "yellow>orange,empty"
    elif orange_n > max(min_px, yellow_n) and orange_n > empty_n:
        decision = "orange"
        reason = "orange>yellow,empty"
    elif orange_n > min_px and yellow_n > min_px and max(orange_n, yellow_n) > empty_n:
        union = cv2.bitwise_or(orange_m, yellow_m)
        if cv2.countNonZero(union) > 0:
            hue = roi_hsv[:, :, 0]
            mean_h = float(hue[union > 0].mean())
            decision = "yellow" if mean_h >= 22 else "orange"
            reason = f"mean_h={mean_h:.1f}"

    center_bgr = warped_bgr[cy, cx].tolist()
    center_rgb = [int(center_bgr[2]), int(center_bgr[1]), int(center_bgr[0])]
    center_hsv = hsv[cy, cx].tolist()

    circle_vals = roi_hsv[circle > 0].reshape(-1, 3) if circle_area > 0 else np.empty((0, 3), np.uint8)
    if circle_vals.size:
        p05 = np.percentile(circle_vals, 5, axis=0).tolist()
        p50 = np.percentile(circle_vals, 50, axis=0).tolist()
        p95 = np.percentile(circle_vals, 95, axis=0).tolist()
    else:
        p05 = p50 = p95 = [0, 0, 0]

    overlay = warped_bgr.copy()
    cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), 2)
    cv2.putText(
        overlay,
        f"r{row} c{col} {decision or '-'}",
        (max(0, cx - radius), max(15, cy - radius - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    stats: dict[str, Any] = {
        "row": row,
        "col": col,
        "decision": decision,
        "reason": reason,
        "center_rgb": center_rgb,
        "center_hsv": [int(center_hsv[0]), int(center_hsv[1]), int(center_hsv[2])],
        "center_hue_deg": int(center_hsv[0]) * 2,
        "circle_px": circle_area,
        "min_px": min_px,
        "orange_px": orange_n,
        "yellow_px": yellow_n,
        "empty_px": empty_n,
        "red_px": red_n,
        "empty_custom_px": empty_custom_n,
        "orange_frac": (orange_n / circle_area) if circle_area else 0.0,
        "yellow_frac": (yellow_n / circle_area) if circle_area else 0.0,
        "empty_frac": (empty_n / circle_area) if circle_area else 0.0,
        "hsv_p05": [float(p05[0]), float(p05[1]), float(p05[2])],
        "hsv_p50": [float(p50[0]), float(p50[1]), float(p50[2])],
        "hsv_p95": [float(p95[0]), float(p95[1]), float(p95[2])],
    }

    return stats, overlay


def init_session_state():
    """Initialize session state from settings and calibration file."""
    defaults = Calibration()

    # HTTP receiver (simpler than ZMQ)
    if "receiver" not in st.session_state:
        st.session_state.receiver = HTTPFrameReceiver(
            host=settings.stream.host,
            port=8080,  # Default HTTP port
            output_dir=settings.stream.output_dir,
        )

    # Stream settings
    if "emitter_host" not in st.session_state:
        st.session_state.emitter_host = settings.stream.host
    if "emitter_port" not in st.session_state:
        st.session_state.emitter_port = 8080  # HTTP port

    # Corner mode from settings
    if "corner_mode" not in st.session_state:
        st.session_state.corner_mode = settings.ui.corner_mode
    if "grid_method" not in st.session_state:
        st.session_state.grid_method = settings.ui.grid_method

    # Calibration defaults
    if "min_cell_frac" not in st.session_state:
        st.session_state.min_cell_frac = defaults.min_cell_frac
    if "cell_radius_frac" not in st.session_state:
        st.session_state.cell_radius_frac = defaults.cell_radius_frac

    # HSV defaults
    st.session_state.setdefault("hsv_blue_h", defaults.blue_h)
    st.session_state.setdefault("hsv_blue_s", defaults.blue_s)
    st.session_state.setdefault("hsv_blue_v", defaults.blue_v)
    st.session_state.setdefault("hsv_orange_h", defaults.orange_h)
    st.session_state.setdefault("hsv_orange_s", defaults.orange_s)
    st.session_state.setdefault("hsv_orange_v", defaults.orange_v)
    st.session_state.setdefault("hsv_yellow_h", defaults.yellow_h)
    st.session_state.setdefault("hsv_yellow_s", defaults.yellow_s)
    st.session_state.setdefault("hsv_yellow_v", defaults.yellow_v)
    st.session_state.setdefault("hsv_empty_enabled", False)
    st.session_state.setdefault("hsv_empty_h", defaults.empty_h or (0, 0))
    st.session_state.setdefault("hsv_empty_s", defaults.empty_s or (0, 0))
    st.session_state.setdefault("hsv_empty_v", defaults.empty_v or (0, 0))

    # Auto-load calibration on first run
    if "calibration_loaded" not in st.session_state:
        cal = load_calibration_file(settings.vision.calibration_path)
        if cal:
            apply_calibration(cal)
        st.session_state.calibration_loaded = True


def apply_calibration(cal: Calibration | None) -> None:
    """Apply calibration to session state."""
    if cal is None:
        return
    st.session_state.min_cell_frac = float(cal.min_cell_frac)
    st.session_state.cell_radius_frac = float(cal.cell_radius_frac)

    st.session_state.hsv_blue_h = tuple(cal.blue_h)
    st.session_state.hsv_blue_s = tuple(cal.blue_s)
    st.session_state.hsv_blue_v = tuple(cal.blue_v)
    st.session_state.hsv_orange_h = tuple(cal.orange_h)
    st.session_state.hsv_orange_s = tuple(cal.orange_s)
    st.session_state.hsv_orange_v = tuple(cal.orange_v)
    st.session_state.hsv_yellow_h = tuple(cal.yellow_h)
    st.session_state.hsv_yellow_s = tuple(cal.yellow_s)
    st.session_state.hsv_yellow_v = tuple(cal.yellow_v)

    if cal.empty_h is not None and cal.empty_s is not None and cal.empty_v is not None:
        st.session_state.hsv_empty_enabled = True
        st.session_state.hsv_empty_h = tuple(cal.empty_h)
        st.session_state.hsv_empty_s = tuple(cal.empty_s)
        st.session_state.hsv_empty_v = tuple(cal.empty_v)
    else:
        st.session_state.hsv_empty_enabled = False

    if cal.corners and len(cal.corners) == 4:
        pts = [(int(p[0]), int(p[1])) for p in cal.corners]
        st.session_state.corner_tl_x, st.session_state.corner_tl_y = pts[0]
        st.session_state.corner_tr_x, st.session_state.corner_tr_y = pts[1]
        st.session_state.corner_br_x, st.session_state.corner_br_y = pts[2]
        st.session_state.corner_bl_x, st.session_state.corner_bl_y = pts[3]
        # Auto-switch to fixed corners mode when calibration has corners
        st.session_state.corner_mode = "fixed"


def corners_from_state() -> list[tuple[int, int]] | None:
    """Get corners from session state."""
    keys = [
        "corner_tl_x", "corner_tl_y", "corner_tr_x", "corner_tr_y",
        "corner_br_x", "corner_br_y", "corner_bl_x", "corner_bl_y",
    ]
    if not all(k in st.session_state for k in keys):
        return None
    try:
        return [
            (int(st.session_state.corner_tl_x), int(st.session_state.corner_tl_y)),
            (int(st.session_state.corner_tr_x), int(st.session_state.corner_tr_y)),
            (int(st.session_state.corner_br_x), int(st.session_state.corner_br_y)),
            (int(st.session_state.corner_bl_x), int(st.session_state.corner_bl_y)),
        ]
    except Exception:
        return None


def calibration_from_state(corners: list[tuple[int, int]] | None = None) -> Calibration:
    """Build Calibration object from session state."""
    corners_val = corners if corners is not None else corners_from_state()
    empty_enabled = bool(st.session_state.get("hsv_empty_enabled", False))
    return Calibration(
        min_cell_frac=float(st.session_state.get("min_cell_frac", Calibration.min_cell_frac)),
        cell_radius_frac=float(st.session_state.get("cell_radius_frac", Calibration.cell_radius_frac)),
        corners=corners_val,
        blue_h=tuple(st.session_state.get("hsv_blue_h", Calibration.blue_h)),
        blue_s=tuple(st.session_state.get("hsv_blue_s", Calibration.blue_s)),
        blue_v=tuple(st.session_state.get("hsv_blue_v", Calibration.blue_v)),
        orange_h=tuple(st.session_state.get("hsv_orange_h", Calibration.orange_h)),
        orange_s=tuple(st.session_state.get("hsv_orange_s", Calibration.orange_s)),
        orange_v=tuple(st.session_state.get("hsv_orange_v", Calibration.orange_v)),
        yellow_h=tuple(st.session_state.get("hsv_yellow_h", Calibration.yellow_h)),
        yellow_s=tuple(st.session_state.get("hsv_yellow_s", Calibration.yellow_s)),
        yellow_v=tuple(st.session_state.get("hsv_yellow_v", Calibration.yellow_v)),
        empty_h=tuple(st.session_state.get("hsv_empty_h", (0, 0))) if empty_enabled else None,
        empty_s=tuple(st.session_state.get("hsv_empty_s", (0, 0))) if empty_enabled else None,
        empty_v=tuple(st.session_state.get("hsv_empty_v", (0, 0))) if empty_enabled else None,
    )


def render_sidebar(frame_shape: tuple[int, int] | None) -> dict:
    """Render sidebar controls."""
    st.sidebar.header("Stream")

    def capture_frame_callback():
        host_val = st.session_state.emitter_host.strip()
        port_val = int(st.session_state.emitter_port)
        st.session_state.receiver.update_settings(host_val, port_val)
        frame = st.session_state.receiver.capture_frame(timeout=5.0, save=True)
        if frame is not None:
            st.session_state._capture_status = "ok"
        else:
            err = st.session_state.receiver.last_error or "timeout"
            st.session_state._capture_status = f"error:{err}"

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.text_input("Host", key="emitter_host")
    with col2:
        st.number_input("Port", min_value=1, max_value=65535, key="emitter_port", label_visibility="collapsed")

    col_capture, col_live = st.sidebar.columns(2)
    with col_capture:
        st.button("📷 Capture", type="primary", on_click=capture_frame_callback)
    with col_live:
        live_mode = st.toggle("🔴 Live", key="live_mode", help="Continuous capture & detect")
    
    capture_status = st.session_state.get("_capture_status") or ""
    if capture_status == "ok":
        st.sidebar.success("Frame captured!")
        st.session_state._capture_status = None
    elif capture_status.startswith("error:"):
        st.sidebar.error(f"Failed: {capture_status[6:]}")
        st.session_state._capture_status = None

    # Live mode settings
    if live_mode:
        st.sidebar.slider("Refresh interval (s)", 1.0, 5.0, 2.0, 0.5, key="live_interval")

    st.sidebar.markdown("---")

    # Corner mode
    corner_mode = st.sidebar.radio(
        "Board Detection",
        options=["auto", "fixed"],
        format_func=lambda m: "Auto (blue board)" if m == "auto" else "Fixed corners",
        key="corner_mode",
        horizontal=True,
    )
    use_manual = corner_mode == "fixed"

    # Grid method
    st.sidebar.selectbox(
        "Grid Method",
        options=["sample_hsv", "hybrid", "hough"],
        format_func=lambda v: {
            "sample_hsv": "HSV Sampling",
            "hybrid": "Hybrid (HSV + Hough)",
            "hough": "Hough Circles",
        }[v],
        key="grid_method",
    )

    st.sidebar.markdown("---")

    # HSV Thresholds in expander
    with st.sidebar.expander("HSV Thresholds", expanded=False):
        st.markdown("**Blue (board)**")
        st.slider("H", 0, 179, key="hsv_blue_h")
        st.slider("S", 0, 255, key="hsv_blue_s")
        st.slider("V", 0, 255, key="hsv_blue_v")

        st.markdown("**Orange (balls)**")
        st.slider("H", 0, 179, key="hsv_orange_h")
        st.slider("S", 0, 255, key="hsv_orange_s")
        st.slider("V", 0, 255, key="hsv_orange_v")

        st.markdown("**Yellow (balls)**")
        st.slider("H", 0, 179, key="hsv_yellow_h")
        st.slider("S", 0, 255, key="hsv_yellow_s")
        st.slider("V", 0, 255, key="hsv_yellow_v")

        st.markdown("**Empty/Background (optional)**")
        empty_enabled = st.checkbox("Enable", key="hsv_empty_enabled")
        st.slider("H", 0, 179, key="hsv_empty_h", disabled=not empty_enabled)
        st.slider("S", 0, 255, key="hsv_empty_s", disabled=not empty_enabled)
        st.slider("V", 0, 255, key="hsv_empty_v", disabled=not empty_enabled)

    # Cell detection params
    with st.sidebar.expander("Cell Detection", expanded=False):
        min_cell_frac = st.slider("Min cell fraction", 0.05, 0.30, step=0.01, key="min_cell_frac")
        cell_radius_frac = st.slider("Cell radius fraction", 0.20, 0.45, step=0.01, key="cell_radius_frac")

    config = calibration_from_state().to_hsv_config()

    # Manual corners
    manual_corners = None
    if use_manual and frame_shape is not None:
        height, width = frame_shape
        with st.sidebar.expander("Manual Corners", expanded=True):
            # Initialize defaults if not set
            if "corner_tl_x" not in st.session_state:
                st.session_state.corner_tl_x = int(width * 0.05)
                st.session_state.corner_tl_y = int(height * 0.25)
                st.session_state.corner_tr_x = int(width * 0.95)
                st.session_state.corner_tr_y = int(height * 0.25)
                st.session_state.corner_br_x = int(width * 0.95)
                st.session_state.corner_br_y = int(height * 0.95)
                st.session_state.corner_bl_x = int(width * 0.05)
                st.session_state.corner_bl_y = int(height * 0.95)

            def auto_detect_corners():
                """Callback to auto-detect corners before widgets render."""
                try:
                    cfg = calibration_from_state().to_hsv_config()
                    res = run_detection(settings.stream.frame_path, config=cfg)
                    if res.board_corners and len(res.board_corners) == 4:
                        pts = [(int(p[0]), int(p[1])) for p in res.board_corners]
                        st.session_state.corner_tl_x, st.session_state.corner_tl_y = pts[0]
                        st.session_state.corner_tr_x, st.session_state.corner_tr_y = pts[1]
                        st.session_state.corner_br_x, st.session_state.corner_br_y = pts[2]
                        st.session_state.corner_bl_x, st.session_state.corner_bl_y = pts[3]
                        st.session_state._autodetect_status = "ok"
                    else:
                        st.session_state._autodetect_status = "no_board"
                except Exception as e:
                    st.session_state._autodetect_status = f"error: {e}"

            st.button("Auto-detect from blue board", on_click=auto_detect_corners)

            status = st.session_state.get("_autodetect_status")
            if status == "ok":
                st.success("Corners updated!")
                st.session_state._autodetect_status = None
            elif status == "no_board":
                st.warning("No board detected")
                st.session_state._autodetect_status = None
            elif status and status.startswith("error:"):
                st.error(f"Detection failed: {status[7:]}")
                st.session_state._autodetect_status = None

            c1, c2 = st.columns(2)
            with c1:
                st.number_input("TL x", 0, width - 1, key="corner_tl_x")
                st.number_input("TL y", 0, height - 1, key="corner_tl_y")
                st.number_input("BL x", 0, width - 1, key="corner_bl_x")
                st.number_input("BL y", 0, height - 1, key="corner_bl_y")
            with c2:
                st.number_input("TR x", 0, width - 1, key="corner_tr_x")
                st.number_input("TR y", 0, height - 1, key="corner_tr_y")
                st.number_input("BR x", 0, width - 1, key="corner_br_x")
                st.number_input("BR y", 0, height - 1, key="corner_br_y")

        manual_corners = corners_from_state()

    st.sidebar.markdown("---")

    # Save/Load calibration
    def save_calibration_callback():
        export_data = calibration_from_state(corners=corners_from_state())
        save_calibration_file(settings.vision.calibration_path, export_data)
        st.session_state._save_status = "ok"

    def reload_calibration_callback():
        cal = load_calibration_file(settings.vision.calibration_path)
        if cal:
            apply_calibration(cal)
            st.session_state._reload_status = "ok"
        else:
            st.session_state._reload_status = "not_found"

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("💾 Save", on_click=save_calibration_callback)
        if st.session_state.get("_save_status") == "ok":
            st.success("Saved!")
            st.session_state._save_status = None
    with col2:
        st.button("🔄 Reload", on_click=reload_calibration_callback)
        if st.session_state.get("_reload_status") == "ok":
            st.success("Loaded!")
            st.session_state._reload_status = None
        elif st.session_state.get("_reload_status") == "not_found":
            st.warning("No calibration file")
            st.session_state._reload_status = None

    st.sidebar.caption(f"Config: `{settings.vision.calibration_path}`")

    debug = st.sidebar.checkbox("Debug mode", value=False)

    return {
        "debug": debug,
        "config": config,
        "manual_corners": manual_corners if use_manual else None,
        "min_cell_frac": float(st.session_state.min_cell_frac),
        "cell_radius_frac": float(st.session_state.cell_radius_frac),
    }


def main():
    """Streamlit entry point."""
    st.set_page_config(page_title="Connect4 Calibrator", layout="wide")
    st.title("Connect4 Vision Calibrator")

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            min-width: {settings.ui.sidebar_width_px}px !important;
            max-width: {settings.ui.sidebar_width_px}px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Check for existing frame
    frame_path = settings.stream.frame_path
    warped_path = os.path.join(settings.stream.output_dir, "warped.jpg")
    frame_shape = None
    if os.path.exists(frame_path):
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame_shape = frame.shape[:2]

    cfg = render_sidebar(frame_shape)
    live_mode = st.session_state.get("live_mode", False)
    
    # Live mode: capture + detect first, then display
    if live_mode:
        receiver = st.session_state.receiver
        receiver.update_settings(
            st.session_state.emitter_host.strip(),
            int(st.session_state.emitter_port),
        )
        
        interval = float(st.session_state.get("live_interval", 2.0))
        start_time = time.time()
        
        # Capture frame
        frame = receiver.capture_frame(timeout=3.0, save=True)
        
        if frame is None:
            st.error(f"🔴 Capture failed: {receiver.last_error}")
            st.info("Start HTTP server: `python scripts/stream_http.py --port 8080`")
            time.sleep(interval)
            st.rerun()
        
        # Small delay to ensure file write is complete
        time.sleep(0.05)
        
        # Run detection
        result = run_detection(
            frame_path,
            debug=cfg["debug"],
            config=cfg["config"],
            manual_corners=cfg["manual_corners"],
            min_cell_frac=cfg["min_cell_frac"],
            cell_radius_frac=cfg["cell_radius_frac"],
            grid_method=st.session_state.grid_method,
        )
        
        # Save warped to file (avoids Streamlit media cache issues)
        if result.warped_image is not None:
            tmp_warped = os.path.join(settings.stream.output_dir, ".warped_tmp.jpg")
            cv2.imwrite(tmp_warped, result.warped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.replace(tmp_warped, warped_path)
        
        elapsed = time.time() - start_time
        st.caption(f"🔴 Live | {elapsed*1000:.0f}ms | Next in {interval}s")
    else:
        result = None
        
        # Manual capture button
        def do_capture():
            receiver = st.session_state.receiver
            receiver.update_settings(
                st.session_state.emitter_host.strip(),
                int(st.session_state.emitter_port),
            )
            f = receiver.capture_frame(timeout=5.0, save=True)
            st.session_state._capture_ok = f is not None
            st.session_state._capture_err = receiver.last_error if f is None else None
        
        def do_detect():
            st.session_state._do_detect = True
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("📷 Capture", on_click=do_capture, use_container_width=True)
        with col_btn2:
            st.button("🔍 Detect", on_click=do_detect, type="primary", use_container_width=True,
                     disabled=not os.path.exists(frame_path))
        
        if st.session_state.get("_capture_ok"):
            st.success("Frame captured!")
            st.session_state._capture_ok = False
        if st.session_state.get("_capture_err"):
            st.error(f"Capture failed: {st.session_state._capture_err}")
            st.session_state._capture_err = None
        
        # Run detection if requested
        if st.session_state.get("_do_detect") and os.path.exists(frame_path):
            st.session_state._do_detect = False
            result = run_detection(
                frame_path,
                debug=cfg["debug"],
                config=cfg["config"],
                manual_corners=cfg["manual_corners"],
                min_cell_frac=cfg["min_cell_frac"],
                cell_radius_frac=cfg["cell_radius_frac"],
                grid_method=st.session_state.grid_method,
            )
            if result.warped_image is not None:
                tmp_warped = os.path.join(settings.stream.output_dir, ".warped_tmp.jpg")
                cv2.imwrite(tmp_warped, result.warped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(tmp_warped, warped_path)
            st.session_state._last_result = result
        
        # Use cached result
        if result is None:
            result = st.session_state.get("_last_result")

    # Helper to check if image file is valid
    def is_valid_image(path: str) -> bool:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 100:  # Too small to be valid
            return False
        # Try to read with OpenCV
        img = cv2.imread(path)
        return img is not None

    # Display results (same for live and manual mode)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection")
        if result:
            if result.error:
                st.error(result.error)
            else:
                st.code(grid_to_ascii(result.grid), language=None)
                orange_n = len(result.balls.get("orange", []))
                yellow_n = len(result.balls.get("yellow", []))
                st.caption(f"{orange_n} 🟠  {yellow_n} 🟡")
                
                # Show warped from file (validate first)
                if is_valid_image(warped_path):
                    st.image(warped_path, caption="Warped view", width="stretch")
        else:
            st.info("Click **Capture** then **Detect**, or enable **Live** mode")
    
    with col2:
        st.subheader("Camera")
        if is_valid_image(frame_path):
            # Draw corner overlay if in fixed mode (only when not live to avoid lag)
            corners = cfg.get("manual_corners")
            if corners and len(corners) == 4 and not live_mode:
                frame_img = cv2.imread(frame_path)
                if frame_img is not None:
                    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_img, [pts], True, (0, 255, 0), 2)
                    for i, (x, y) in enumerate(corners):
                        cv2.circle(frame_img, (x, y), 8, (0, 0, 255), -1)
                        cv2.putText(frame_img, ["TL", "TR", "BR", "BL"][i], (x + 10, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Display directly as numpy array (no file write)
                    st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), caption="Camera + corners", width="stretch")
                else:
                    st.image(frame_path, caption="Latest frame", width="stretch")
            else:
                st.image(frame_path, caption="Latest frame", width="stretch")
        else:
            st.info("No valid frame captured yet")
    
    # Live mode: wait and rerun
    if live_mode:
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
