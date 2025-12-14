"""Calibration helpers for Mission 2 board detection.

This module defines the JSON schema used by the Streamlit calibrator app and
provides utilities to load/save and convert calibration to `HSVConfig`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .board_detector import HSVConfig


def _parse_range2(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return default
    return default


def _parse_range2_optional(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def _parse_corners(value: Any) -> list[tuple[int, int]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        pts = [(int(p[0]), int(p[1])) for p in value]
    except Exception:
        return None
    return pts if len(pts) == 4 else None


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class Calibration:
    """Serializable calibration used by the vision pipeline."""

    min_cell_frac: float = 0.11
    cell_radius_frac: float = 0.33
    corners: list[tuple[int, int]] | None = None

    blue_h: tuple[int, int] = (100, 130)
    blue_s: tuple[int, int] = (100, 255)
    blue_v: tuple[int, int] = (50, 255)

    orange_h: tuple[int, int] = (5, 25)
    orange_s: tuple[int, int] = (70, 255)
    orange_v: tuple[int, int] = (80, 255)

    yellow_h: tuple[int, int] = (18, 45)
    yellow_s: tuple[int, int] = (40, 255)
    yellow_v: tuple[int, int] = (80, 255)

    # Optional extra "empty/background" HSV thresholds, useful when empty holes
    # show a non-red background (wood/wall) that overlaps with ball colors.
    empty_h: tuple[int, int] | None = None
    empty_s: tuple[int, int] | None = None
    empty_v: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        hsv = (data.get("hsv") or {}) if isinstance(data, dict) else {}

        def rng(color: str, channel: str, default: tuple[int, int]) -> tuple[int, int]:
            section = hsv.get(color) if isinstance(hsv, dict) else None
            if isinstance(section, dict):
                return _parse_range2(section.get(channel), default)
            return default

        def rng_optional(color: str, channel: str) -> tuple[int, int] | None:
            section = hsv.get(color) if isinstance(hsv, dict) else None
            if isinstance(section, dict) and channel in section:
                return _parse_range2_optional(section.get(channel))
            return None

        corners = _parse_corners(data.get("corners") if isinstance(data, dict) else None)

        return cls(
            min_cell_frac=_parse_float(data.get("min_cell_frac"), cls.min_cell_frac) if isinstance(data, dict) else cls.min_cell_frac,
            cell_radius_frac=_parse_float(data.get("cell_radius_frac"), cls.cell_radius_frac)
            if isinstance(data, dict)
            else cls.cell_radius_frac,
            corners=corners,
            blue_h=rng("blue", "h", cls.blue_h),
            blue_s=rng("blue", "s", cls.blue_s),
            blue_v=rng("blue", "v", cls.blue_v),
            orange_h=rng("orange", "h", cls.orange_h),
            orange_s=rng("orange", "s", cls.orange_s),
            orange_v=rng("orange", "v", cls.orange_v),
            yellow_h=rng("yellow", "h", cls.yellow_h),
            yellow_s=rng("yellow", "s", cls.yellow_s),
            yellow_v=rng("yellow", "v", cls.yellow_v),
            empty_h=rng_optional("empty", "h"),
            empty_s=rng_optional("empty", "s"),
            empty_v=rng_optional("empty", "v"),
        )

    def to_dict(self) -> dict[str, Any]:
        hsv: dict[str, Any] = {
            "blue": {"h": list(self.blue_h), "s": list(self.blue_s), "v": list(self.blue_v)},
            "orange": {"h": list(self.orange_h), "s": list(self.orange_s), "v": list(self.orange_v)},
            "yellow": {"h": list(self.yellow_h), "s": list(self.yellow_s), "v": list(self.yellow_v)},
        }

        if self.empty_h is not None and self.empty_s is not None and self.empty_v is not None:
            hsv["empty"] = {
                "h": list(self.empty_h),
                "s": list(self.empty_s),
                "v": list(self.empty_v),
            }

        return {
            "min_cell_frac": float(self.min_cell_frac),
            "cell_radius_frac": float(self.cell_radius_frac),
            "corners": self.corners,
            "hsv": hsv,
        }

    def to_hsv_config(self) -> HSVConfig:
        kwargs: dict[str, Any] = dict(
            blue_lo=(int(self.blue_h[0]), int(self.blue_s[0]), int(self.blue_v[0])),
            blue_hi=(int(self.blue_h[1]), int(self.blue_s[1]), int(self.blue_v[1])),
            orange_lo=(int(self.orange_h[0]), int(self.orange_s[0]), int(self.orange_v[0])),
            orange_hi=(int(self.orange_h[1]), int(self.orange_s[1]), int(self.orange_v[1])),
            yellow_lo=(int(self.yellow_h[0]), int(self.yellow_s[0]), int(self.yellow_v[0])),
            yellow_hi=(int(self.yellow_h[1]), int(self.yellow_s[1]), int(self.yellow_v[1])),
        )

        if self.empty_h is not None and self.empty_s is not None and self.empty_v is not None:
            kwargs["empty_lo"] = (
                int(self.empty_h[0]),
                int(self.empty_s[0]),
                int(self.empty_v[0]),
            )
            kwargs["empty_hi"] = (
                int(self.empty_h[1]),
                int(self.empty_s[1]),
                int(self.empty_v[1]),
            )

        return HSVConfig(**kwargs)


def load_calibration_file(path: str) -> Calibration | None:
    """Load a calibration file from disk (returns None when missing/invalid)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return None
        return Calibration.from_dict(data)
    except Exception:
        return None


def save_calibration_file(path: str, calibration: Calibration) -> None:
    """Save a calibration file to disk (creates parent dirs)."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibration.to_dict(), f, indent=2)
