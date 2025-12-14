"""
Configuration management using pydantic-settings.

Loads settings from environment variables and .env file.
Provides type-safe access with validation.
"""

import os
from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────
# MODE ENUMS
# ─────────────────────────────────────────────────────────────


class RobotMode(str, Enum):
    """Robot execution mode."""

    MOCK = "mock"  # Shows SmolVLA instruction, doesn't execute
    SMOLVLA = "smolvla"  # Real SmolVLA execution


class VisionMode(str, Enum):
    """Vision detection mode."""

    MOCK = "mock"  # Returns predefined/manual board state
    REAL = "real"  # Uses camera + board_detector


# ─────────────────────────────────────────────────────────────
# NESTED SETTINGS
# ─────────────────────────────────────────────────────────────


class RobotSettings(BaseSettings):
    """Robot arm configuration."""

    model_config = SettingsConfigDict(env_prefix="ROBOT_")

    mode: RobotMode = RobotMode.MOCK
    port: str = "/dev/ttyACM1"
    home_position: list[float] = Field(
        default=[0.0, -90.0, 90.0, 0.0, 0.0, 0.0],
        description="Home position in degrees for 6 joints",
    )
    move_delay: float = Field(
        default=1.0,
        description="Simulated move delay in seconds (mock mode)",
    )


class VisionSettings(BaseSettings):
    """Vision/camera configuration."""

    model_config = SettingsConfigDict(env_prefix="VISION_")

    mode: VisionMode = VisionMode.MOCK
    calibration_path: str = "mission2/vision/calibration.json"


class StreamSettings(BaseSettings):
    """ZMQ stream configuration."""

    model_config = SettingsConfigDict(env_prefix="STREAM_")

    host: str = "192.168.1.2"
    port: int = 5555
    output_dir: str = "outputs/stream"
    frame_path: str = "outputs/stream/latest.jpg"
    capture_timeout_ms: int = 5000


class GameSettings(BaseSettings):
    """Game rules configuration."""

    model_config = SettingsConfigDict(env_prefix="GAME_")

    board_size: int = 5
    win_length: int = 4


class AISettings(BaseSettings):
    """AI player configuration."""

    model_config = SettingsConfigDict(env_prefix="AI_")

    depth: int = Field(default=5, ge=1, le=10, description="Minimax search depth")
    player: Literal["orange", "yellow"] = "orange"


class TARSSettings(BaseSettings):
    """TARS personality configuration."""

    model_config = SettingsConfigDict(env_prefix="TARS_")

    enabled: bool = False
    humor: int = Field(default=75, ge=0, le=100)
    honesty: int = Field(default=90, ge=0, le=100)
    gemini_model: str = "gemini-1.5-flash"
    gemini_api_key: str = ""


class UISettings(BaseSettings):
    """UI/calibrator configuration."""

    model_config = SettingsConfigDict(env_prefix="UI_")

    sidebar_width_px: int = 520
    corner_mode: Literal["auto", "fixed"] = "auto"
    grid_method: Literal["sample_hsv", "hough", "hybrid"] = "sample_hsv"


# ─────────────────────────────────────────────────────────────
# MAIN SETTINGS
# ─────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Main application settings.

    Loads from .env file and environment variables.
    Environment variables take precedence over .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars
    )

    robot: RobotSettings = Field(default_factory=RobotSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    game: GameSettings = Field(default_factory=GameSettings)
    ai: AISettings = Field(default_factory=AISettings)
    tars: TARSSettings = Field(default_factory=TARSSettings)
    ui: UISettings = Field(default_factory=UISettings)


# ─────────────────────────────────────────────────────────────
# SINGLETON ACCESS
# ─────────────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        # Backward-compat env aliases (older .env used MISSION2_* keys).
        aliases = [
            ("MISSION2_CALIBRATION_PATH", "VISION_CALIBRATION_PATH"),
            ("MISSION2_EMITTER_HOST", "STREAM_HOST"),
            ("MISSION2_EMITTER_PORT", "STREAM_PORT"),
            ("MISSION2_CAPTURE_TIMEOUT_MS", "STREAM_CAPTURE_TIMEOUT_MS"),
            ("MISSION2_SIDEBAR_WIDTH_PX", "UI_SIDEBAR_WIDTH_PX"),
            ("MISSION2_CORNER_MODE", "UI_CORNER_MODE"),
            ("MISSION2_GRID_METHOD", "UI_GRID_METHOD"),
        ]
        for legacy_key, new_key in aliases:
            if os.environ.get(new_key, "").strip():
                continue
            legacy_value = os.environ.get(legacy_key, "").strip()
            if legacy_value:
                os.environ[new_key] = legacy_value
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
