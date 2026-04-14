from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    camera_index: int = 0
    frame_width: int = 320
    frame_height: int = 240
    touch_port: str = "D4"
    oled_width: int = 128
    oled_height: int = 64
    oled_addr: int = 0x3C
    face_on_threshold: int = 3
    face_off_threshold: int = 6
    touch_cooldown: float = 0.4
    sleep_timeout: float = 15.0
    blink_interval: float = 4.0
    blink_duration: float = 0.12
    loop_delay: float = 0.03
    auto_enroll_frames: int = 10
    touch_affinity_increment: int = 15
    face_match_threshold: float = 0.14
    face_signature_size: int = 32
    max_focal_points: int = 16
    database_path: Path = Path("friend.db")
    cascade_path: str | None = None
    use_simulated_hardware: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            camera_index=int(os.getenv("FRIEND_CAMERA_INDEX", "0")),
            frame_width=int(os.getenv("FRIEND_FRAME_WIDTH", "320")),
            frame_height=int(os.getenv("FRIEND_FRAME_HEIGHT", "240")),
            touch_port=os.getenv("FRIEND_TOUCH_PORT", "D4"),
            oled_width=int(os.getenv("FRIEND_OLED_WIDTH", "128")),
            oled_height=int(os.getenv("FRIEND_OLED_HEIGHT", "64")),
            oled_addr=int(os.getenv("FRIEND_OLED_ADDR", "0x3C"), 0),
            face_on_threshold=int(os.getenv("FRIEND_FACE_ON_THRESHOLD", "3")),
            face_off_threshold=int(os.getenv("FRIEND_FACE_OFF_THRESHOLD", "6")),
            touch_cooldown=float(os.getenv("FRIEND_TOUCH_COOLDOWN", "0.4")),
            sleep_timeout=float(os.getenv("FRIEND_SLEEP_TIMEOUT", "15.0")),
            blink_interval=float(os.getenv("FRIEND_BLINK_INTERVAL", "4.0")),
            blink_duration=float(os.getenv("FRIEND_BLINK_DURATION", "0.12")),
            loop_delay=float(os.getenv("FRIEND_LOOP_DELAY", "0.03")),
            auto_enroll_frames=int(os.getenv("FRIEND_AUTO_ENROLL_FRAMES", "10")),
            touch_affinity_increment=int(
                os.getenv("FRIEND_TOUCH_AFFINITY_INCREMENT", "15")
            ),
            face_match_threshold=float(
                os.getenv("FRIEND_FACE_MATCH_THRESHOLD", "0.14")
            ),
            face_signature_size=int(os.getenv("FRIEND_FACE_SIGNATURE_SIZE", "32")),
            max_focal_points=int(os.getenv("FRIEND_MAX_FOCAL_POINTS", "16")),
            database_path=Path(os.getenv("FRIEND_DB_PATH", "friend.db")),
            cascade_path=os.getenv("FRIEND_CASCADE_PATH") or None,
            use_simulated_hardware=_read_bool("FRIEND_SIMULATED_HARDWARE", False),
        )
