from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EmotionState(StrEnum):
    SLEEP = "sleep"
    NEUTRAL = "neutral"
    HAPPY = "happy"
    LOVE = "love"
    BLINK = "blink"


@dataclass(slots=True)
class PersonProfile:
    id: int
    display_name: str
    face_signature: list[float]
    focal_points: list[tuple[float, float]]
    affinity: int
    created_at: str
    last_seen_at: str | None = None


@dataclass(slots=True)
class FaceObservation:
    bbox: tuple[int, int, int, int]
    signature: list[float]
    focal_points: list[tuple[float, float]]
    confidence: float
    person: PersonProfile | None = None


@dataclass(slots=True)
class RuntimeStatus:
    state: EmotionState
    face_present: bool
    tracked_person: PersonProfile | None
    observations: list[FaceObservation] = field(default_factory=list)
