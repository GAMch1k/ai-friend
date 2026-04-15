from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from ai_friend.app import FriendRuntime
from ai_friend.config import Settings
from ai_friend.hardware import BaseCamera, BaseDisplay, BaseTouchSensor, HardwareBundle
from ai_friend.models import EmotionState, FaceObservation
from ai_friend.storage import FriendRepository


class FakeDisplay(BaseDisplay):
    def __init__(self) -> None:
        self.rendered: list[EmotionState] = []

    def render(self, state: EmotionState) -> None:
        self.rendered.append(state)


class FakeTouchSensor(BaseTouchSensor):
    def __init__(self, events: list[bool]) -> None:
        self.events = events

    def poll_event(self, now: float, cooldown: float) -> bool:
        if not self.events:
            return False
        return self.events.pop(0)


class FakeCamera(BaseCamera):
    def read(self):
        return object()


@dataclass
class FakeVision:
    sequences: list[list[FaceObservation]]

    def analyze(self, frame) -> list[FaceObservation]:
        if self.sequences:
            return self.sequences.pop(0)
        return []


class FriendRuntimeTests(unittest.TestCase):
    def test_auto_enroll_and_touch_updates_affinity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FriendRepository(Path(temp_dir) / "friend.db")
            try:
                settings = Settings(
                    database_path=Path(temp_dir) / "friend.db",
                    face_on_threshold=1,
                    face_off_threshold=1,
                    auto_enroll_frames=2,
                    sleep_timeout=60.0,
                    blink_interval=999.0,
                )
                display = FakeDisplay()
                touch = FakeTouchSensor([False, False, True])
                camera = FakeCamera()
                unknown = FaceObservation(
                    bbox=(0, 0, 50, 50),
                    signature=[0.1, 0.2, 0.3],
                    focal_points=[(0.2, 0.2)],
                    confidence=0.0,
                    person=None,
                )
                vision = FakeVision([[unknown], [unknown], [unknown]])
                runtime = FriendRuntime(
                    settings=settings,
                    repository=repository,
                    hardware=HardwareBundle(display=display, touch_sensor=touch, camera=camera),
                    vision=vision,
                )

                first = runtime.tick(now=1.0)
                self.assertTrue(first.face_present)
                self.assertIsNone(first.tracked_person)
                self.assertEqual(first.state, EmotionState.NEUTRAL)

                second = runtime.tick(now=2.0)
                self.assertIsNotNone(second.tracked_person)
                self.assertEqual(second.tracked_person.affinity, 0)

                third = runtime.tick(now=3.0)
                self.assertIsNotNone(third.tracked_person)
                self.assertEqual(third.tracked_person.affinity, 15)
            finally:
                repository.close()

    def test_missing_face_returns_to_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FriendRepository(Path(temp_dir) / "friend.db")
            try:
                settings = Settings(
                    database_path=Path(temp_dir) / "friend.db",
                    face_on_threshold=1,
                    face_off_threshold=1,
                    sleep_timeout=1.0,
                    blink_interval=999.0,
                )
                display = FakeDisplay()
                touch = FakeTouchSensor([False, False])
                camera = FakeCamera()
                known = repository.create_person(
                    signature=[0.1, 0.2, 0.3],
                    focal_points=[(0.3, 0.3)],
                )
                observed = FaceObservation(
                    bbox=(0, 0, 50, 50),
                    signature=known.face_signature,
                    focal_points=known.focal_points,
                    confidence=0.9,
                    person=known,
                )
                vision = FakeVision([[observed], []])
                runtime = FriendRuntime(
                    settings=settings,
                    repository=repository,
                    hardware=HardwareBundle(display=display, touch_sensor=touch, camera=camera),
                    vision=vision,
                )

                awake = runtime.tick(now=1.0)
                self.assertEqual(awake.state, EmotionState.NEUTRAL)

                asleep = runtime.tick(now=3.0)
                self.assertEqual(asleep.state, EmotionState.SLEEP)
            finally:
                repository.close()

    def test_recognized_face_prints_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = FriendRepository(Path(temp_dir) / "friend.db")
            try:
                settings = Settings(
                    database_path=Path(temp_dir) / "friend.db",
                    face_on_threshold=1,
                    face_off_threshold=1,
                    sleep_timeout=60.0,
                    blink_interval=999.0,
                )
                display = FakeDisplay()
                touch = FakeTouchSensor([False])
                camera = FakeCamera()
                known = repository.create_person(
                    signature=[0.1, 0.2, 0.3],
                    focal_points=[(0.3, 0.3)],
                    display_name="Alice",
                )
                observed = FaceObservation(
                    bbox=(0, 0, 50, 50),
                    signature=known.face_signature,
                    focal_points=known.focal_points,
                    confidence=0.9,
                    person=known,
                )
                vision = FakeVision([[observed]])
                runtime = FriendRuntime(
                    settings=settings,
                    repository=repository,
                    hardware=HardwareBundle(display=display, touch_sensor=touch, camera=camera),
                    vision=vision,
                )

                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    runtime.tick(now=1.0)

                self.assertIn("RECOGNIZED_FACE name=Alice", buffer.getvalue())
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
