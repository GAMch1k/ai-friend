from __future__ import annotations

import time
import unittest

from ai_friend.vision import AsyncFaceRecognitionService


class FakeRecognitionService:
    def __init__(self) -> None:
        self.frames: list[object] = []

    def analyze(self, frame) -> list[object]:
        time.sleep(0.01)
        self.frames.append(frame)
        return [frame]


class AsyncFaceRecognitionServiceTests(unittest.TestCase):
    def test_worker_processes_frames_outside_main_thread(self) -> None:
        service = AsyncFaceRecognitionService(FakeRecognitionService())
        try:
            first = service.analyze("frame-1")
            self.assertEqual(first, [])

            latest: list[object] = []
            deadline = time.time() + 1.0
            while time.time() < deadline:
                latest = service.analyze(None)
                if latest == ["frame-1"]:
                    break
                time.sleep(0.01)

            self.assertEqual(latest, ["frame-1"])
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
