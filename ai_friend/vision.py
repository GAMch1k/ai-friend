from __future__ import annotations

import threading
from pathlib import Path

from .config import Settings
from .models import FaceObservation
from .storage import FriendRepository


class FaceRecognitionService:
    def __init__(self, settings: Settings, repository: FriendRepository) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for face detection and recognition."
            ) from exc

        self.cv2 = cv2
        self.settings = settings
        self.repository = repository

        cascade_path = settings.cascade_path or self._default_cascade_path()
        if not Path(cascade_path).exists():
            raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {cascade_path}")

    def analyze(self, frame) -> list[FaceObservation]:
        if frame is None:
            return []

        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )

        observations: list[FaceObservation] = []
        for x, y, width, height in faces:
            face_region = gray[y : y + height, x : x + width]
            if face_region.size == 0:
                continue

            signature = self._extract_signature(face_region)
            focal_points = self._extract_focal_points(face_region)
            person, confidence = self.repository.find_best_match(
                signature,
                self.settings.face_match_threshold,
            )
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(width), int(height)),
                    signature=signature,
                    focal_points=focal_points,
                    confidence=confidence,
                    person=person,
                )
            )

        observations.sort(key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)
        return observations

    def _default_cascade_path(self) -> str:
        if hasattr(self.cv2, "data") and hasattr(self.cv2.data, "haarcascades"):
            return str(Path(self.cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        return "haarcascade_frontalface_default.xml"

    def _extract_signature(self, face_region) -> list[float]:
        size = self.settings.face_signature_size
        resized = self.cv2.resize(face_region, (size, size))
        normalized = resized.astype("float32") / 255.0
        return [round(float(value), 6) for value in normalized.reshape(-1).tolist()]

    def _extract_focal_points(self, face_region) -> list[tuple[float, float]]:
        points = self.cv2.goodFeaturesToTrack(
            face_region,
            maxCorners=self.settings.max_focal_points,
            qualityLevel=0.05,
            minDistance=4,
        )
        height, width = face_region.shape[:2]
        if points is None or width == 0 or height == 0:
            return []

        normalized_points: list[tuple[float, float]] = []
        for point in points:
            x_coord, y_coord = point.ravel()
            normalized_points.append(
                (
                    round(float(x_coord) / float(width), 4),
                    round(float(y_coord) / float(height), 4),
                )
            )
        return normalized_points


class AsyncFaceRecognitionService:
    def __init__(self, service: FaceRecognitionService) -> None:
        self.service = service
        self._condition = threading.Condition()
        self._latest_observations: list[FaceObservation] = []
        self._pending_frame = None
        self._stop_requested = False
        self._worker_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="face-recognition-worker",
            daemon=True,
        )
        self._thread.start()

    def analyze(self, frame) -> list[FaceObservation]:
        self._raise_worker_error()
        if frame is not None:
            frame_to_queue = frame.copy() if hasattr(frame, "copy") else frame
            with self._condition:
                self._pending_frame = frame_to_queue
                self._condition.notify()

        with self._condition:
            return list(self._latest_observations)

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._pending_frame = None
            self._condition.notify_all()

        self._thread.join(timeout=2.0)
        self._raise_worker_error()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_frame is None and not self._stop_requested:
                    self._condition.wait()

                if self._stop_requested:
                    return

                frame = self._pending_frame
                self._pending_frame = None

            try:
                observations = self.service.analyze(frame)
            except BaseException as exc:
                with self._condition:
                    self._worker_error = exc
                    self._stop_requested = True
                return

            with self._condition:
                self._latest_observations = observations

    def _raise_worker_error(self) -> None:
        with self._condition:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Face recognition worker failed") from error
