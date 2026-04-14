from __future__ import annotations

import time

from .config import Settings
from .hardware import HardwareBundle, build_hardware
from .models import EmotionState, FaceObservation, PersonProfile, RuntimeStatus
from .storage import FriendRepository, utc_now_iso
from .vision import FaceRecognitionService


def state_from_affinity(affinity: int) -> EmotionState:
    if affinity >= 60:
        return EmotionState.LOVE
    if affinity >= 20:
        return EmotionState.HAPPY
    return EmotionState.NEUTRAL


class FriendRuntime:
    def __init__(
        self,
        settings: Settings,
        repository: FriendRepository,
        hardware: HardwareBundle,
        vision: FaceRecognitionService | object,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.hardware = hardware
        self.vision = vision

        now = time.time()
        self.face_seen_count = 0
        self.face_missing_count = 0
        self.face_present = False
        self.last_seen_face_time = 0.0
        self.last_rendered_state: EmotionState | None = None
        self.next_blink_time = now + settings.blink_interval
        self.tracked_person_id: int | None = None
        self.pending_unknown_frames = 0

    def run_forever(self) -> None:
        print("START AI FRIEND")
        self._render(EmotionState.SLEEP)
        try:
            while True:
                self.tick()
                time.sleep(self.settings.loop_delay)
        except KeyboardInterrupt:
            print("STOPPED")
        finally:
            self.hardware.camera.close()
            self._render(EmotionState.SLEEP)
            self.repository.close()

    def tick(self, now: float | None = None) -> RuntimeStatus:
        current_time = time.time() if now is None else now
        frame = self.hardware.camera.read()
        observations = self.vision.analyze(frame)
        primary = observations[0] if observations else None

        self._update_face_presence(primary)
        tracked_person = self._update_tracking(primary, current_time)

        if self.hardware.touch_sensor.poll_event(
            current_time, self.settings.touch_cooldown
        ) and tracked_person is not None:
            tracked_person = self.repository.update_affinity(
                tracked_person.id,
                self.settings.touch_affinity_increment,
            )

        state = self._current_state(current_time, tracked_person)
        self._maybe_blink(current_time, state)
        self._render(state)
        return RuntimeStatus(
            state=state,
            face_present=self.face_present,
            tracked_person=tracked_person,
            observations=observations,
        )

    def _update_face_presence(self, primary: FaceObservation | None) -> None:
        seen_now = primary is not None
        if seen_now:
            self.face_seen_count += 1
            self.face_missing_count = 0
        else:
            self.face_missing_count += 1
            self.face_seen_count = 0

        if not self.face_present and self.face_seen_count >= self.settings.face_on_threshold:
            self.face_present = True

        if self.face_present and self.face_missing_count >= self.settings.face_off_threshold:
            self.face_present = False
            self.tracked_person_id = None
            self.pending_unknown_frames = 0

    def _update_tracking(
        self, primary: FaceObservation | None, current_time: float
    ) -> PersonProfile | None:
        if primary is None:
            return self.current_person()

        self.last_seen_face_time = current_time
        if primary.person is not None:
            self.tracked_person_id = primary.person.id
            self.pending_unknown_frames = 0
            return self.repository.record_seen(primary.person.id, utc_now_iso())

        if self.tracked_person_id is None:
            self.pending_unknown_frames += 1
            if self.pending_unknown_frames >= self.settings.auto_enroll_frames:
                created = self.repository.create_person(
                    primary.signature,
                    primary.focal_points,
                )
                self.tracked_person_id = created.id
                self.pending_unknown_frames = 0
                return self.repository.record_seen(created.id, utc_now_iso())
            return None

        self.pending_unknown_frames = 0
        return self.repository.record_seen(self.tracked_person_id, utc_now_iso())

    def _current_state(
        self, now: float, tracked_person: PersonProfile | None
    ) -> EmotionState:
        if now - self.last_seen_face_time > self.settings.sleep_timeout:
            return EmotionState.SLEEP
        affinity = tracked_person.affinity if tracked_person is not None else 0
        return state_from_affinity(affinity)

    def _maybe_blink(self, now: float, state: EmotionState) -> None:
        if state == EmotionState.SLEEP or now < self.next_blink_time:
            return
        self._render(EmotionState.BLINK)
        time.sleep(self.settings.blink_duration)
        self.next_blink_time = now + self.settings.blink_interval

    def _render(self, state: EmotionState) -> None:
        if state == self.last_rendered_state:
            return
        self.hardware.display.render(state)
        self.last_rendered_state = state

    def current_person(self) -> PersonProfile | None:
        if self.tracked_person_id is None:
            return None
        return self.repository.get_person(self.tracked_person_id)


def main() -> None:
    settings = Settings.from_env()
    repository = FriendRepository(settings.database_path)
    hardware = build_hardware(settings)
    vision = FaceRecognitionService(settings, repository)
    runtime = FriendRuntime(settings, repository, hardware, vision)
    runtime.run_forever()


if __name__ == "__main__":
    main()
