from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .models import EmotionState


class BaseDisplay:
    def render(self, state: EmotionState) -> None:
        raise NotImplementedError


class BaseTouchSensor:
    def poll_event(self, now: float, cooldown: float) -> bool:
        raise NotImplementedError


class BaseCamera:
    def read(self):
        raise NotImplementedError

    def close(self) -> None:
        return None


class ConsoleDisplay(BaseDisplay):
    def __init__(self) -> None:
        self.last_state: EmotionState | None = None

    def render(self, state: EmotionState) -> None:
        if state != self.last_state:
            print(f"[display] {state.value}")
            self.last_state = state


class NoopTouchSensor(BaseTouchSensor):
    def poll_event(self, now: float, cooldown: float) -> bool:
        return False


class OpenCVCamera(BaseCamera):
    def __init__(self, settings: Settings) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for camera capture. Install dependencies first."
            ) from exc

        self.cv2 = cv2
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        self.capture = cv2.VideoCapture(settings.camera_index, backend)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        self.capture.release()


class PiTopTouchSensor(BaseTouchSensor):
    def __init__(self, touch_port: str) -> None:
        try:
            from pitop import Button  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pi-top touch support requires the pitop package on the device."
            ) from exc

        self.button = Button(touch_port)
        self.last_touch_state = self.button.is_pressed
        self.last_touch_time = 0.0

    def poll_event(self, now: float, cooldown: float) -> bool:
        state = self.button.is_pressed
        event = False

        if self.last_touch_state is True and state is False:
            if now - self.last_touch_time > cooldown:
                event = True
                self.last_touch_time = now

        self.last_touch_state = state
        return event


class PiOledDisplay(BaseDisplay):
    def __init__(self, settings: Settings) -> None:
        try:
            import adafruit_platformdetect.constants.chips as ap_chip  # type: ignore
            import adafruit_ssd1306  # type: ignore
            import board  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OLED support requires Adafruit and Pillow packages on the device."
            ) from exc

        if not hasattr(ap_chip, "RP2350"):
            ap_chip.RP2350 = -1

        self.oled_width = settings.oled_width
        self.oled_height = settings.oled_height
        i2c = board.I2C()
        self.oled = adafruit_ssd1306.SSD1306_I2C(
            settings.oled_width,
            settings.oled_height,
            i2c,
            addr=settings.oled_addr,
        )
        self.image = Image.new("1", (settings.oled_width, settings.oled_height))
        self.draw = ImageDraw.Draw(self.image)

    def render(self, state: EmotionState) -> None:
        self.draw.rectangle(
            (0, 0, self.oled_width, self.oled_height),
            outline=0,
            fill=0,
        )

        if state == EmotionState.SLEEP:
            self.draw.line((25, 34, 55, 34), fill=1)
            self.draw.line((75, 34, 105, 34), fill=1)
        elif state == EmotionState.BLINK:
            self.draw.line((25, 32, 55, 32), fill=1)
            self.draw.line((75, 32, 105, 32), fill=1)
        else:
            self.draw.ellipse((22, 18, 58, 48), outline=1, fill=1)
            self.draw.ellipse((70, 18, 106, 48), outline=1, fill=1)
            if state == EmotionState.NEUTRAL:
                left_pupil = (38, 30, 42, 34)
                right_pupil = (86, 30, 90, 34)
            elif state == EmotionState.HAPPY:
                left_pupil = (35, 28, 45, 38)
                right_pupil = (83, 28, 93, 38)
            else:
                left_pupil = (32, 25, 48, 41)
                right_pupil = (80, 25, 96, 41)
            self.draw.ellipse(left_pupil, outline=0, fill=0)
            self.draw.ellipse(right_pupil, outline=0, fill=0)

        self.oled.image(self.image)
        self.oled.show()


@dataclass(slots=True)
class HardwareBundle:
    display: BaseDisplay
    touch_sensor: BaseTouchSensor
    camera: BaseCamera


def build_hardware(settings: Settings) -> HardwareBundle:
    camera = OpenCVCamera(settings)

    if settings.use_simulated_hardware:
        return HardwareBundle(
            display=ConsoleDisplay(),
            touch_sensor=NoopTouchSensor(),
            camera=camera,
        )

    try:
        display = PiOledDisplay(settings)
    except RuntimeError:
        display = ConsoleDisplay()

    try:
        touch_sensor = PiTopTouchSensor(settings.touch_port)
    except RuntimeError:
        touch_sensor = NoopTouchSensor()

    return HardwareBundle(display=display, touch_sensor=touch_sensor, camera=camera)
