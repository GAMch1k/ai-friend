from __future__ import annotations

import os
import time
import cv2

# --- Fix für Adafruit Versionsproblem ---
import adafruit_platformdetect.constants.chips as ap_chip
if not hasattr(ap_chip, "RP2350"):
    ap_chip.RP2350 = -1

import board
import digitalio
import adafruit_ssd1306
from PIL import Image, ImageDraw
from pitop import Button

# ----------------------------
# CONFIG
# ----------------------------
CAM_INDEX = 0
TOUCH_PORT = "D4"
OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_ADDR = 0x3C

CASCADE_PATH = "haarcascade_frontalface_default.xml"

FACE_ON_THRESHOLD = 3
FACE_OFF_THRESHOLD = 6
TOUCH_COOLDOWN = 0.4

# ----------------------------
# OLED SETUP
# ----------------------------
i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDR)

image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
draw = ImageDraw.Draw(image)

def oled_clear():
    draw.rectangle((0, 0, OLED_WIDTH, OLED_HEIGHT), outline=0, fill=0)

def oled_show():
    oled.image(image)
    oled.show()

def draw_sleep():
    oled_clear()
    draw.line((25, 34, 55, 34), fill=1)
    draw.line((75, 34, 105, 34), fill=1)
    oled_show()

def draw_blink():
    oled_clear()
    draw.line((25, 32, 55, 32), fill=1)
    draw.line((75, 32, 105, 32), fill=1)
    oled_show()

def draw_neutral():
    oled_clear()
    draw.ellipse((22, 18, 58, 48), outline=1, fill=1)
    draw.ellipse((70, 18, 106, 48), outline=1, fill=1)
    draw.ellipse((38, 30, 42, 34), outline=0, fill=0)
    draw.ellipse((86, 30, 90, 34), outline=0, fill=0)
    oled_show()

def draw_happy():
    oled_clear()
    draw.ellipse((22, 18, 58, 48), outline=1, fill=1)
    draw.ellipse((70, 18, 106, 48), outline=1, fill=1)
    draw.ellipse((35, 28, 45, 38), outline=0, fill=0)
    draw.ellipse((83, 28, 93, 38), outline=0, fill=0)
    oled_show()

def draw_love():
    oled_clear()
    draw.ellipse((22, 18, 58, 48), outline=1, fill=1)
    draw.ellipse((70, 18, 106, 48), outline=1, fill=1)
    draw.ellipse((32, 25, 48, 41), outline=0, fill=0)
    draw.ellipse((80, 25, 96, 41), outline=0, fill=0)
    oled_show()

def render_state(state: str):
    if state == "sleep":
        draw_sleep()
    elif state == "neutral":
        draw_neutral()
    elif state == "happy":
        draw_happy()
    elif state == "love":
        draw_love()
    elif state == "blink":
        draw_blink()

# ----------------------------
# TOUCH SETUP (FIXED)
# ----------------------------
touch = Button(TOUCH_PORT)
last_touch_state = touch.is_pressed
last_touch_time = 0.0
last_touch_release_time = 0.0

def poll_touch_event():
    global last_touch_state, last_touch_time, last_touch_release_time

    state = touch.is_pressed
    event = False
    now = time.time()

    # active-low: True = idle, False = touched
    if last_touch_state is True and state is False:
        if now - last_touch_time > TOUCH_COOLDOWN:
            event = True
            last_touch_time = now

    if state is True:
        last_touch_release_time = now

    last_touch_state = state
    return event

# ----------------------------
# CAMERA SETUP (FIXED)
# ----------------------------
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Cascade file not found: {CASCADE_PATH}")

cascade = cv2.CascadeClassifier(CASCADE_PATH)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# weniger Auflösung = weniger RAM + stabiler
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

face_seen_count = 0
face_missing_count = 0
face_present = False

def poll_face_state():
    global face_seen_count, face_missing_count, face_present

    ret, frame = cap.read()
    if not ret:
        return face_present

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )

    seen_now = len(faces) > 0

    if seen_now:
        face_seen_count += 1
        face_missing_count = 0
    else:
        face_missing_count += 1
        face_seen_count = 0

    if not face_present and face_seen_count >= FACE_ON_THRESHOLD:
        face_present = True

    if face_present and face_missing_count >= FACE_OFF_THRESHOLD:
        face_present = False

    return face_present

# ----------------------------
# MAIN SYSTEM
# ----------------------------
affinity = 0
last_seen_face_time = 0
last_rendered_state = None
next_blink_time = time.time() + 4

def state_from_affinity(value: int):
    if value >= 60:
        return "love"
    elif value >= 20:
        return "happy"
    return "neutral"

try:
    print("START AI FRIEND")
    render_state("sleep")

    while True:
        now = time.time()

        current_face = poll_face_state()
        if current_face:
            last_seen_face_time = now

        if poll_touch_event():
            if current_face:
                affinity = min(100, affinity + 15)
            else:
                affinity = min(100, affinity + 8)
            print(f"TOUCH -> affinity={affinity}")

        if affinity > 0 and now - last_touch_time > 6:
            affinity -= 1

        if now - last_seen_face_time > 15:
            state = "sleep"
        else:
            state = state_from_affinity(affinity)

        if state != "sleep" and now >= next_blink_time:
            render_state("blink")
            time.sleep(0.12)
            render_state(state)
            next_blink_time = now + 4

        if state != last_rendered_state:
            render_state(state)
            print(f"FACE={current_face} state={state} affinity={affinity}")
            last_rendered_state = state

        time.sleep(0.03)

except KeyboardInterrupt:
    print("STOPPED")

finally:
    cap.release()
    render_state("sleep")
