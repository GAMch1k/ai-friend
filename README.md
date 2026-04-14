# AI Friend

This project turns the original `friend_core_final.py` prototype into a small Python app for a Raspberry Pi or pi-top style robot with:

- camera-based face detection
- local face profile storage in SQLite
- per-person affinity
- touch-based affinity increases
- OLED emotion rendering

## Structure

- `friend_core_final.py`: original prototype kept for reference
- `ai_friend/`: modular runtime package
- `run_friend.py`: simple entrypoint
- `tests/`: small unittest suite for storage and runtime logic

## Runtime Notes

The app targets the same hardware assumptions as the prototype:

- OpenCV camera capture
- pi-top touch button on `D4`
- SSD1306 OLED over I2C

If the OLED or touch libraries are missing, the app falls back to console display and a no-op touch sensor. Camera capture still requires OpenCV.

## Setup

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

## Run

```powershell
venv\Scripts\python.exe run_friend.py
```

Useful environment variables:

- `FRIEND_DB_PATH`
- `FRIEND_CAMERA_INDEX`
- `FRIEND_AUTO_ENROLL_FRAMES`
- `FRIEND_FACE_MATCH_THRESHOLD`
- `FRIEND_SIMULATED_HARDWARE`

## Recognition Approach

This version keeps the project simple:

- detect faces with Haar cascades
- build a compact grayscale face signature
- save focal points from `goodFeaturesToTrack`
- match against saved signatures in SQLite

It is intentionally lightweight for a school project, not production-grade biometrics.
