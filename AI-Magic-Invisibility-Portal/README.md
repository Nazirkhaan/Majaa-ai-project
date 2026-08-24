# AI Magic Invisibility Portal

A real-time "invisibility" effect driven by hand gestures. Point your index finger
at the camera to open a glowing portal that reveals a pre-captured background,
making your hand (or anything behind it) appear to disappear through the magic
opening.

Built with Python, OpenCV, MediaPipe, and NumPy.

## Features

- **Gesture-driven portal**: track your index fingertip to move the portal; pinch
  your thumb and index finger to resize it.
- **Four control gestures** with a 1-second hold timer to prevent accidental triggers:
  - **OK sign** — capture / refresh the background
  - **Peace sign** — toggle portal visibility
  - **Closed fist** — pause / resume the effect
  - **Open palm** — restore defaults
- **Three portal shapes**: circle, square, hexagon (cycle with the `T` key).
- **Glowing border**: multi-layer halo and rim rendered with Gaussian blur and
  additive color mixing.
- **Lightweight HUD**: live status, tracking state, active gesture, and FPS.
- **Graceful edge cases**: lost hand tracking, missing webcam, and low-light
  detection drops are all handled without crashing.

## Requirements

- Python 3.9 - 3.11
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> **Note on MediaPipe versions**: MediaPipe 0.11+ removed the legacy
> `mp.solutions.hands` API used by this project. `requirements.txt` pins
> `mediapipe>=0.10.7,<0.11.0` to keep the classic API working.

## Usage

Run the application:

```bash
python main.py
```

On startup the app waits 3 seconds, then automatically captures the baseline
background. **Keep your hand out of view until capture finishes.** You can
re-capture at any time with the `C` key.

Hold a gesture for about 1 second to trigger its action:

| Gesture | Action |
|---|---|
| OK sign (thumb + index touching, other fingers up) | Capture background |
| Peace sign (index + middle up) | Toggle portal visibility |
| Closed fist (all fingers curled) | Pause / resume the effect |
| Open palm (all fingers extended) | Reset to default state |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Recapture / refresh background |
| `B` | Toggle portal visibility |
| `T` | Cycle portal shape (circle / square / hexagon) |
| `P` | Pause / resume the effect |
| `F` | Toggle fullscreen |
| `Q` / `ESC` | Quit |

## Project Structure

```
AI-Magic-Invisibility-Portal/
├── main.py                # Main application loop, frame capture, and orchestration
├── gesture_recognizer.py  # MediaPipe hand tracking, gesture state machine, shape/size math
├── portal.py              # Portal rendering, masking, glowing effects, background blending
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## How It Works

1. Each frame is captured from the webcam and mirrored horizontally.
2. MediaPipe Hands locates the hand; the index fingertip (EMA-smoothed) drives the
   portal position and the thumb-index pinch distance drives its radius.
3. Gestures are classified from finger-extended states and debounced through a
   1-second hold timer.
4. A feathered circular/polygonal mask blends the pre-captured background into the
   portal region of the live frame, and a glowing border is added additively.

## Troubleshooting

- **"Could not open the webcam"**: verify the camera is connected and not in use
  by another application.
- **No hand detected**: improve lighting and keep your hand fully in frame.
- **Gestures not triggering**: hold the pose still for a full second.
