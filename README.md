# AI Magic Invisibility Portal

A real-time full-body "ghost" effect powered by MediaPipe. The camera sees your
whole silhouette; opening a hand pinch turns your body transparent, revealing a
pre-captured background through you — like a magic cloak of invisibility. Both
hands are tracked and drawn as glowing skeletons on screen.

Built with Python, OpenCV, MediaPipe (Tasks API), and NumPy.

## Features

- **Full-body ghost effect**: live selfie segmentation blends your silhouette
  toward the pre-captured background, so your body turns see-through.
- **Pinch-driven opacity**: the wider your thumb-index pinch, the more ghostly
  you become (`PORTAL %` in the HUD tracks it live).
- **Two-hand tracking**: up to two hands tracked simultaneously (`HANDS` count
  in the HUD), each drawn as a 21-landmark skeleton overlay.
- **Magic silhouette glow**: an additive neon halo hugs the body boundary.
- **Four control gestures** with a 1-second hold timer to prevent accidental triggers:
  - **OK sign** — capture / refresh the background
  - **Peace sign** — toggle ghost visibility
  - **Closed fist** — pause / resume the effect
  - **Open palm** — restore defaults
- **Live HUD**: top status bar plus a bottom stats bar (`PORTAL %`, `HANDS`,
  `PINCH`, gesture, FPS) updating every frame.
- **Graceful edge cases**: lost hand tracking, missing webcam, and low-light
  detection drops are all handled without crashing.

## Requirements

- Python 3.9 - 3.14
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

On first run the app downloads two MediaPipe models into `models/` automatically:

- `hand_landmarker.task` (~7.8 MB) — two-hand tracking
- `selfie_segmenter.tflite` (~250 KB) — body silhouette segmentation

You can instead download them manually:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -L -o models/selfie_segmenter.tflite \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
```

## Usage

Run the application:

```bash
python main.py
```

On startup the app waits 3 seconds, then automatically captures the baseline
background. **Keep yourself out of view until capture finishes.** You can
re-capture at any time with the `C` key.

To use the ghost effect:

1. Step into frame and hold up both hands.
2. Spread your thumb and index finger apart — your body turns transparent in
   proportion to the pinch.
3. Close the pinch again to become solid.

Hold a control gesture for about 1 second to trigger its action:

| Gesture | Action |
|---|---|
| OK sign (thumb + index touching, other fingers up) | Capture background |
| Peace sign (index + middle up) | Toggle ghost visibility |
| Closed fist (all fingers curled) | Pause / resume the effect |
| Open palm (all fingers extended) | Reset to default state |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Recapture / refresh background |
| `B` | Toggle ghost visibility |
| `P` | Pause / resume the effect |
| `F` | Toggle fullscreen |
| `Q` / `ESC` | Quit |

## Project Structure

```
AI-Magic-Invisibility-Portal/
├── main.py                # Main application loop, frame capture, and orchestration
├── gesture_recognizer.py  # Two-hand tracking, gestures, skeleton + pinch math
├── segmenter.py           # Selfie segmentation (body silhouette mask)
├── portal.py              # Ghost compositing, background blending, silhouette glow
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## How It Works

1. Each frame is captured from the webcam and mirrored horizontally.
2. The MediaPipe `HandLandmarker` (Tasks API) tracks up to two hands; their
   21-landmark skeletons are drawn on screen, and a pinch ratio is computed from
   the thumb-index distance normalized by hand size.
3. The MediaPipe `ImageSegmenter` (selfie model) produces a soft person mask.
4. The widest pinch across both hands sets the ghost opacity, which is
   EMA-smoothed and fed to the compositor: the person silhouette is alpha-blended
   toward the pre-captured background, with an additive glow around the boundary.
5. Control gestures are classified from finger-extended states and debounced
   through a 1-second hold timer.

## Troubleshooting

- **"Could not open the webcam"**: verify the camera is connected and not in use
  by another application.
- **No hand detected**: improve lighting and keep your hands fully in frame.
- **Ghost won't activate**: make sure the background was captured (`BG: READY` in
  the HUD) and spread your thumb and index fingers wide.
- **Gestures not triggering**: hold the pose still for a full second.
