"""Main application loop, frame capture, and orchestration.

Piecing everything together:

* Opens the webcam and runs a short warm-up that auto-captures the baseline
  background.
* Each frame is mirrored, analyzed by :class:`GestureRecognizer`, and composited
  by :class:`PortalRenderer`.
* Gesture actions and keyboard shortcuts drive the same control states.
* A minimal HUD reports status and shows transient event feedback.

Keyboard shortcuts:
    C       recapture / refresh the background
    B       toggle portal visibility
    T       cycle portal shape (circle / square / hexagon)
    P       pause / resume the effect
    F       toggle fullscreen
    Q, ESC  quit
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from gesture_recognizer import Gesture, GestureRecognizer
from portal import PortalRenderer, PortalShape

WINDOW_NAME = "AI Magic Invisibility Portal"
WARMUP_SECONDS = 3.0
DEFAULT_RADIUS = 120.0
EVENT_TTL_SECONDS = 2.0

Font = cv2.FONT_HERSHEY_SIMPLEX


def _draw_status_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int],
) -> None:
    """Draw a translucent status panel with colored text lines."""
    x, y = origin
    pad = 10
    line_h = 24
    scale = 0.6

    panel_w = 0
    for text, _ in lines:
        text_w = cv2.getTextSize(text, Font, scale, 1)[0][0]
        panel_w = max(panel_w, text_w)
    panel_w = min(panel_w + pad * 2, frame.shape[1] - x)
    panel_h = line_h * len(lines) + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0)
    frame[:] = blended

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x, y + i * line_h + 18), Font, scale, color, 1, cv2.LINE_AA)


def _draw_hint_bar(frame: np.ndarray) -> None:
    """Draw the keyboard hint bar pinned to the bottom edge."""
    hint = "[C] recapture  [B] portal  [T] shape  [P] pause  [F] fullscreen  [Q] quit"
    scale = 0.5
    h, w = frame.shape[:2]
    text_w = cv2.getTextSize(hint, Font, scale, 1)[0][0]
    x = (w - text_w) // 2
    y = h - 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 12, y - 22), (x + text_w + 12, y + 8), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0.0)
    frame[:] = blended
    cv2.putText(frame, hint, (x, y), Font, scale, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_center_event(frame: np.ndarray, message: str) -> None:
    """Draw a transient event message in the upper-center of the frame."""
    scale = 0.9
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(message, Font, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    y = 90
    cv2.putText(frame, message, (x, y), Font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _warmup(cap: cv2.VideoCapture, renderer: PortalRenderer, seconds: float) -> bool:
    """Run the startup countdown and auto-capture the background frame.

    Returns:
        ``True`` when a background was captured successfully.
    """
    start = time.monotonic()
    last = None
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        last = frame

        remaining = seconds - (time.monotonic() - start)
        _draw_center_event(frame, f"Clear the frame - capturing in {remaining:0.1f}s")
        cv2.putText(
            frame,
            "Keep your hand out of view until capture finishes",
            (frame.shape[1] // 2 - 220, 120),
            Font,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False

    return renderer.capture_background(last) if last is not None else False


def _toggle_fullscreen(window: str, enabled: bool) -> bool:
    """Toggle the named OpenCV window between normal and fullscreen."""
    prop = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)
    return enabled


def _handle_gesture_action(
    action: Gesture | None,
    *,
    renderer: PortalRenderer,
    portal_visible: bool,
    paused: bool,
    shape: PortalShape,
) -> tuple[bool, bool, PortalShape, str]:
    """Apply a triggered gesture action, returning updated control state."""
    event = ""
    if action is Gesture.OK_SIGN:
        event = "Background captured"
    elif action is Gesture.PEACE_SIGN:
        portal_visible = not portal_visible
        event = "Portal active" if portal_visible else "Portal hidden"
    elif action is Gesture.CLOSED_FIST:
        paused = not paused
        event = "Paused" if paused else "Resumed"
    elif action is Gesture.OPEN_PALM:
        portal_visible = True
        paused = False
        shape = PortalShape.CIRCLE
        event = "Defaults restored"
    return portal_visible, paused, shape, event


def _run() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam (index 0). Is a camera connected?")
        return

    recognizer = GestureRecognizer()
    renderer = PortalRenderer()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"[INFO] Warming up for {WARMUP_SECONDS:.0f}s and capturing background...")
    if not _warmup(cap, renderer, WARMUP_SECONDS):
        print("[INFO] Warm-up aborted. You can press [C] later to capture the background.")
    else:
        print("[INFO] Background captured.")

    portal_visible = True
    paused = False
    fullscreen = False
    shape = PortalShape.CIRCLE

    last_position: tuple[float, float] | None = None
    last_radius: float | None = None
    last_frame: np.ndarray | None = None

    event_message = ""
    event_time = 0.0
    fps = 0.0
    frame_start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)

            result = recognizer.process(frame)

            if result.position is not None:
                last_position = result.position
            if result.radius is not None:
                last_radius = result.radius

            portal_visible, paused, shape, gesture_event = _handle_gesture_action(
                result.action,
                renderer=renderer,
                portal_visible=portal_visible,
                paused=paused,
                shape=shape,
            )
            renderer.set_shape(shape)

            if gesture_event:
                event_message = gesture_event
                event_time = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                if renderer.capture_background(frame):
                    event_message = "Background captured"
                    event_time = time.monotonic()
            elif key == ord("b"):
                portal_visible = not portal_visible
                event_message = "Portal active" if portal_visible else "Portal hidden"
                event_time = time.monotonic()
            elif key == ord("t"):
                shape = renderer.cycle_shape()
                event_message = f"Shape: {shape.value.title()}"
                event_time = time.monotonic()
            elif key == ord("p"):
                paused = not paused
                event_message = "Paused" if paused else "Resumed"
                event_time = time.monotonic()
            elif key == ord("f"):
                fullscreen = _toggle_fullscreen(WINDOW_NAME, not fullscreen)

            # Compose (or freeze) the display frame.
            if paused and last_frame is not None:
                display = last_frame.copy()
            else:
                display = renderer.compose(
                    frame,
                    last_position,
                    last_radius,
                    visible=portal_visible,
                )
                last_frame = display

            # FPS estimate.
            now = time.monotonic()
            delta = now - frame_start
            if delta > 0:
                fps = fps * 0.9 + (1.0 / delta) * 0.1
            frame_start = now

            # HUD.
            tracking = "HAND" if result.hand_visible else "LOST"
            _draw_status_panel(
                display,
                [
                    ("AI Magic Invisibility Portal", (255, 255, 255)),
                    (f"Portal: {'ACTIVE' if portal_visible else 'HIDDEN'}", (0, 255, 120) if portal_visible else (0, 0, 255)),
                    (f"Background: {'READY' if renderer.background_captured else 'NOT CAPTURED'}", (0, 255, 255)),
                    (f"Tracking: {tracking}", (255, 200, 0)),
                    (f"Gesture: {result.gesture.label if result.gesture else 'NONE'}", (200, 200, 200)),
                    (f"Shape: {shape.value.title()}   FPS: {fps:5.1f}", (200, 200, 200)),
                ],
                (12, 40),
            )
            _draw_hint_bar(display)

            if event_message and (now - event_time) < EVENT_TTL_SECONDS:
                _draw_center_event(display, event_message)

            cv2.imshow(WINDOW_NAME, display)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point for the AI Magic Invisibility Portal."""
    try:
        _run()
    except Exception as exc:  # noqa: BLE001 - surface any runtime failure cleanly
        print(f"[ERROR] Application failed: {exc}")


if __name__ == "__main__":
    main()
