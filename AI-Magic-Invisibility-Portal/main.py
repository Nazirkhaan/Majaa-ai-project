"""Main application loop, frame capture, and orchestration.

Pieces everything together:

* Two-hand tracking via :class:`gesture_recognizer.GestureRecognizer`.
* Full-body selfie segmentation via :class:`segmenter.SelfieSegmenter`.
* Ghost compositing via :class:`portal.GhostCompositor`.

Each frame the person silhouette is blended toward the pre-captured background,
with the transparency driven by the hand pinch (the more open the pinch, the
more "ghostly" the body). Hand skeletons and a live HUD are drawn on top.

Keyboard shortcuts:
    C       recapture / refresh the background
    B       toggle ghost visibility
    P       pause / resume the effect
    F       toggle fullscreen
    Q, ESC  quit
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from gesture_recognizer import HAND_CONNECTIONS, Gesture, GestureRecognizer
from portal import GhostCompositor
from segmenter import SelfieSegmenter

WINDOW_NAME = "AI Magic Invisibility Portal"
WARMUP_SECONDS = 3.0
EVENT_TTL_SECONDS = 2.0
OPACITY_SMOOTHING = 0.2

Font = cv2.FONT_HERSHEY_SIMPLEX


def _draw_skeletons(frame: np.ndarray, hands) -> None:
    """Draw the 21-landmark skeleton (bones + joints) for each tracked hand."""
    bone_color = (255, 220, 60)
    joint_color = (255, 255, 255)
    for hand in hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, (int(hand.landmarks[a][0]), int(hand.landmarks[a][1])),
                     (int(hand.landmarks[b][0]), int(hand.landmarks[b][1])), bone_color, 2, cv2.LINE_AA)
        for x, y in hand.landmarks:
            cv2.circle(frame, (int(x), int(y)), 4, joint_color, -1, cv2.LINE_AA)


def _draw_translucent_bar(frame: np.ndarray, y_top: int, height: int, alpha: float = 0.45) -> None:
    """Draw a full-width translucent strip used by the HUD bars."""
    x0, y0 = 0, y_top
    x1, y1 = frame.shape[1], y_top + height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
    frame[:] = blended


def _draw_top_bar(frame: np.ndarray, ghost_visible: bool, bg_ready: bool, hands: int) -> None:
    """Draw the top status bar with title and quick status chips."""
    _draw_translucent_bar(frame, 0, 34)
    cv2.putText(frame, "AI MAGIC INVISIBILITY PORTAL", (12, 24), Font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    ghost = "GHOST: ACTIVE" if ghost_visible else "GHOST: HIDDEN"
    ghost_color = (0, 255, 120) if ghost_visible else (0, 0, 255)
    bg = "BG: READY" if bg_ready else "BG: NOT CAPTURED"
    bg_color = (0, 255, 255) if bg_ready else (0, 0, 255)

    x = frame.shape[1] - 12
    for text, color in ((bg, bg_color), (ghost, ghost_color)):
        text_w = cv2.getTextSize(text, Font, 0.55, 1)[0][0]
        x -= text_w + 16
        cv2.putText(frame, text, (x, 24), Font, 0.55, color, 1, cv2.LINE_AA)


def _draw_stats_bar(frame: np.ndarray, opacity: float, hands: int, pinch_px: float,
                    gesture: Gesture | None, fps: float) -> None:
    """Draw the bottom stats bar (PORTAL %, HANDS, PINCH, FPS)."""
    h = frame.shape[0]
    _draw_translucent_bar(frame, h - 34, 34)

    portal_pct = round(opacity * 100.0)
    portal = f"PORTAL {portal_pct}%"
    portal_color = (0, 255, 120) if portal_pct > 10 else (200, 200, 200)
    hands_text = f"HANDS {hands}"
    pinch = f"PINCH {round(pinch_px)}px"
    gesture_text = f"GESTURE {gesture.label if gesture else 'NONE'}"
    fps_text = f"{fps:4.1f} FPS"

    x = 12
    y = h - 12
    for text, color in ((portal, portal_color), (hands_text, (255, 255, 255)),
                        (pinch, (255, 255, 255)), (gesture_text, (255, 255, 255))):
        cv2.putText(frame, text, (x, y), Font, 0.55, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, Font, 0.55, 1)[0][0] + 18

    text_w = cv2.getTextSize(fps_text, Font, 0.55, 1)[0][0]
    cv2.putText(frame, fps_text, (frame.shape[1] - text_w - 12, y), Font, 0.55, (255, 255, 0), 1, cv2.LINE_AA)


def _draw_center_event(frame: np.ndarray, message: str) -> None:
    """Draw a transient event message in the upper-center of the frame."""
    scale = 0.8
    thickness = 2
    (text_w, _), _ = cv2.getTextSize(message, Font, scale, thickness)
    x = (frame.shape[1] - text_w) // 2
    cv2.putText(frame, message, (x, 84), Font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _warmup(cap: cv2.VideoCapture, compositor: GhostCompositor, seconds: float) -> bool:
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
            "Keep out of view until capture finishes",
            (frame.shape[1] // 2 - 150, 108),
            Font,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False

    return compositor.capture_background(last) if last is not None else False


def _toggle_fullscreen(window: str, enabled: bool) -> bool:
    """Toggle the named OpenCV window between normal and fullscreen."""
    prop = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, prop)
    return enabled


def _handle_gesture_action(
    action: Gesture | None,
    *,
    compositor: GhostCompositor,
    ghost_visible: bool,
    paused: bool,
) -> tuple[bool, bool, str]:
    """Apply a triggered gesture action, returning updated control state."""
    event = ""
    if action is Gesture.OK_SIGN:
        event = "Background captured"
    elif action is Gesture.PEACE_SIGN:
        ghost_visible = not ghost_visible
        event = "Ghost active" if ghost_visible else "Ghost hidden"
    elif action is Gesture.CLOSED_FIST:
        paused = not paused
        event = "Paused" if paused else "Resumed"
    elif action is Gesture.OPEN_PALM:
        ghost_visible = True
        paused = False
        event = "Defaults restored"
    return ghost_visible, paused, event


def _run() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam (index 0). Is a camera connected?")
        return

    recognizer = GestureRecognizer()
    segmenter = SelfieSegmenter()
    compositor = GhostCompositor()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"[INFO] Warming up for {WARMUP_SECONDS:.0f}s and capturing background...")
    if not _warmup(cap, compositor, WARMUP_SECONDS):
        print("[INFO] Warm-up aborted. You can press [C] later to capture the background.")
    else:
        print("[INFO] Background captured.")

    ghost_visible = True
    paused = False
    fullscreen = False

    smooth_opacity = 0.0
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
            person_mask = segmenter.segment(frame)

            ghost_visible, paused, gesture_event = _handle_gesture_action(
                result.action,
                compositor=compositor,
                ghost_visible=ghost_visible,
                paused=paused,
            )
            if gesture_event:
                event_message = gesture_event
                event_time = time.monotonic()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                if compositor.capture_background(frame):
                    event_message = "Background captured"
                    event_time = time.monotonic()
            elif key == ord("b"):
                ghost_visible = not ghost_visible
                event_message = "Ghost active" if ghost_visible else "Ghost hidden"
                event_time = time.monotonic()
            elif key == ord("p"):
                paused = not paused
                event_message = "Paused" if paused else "Resumed"
                event_time = time.monotonic()
            elif key == ord("f"):
                fullscreen = _toggle_fullscreen(WINDOW_NAME, not fullscreen)

            # Ghost opacity is driven by the widest pinch across both hands.
            target_opacity = max((hand.pinch_ratio for hand in result.hands), default=0.0)
            smooth_opacity = smooth_opacity * (1.0 - OPACITY_SMOOTHING) + target_opacity * OPACITY_SMOOTHING

            # Compose (or freeze) the display frame.
            if paused and last_frame is not None:
                display = last_frame.copy()
                _draw_center_event(display, "PAUSED")
            else:
                display = compositor.compose(frame, person_mask, smooth_opacity, visible=ghost_visible)
                _draw_skeletons(display, result.hands)
                last_frame = display

            # FPS estimate.
            now = time.monotonic()
            delta = now - frame_start
            if delta > 0:
                fps = fps * 0.9 + (1.0 / delta) * 0.1
            frame_start = now

            # HUD.
            max_pinch = max((hand.pinch_px for hand in result.hands), default=0.0)
            _draw_top_bar(display, ghost_visible, compositor.background_captured, result.hand_count)
            _draw_stats_bar(display, smooth_opacity, result.hand_count, max_pinch, result.hands[0].gesture if result.hands else None, fps)

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
