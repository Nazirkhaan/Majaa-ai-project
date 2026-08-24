"""MediaPipe hand tracking, gesture state machine, and portal shape/size estimation.

This module wraps MediaPipe's Hands solution and exposes a high-level API that
the main loop can use to drive the invisibility portal:

* Continuous, de-jittered portal center (EMA-smoothed index finger tip).
* A dynamic portal radius derived from the thumb<->index pinch distance.
* Debounced gesture triggers guarded by a 1-second hold timer.

Coordinate convention: all returned pixel coordinates refer to the *mirrored*
frame handed to :meth:`GestureRecognizer.process`, so they align with what the
user sees on screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

# MediaPipe layout differs across releases: classic wheels expose the legacy
# API as ``mediapipe.solutions``, while newer single-module builds place it at
# ``mediapipe.python.solutions``. Try the standard path first, then fall back.
try:  # pragma: no cover - depends on the installed wheel
    from mediapipe.solutions import hands as _mp_hands  # type: ignore[attr-defined]
except (ImportError, ModuleNotFoundError):
    from mediapipe.python.solutions import hands as _mp_hands

# Hand landmark indices (see mediapipe.solutions.hands.HandLandmark).
_THUMB_TIP = 4
_THUMB_IP = 3
_THUMB_MCP = 2
_INDEX_TIP = 8
_INDEX_PIP = 6
_INDEX_MCP = 5
_MIDDLE_TIP = 12
_MIDDLE_PIP = 10
_MIDDLE_MCP = 9
_RING_TIP = 16
_RING_PIP = 14
_RING_MCP = 13
_PINKY_TIP = 20
_PINKY_PIP = 18
_PINKY_MCP = 17


class Gesture(Enum):
    """High-level gestures that control the invisibility portal."""

    OK_SIGN = auto()        # capture / refresh background
    PEACE_SIGN = auto()     # toggle portal visibility
    CLOSED_FIST = auto()    # pause / resume the effect
    OPEN_PALM = auto()      # reset to default state

    @property
    def label(self) -> str:
        """Human readable name used for the HUD."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class GestureResult:
    """Per-frame output of :meth:`GestureRecognizer.process`."""

    position: tuple[float, float] | None = None
    radius: float | None = None
    gesture: Gesture | None = None
    action: Gesture | None = None
    hand_visible: bool = False


class GestureRecognizer:
    """Tracks a hand and turns MediaPipe landmarks into portal parameters."""

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        smoothing_alpha: float = 0.35,
        min_radius: float = 20.0,
        max_radius: float = 250.0,
        radius_scale: float = 2.0,
        hold_seconds: float = 1.0,
    ) -> None:
        """Configure the recognizer.

        Args:
            max_num_hands: Maximum hands tracked simultaneously.
            min_detection_confidence: MediaPipe detection confidence threshold.
            min_tracking_confidence: MediaPipe tracking confidence threshold.
            smoothing_alpha: EMA weight (0..1) for portal position. Higher = snappier.
            min_radius: Lower clamp for the portal radius (px).
            max_radius: Upper clamp for the portal radius (px).
            radius_scale: Multiplier applied to the thumb<->index pinch distance.
            hold_seconds: How long a gesture must be held before it triggers.
        """
        self._min_radius = float(min_radius)
        self._max_radius = float(max_radius)
        self._radius_scale = float(radius_scale)
        self._alpha = float(smoothing_alpha)
        self._hold_seconds = float(hold_seconds)

        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Smoothing / state machine state.
        self._smooth_pos: tuple[float, float] | None = None
        self._current: Gesture | None = None
        self._hold_started: float = 0.0
        self._last_triggered: Gesture | None = None

    # ------------------------------------------------------------------ public

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp: float | None = None,
    ) -> GestureResult:
        """Analyze one BGR frame.

        Args:
            frame_bgr: A BGR frame (preferably already mirrored).
            timestamp: Optional monotonic timestamp override (mostly for tests).

        Returns:
            A :class:`GestureResult` describing position, radius and any
            debounced gesture action triggered on this frame.
        """
        now = time.monotonic() if timestamp is None else timestamp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._reset_tracking()
            return GestureResult(hand_visible=False)

        landmarks = results.multi_hand_landmarks[0].landmark
        height, width = frame_bgr.shape[:2]

        position = self._smooth_position(
            (landmarks[_INDEX_TIP].x * width, landmarks[_INDEX_TIP].y * height)
        )

        pinch_px = self._landmark_distance_px(landmarks[_THUMB_TIP], landmarks[_INDEX_TIP], width, height)
        radius = float(np.clip(pinch_px * self._radius_scale, self._min_radius, self._max_radius))

        gesture = self._classify_gesture(landmarks)
        action = self._update_state_machine(gesture, now)

        return GestureResult(
            position=position,
            radius=radius,
            gesture=gesture,
            action=action,
            hand_visible=True,
        )

    def reset(self) -> None:
        """Forget tracking history and the gesture state machine."""
        self._reset_tracking()

    # ------------------------------------------------------------- gesture math

    def _classify_gesture(self, lm) -> Gesture | None:
        """Classify the current hand pose from its landmarks."""
        thumb_ext = self._is_thumb_extended(lm)
        index_ext = self._is_finger_extended(lm, _INDEX_TIP, _INDEX_PIP)
        middle_ext = self._is_finger_extended(lm, _MIDDLE_TIP, _MIDDLE_PIP)
        ring_ext = self._is_finger_extended(lm, _RING_TIP, _RING_PIP)
        pinky_ext = self._is_finger_extended(lm, _PINKY_TIP, _PINKY_PIP)

        if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
            return Gesture.OPEN_PALM
        if not thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return Gesture.CLOSED_FIST
        # OK: thumb & index folded into a ring, the remaining three extended.
        if (
            not thumb_ext
            and not index_ext
            and middle_ext
            and ring_ext
            and pinky_ext
            and self._landmark_distance(lm[_THUMB_TIP], lm[_INDEX_TIP]) < 0.12
        ):
            return Gesture.OK_SIGN
        # Peace: index & middle extended, thumb, ring and pinky folded.
        if (
            not thumb_ext
            and index_ext
            and middle_ext
            and not ring_ext
            and not pinky_ext
        ):
            return Gesture.PEACE_SIGN
        return None

    @staticmethod
    def _is_finger_extended(lm, tip_id: int, pip_id: int) -> bool:
        """A finger is extended when its tip sits clearly above its PIP joint."""
        return bool(lm[tip_id].y < lm[pip_id].y - 0.03)

    @staticmethod
    def _is_thumb_extended(lm) -> bool:
        """Hand-agnostic thumb check based on reach relative to the index MCP.

        When the thumb is extended its tip is farther from the index knuckle
        than the thumb IP joint is; when folded it collapses toward the palm.
        """
        tip_to_index_mcp = GestureRecognizer._landmark_distance(lm[_THUMB_TIP], lm[_INDEX_MCP])
        ip_to_index_mcp = GestureRecognizer._landmark_distance(lm[_THUMB_IP], lm[_INDEX_MCP])
        return bool(tip_to_index_mcp > ip_to_index_mcp + 0.02)

    @staticmethod
    def _landmark_distance(a, b) -> float:
        """Euclidean distance between two normalized landmarks."""
        return float(np.hypot(a.x - b.x, a.y - b.y))

    @staticmethod
    def _landmark_distance_px(a, b, width: int, height: int) -> float:
        """Euclidean distance between two landmarks in pixel space."""
        dx = (a.x - b.x) * width
        dy = (a.y - b.y) * height
        return float(np.hypot(dx, dy))

    # ------------------------------------------------------------ state machine

    def _update_state_machine(self, gesture: Gesture | None, now: float) -> Gesture | None:
        """Debounce gesture triggers with a hold timer.

        A gesture must be held continuously for ``hold_seconds`` before firing.
        After firing, the same gesture cannot re-fire until it has been released
        (or replaced by another gesture).
        """
        if gesture is None:
            self._current = None
            self._hold_started = 0.0
            self._last_triggered = None
            return None

        if gesture is not self._current:
            self._current = gesture
            self._hold_started = now

        if (now - self._hold_started) >= self._hold_seconds and gesture is not self._last_triggered:
            self._last_triggered = gesture
            return gesture
        return None

    # ----------------------------------------------------------------- helpers

    def _smooth_position(self, raw: tuple[float, float]) -> tuple[float, float]:
        """Exponential moving average to suppress fingertip jitter."""
        if self._smooth_pos is None:
            self._smooth_pos = raw
            return raw
        alpha = self._alpha
        x = alpha * raw[0] + (1.0 - alpha) * self._smooth_pos[0]
        y = alpha * raw[1] + (1.0 - alpha) * self._smooth_pos[1]
        self._smooth_pos = (x, y)
        return self._smooth_pos

    def _reset_tracking(self) -> None:
        """Reset smoothing and gesture state when the hand is lost."""
        self._smooth_pos = None
        self._current = None
        self._hold_started = 0.0
        self._last_triggered = None
