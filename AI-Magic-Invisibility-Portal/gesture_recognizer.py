"""Two-hand tracking (MediaPipe Tasks API), gesture state machine, and pinch math.

This module uses MediaPipe's modern Tasks API (``HandLandmarker``), which is the
only hand tracking API shipped in MediaPipe 0.11+ / 1.x and is required on
Python 3.12+. The legacy ``mp.solutions.hands`` API was removed there.

It tracks up to two hands simultaneously and reports, per hand:

* The 21-landmark pixel skeleton (for on-screen overlay drawing).
* A pinch distance (thumb<->index) and a size-normalized pinch ratio.
* A classified control gesture.

A single debounced state machine turns the first hand's gesture into actions.

Coordinate convention: all returned pixel coordinates refer to the *mirrored*
frame handed to :meth:`GestureRecognizer.process`, so they align with what the
user sees on screen.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Official MediaPipe hand landmarker model (float16, ~7.8 MB).
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# Hand landmark indices (standard 21-landmark hand topology).
_WRIST = 0
_THUMB_MCP = 2
_THUMB_IP = 3
_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_PIP = 6
_INDEX_TIP = 8
_MIDDLE_MCP = 9
_RING_TIP = 16
_RING_PIP = 14
_PINKY_TIP = 20
_PINKY_PIP = 18
_PINKY_MCP = 17

# Standard MediaPipe hand skeleton connections (bone pairs to draw).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


class Gesture(Enum):
    """High-level gestures that control the invisibility effect."""

    OK_SIGN = auto()        # capture / refresh background
    PEACE_SIGN = auto()     # toggle ghost visibility
    CLOSED_FIST = auto()    # pause / resume the effect
    OPEN_PALM = auto()      # reset to default state

    @property
    def label(self) -> str:
        """Human readable name used for the HUD."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class HandInfo:
    """Per-hand tracking data for one frame."""

    handedness: str = ""
    landmarks: tuple[tuple[float, float], ...] = ()
    index_tip: tuple[float, float] = (0.0, 0.0)
    pinch_px: float = 0.0
    pinch_ratio: float = 0.0
    gesture: Gesture | None = None


@dataclass(frozen=True)
class GestureResult:
    """Per-frame output of :meth:`GestureRecognizer.process`."""

    hands: tuple[HandInfo, ...] = ()
    hand_count: int = 0
    action: Gesture | None = None


def _ensure_model_file(model_path: str | None = None) -> str:
    """Return a usable model path, downloading the model if it is missing.

    Args:
        model_path: Optional explicit path to ``hand_landmarker.task``. When
            omitted, ``models/hand_landmarker.task`` next to this module is used.

    Returns:
        The resolved, existing model file path.

    Raises:
        RuntimeError: If the model cannot be downloaded.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Hand landmarker model not found at {path}")
    print(f"[INFO] Downloading from {HAND_LANDMARKER_MODEL_URL} ...")

    request = urllib.request.Request(HAND_LANDMARKER_MODEL_URL, headers={"User-Agent": "AI-Magic-Invisibility-Portal"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as handle:
            total = int(response.headers.get("Content-Length", 0)) or None
            downloaded = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r[INFO] Downloading model: {downloaded / 1_000_000:.1f} / {total / 1_000_000:.1f} MB", end="", flush=True)
        if total:
            print()
    except Exception as exc:
        raise RuntimeError(f"Failed to download the hand landmarker model to {path}: {exc}") from exc

    return str(path)


class GestureRecognizer:
    """Tracks up to two hands and turns their landmarks into usable data."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        hold_seconds: float = 1.0,
    ) -> None:
        """Configure the recognizer.

        Args:
            model_path: Optional path to the ``hand_landmarker.task`` model.
                Downloaded automatically to ``models/`` when omitted.
            max_num_hands: Maximum hands tracked simultaneously (default 2).
            min_detection_confidence: Hand detection confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
            hold_seconds: How long a gesture must be held before it triggers.
        """
        self._max_num_hands = max(1, int(max_num_hands))
        self._hold_seconds = float(hold_seconds)

        model_file = _ensure_model_file(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_file),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = 0

        # Gesture state machine state.
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
            A :class:`GestureResult` with per-hand data and any debounced action.
        """
        now = time.monotonic() if timestamp is None else timestamp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(image, self._next_timestamp_ms(now))

        hands: list[HandInfo] = []
        if results.hand_landmarks:
            height, width = frame_bgr.shape[:2]
            for index, landmarks in enumerate(results.hand_landmarks):
                handedness = ""
                if index < len(results.handedness) and results.handedness[index]:
                    handedness = results.handedness[index][0].category_name or ""

                pixel = tuple((lm.x * width, lm.y * height) for lm in landmarks)
                pinch_px = self._landmark_distance_px(landmarks[_THUMB_TIP], landmarks[_INDEX_TIP], width, height)
                hand_scale = self._landmark_distance_px(landmarks[_WRIST], landmarks[_MIDDLE_MCP], width, height)
                pinch_ratio = float(np.clip(pinch_px / max(hand_scale * 2.0, 1e-3), 0.0, 1.0))

                hands.append(
                    HandInfo(
                        handedness=handedness,
                        landmarks=pixel,
                        index_tip=pixel[_INDEX_TIP],
                        pinch_px=pinch_px,
                        pinch_ratio=pinch_ratio,
                        gesture=self._classify_gesture(landmarks),
                    )
                )

        if not hands:
            self._reset_tracking()
            return GestureResult(hands=(), hand_count=0)

        action = self._update_state_machine(hands[0].gesture, now)
        return GestureResult(hands=tuple(hands), hand_count=len(hands), action=action)

    def reset(self) -> None:
        """Forget gesture state machine history."""
        self._reset_tracking()

    # ------------------------------------------------------------- gesture math

    def _classify_gesture(self, lm) -> Gesture | None:
        """Classify the current hand pose from its landmarks."""
        thumb_ext = self._is_thumb_extended(lm)
        index_ext = self._is_finger_extended(lm, _INDEX_TIP, _INDEX_PIP)
        middle_ext = self._is_finger_extended(lm, 12, 10)
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
        if not thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext:
            return Gesture.PEACE_SIGN
        return None

    @staticmethod
    def _is_finger_extended(lm, tip_id: int, pip_id: int) -> bool:
        """A finger is extended when its tip sits clearly above its PIP joint."""
        return bool(lm[tip_id].y < lm[pip_id].y - 0.03)

    @staticmethod
    def _is_thumb_extended(lm) -> bool:
        """Hand-agnostic thumb check based on reach relative to the index MCP."""
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
        """Debounce gesture triggers with a hold timer."""
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

    def _next_timestamp_ms(self, now: float) -> int:
        """Return a strictly increasing timestamp (ms) required by VIDEO mode."""
        timestamp_ms = int(now * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _reset_tracking(self) -> None:
        """Reset gesture state when no hand is visible."""
        self._current = None
        self._hold_started = 0.0
        self._last_triggered = None
