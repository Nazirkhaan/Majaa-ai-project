"""MediaPipe selfie segmentation for the full-body ghost effect.

Wraps MediaPipe's Tasks API ``ImageSegmenter`` with the selfie segmentation
model. Each frame it produces a soft float32 mask in ``[0, 1]`` marking the
person silhouette, which :class:`portal.GhostCompositor` uses to blend the
pre-captured background through the body.

The model (``selfie_segmenter.tflite``) is downloaded automatically on first
use to ``models/`` next to this module unless an explicit ``model_path`` is
given.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Official MediaPipe selfie segmentation model (float16, ~250 KB).
SELFIE_SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"


def _ensure_model_file(model_path: str | None = None) -> str:
    """Return a usable model path, downloading the model if it is missing.

    Args:
        model_path: Optional explicit path to ``selfie_segmenter.tflite``.

    Returns:
        The resolved, existing model file path.

    Raises:
        RuntimeError: If the model cannot be downloaded.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Selfie segmenter model not found at {path}")
    print(f"[INFO] Downloading from {SELFIE_SEGMENTER_MODEL_URL} ...")

    request = urllib.request.Request(SELFIE_SEGMENTER_MODEL_URL, headers={"User-Agent": "AI-Magic-Invisibility-Portal"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"Failed to download the selfie segmenter model to {path}: {exc}") from exc

    return str(path)


class SelfieSegmenter:
    """Computes a per-frame person silhouette mask from a webcam frame."""

    def __init__(self, *, model_path: str | None = None) -> None:
        """Configure the segmenter.

        Args:
            model_path: Optional path to ``selfie_segmenter.tflite``. Downloaded
                automatically to ``models/`` when omitted.
        """
        model_file = _ensure_model_file(model_path)
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_file),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._last_timestamp_ms: int = 0

    # ------------------------------------------------------------------ public

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Segment the person in a BGR frame.

        Args:
            frame_bgr: A BGR frame (preferably already mirrored).

        Returns:
            A float32 mask of shape ``(H, W)`` with values in ``[0, 1]``, where
            ``1`` marks the person silhouette.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._segmenter.segment_for_video(image, self._next_timestamp_ms())

        height, width = frame_bgr.shape[:2]
        if not results.confidence_masks:
            return np.zeros((height, width), dtype=np.float32)

        view = np.asarray(results.confidence_masks[0].numpy_view())
        mask = view[..., 0] if view.ndim == 3 else view
        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    # ----------------------------------------------------------------- helpers

    def _next_timestamp_ms(self) -> int:
        """Return a strictly increasing timestamp (ms) required by VIDEO mode."""
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms
