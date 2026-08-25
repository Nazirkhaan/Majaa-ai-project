"""Ghost compositing: full-body silhouette transparency with a magic glow.

The shaped portal window is replaced by live selfie segmentation. The person's
silhouette (a soft mask produced by :class:`segmenter.SelfieSegmenter`) is
blended toward the pre-captured background to create a "ghost" / see-through
body effect. A pinch gesture controls how transparent the ghost becomes.

Visual treatment:

* Feathered alpha compositing so the body edge stays soft.
* An additive silhouette glow (Gaussian halo carved around the body boundary)
  rendered at half resolution for performance.
"""

from __future__ import annotations

import cv2
import numpy as np


class GhostCompositor:
    """Blends the person silhouette into the background to make a ghost."""

    def __init__(
        self,
        *,
        glow_color: tuple[int, int, int] = (255, 200, 80),
        glow_intensity: float = 60.0,
        glow_scale: float = 0.5,
    ) -> None:
        """Configure the compositor.

        Args:
            glow_color: BGR color used for the silhouette glow.
            glow_intensity: Peak additive brightness of the glow (0..255).
            glow_scale: Downscale factor for glow computation (0.5 = half res).
        """
        self._glow_color = tuple(int(c) for c in glow_color)
        self._glow_intensity = float(glow_intensity)
        self._glow_scale = float(glow_scale)

        self._bg_frame: np.ndarray | None = None

    # ------------------------------------------------------------------ public

    @property
    def background_captured(self) -> bool:
        """Whether a baseline background frame is available."""
        return self._bg_frame is not None

    def capture_background(self, frame: np.ndarray) -> bool:
        """Store *frame* as the baseline background to reveal through the body.

        Returns:
            ``True`` when a frame was stored.
        """
        if frame is None or frame.size == 0:
            return False
        self._bg_frame = frame.copy()
        return True

    def compose(
        self,
        frame: np.ndarray,
        person_mask: np.ndarray | None,
        opacity: float,
        *,
        visible: bool = True,
    ) -> np.ndarray:
        """Compose the final frame with the ghost effect applied.

        Args:
            frame: Current webcam frame (BGR).
            person_mask: Float32 silhouette mask in ``[0, 1]``, or ``None``.
            opacity: Ghost transparency in ``[0, 1]`` (1 = fully see-through).
            visible: Whether the ghost effect should be rendered at all.

        Returns:
            A new composite BGR frame.
        """
        if not self.background_captured or not visible or person_mask is None:
            return frame.copy()

        alpha = person_mask * float(np.clip(opacity, 0.0, 1.0))

        fg = frame.astype(np.float32)
        bg = self._bg_frame.astype(np.float32)
        blend = fg * (1.0 - alpha[..., None]) + bg * alpha[..., None]
        composite = np.clip(blend, 0.0, 255.0).astype(np.uint8)

        glow = self._build_glow(composite.shape, person_mask, float(opacity))
        return cv2.add(composite, glow)

    # -------------------------------------------------------------------- glow

    def _build_glow(
        self,
        frame_shape: tuple[int, int, int],
        person_mask: np.ndarray,
        opacity: float,
    ) -> np.ndarray:
        """Build an additive halo around the person's silhouette.

        The mask is blurred twice with different kernels; subtracting the two
        produces a soft band that hugs the body boundary. It is tinted with
        ``glow_color``, scaled by the ghost ``opacity``, and added to the frame,
        so clipping yields a neon bloom.
        """
        height, width = frame_shape[:2]
        scale = self._glow_scale
        small_w = max(1, round(width * scale))
        small_h = max(1, round(height * scale))

        if scale != 1.0:
            small_mask = cv2.resize(person_mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_mask = person_mask

        outer = cv2.GaussianBlur(small_mask, (0, 0), sigmaX=6.0)
        inner = cv2.GaussianBlur(small_mask, (0, 0), sigmaX=1.5)
        ring = np.clip(outer - inner, 0.0, 1.0) * float(np.clip(opacity, 0.0, 1.0))

        color = np.asarray(self._glow_color, dtype=np.float32)
        glow = np.zeros((small_h, small_w, 3), np.float32)
        glow[..., 0] = ring * color[0] * self._glow_intensity / 255.0
        glow[..., 1] = ring * color[1] * self._glow_intensity / 255.0
        glow[..., 2] = ring * color[2] * self._glow_intensity / 255.0

        glow = np.clip(glow, 0.0, 255.0).astype(np.uint8)
        if scale != 1.0:
            glow = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
        return glow
