"""Portal math, rendering, masking, glowing effects, and background blending.

The :class:`PortalRenderer` implements the visual core of the application:

1. Stores a baseline background frame to "see through" the portal.
2. Builds a soft, feathered binary mask (Circle / Square / Hexagon) centered on
   the tracked fingertip with a dynamic radius.
3. Blends the pre-captured background into the masked region of the live frame.
4. Paints a multi-layered glowing border (Gaussian halo + crisp inner rim)
   using additive color mixing for a neon "magic portal" look.

All heavy lifting uses vectorized NumPy operations to stay close to real-time.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class PortalShape(Enum):
    """Supported portal shapes."""

    CIRCLE = "circle"
    SQUARE = "square"
    HEXAGON = "hexagon"


class PortalRenderer:
    """Renders the invisibility portal and its glow effects."""

    def __init__(
        self,
        *,
        glow_color: tuple[int, int, int] = (255, 180, 40),
        glow_scale: float = 0.5,
        edge_feather: float = 0.05,
    ) -> None:
        """Configure the renderer.

        Args:
            glow_color: BGR color used for the portal glow.
            glow_scale: Downscale factor for glow computation (0.5 = half res).
            edge_feather: Mask feather size as a fraction of the portal radius.
        """
        self._glow_color = tuple(int(c) for c in glow_color)
        self._glow_scale = float(glow_scale)
        self._edge_feather = float(edge_feather)

        self._bg_frame: np.ndarray | None = None
        self._shape = PortalShape.CIRCLE

    # ------------------------------------------------------------------ public

    @property
    def shape(self) -> PortalShape:
        """Currently selected portal shape."""
        return self._shape

    @property
    def background_captured(self) -> bool:
        """Whether a baseline background frame is available."""
        return self._bg_frame is not None

    def capture_background(self, frame: np.ndarray) -> bool:
        """Store *frame* as the baseline background to reveal through the portal.

        Returns:
            ``True`` when a frame was stored.
        """
        if frame is None or frame.size == 0:
            return False
        self._bg_frame = frame.copy()
        return True

    def set_shape(self, shape: PortalShape) -> PortalShape:
        """Select the portal shape explicitly."""
        self._shape = shape
        return self._shape

    def cycle_shape(self) -> PortalShape:
        """Advance to the next portal shape in the enum order."""
        shapes = list(PortalShape)
        index = shapes.index(self._shape)
        self._shape = shapes[(index + 1) % len(shapes)]
        return self._shape

    def compose(
        self,
        frame: np.ndarray,
        position: tuple[float, float] | None,
        radius: float | None,
        *,
        visible: bool = True,
    ) -> np.ndarray:
        """Compose the final frame with the invisibility portal applied.

        Args:
            frame: Current webcam frame (BGR).
            position: Portal center in pixel coordinates, or ``None`` to disable.
            radius: Portal radius in pixels, or ``None`` to disable.
            visible: Whether the portal effect should be rendered at all.

        Returns:
            A new composite BGR frame.
        """
        if (
            not self.background_captured
            or not visible
            or position is None
            or radius is None
        ):
            return frame.copy()

        soft_mask = self._build_soft_mask(frame.shape, position, radius)

        fg = frame.astype(np.float32)
        bg = self._bg_frame.astype(np.float32)
        blend = fg * (1.0 - soft_mask[..., None]) + bg * soft_mask[..., None]
        composite = np.clip(blend, 0.0, 255.0).astype(np.uint8)

        glow = self._build_glow(composite.shape, position, radius)
        return cv2.add(composite, glow)

    # ----------------------------------------------------------------- masking

    def _build_soft_mask(
        self,
        frame_shape: tuple[int, int, int],
        center: tuple[float, float],
        radius: float,
    ) -> np.ndarray:
        """Build a feathered binary mask for the selected portal shape.

        Returns:
            Float32 mask in ``[0, 1]`` where ``1`` marks portal interior.
        """
        height, width = frame_shape[:2]
        points = self._shape_points(center, radius)

        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [points], 255)

        sigma = max(int(radius * self._edge_feather), 2)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        return mask.astype(np.float32) / 255.0

    def _shape_points(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """Return the polygon vertices for the current shape (int32 Nx2)."""
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)

        if self._shape is PortalShape.CIRCLE:
            angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
            points = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
        elif self._shape is PortalShape.SQUARE:
            points = np.array(
                [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]],
                dtype=np.float32,
            )
        else:  # HEXAGON (pointy-top)
            angles = np.pi / 6.0 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
            points = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)

        return points.astype(np.int32)

    # -------------------------------------------------------------------- glow

    def _build_glow(
        self,
        frame_shape: tuple[int, int, int],
        center: tuple[float, float],
        radius: float,
    ) -> np.ndarray:
        """Build a multi-layered glowing border (halo + rim) as an additive layer.

        The glow is computed at a downscaled resolution for performance and then
        resized back to the full frame size. Layers:

        * A wide, dim Gaussian halo just outside the portal edge.
        * A narrow, bright rim tracing the portal perimeter.

        Both are tinted with ``glow_color``; when added to the frame via
        :func:`cv2.add`, clipping produces a natural neon bloom.
        """
        height, width = frame_shape[:2]
        scale = self._glow_scale
        small_w = max(1, round(width * scale))
        small_h = max(1, round(height * scale))
        small_center = (round(center[0] * scale), round(center[1] * scale))
        small_radius = max(1.0, radius * scale)

        # Outer soft halo.
        outer = np.zeros((small_h, small_w), np.float32)
        cv2.fillPoly(outer, [self._shape_points(small_center, small_radius)], 1.0)
        outer = cv2.GaussianBlur(outer, (0, 0), sigmaX=max(small_radius * 0.30, 6.0))

        # Inner shape used to carve the ring out of the halo.
        inner = np.zeros((small_h, small_w), np.float32)
        cv2.fillPoly(
            inner,
            [self._shape_points(small_center, small_radius * 0.80)],
            1.0,
        )
        inner = cv2.GaussianBlur(inner, (0, 0), sigmaX=max(small_radius * 0.10, 3.0))

        halo = np.clip(outer - inner, 0.0, 1.0)

        # Crisp perimeter rim.
        rim = np.zeros((small_h, small_w), np.float32)
        cv2.polylines(
            rim,
            [self._shape_points(small_center, small_radius)],
            True,
            1.0,
            thickness=max(2, int(2.0 * scale)),
        )
        rim = cv2.GaussianBlur(rim, (0, 0), sigmaX=1.0)

        color = np.asarray(self._glow_color, dtype=np.float32)
        glow = np.zeros((small_h, small_w, 3), np.float32)
        glow[..., 0] = halo * color[0] * 1.0 + rim * color[0] * 1.2
        glow[..., 1] = halo * color[1] * 1.0 + rim * color[1] * 1.2
        glow[..., 2] = halo * color[2] * 1.0 + rim * color[2] * 1.2

        glow = np.clip(glow, 0.0, 255.0).astype(np.uint8)
        if scale != 1.0:
            glow = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
        return glow
