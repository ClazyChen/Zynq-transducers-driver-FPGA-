"""
Field builder module (NumPy version).
Computes acoustic field intensity from phase/amplitude matrices.
"""

import numpy as np
from .weight_matrix import calculate_weight_matrix


class FieldBuilder:
    """Field builder: computes acoustic field intensity from phase/amplitude matrices."""

    def __init__(
        self,
        f: float,
        array_width: int,
        array_height: int,
        d: float,
        z: float,
        image_resolution: int,
        image_size: float,
        c: float = 343000.0
    ):
        self.f = f
        self.array_width = array_width
        self.array_height = array_height
        self.d = d
        self.z = z
        self.image_resolution = image_resolution
        self.image_size = image_size
        self.c = c

        self.X = None
        self.Y = None
        self._grid_computed = False

        self._weight_matrix = None
        self._weight_matrix_computed = False

    def _compute_grid(self):
        """Create coordinate grid (lazy computation)."""
        if not self._grid_computed:
            x = np.linspace(-self.image_size/2, self.image_size/2, self.image_resolution)
            y = np.linspace(-self.image_size/2, self.image_size/2, self.image_resolution)
            self.X, self.Y = np.meshgrid(x, y)
            self._grid_computed = True

    def _compute_weight_matrix(self):
        """Precompute weight matrix (lazy computation)."""
        if not self._weight_matrix_computed:
            self._compute_grid()

            x_flat = self.X.flatten()
            y_flat = self.Y.flatten()
            points = np.stack([
                x_flat,
                y_flat,
                np.full(self.image_resolution * self.image_resolution, self.z)
            ], axis=1)

            self._weight_matrix = calculate_weight_matrix(
                f=self.f,
                width=self.array_width,
                height=self.array_height,
                d=self.d,
                points=points,
                c=self.c
            )
            self._weight_matrix_computed = True

    def build_field(self, phase: np.ndarray, amplitude: np.ndarray = None) -> np.ndarray:
        """Compute acoustic field intensity from phase and amplitude matrices.

        Args:
            phase: Phase matrix (radians), shape (array_width, array_height)
            amplitude: Amplitude matrix (optional), shape (array_width, array_height), range [0, 1]
                       If None, defaults to all ones.

        Returns:
            intensity: Intensity field, shape (image_resolution, image_resolution)
        """
        phase = np.asarray(phase, dtype=np.float32)
        if phase.shape != (self.array_width, self.array_height):
            raise ValueError(
                f"Phase shape mismatch: expected ({self.array_width}, {self.array_height}), got {phase.shape}"
            )

        if amplitude is None:
            amplitude = np.ones_like(phase)
        else:
            amplitude = np.asarray(amplitude, dtype=np.float32)
            if amplitude.shape != (self.array_width, self.array_height):
                raise ValueError(
                    f"Amplitude shape mismatch: expected ({self.array_width}, {self.array_height}), got {amplitude.shape}"
                )

        self._compute_weight_matrix()

        signal = np.exp(1j * phase) * amplitude
        signal = signal[np.newaxis, :, :]

        total_signal = np.sum(self._weight_matrix * signal, axis=(1, 2))
        intensity = np.abs(total_signal)
        intensity = intensity.reshape(self.image_resolution, self.image_resolution)

        return intensity

    def build_field_at_points(
        self,
        phase: np.ndarray,
        points: np.ndarray,
        amplitude: np.ndarray = None
    ) -> np.ndarray:
        """Compute field intensity at specified points.

        Args:
            phase: Phase matrix (radians), shape (array_width, array_height)
            points: Target coordinates, shape (n, 3), each row (x, y, z)
            amplitude: Amplitude matrix (optional), shape (array_width, array_height)

        Returns:
            intensity: Intensity array, shape (n,)
        """
        phase = np.asarray(phase, dtype=np.float32)
        if phase.shape != (self.array_width, self.array_height):
            raise ValueError(
                f"Phase shape mismatch: expected ({self.array_width}, {self.array_height}), got {phase.shape}"
            )

        if amplitude is None:
            amplitude = np.ones_like(phase)
        else:
            amplitude = np.asarray(amplitude, dtype=np.float32)
            if amplitude.shape != (self.array_width, self.array_height):
                raise ValueError(
                    f"Amplitude shape mismatch: expected ({self.array_width}, {self.array_height}), got {amplitude.shape}"
                )

        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points shape should be (n, 3), got {points.shape}")

        weight_matrix = calculate_weight_matrix(
            f=self.f,
            width=self.array_width,
            height=self.array_height,
            d=self.d,
            points=points,
            c=self.c
        )

        signal = np.exp(1j * phase) * amplitude
        signal = signal[np.newaxis, :, :]

        total_signal = np.sum(weight_matrix * signal, axis=(1, 2))
        intensity = np.abs(total_signal)

        return intensity
