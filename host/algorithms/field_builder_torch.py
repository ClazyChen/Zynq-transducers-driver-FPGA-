"""
Field builder module (PyTorch version).
Computes acoustic field intensity from phase/amplitude matrices, supports autograd.
"""

import numpy as np
import torch
from typing import Optional
from .weight_matrix import calculate_weight_matrix


class FieldBuilderTorch:
    """Field builder (PyTorch version): computes acoustic field intensity from phase/amplitude matrices."""

    def __init__(
        self,
        f: float,
        array_width: int,
        array_height: int,
        d: float,
        z: float,
        image_resolution: int,
        image_size: float,
        c: float = 343000.0,
        device: str = None
    ):
        self.f = f
        self.array_width = array_width
        self.array_height = array_height
        self.d = d
        self.z = z
        self.image_resolution = image_resolution
        self.image_size = image_size
        self.c = c

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        x = np.linspace(-image_size/2, image_size/2, image_resolution)
        y = np.linspace(-image_size/2, image_size/2, image_resolution)
        self.X, self.Y = np.meshgrid(x, y)

        self._weight_matrix = None
        self._weight_matrix_computed = False

    def _compute_weight_matrix(self):
        """Precompute weight matrix (lazy computation)."""
        if not self._weight_matrix_computed:
            x_flat = self.X.flatten()
            y_flat = self.Y.flatten()
            points = np.stack([
                x_flat,
                y_flat,
                np.full(self.image_resolution * self.image_resolution, self.z)
            ], axis=1)

            weight_matrix_np = calculate_weight_matrix(
                f=self.f,
                width=self.array_width,
                height=self.array_height,
                d=self.d,
                points=points,
                c=self.c
            )

            self._weight_matrix = torch.from_numpy(weight_matrix_np).to(
                dtype=torch.complex64,
                device=self.device
            )
            self._weight_matrix_computed = True

    def build_field(self, phase: torch.Tensor, amplitude: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute acoustic field intensity from phase and amplitude.

        Args:
            phase: Phase matrix (radians), shape (array_width, array_height)
            amplitude: Amplitude matrix (optional), shape (array_width, array_height), range [0, 1]

        Returns:
            intensity: Intensity field, shape (image_resolution, image_resolution)
        """
        if isinstance(phase, torch.Tensor):
            phase = phase.to(self.device)
        else:
            phase = torch.tensor(phase, dtype=torch.float32, device=self.device)

        if phase.shape != (self.array_width, self.array_height):
            raise ValueError(
                f"Phase shape mismatch: expected ({self.array_width}, {self.array_height}), got {phase.shape}"
            )

        if amplitude is None:
            amplitude = torch.ones_like(phase)
        else:
            if isinstance(amplitude, torch.Tensor):
                amplitude = amplitude.to(self.device)
            else:
                amplitude = torch.tensor(amplitude, dtype=torch.float32, device=self.device)

            if amplitude.shape != (self.array_width, self.array_height):
                raise ValueError(
                    f"Amplitude shape mismatch: expected ({self.array_width}, {self.array_height}), got {amplitude.shape}"
                )

        self._compute_weight_matrix()

        phase_f32 = phase.to(torch.float32)
        signal_real = torch.cos(phase_f32) * amplitude
        signal_imag = torch.sin(phase_f32) * amplitude
        signal = torch.complex(signal_real, signal_imag)
        signal = signal.unsqueeze(0)

        total_signal = torch.sum(self._weight_matrix * signal, dim=(1, 2))
        intensity = torch.abs(total_signal)
        intensity = intensity.reshape(self.image_resolution, self.image_resolution)

        return intensity

    def build_field_batch(self, phase: torch.Tensor, amplitude: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batch compute acoustic field intensity.

        Args:
            phase: Phase matrix, shape (batch_size, array_width, array_height) or (batch_size, 1, array_width, array_height)
            amplitude: Amplitude matrix (optional), same shape rules

        Returns:
            intensity: shape (batch_size, image_resolution, image_resolution)
        """
        if isinstance(phase, torch.Tensor):
            phase = phase.to(self.device)
        else:
            phase = torch.tensor(phase, dtype=torch.float32, device=self.device)

        if phase.dim() == 4:
            phase = phase.squeeze(1)
        elif phase.dim() == 2:
            phase = phase.unsqueeze(0)

        if phase.dim() != 3 or phase.shape[1:] != (self.array_width, self.array_height):
            raise ValueError(
                f"Phase shape mismatch: expected (batch_size, {self.array_width}, {self.array_height}), got {phase.shape}"
            )

        batch_size = phase.shape[0]

        if amplitude is None:
            amplitude = torch.ones_like(phase)
        else:
            if isinstance(amplitude, torch.Tensor):
                amplitude = amplitude.to(self.device)
            else:
                amplitude = torch.tensor(amplitude, dtype=torch.float32, device=self.device)

            if amplitude.dim() == 4:
                amplitude = amplitude.squeeze(1)
            elif amplitude.dim() == 2:
                amplitude = amplitude.unsqueeze(0)

            if amplitude.dim() != 3 or amplitude.shape != phase.shape:
                raise ValueError(
                    f"Amplitude shape mismatch: expected {phase.shape}, got {amplitude.shape}"
                )

        self._compute_weight_matrix()

        phase_f32 = phase.to(torch.float32)
        signal_real = torch.cos(phase_f32) * amplitude
        signal_imag = torch.sin(phase_f32) * amplitude
        signal = torch.complex(signal_real, signal_imag)

        weight_matrix_expanded = self._weight_matrix.unsqueeze(0)
        signal_expanded = signal.unsqueeze(1)

        total_signal = torch.sum(weight_matrix_expanded * signal_expanded, dim=(2, 3))
        intensity = torch.abs(total_signal)
        intensity = intensity.reshape(batch_size, self.image_resolution, self.image_resolution)

        return intensity

    def get_weight_matrix(self) -> torch.Tensor:
        """Get precomputed weight matrix."""
        self._compute_weight_matrix()
        return self._weight_matrix.clone()
