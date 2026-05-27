"""
Convert phase/amplitude matrices to FPGA BRAM row format.
"""

import numpy as np


class ControlMatrixConverter:
    """Convert algorithm outputs (phase, amplitude) to BRAM-compatible 32-bit rows."""

    def __init__(self, div: int = 30):
        self.div = div
        self._div_u32 = np.uint32(div)

    def convert(
        self,
        phase: np.ndarray,
        amplitude: np.ndarray,
        cycle_index: int,
    ) -> np.ndarray:
        """Convert phase and amplitude to 64 BRAM rows."""
        return self.convert_batch(
            phase[np.newaxis, ...],
            amplitude[np.newaxis, ...],
            cycle_index,
        )[0]

    def convert_batch(
        self,
        phase_batch: np.ndarray,
        amplitude_batch: np.ndarray,
        cycle_index_start: int,
    ) -> np.ndarray:
        """Vectorized convert (B, 8, 8) -> (B, 64) uint32."""
        phase = np.asarray(phase_batch, dtype=np.float32)
        amplitude = np.asarray(amplitude_batch, dtype=np.float32)
        if phase.ndim != 3 or phase.shape[1:] != (8, 8):
            raise ValueError(f"phase_batch must be (B, 8, 8), got {phase.shape}")
        if amplitude.shape != phase.shape:
            raise ValueError(
                f"amplitude_batch shape {amplitude.shape} != phase {phase.shape}"
            )

        b = phase.shape[0]
        div = self.div
        two_pi = np.float32(2.0 * np.pi)

        phase_norm = (phase + np.pi) / two_pi
        phase_idx = (np.round(phase_norm * div).astype(np.uint32)) % div

        clipped_amp = np.clip(amplitude, 0.0, 1.0)
        duty_float = np.arcsin(clipped_amp) / np.pi * div
        duty_idx = np.round(duty_float).astype(np.uint32)
        np.clip(duty_idx, 0, div, out=duty_idx)

        ci = (np.arange(b, dtype=np.uint32) + np.uint32(cycle_index_start)) & np.uint32(
            0xFFFF
        )
        packed = (
            (ci[:, np.newaxis, np.newaxis] << np.uint32(16))
            | (duty_idx << np.uint32(8))
            | phase_idx
        )
        return packed.reshape(b, 64)

    def convert_patterns_batch(
        self,
        phase_batch: np.ndarray,
        amplitude_batch: np.ndarray,
    ) -> np.ndarray:
        """Pack (duty, phase) only in low 16 bits; runtime ORs cycle_index in high 16."""
        packed = self.convert_batch(phase_batch, amplitude_batch, 0)
        return packed & np.uint32(0x0000FFFF)
