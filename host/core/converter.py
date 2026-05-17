"""
Convert phase/amplitude matrices to FPGA BRAM row format.
"""

import numpy as np


class ControlMatrixConverter:
    """Convert algorithm outputs (phase, amplitude) to BRAM-compatible 32-bit rows."""

    def __init__(self, div: int = 30):
        self.div = div

    def convert(
        self,
        phase: np.ndarray,
        amplitude: np.ndarray,
        cycle_index: int,
    ) -> np.ndarray:
        """Convert phase and amplitude to 64 BRAM rows.

        Args:
            phase: (8, 8) radians, range [-pi, pi]
            amplitude: (8, 8) range [0, 1]
            cycle_index: 16-bit unsigned, strictly increasing

        Returns:
            bram_rows: (64,) uint32 array, little-endian
        """
        phase = np.asarray(phase, dtype=np.float64)
        amplitude = np.asarray(amplitude, dtype=np.float64)

        if phase.shape != (8, 8):
            raise ValueError(f"phase must be (8, 8), got {phase.shape}")
        if amplitude.shape != (8, 8):
            raise ValueError(f"amplitude must be (8, 8), got {amplitude.shape}")

        # Map phase [-pi, pi] -> [0, div-1]
        phase_norm = (phase + np.pi) / (2.0 * np.pi)
        phase_idx = np.round(phase_norm * self.div).astype(np.uint32) % self.div

        # Map amplitude [0, 1] -> duty [0, div] via D = arcsin(A) / pi
        # arcsin(1) / pi = 0.5, so max duty corresponds to 50% of div
        clipped_amp = np.clip(amplitude, 0.0, 1.0)
        duty_float = np.arcsin(clipped_amp) / np.pi * self.div
        duty_idx = np.round(duty_float).astype(np.uint32)
        duty_idx = np.clip(duty_idx, 0, self.div)

        # Pack: [31:16] cycle_index, [15:8] duty, [7:0] phase
        bram_rows = (
            (np.uint32(cycle_index) << 16)
            | (duty_idx << 8)
            | phase_idx
        )

        # Flatten in row-major order: transducer_index = row * 8 + col
        bram_rows = bram_rows.flatten()

        return bram_rows

    def convert_batch(
        self,
        phase_batch: np.ndarray,
        amplitude_batch: np.ndarray,
        cycle_index_start: int,
    ) -> np.ndarray:
        """Convert a batch of phase/amplitude matrices.

        Args:
            phase_batch: (B, 8, 8) radians
            amplitude_batch: (B, 8, 8) range [0, 1]
            cycle_index_start: starting cycle index

        Returns:
            bram_rows_batch: (B, 64) uint32 array
        """
        B = phase_batch.shape[0]
        results = []
        for i in range(B):
            ci = (cycle_index_start + i) & 0xFFFF  # 16-bit wrap
            rows = self.convert(phase_batch[i], amplitude_batch[i], ci)
            results.append(rows)
        return np.stack(results, axis=0)
