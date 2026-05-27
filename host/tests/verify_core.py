"""
Quick verification of core logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np

from core.converter import ControlMatrixConverter
from core.device_client import DeviceClient


def test_converter():
    print("Testing ControlMatrixConverter...")
    conv = ControlMatrixConverter(div=30)

    phase = np.zeros((8, 8), dtype=np.float32)
    amplitude = np.ones((8, 8), dtype=np.float32)

    rows = conv.convert(phase, amplitude, cycle_index=1)
    assert rows.shape == (64,), f"Expected (64,), got {rows.shape}"
    assert rows.dtype == np.uint32

    # Check row0 format: [31:16] cycle_index, [15:8] duty, [7:0] phase
    row0 = rows[0]
    ci = (row0 >> 16) & 0xFFFF
    duty = (row0 >> 8) & 0xFF
    ph = row0 & 0xFF

    assert ci == 1, f"cycle_index mismatch: {ci}"
    # amplitude=1 -> arcsin(1)/pi * 30 = 0.5 * 30 = 15
    assert duty == 15, f"duty for amp=1 should be 15, got {duty}"
    # phase=0 -> (0+pi)/(2pi)*30 = 15
    assert ph == 15, f"phase for phase=0 should be 15, got {ph}"

    # Test with phase = -pi
    phase_neg = np.full((8, 8), -np.pi, dtype=np.float32)
    rows2 = conv.convert(phase_neg, np.zeros((8, 8)), cycle_index=2)
    row0_2 = rows2[0]
    ph2 = row0_2 & 0xFF
    assert ph2 == 0, f"phase for -pi should be 0, got {ph2}"

    # Test with phase = pi (wrap to 0)
    phase_pos = np.full((8, 8), np.pi, dtype=np.float32)
    rows3 = conv.convert(phase_pos, np.zeros((8, 8)), cycle_index=3)
    row0_3 = rows3[0]
    ph3 = row0_3 & 0xFF
    assert ph3 == 0, f"phase for +pi should wrap to 0, got {ph3}"

    print("  ControlMatrixConverter OK")
    return True


def test_device_client_mock():
    print("Testing DeviceClient (mock mode)...")
    client = DeviceClient(mock=True)
    assert client.connect()
    assert client.connected

    rows = np.arange(64, dtype=np.uint32)
    assert client.send_frame(1, rows)
    assert client.send_frame(2, rows)

    batch = np.stack([rows, rows + 1], axis=0)
    batch[0, 0] = (1 << 16) | (batch[0, 0] & 0xFFFF)
    batch[1, 0] = (2 << 16) | (batch[1, 0] & 0xFFFF)
    assert client.send_burst(batch)

    stats = client.get_stats()
    assert stats["frames_sent"] == 4
    assert stats["connected"]

    client.disconnect()
    assert not client.connected
    print("  DeviceClient OK")
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_converter()
    all_ok &= test_device_client_mock()

    if all_ok:
        print("\nAll core tests PASSED.")
    else:
        print("\nSome core tests FAILED.")
        sys.exit(1)
