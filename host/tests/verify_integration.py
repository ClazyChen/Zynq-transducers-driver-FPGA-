"""
Integration test: verify renderer static/dynamic loops with mock device client.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import time
import numpy as np

from PySide6.QtCore import QCoreApplication

from algorithms.engine import UNetEngine, GSPATEngine
from core.device_client import DeviceClient
from core.renderer import RenderController

import config as cfg


def _expand_burst_frames(captured: list) -> list:
    """Flatten send_burst captures into per-frame (cycle_index, rows) list."""
    frames = []
    for batch in captured:
        for i in range(batch.shape[0]):
            rows = batch[i]
            ci = (rows[0] >> 16) & 0xFFFF
            frames.append((ci, rows.copy()))
    return frames


def test_renderer_static():
    print("Testing RenderController (static mode)...")

    unet = UNetEngine(str(cfg.CHECKPOINT_PATH), base_channels=cfg.UNET_BASE_CHANNELS, use_compile=False)
    gs = GSPATEngine()
    client = DeviceClient(mock=True)
    client.connect()

    # Skip field visualization in loop for throughput (GUI uses field_builder separately)
    renderer = RenderController(unet, gs, client, field_builder=None)
    renderer.configure(
        foci=[(64.0, 64.0), (70.0, 70.0)],
        mode="static",
        algorithm="unet",
        lm_freq=25.0,
        lm_amp=4.0,
        lm_samples=12,
        lm_direction="x",
        dwell_time_ms=500,
    )

    captured = []
    orig_send_burst = client.send_burst

    def capture_burst(batch):
        captured.append(batch.copy())
        return orig_send_burst(batch)

    client.send_burst = capture_burst

    renderer.start()
    time.sleep(1.0)
    renderer.stop()

    frames = _expand_burst_frames(captured)
    print(f"  Captured {len(frames)} frames in {len(captured)} bursts")
    prime_min = int(cfg.BURST_NOMINAL_FRAMES * cfg.BURST_PRIME_MULTIPLIER)
    assert captured[0].shape[0] >= prime_min, (
        f"First burst expected >={prime_min} frames, got {captured[0].shape[0]}"
    )
    assert len(captured) >= 2, f"Expected multiple bursts, got {len(captured)}"
    assert len(frames) >= 2000, f"Expected >=2000 frames in 1.0s, got {len(frames)}"
    assert len(frames) <= 50000, f"Expected <=50000 frames in 1.0s, got {len(frames)}"

    cis = [f[0] for f in frames]
    for i in range(1, len(cis)):
        expected = (cis[i - 1] + 1) & 0xFFFF
        assert cis[i] == expected, (
            f"Cycle index discontinuity at frame {i}: {cis[i-1]} -> {cis[i]}"
        )

    _, first_rows = frames[0]
    assert first_rows.shape == (64,), f"Expected (64,), got {first_rows.shape}"
    assert first_rows.dtype == np.uint32

    row0 = first_rows[0]
    ci = (row0 >> 16) & 0xFFFF
    duty = (row0 >> 8) & 0xFF
    ph = row0 & 0xFF
    print(f"  First frame row0: cycle_index={ci}, duty={duty}, phase={ph}")

    print("  Static mode OK\n")
    return True


def test_renderer_dynamic():
    print("Testing RenderController (dynamic mode)...")

    unet = UNetEngine(str(cfg.CHECKPOINT_PATH), base_channels=cfg.UNET_BASE_CHANNELS, use_compile=False)
    gs = GSPATEngine()
    client = DeviceClient(mock=True)
    client.connect()
    fb = None

    renderer = RenderController(unet, gs, client, fb)
    renderer.configure(
        foci=[(60.0, 60.0), (70.0, 70.0)],
        mode="dynamic",
        algorithm="gs_pat",
        lm_freq=25.0,
        lm_amp=4.0,
        lm_samples=12,
        lm_direction="y",
        dwell_time_ms=200,
    )

    captured = []
    orig_send_burst = client.send_burst

    def capture_burst(batch):
        captured.append(batch.copy())
        return orig_send_burst(batch)

    client.send_burst = capture_burst

    renderer.start()
    time.sleep(1.0)
    renderer.stop()

    frames = _expand_burst_frames(captured)
    print(f"  Captured {len(frames)} frames in {len(captured)} bursts")
    assert len(captured) >= 2, f"Expected multiple bursts, got {len(captured)}"
    assert len(frames) >= 2000, f"Expected >=2000 frames in 1.0s, got {len(frames)}"
    assert len(frames) <= 50000, f"Expected <=50000 frames in 1.0s, got {len(frames)}"

    print("  Dynamic mode OK\n")
    return True


if __name__ == "__main__":
    app = QCoreApplication(sys.argv)

    all_ok = True
    all_ok &= test_renderer_static()
    all_ok &= test_renderer_dynamic()

    if all_ok:
        print("All integration tests PASSED.")
    else:
        print("Some integration tests FAILED.")
        sys.exit(1)
