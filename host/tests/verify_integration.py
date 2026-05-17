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
from algorithms.field_builder_torch import FieldBuilderTorch
from core.device_client import DeviceClient
from core.renderer import RenderController

import config as cfg


def test_renderer_static():
    print("Testing RenderController (static mode)...")

    unet = UNetEngine(str(cfg.CHECKPOINT_PATH), base_channels=cfg.UNET_BASE_CHANNELS, use_compile=False)
    gs = GSPATEngine()
    client = DeviceClient(mock=True)
    client.connect()

    fb = FieldBuilderTorch(
        f=cfg.F, array_width=cfg.ARRAY_WIDTH, array_height=cfg.ARRAY_HEIGHT,
        d=cfg.D, z=cfg.Z, image_resolution=cfg.IMAGE_RESOLUTION,
        image_size=cfg.IMAGE_SIZE, c=cfg.c,
    )

    renderer = RenderController(unet, gs, client, fb)
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

    # Collect a few frames
    frames = []
    orig_send = client.send_frame

    def capture_send(ci, rows):
        frames.append((ci, rows.copy()))
        return orig_send(ci, rows)

    client.send_frame = capture_send

    renderer.start()
    time.sleep(0.5)  # Let it run for 0.5s (~150 frames at 300Hz)
    renderer.stop()

    print(f"  Captured {len(frames)} frames")
    assert len(frames) >= 30, f"Expected >=30 frames, got {len(frames)}"

    # Verify cycle index continuity
    cis = [f[0] for f in frames]
    for i in range(1, len(cis)):
        expected = (cis[i-1] + 1) & 0xFFFF
        assert cis[i] == expected, f"Cycle index discontinuity at frame {i}: {cis[i-1]} -> {cis[i]}"

    # Verify BRAM row format
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
    fb = None  # Skip visualization for speed

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

    frames = []
    orig_send = client.send_frame

    def capture_send(ci, rows):
        frames.append((ci, rows.copy()))
        return orig_send(ci, rows)

    client.send_frame = capture_send

    renderer.start()
    time.sleep(1.0)
    renderer.stop()

    print(f"  Captured {len(frames)} frames")
    assert len(frames) > 100

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
