"""
Benchmark host streaming path (build + mock send) without GUI/device.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np

import config as cfg
from core.converter import ControlMatrixConverter
from core.device_client import DeviceClient
from core.renderer import RenderController
from algorithms.engine import GSPATEngine


def bench_converter():
    conv = ControlMatrixConverter(div=cfg.DIV)
    phases = np.random.randn(512, 8, 8).astype(np.float32) * 0.5
    amps = np.random.rand(512, 8, 8).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(20):
        conv.convert_batch(phases, amps, 1)
    dt = time.perf_counter() - t0
    mbps = (512 * 256 * 20) / dt / 1e6
    print(f"convert_batch 512 frames x20: {mbps:.1f} MB/s")


def bench_renderer_path():
    gs = GSPATEngine()
    client = DeviceClient(mock=True)
    client.connect()
    renderer = RenderController(None, gs, client, field_builder=None)
  # GS-PAT only for speed without checkpoint
    if renderer.unet_engine is None:
        pass
    renderer.configure(
        foci=[(64.0, 64.0)],
        mode="static",
        algorithm="gs_pat",
        lm_freq=25.0,
        lm_samples=12,
    )

    n = 512
    t0 = time.perf_counter()
    for _ in range(40):
        bram = renderer._build_bram_batch(0, n)
        client.send_burst(bram)
    dt = time.perf_counter() - t0
    mbps = (n * 256 * 40) / dt / 1e6
    print(f"build_bram_batch + mock send 512 x40: {mbps:.1f} MB/s")


if __name__ == "__main__":
    print("Streaming benchmark (target ~10 MB/s wire rate at 40 kHz)")
    bench_converter()
    bench_renderer_path()
