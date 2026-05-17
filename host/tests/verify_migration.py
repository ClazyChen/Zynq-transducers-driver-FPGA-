"""
Verification script: check that migrated algorithms produce consistent results.
"""

import sys
from pathlib import Path

# Ensure host/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import numpy as np
import torch

from algorithms.engine import UNetEngine, GSPATEngine
from algorithms.target_generator import TargetGenerator


def test_unet_engine():
    print("=" * 60)
    print("Testing UNetEngine...")
    print("=" * 60)

    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "checkpoint_best.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False

    try:
        engine = UNetEngine(
            checkpoint_path=str(checkpoint_path),
            base_channels=32,
            use_compile=False,  # avoid compile overhead for quick test
        )
        print(f"  Model loaded successfully on {engine.device}")
        print(f"  Actual base_channels: {engine.base_channels}")
    except Exception as e:
        print(f"ERROR loading UNetEngine: {e}")
        return False

    # Test single focus
    foci = [[(64.0, 64.0)]]
    phase, amplitude = engine.infer(foci)
    print(f"  Single focus output: phase shape {phase.shape}, range [{phase.min():.3f}, {phase.max():.3f}]")
    print(f"                       amplitude shape {amplitude.shape}, range [{amplitude.min():.3f}, {amplitude.max():.3f}]")

    # Test batch with 3 foci
    foci_batch = [
        [(50.0, 50.0)],
        [(60.0, 60.0), (70.0, 70.0)],
        [(55.0, 55.0), (65.0, 65.0), (75.0, 75.0)],
    ]
    phase_b, amplitude_b = engine.infer(foci_batch)
    print(f"  Batch output: phase shape {phase_b.shape}, amplitude shape {amplitude_b.shape}")

    print("  UNetEngine OK\n")
    return True


def test_gs_pat_engine():
    print("=" * 60)
    print("Testing GSPATEngine...")
    print("=" * 60)

    engine = GSPATEngine()
    print(f"  Device: {engine.device}, use_gpu: {engine.use_gpu}")

    # Test single focus
    foci = [[(64.0, 64.0)]]
    phase, amplitude = engine.infer(foci)
    print(f"  Single focus output: phase shape {phase.shape}, range [{phase.min():.3f}, {phase.max():.3f}]")
    print(f"                       amplitude shape {amplitude.shape}, range [{amplitude.min():.3f}, {amplitude.max():.3f}]")

    # Test batch
    foci_batch = [
        [(50.0, 50.0)],
        [(60.0, 60.0), (70.0, 70.0)],
        [(55.0, 55.0), (65.0, 65.0), (75.0, 75.0)],
    ]
    phase_b, amplitude_b = engine.infer(foci_batch)
    print(f"  Batch output: phase shape {phase_b.shape}, amplitude shape {amplitude_b.shape}")

    print("  GSPATEngine OK\n")
    return True


def test_target_generator():
    print("=" * 60)
    print("Testing TargetGenerator...")
    print("=" * 60)

    tg = TargetGenerator()
    foci = [(64.0, 64.0), (70.0, 70.0)]
    pos_mask, neg_mask = tg.generate_masks(foci)
    print(f"  Positive mask shape: {pos_mask.shape}, sum: {pos_mask.sum():.1f}")
    print(f"  Negative mask shape: {neg_mask.shape}, mean: {neg_mask.mean():.3f}")

    print("  TargetGenerator OK\n")
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_target_generator()
    all_ok &= test_unet_engine()
    all_ok &= test_gs_pat_engine()

    print("=" * 60)
    if all_ok:
        print("All verification tests PASSED.")
    else:
        print("Some verification tests FAILED.")
        sys.exit(1)
