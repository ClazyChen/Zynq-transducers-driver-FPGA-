"""
End-to-end loopback test: PC-side quantization error verification.

Tests the full chain:
  Focus positions → Algorithm inference → Converter quantization
  → Inverse converter → Reconstructed field
  → Comparison with original reference field

Also generates BRAM binary files for Scala-side loopback verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import time
import json
import struct
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from algorithms.engine import UNetEngine, GSPATEngine
from algorithms.field_builder_torch import FieldBuilderTorch
from core.converter import ControlMatrixConverter
import config as cfg


# ---------------------------------------------------------------------------
# LUT generation (Python replica of LutGenerator.generateLut)
# ---------------------------------------------------------------------------
def generate_lut(div: int = 30, phase_duty_bits: int = 5):
    """Generate the same LUT as Scala LutGenerator."""
    lut_size = 1 << phase_duty_bits
    lut = []
    for duty in range(lut_size):
        row = []
        for phase in range(lut_size):
            bits = []
            for frame in range(div):
                pos = (frame - phase + div * 2) % div
                bits.append(1 if pos < duty else 0)
            # Pack bits: frame 0 is LSB (same as Scala)
            bitmap = sum(bit << idx for idx, bit in enumerate(bits))
            row.append(bitmap)
        lut.append(row)
    return lut


def decode_bram_row(row: int) -> tuple:
    """Decode a 32-bit BRAM row into (cycle_index, duty, phase)."""
    cycle_index = (row >> 16) & 0xFFFF
    duty = (row >> 8) & 0xFF
    phase = row & 0xFF
    return cycle_index, duty, phase


def inverse_converter(bram_rows: np.ndarray, div: int = 30) -> tuple:
    """Reconstruct quantized phase and amplitude from BRAM rows."""
    phase_idx = np.zeros((8, 8), dtype=np.float32)
    duty_idx = np.zeros((8, 8), dtype=np.float32)

    for idx in range(64):
        row = bram_rows[idx]
        _, duty, phase = decode_bram_row(row)
        r = idx // 8
        c = idx % 8
        phase_idx[r, c] = phase
        duty_idx[r, c] = duty

    # Reverse quantization formulas
    # phase_idx = round((phase + pi) / (2*pi) * div) % div
    phase_rad = (phase_idx / div) * 2.0 * np.pi - np.pi

    # duty = round(arcsin(amplitude) / pi * div)
    # amplitude = sin(duty / div * pi)
    amplitude = np.sin(duty_idx / div * np.pi)
    amplitude = np.clip(amplitude, 0.0, 1.0)

    return phase_rad, amplitude


def compute_metrics(reference: np.ndarray, reconstructed: np.ndarray, foci: list) -> dict:
    """Compute similarity metrics between two intensity fields.
    
    For multi-focus cases, computes per-focus peak error by searching
    around each preset focus position rather than using global max.
    """
    mse = np.mean((reference - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    max_ref = reference.max()
    max_rec = reconstructed.max()
    peak_err = abs(max_ref - max_rec)
    rel_peak_err = peak_err / max_ref if max_ref > 0 else 0.0

    # Per-focus position error: search in a small window around each focus
    focus_radius_px = int(8.0 * cfg.IMAGE_RESOLUTION / cfg.IMAGE_SIZE)  # 8mm radius
    focus_errors = []
    for fx_mm, fy_mm in foci:
        # Convert mm to pixel coords (origin at top-left)
        cx = int(fx_mm * cfg.IMAGE_RESOLUTION / cfg.IMAGE_SIZE)
        cy = int(fy_mm * cfg.IMAGE_RESOLUTION / cfg.IMAGE_SIZE)
        
        # Extract window around focus from reference and reconstructed
        x0 = max(0, cx - focus_radius_px)
        x1 = min(cfg.IMAGE_RESOLUTION, cx + focus_radius_px + 1)
        y0 = max(0, cy - focus_radius_px)
        y1 = min(cfg.IMAGE_RESOLUTION, cy + focus_radius_px + 1)
        
        ref_window = reference[y0:y1, x0:x1]
        rec_window = reconstructed[y0:y1, x0:x1]
        
        ref_peak_local = np.unravel_index(np.argmax(ref_window), ref_window.shape)
        rec_peak_local = np.unravel_index(np.argmax(rec_window), rec_window.shape)
        
        ref_peak_global = (ref_peak_local[0] + y0, ref_peak_local[1] + x0)
        rec_peak_global = (rec_peak_local[0] + y0, rec_peak_local[1] + x0)
        
        pixel_dist = np.sqrt(
            (ref_peak_global[0] - rec_peak_global[0])**2 +
            (ref_peak_global[1] - rec_peak_global[1])**2
        )
        mm_dist = pixel_dist * (cfg.IMAGE_SIZE / cfg.IMAGE_RESOLUTION)
        focus_errors.append(mm_dist)
    
    max_focus_err = max(focus_errors) if focus_errors else 0.0
    mean_focus_err = np.mean(focus_errors) if focus_errors else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "peak_ref": float(max_ref),
        "peak_rec": float(max_rec),
        "peak_abs_err": float(peak_err),
        "peak_rel_err": float(rel_peak_err),
        "focus_errors_mm": [float(e) for e in focus_errors],
        "max_focus_err_mm": float(max_focus_err),
        "mean_focus_err_mm": float(mean_focus_err),
    }


def run_loopback_test(
    foci: list,
    algorithm: str,
    engine,
    field_builder: FieldBuilderTorch,
    converter: ControlMatrixConverter,
    test_name: str,
    output_dir: Path,
):
    """Run one loopback test case and save results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name} | Algorithm: {algorithm} | Foci: {foci}")
    print(f"{'='*60}")

    # 1. Algorithm inference
    phase_cont, amplitude_cont = engine.infer([foci])
    phase_cont = phase_cont[0]
    amplitude_cont = amplitude_cont[0]
    print(f"  Continuous phase range: [{phase_cont.min():.3f}, {phase_cont.max():.3f}] rad")
    print(f"  Continuous amplitude range: [{amplitude_cont.min():.3f}, {amplitude_cont.max():.3f}]")

    # 2. Reference field (continuous values)
    intensity_ref = field_builder.build_field(
        torch.from_numpy(phase_cont).float(),
        torch.from_numpy(amplitude_cont).float(),
    ).cpu().numpy()

    # 3. Converter quantization
    bram_rows = converter.convert(phase_cont, amplitude_cont, cycle_index=1)
    print(f"  BRAM rows shape: {bram_rows.shape}, dtype: {bram_rows.dtype}")

    # 4. Save BRAM binary for Scala loopback test
    bram_file = output_dir / f"{test_name}_bram.bin"
    with open(bram_file, 'wb') as f:
        for row in bram_rows:
            f.write(struct.pack('<I', int(row)))
    print(f"  BRAM binary saved: {bram_file}")

    # 5. Save metadata JSON
    meta = {
        "test_name": test_name,
        "algorithm": algorithm,
        "foci": foci,
        "phase_continuous": phase_cont.tolist(),
        "amplitude_continuous": amplitude_cont.tolist(),
        "bram_rows": bram_rows.tolist(),
    }
    meta_file = output_dir / f"{test_name}_meta.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 6. Inverse converter: reconstruct quantized phase/amplitude
    phase_quant, amplitude_quant = inverse_converter(bram_rows)
    print(f"  Quantized phase range: [{phase_quant.min():.3f}, {phase_quant.max():.3f}] rad")
    print(f"  Quantized amplitude range: [{amplitude_quant.min():.3f}, {amplitude_quant.max():.3f}]")

    # Quantization error on control parameters
    phase_qerr = np.abs(phase_cont - phase_quant).mean()
    amp_qerr = np.abs(amplitude_cont - amplitude_quant).mean()
    print(f"  Mean phase quantization error: {phase_qerr:.4f} rad")
    print(f"  Mean amplitude quantization error: {amp_qerr:.4f}")

    # 7. Reconstructed field (quantized values)
    intensity_rec = field_builder.build_field(
        torch.from_numpy(phase_quant).float(),
        torch.from_numpy(amplitude_quant).float(),
    ).cpu().numpy()

    # 8. Metrics
    metrics = compute_metrics(intensity_ref, intensity_rec, foci)
    print(f"  Reference peak: {metrics['peak_ref']:.4f}")
    print(f"  Reconstructed peak: {metrics['peak_rec']:.4f}")
    print(f"  Peak relative error: {metrics['peak_rel_err']*100:.2f}%")
    focus_errs_str = ", ".join(f"{e:.3f}" for e in metrics['focus_errors_mm'])
    print(f"  Focus position errors: [{focus_errs_str}] mm (max: {metrics['max_focus_err_mm']:.3f} mm)")
    print(f"  RMSE: {metrics['rmse']:.6f}")

    # 9. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(intensity_ref, origin='upper', cmap='hot', vmin=0)
    axes[0].set_title('参考声场 (连续值)')
    axes[0].set_xlabel('X (px)')
    axes[0].set_ylabel('Y (px)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(intensity_rec, origin='upper', cmap='hot', vmin=0)
    axes[1].set_title('重建声场 (量化值)')
    axes[1].set_xlabel('X (px)')
    axes[1].set_ylabel('Y (px)')
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(intensity_ref - intensity_rec)
    im2 = axes[2].imshow(diff, origin='upper', cmap='viridis')
    axes[2].set_title('绝对误差')
    axes[2].set_xlabel('X (px)')
    axes[2].set_ylabel('Y (px)')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"{test_name} | {algorithm} | 峰值误差: {metrics['peak_rel_err']*100:.2f}%")
    plt.tight_layout()
    plot_file = output_dir / f"{test_name}_comparison.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"  Comparison plot saved: {plot_file}")

    return metrics, bram_file


def main():
    output_dir = Path(__file__).parent.parent / "loopback_output"
    output_dir.mkdir(exist_ok=True)

    # Initialize engines
    print("Loading engines...")
    unet = UNetEngine(str(cfg.CHECKPOINT_PATH), base_channels=cfg.UNET_BASE_CHANNELS, use_compile=False)
    gs = GSPATEngine()
    fb = FieldBuilderTorch(
        f=cfg.F, array_width=cfg.ARRAY_WIDTH, array_height=cfg.ARRAY_HEIGHT,
        d=cfg.D, z=cfg.Z, image_resolution=cfg.IMAGE_RESOLUTION,
        image_size=cfg.IMAGE_SIZE, c=cfg.c,
    )
    conv = ControlMatrixConverter(div=cfg.DIV)

    # Pre-generate LUT for validation
    print("Generating LUT...")
    lut = generate_lut(div=cfg.DIV, phase_duty_bits=cfg.PHASE_DUTY_BITS)
    print(f"LUT shape: {len(lut)} x {len(lut[0])}")

    # Test cases
    test_cases = [
        ("unet_1focus_center", "unet", [(64.0, 64.0)]),
        ("unet_1focus_offset", "unet", [(50.0, 70.0)]),
        ("unet_2foci", "unet", [(55.0, 55.0), (75.0, 75.0)]),
        ("unet_3foci", "unet", [(50.0, 60.0), (64.0, 64.0), (78.0, 70.0)]),
        ("gspat_1focus", "gs_pat", [(64.0, 64.0)]),
        ("gspat_2foci", "gs_pat", [(55.0, 55.0), (75.0, 75.0)]),
    ]

    all_metrics = {}
    all_bram_files = []

    for test_name, algo, foci in test_cases:
        engine = unet if algo == "unet" else gs
        metrics, bram_file = run_loopback_test(
            foci=foci,
            algorithm=algo,
            engine=engine,
            field_builder=fb,
            converter=conv,
            test_name=test_name,
            output_dir=output_dir,
        )
        all_metrics[test_name] = metrics
        all_bram_files.append(bram_file)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<25} {'Peak Err %':>12} {'Max Focus(mm)':>14} {'RMSE':>10}")
    print("-" * 65)
    for name, m in all_metrics.items():
        print(f"{name:<25} {m['peak_rel_err']*100:>11.2f}% {m['max_focus_err_mm']:>14.3f} {m['rmse']:>10.6f}")

    # Save summary JSON
    summary_file = output_dir / "loopback_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_file}")
    print(f"BRAM files for Scala test: {len(all_bram_files)}")
    for bf in all_bram_files:
        print(f"  - {bf.name}")

    # Overall pass/fail criteria
    max_acceptable_peak_err = 0.10  # 10%
    max_acceptable_focus_err_mm = 2.0
    all_pass = all(
        m["peak_rel_err"] < max_acceptable_peak_err and
        m["max_focus_err_mm"] < max_acceptable_focus_err_mm
        for m in all_metrics.values()
    )
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL LOOPBACK TESTS PASSED")
    else:
        print("SOME LOOPBACK TESTS FAILED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
