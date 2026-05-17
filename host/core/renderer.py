"""
Render controller: manages the real-time rendering loop, LM trajectory generation,
algorithm inference batching, and frame transmission.
"""

import time
import threading
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from algorithms.engine import UNetEngine, GSPATEngine
from algorithms.field_builder_torch import FieldBuilderTorch
from .converter import ControlMatrixConverter
from .device_client import DeviceClient


class RenderController(QObject):
    """Real-time render controller running in a background thread."""

    # Signals for GUI updates (thread-safe via Qt queued connections)
    # Using object instead of np.ndarray to avoid cross-thread type marshalling issues.
    field_visualization = Signal(object)       # intensity field (128, 128) float32
    status_message = Signal(str)
    fps_updated = Signal(float)
    connection_status = Signal(str)
    current_focus_info = Signal(str)           # e.g. "Focus 1/3"

    def __init__(
        self,
        unet_engine: UNetEngine,
        gs_pat_engine: GSPATEngine,
        device_client: DeviceClient,
        field_builder: Optional[FieldBuilderTorch] = None,
    ):
        super().__init__()
        self.unet_engine = unet_engine
        self.gs_pat_engine = gs_pat_engine
        self.device_client = device_client
        self.converter = ControlMatrixConverter(div=30)
        self.field_builder = field_builder

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Configuration defaults
        self.foci: List[Tuple[float, float]] = []
        self.mode: str = "static"          # "static" or "dynamic"
        self.algorithm: str = "unet"       # "unet" or "gs_pat"
        self.lm_freq: float = 25.0         # Hz
        self.lm_amp: float = 4.0           # mm
        self.lm_samples: int = 12          # samples per LM period
        self.lm_direction: str = "x"       # "x" or "y"
        self.dwell_time_ms: float = 500.0  # ms per focus in dynamic mode

        # Precomputed playback data
        self._precomputed_phases: Optional[np.ndarray] = None
        self._precomputed_amplitudes: Optional[np.ndarray] = None
        self._playback_sequence: Optional[List[int]] = None
        self._dynamic_focus_data: Optional[List[Dict]] = None
        self._cycle_index = 1

        # Visualization throttling
        self._viz_interval = 1.0 / 30.0    # 30 FPS max for GUI
        self._last_viz_time = 0.0

    def configure(
        self,
        foci: List[Tuple[float, float]],
        mode: str = "static",
        algorithm: str = "unet",
        lm_freq: float = 25.0,
        lm_amp: float = 4.0,
        lm_samples: int = 12,
        lm_direction: str = "x",
        dwell_time_ms: float = 500.0,
    ):
        """Update rendering configuration and precompute trajectories."""
        if len(foci) == 0 or len(foci) > 3:
            raise ValueError("焦点数量必须在 1 到 3 之间")
        if mode not in ("static", "dynamic"):
            raise ValueError("模式必须是 'static' 或 'dynamic'")
        if algorithm not in ("unet", "gs_pat"):
            raise ValueError("算法必须是 'unet' 或 'gs_pat'")
        if lm_samples < 2:
            raise ValueError("lm_samples 必须 >= 2")
        if lm_direction not in ("x", "y"):
            raise ValueError("lm_direction 必须是 'x' 或 'y'")

        self.foci = [(float(x), float(y)) for x, y in foci]
        self.mode = mode
        self.algorithm = algorithm
        self.lm_freq = lm_freq
        self.lm_amp = lm_amp
        self.lm_samples = lm_samples
        self.lm_direction = lm_direction
        self.dwell_time_ms = dwell_time_ms

        self._precompute()
        self.status_message.emit(f"已配置: {len(foci)} 个焦点, {mode} 模式, {algorithm}")

    def _precompute(self):
        """Precompute LM trajectories and run batch inference."""
        N = self.lm_samples
        dt = 1.0 / (self.lm_freq * N)
        t = np.arange(N) * dt
        offsets = self.lm_amp * np.sin(2.0 * np.pi * self.lm_freq * t)

        if self.mode == "static":
            configs = []
            for off in offsets:
                foci_positions = []
                for x0, y0 in self.foci:
                    if self.lm_direction == "x":
                        foci_positions.append((x0 + off, y0))
                    else:
                        foci_positions.append((x0, y0 + off))
                configs.append(foci_positions)

            unique_configs, sequence = self._deduplicate_configs(configs)
            phases, amplitudes = self._infer_batch(unique_configs)

            self._precomputed_phases = phases
            self._precomputed_amplitudes = amplitudes
            self._playback_sequence = sequence
            self._dynamic_focus_data = None

        else:  # dynamic
            all_unique_configs = []
            per_focus_meta = []

            for focus in self.foci:
                configs = []
                for off in offsets:
                    if self.lm_direction == "x":
                        configs.append([(focus[0] + off, focus[1])])
                    else:
                        configs.append([(focus[0], focus[1] + off)])

                unique_configs, sequence = self._deduplicate_configs(configs)
                per_focus_meta.append({
                    "n_unique": len(unique_configs),
                    "sequence": sequence,
                })
                all_unique_configs.extend(unique_configs)

            phases, amplitudes = self._infer_batch(all_unique_configs)

            idx = 0
            self._dynamic_focus_data = []
            for meta in per_focus_meta:
                n = meta["n_unique"]
                self._dynamic_focus_data.append({
                    "phases": phases[idx: idx + n],
                    "amplitudes": amplitudes[idx: idx + n],
                    "sequence": meta["sequence"],
                })
                idx += n

            self._precomputed_phases = None
            self._precomputed_amplitudes = None
            self._playback_sequence = None

    def _deduplicate_configs(
        self,
        configs: List[List[Tuple[float, float]]]
    ) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
        """Deduplicate focus configurations and return index mapping."""
        seen: Dict[tuple, int] = {}
        unique: List[List[Tuple[float, float]]] = []
        sequence: List[int] = []
        for cfg in configs:
            key = tuple((round(float(x), 6), round(float(y), 6)) for x, y in cfg)
            if key not in seen:
                seen[key] = len(unique)
                unique.append(cfg)
            sequence.append(seen[key])
        return unique, sequence

    def _infer_batch(
        self,
        configs: List[List[Tuple[float, float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run batch inference on focus configurations."""
        if not configs:
            return np.zeros((0, 8, 8)), np.zeros((0, 8, 8))
        if self.algorithm == "unet":
            return self.unet_engine.infer(configs)
        else:
            return self.gs_pat_engine.infer(configs)

    def start(self):
        """Start the rendering thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._precomputed_phases is None and self._dynamic_focus_data is None:
            raise RuntimeError("Renderer not configured. Call configure() first.")
        if not self.device_client.connected:
            self.connection_status.emit("设备未连接")
            return

        self._stop_event.clear()
        self._cycle_index = 1
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.status_message.emit("渲染已启动")

    def stop(self):
        """Stop the rendering thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.status_message.emit("渲染已停止")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self):
        """Main real-time rendering loop."""
        target_interval = 1.0 / (self.lm_freq * self.lm_samples)
        frame_count = 0
        fps_start_time = time.perf_counter()

        if self.mode == "static":
            self._run_static_loop(target_interval, frame_count, fps_start_time)
        else:
            self._run_dynamic_loop(target_interval, frame_count, fps_start_time)

    def _run_static_loop(self, target_interval: float, frame_count: int, fps_start_time: float):
        """Static mode: cycle through precomputed sequence."""
        seq = self._playback_sequence
        phases = self._precomputed_phases
        amplitudes = self._precomputed_amplitudes
        N = len(seq)
        idx = 0

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            si = seq[idx]
            phase = phases[si]
            amplitude = amplitudes[si]

            self._send_frame(phase, amplitude)
            self._maybe_compute_field(phase, amplitude)

            idx = (idx + 1) % N
            frame_count += 1

            # FPS calculation
            now = time.perf_counter()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(fps)
                frame_count = 0
                fps_start_time = now

            # Precise timing
            sleep_time = target_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _run_dynamic_loop(self, target_interval: float, frame_count: int, fps_start_time: float):
        """Dynamic mode: dwell on each focus then switch."""
        focus_idx = 0
        focus_start_time = time.perf_counter()
        num_foci = len(self._dynamic_focus_data)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # Check dwell time and switch focus if needed
            elapsed_focus = (loop_start - focus_start_time) * 1000.0
            if elapsed_focus >= self.dwell_time_ms:
                focus_idx = (focus_idx + 1) % num_foci
                focus_start_time = loop_start
                self.current_focus_info.emit(f"焦点 {focus_idx + 1}/{num_foci}")

            data = self._dynamic_focus_data[focus_idx]
            seq = data["sequence"]
            phases = data["phases"]
            amplitudes = data["amplitudes"]
            N = len(seq)

            # Local frame index within current focus dwell period
            # We want to cycle through the LM sequence continuously during dwell
            local_idx = int(((loop_start - focus_start_time) * self.lm_freq * self.lm_samples)) % N
            si = seq[local_idx]
            phase = phases[si]
            amplitude = amplitudes[si]

            self._send_frame(phase, amplitude)
            self._maybe_compute_field(phase, amplitude)

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(fps)
                frame_count = 0
                fps_start_time = now

            sleep_time = target_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _send_frame(self, phase: np.ndarray, amplitude: np.ndarray):
        """Convert and send a single frame."""
        bram_rows = self.converter.convert(phase, amplitude, self._cycle_index)
        ok = self.device_client.send_frame(self._cycle_index, bram_rows)
        if ok:
            self._cycle_index = (self._cycle_index + 1) & 0xFFFF
        else:
            self.connection_status.emit("发送失败")

    def _maybe_compute_field(self, phase: np.ndarray, amplitude: np.ndarray):
        """Compute intensity field for visualization (throttled)."""
        if self.field_builder is None:
            return
        now = time.perf_counter()
        if now - self._last_viz_time < self._viz_interval:
            return
        self._last_viz_time = now
        try:
            intensity = self.field_builder.build_field(
                torch.from_numpy(phase).float(),
                torch.from_numpy(amplitude).float(),
            )
            arr = intensity.cpu().numpy()
            self.field_visualization.emit(arr)
        except Exception as e:
            print(f"[Renderer] Field visualization error: {e}")
            import traceback
            traceback.print_exc()
