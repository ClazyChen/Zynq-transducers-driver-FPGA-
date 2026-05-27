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

import config as cfg
from algorithms.engine import UNetEngine, GSPATEngine
from algorithms.field_builder_torch import FieldBuilderTorch
from .converter import ControlMatrixConverter
from .device_client import DeviceClient


class RenderController(QObject):
    """Real-time render controller running in a background thread."""

    field_visualization = Signal(object)
    status_message = Signal(str)
    fps_updated = Signal(float)
    connection_status = Signal(str)
    current_focus_info = Signal(str)

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
        self.converter = ControlMatrixConverter(div=cfg.DIV)
        self.field_builder = field_builder

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.foci: List[Tuple[float, float]] = []
        self.mode: str = "static"
        self.algorithm: str = "unet"
        self.lm_freq: float = 25.0
        self.lm_amp: float = 4.0
        self.lm_samples: int = 12
        self.lm_direction: str = "x"
        self.dwell_time_ms: float = 500.0

        self._precomputed_phases: Optional[np.ndarray] = None
        self._precomputed_amplitudes: Optional[np.ndarray] = None
        self._playback_sequence: Optional[List[int]] = None
        self._dynamic_focus_data: Optional[List[Dict]] = None
        self._frames_per_lm_step: int = 1
        self._frames_per_dwell: int = 1
        self._cycle_index = 1
        self._first_burst = True

        self._viz_interval = 1.0 / cfg.FIELD_VISUALIZATION_FPS
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
        self._frames_per_lm_step = max(
            1,
            int(round(cfg.DEVICE_SAMPLE_RATE / (self.lm_freq * self.lm_samples))),
        )
        self._frames_per_dwell = max(
            1,
            int(round(self.dwell_time_ms * 1e-3 * cfg.DEVICE_SAMPLE_RATE)),
        )

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

        else:
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
        configs: List[List[Tuple[float, float]]],
    ) -> Tuple[List[List[Tuple[float, float]]], List[int]]:
        seen: Dict[tuple, int] = {}
        unique: List[List[Tuple[float, float]]] = []
        sequence: List[int] = []
        for cfg_item in configs:
            key = tuple((round(float(x), 6), round(float(y), 6)) for x, y in cfg_item)
            if key not in seen:
                seen[key] = len(unique)
                unique.append(cfg_item)
            sequence.append(seen[key])
        return unique, sequence

    def _infer_batch(
        self,
        configs: List[List[Tuple[float, float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not configs:
            return np.zeros((0, 8, 8)), np.zeros((0, 8, 8))
        if self.algorithm == "unet":
            return self.unet_engine.infer(configs)
        return self.gs_pat_engine.infer(configs)

    def _resolve_pattern(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """Return (phase, amplitude, focus_idx) for a global frame index."""
        if self.mode == "static":
            seq = self._playback_sequence
            assert seq is not None
            lm_step = (frame_idx // self._frames_per_lm_step) % len(seq)
            si = seq[lm_step]
            return (
                self._precomputed_phases[si],
                self._precomputed_amplitudes[si],
                None,
            )

        assert self._dynamic_focus_data is not None
        num_foci = len(self._dynamic_focus_data)
        focus_idx = (frame_idx // self._frames_per_dwell) % num_foci
        frame_in_dwell = frame_idx % self._frames_per_dwell
        data = self._dynamic_focus_data[focus_idx]
        seq = data["sequence"]
        lm_step = (frame_in_dwell // self._frames_per_lm_step) % len(seq)
        si = seq[lm_step]
        return data["phases"][si], data["amplitudes"][si], focus_idx

    def _build_batch(
        self, global_frame_start: int, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        phase_batch = np.empty((n, 8, 8), dtype=np.float64)
        amp_batch = np.empty((n, 8, 8), dtype=np.float64)
        for i in range(n):
            phase, amplitude, _ = self._resolve_pattern(global_frame_start + i)
            phase_batch[i] = phase
            amp_batch[i] = amplitude
        return phase_batch, amp_batch

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        if self._precomputed_phases is None and self._dynamic_focus_data is None:
            raise RuntimeError("Renderer not configured. Call configure() first.")
        if not self.device_client.connected:
            self.connection_status.emit("设备未连接")
            return

        self._stop_event.clear()
        self._cycle_index = 1
        self._first_burst = True
        self._thread = threading.Thread(target=self._run_burst_loop, daemon=True)
        self._thread.start()
        self.status_message.emit("渲染已启动")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.status_message.emit("渲染已停止")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_burst_loop(self):
        """Send frames in bursts at ~40 kHz average rate."""
        last_anchor = time.perf_counter()
        global_frame = 0
        frame_count = 0
        fps_start_time = last_anchor
        last_focus_idx: Optional[int] = None

        while not self._stop_event.is_set():
            now = time.perf_counter()
            sleep_time = (last_anchor + cfg.BURST_NOMINAL_INTERVAL) - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            now = time.perf_counter()
            n = max(
                cfg.BURST_MIN_FRAMES,
                int(round((now - last_anchor) * cfg.DEVICE_SAMPLE_RATE)),
            )
            if self._first_burst:
                prime_min = int(
                    cfg.BURST_NOMINAL_FRAMES * cfg.BURST_PRIME_MULTIPLIER
                )
                n = max(n, prime_min)
                self._first_burst = False

            phase_batch, amp_batch = self._build_batch(global_frame, n)
            bram_batch = self.converter.convert_batch(
                phase_batch, amp_batch, self._cycle_index
            )

            ok = self.device_client.send_burst(bram_batch)
            if not ok:
                self.connection_status.emit("发送失败")
                break

            if self.mode == "dynamic":
                _, _, focus_idx = self._resolve_pattern(global_frame + n - 1)
                if focus_idx is not None and focus_idx != last_focus_idx:
                    last_focus_idx = focus_idx
                    num_foci = len(self._dynamic_focus_data)
                    self.current_focus_info.emit(f"焦点 {focus_idx + 1}/{num_foci}")

            self._cycle_index = (self._cycle_index + n) & 0xFFFF
            global_frame += n
            last_anchor += n / cfg.DEVICE_SAMPLE_RATE

            self._maybe_compute_field(phase_batch[-1], amp_batch[-1])

            frame_count += n
            now = time.perf_counter()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.fps_updated.emit(fps)
                frame_count = 0
                fps_start_time = now

    def _maybe_compute_field(self, phase: np.ndarray, amplitude: np.ndarray):
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
