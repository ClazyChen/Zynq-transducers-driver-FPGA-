"""
Algorithm engines: unified interface for U-Net and GS-PAT inference.
"""

import math
import warnings
from typing import List, Tuple, Optional

import numpy as np
import torch

from .unet_model import ProposedUNet
from .target_generator import TargetGenerator
from .gs_pat import GS_PAT
from .gs_pat_gpu import GS_PAT_GPU_batch


class UNetEngine:
    """U-Net(32b) inference engine."""

    def __init__(
        self,
        checkpoint_path: str,
        base_channels: int = 32,
        device: Optional[str] = None,
        use_compile: bool = True,
        use_amp: bool = True,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.base_channels = base_channels
        self.use_amp = use_amp and self.device.type == "cuda"

        self.model = ProposedUNet(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            use_negative_mask=False,
            output_size=(8, 8),
            phase_only=False,
        )

        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        if use_compile and self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                warnings.warn(f"torch.compile failed: {e}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self.target_generator = TargetGenerator(
            image_size=128.0,
            image_resolution=128,
            center_region_size=80.0,
            focus_radius=4.68,
            min_foci_distance=8.5,
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with automatic base_channels detection."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Try current base_channels first
        try:
            self.model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass

        # Auto-detect base_channels from first conv weight shape
        first_key = "enc1.conv.0.weight"
        if first_key in state_dict:
            actual_channels = state_dict[first_key].shape[0]
            if actual_channels != self.base_channels:
                warnings.warn(
                    f"Checkpoint base_channels mismatch: expected {self.base_channels}, "
                    f"detected {actual_channels}. Rebuilding model."
                )
                self.base_channels = actual_channels
                self.model = ProposedUNet(
                    in_channels=1,
                    out_channels=1,
                    base_channels=actual_channels,
                    use_negative_mask=False,
                    output_size=(8, 8),
                    phase_only=False,
                )
                self.model.load_state_dict(state_dict, strict=True)
                return

        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")

    def infer(
        self,
        foci_positions_batch: List[List[Tuple[float, float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of focus positions.

        Args:
            foci_positions_batch: List of focus position lists, each inner list is [(x, y), ...]

        Returns:
            phase: (B, 8, 8) numpy array, radians
            amplitude: (B, 8, 8) numpy array, [0, 1]
        """
        positive_masks = []
        for foci in foci_positions_batch:
            pos_mask, _ = self.target_generator.generate_masks(foci)
            positive_masks.append(pos_mask)

        input_tensor = torch.from_numpy(np.stack(positive_masks)).float().unsqueeze(1).to(self.device)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    phase, amplitude = self.model(input_tensor)
            else:
                phase, amplitude = self.model(input_tensor)

        phase = phase.squeeze(1).cpu().numpy()
        amplitude = amplitude.squeeze(1).cpu().numpy()

        return phase, amplitude


class GSPATEngine:
    """GS-PAT inference engine (GPU batch preferred, CPU fallback)."""

    def __init__(
        self,
        f: float = 40e3,
        width: int = 8,
        height: int = 8,
        d: float = 10.0,
        z: float = 50.0,
        c: float = 343000.0,
        max_iter: int = 100,
        device: Optional[str] = None,
    ):
        self.f = f
        self.width = width
        self.height = height
        self.d = d
        self.z = z
        self.c = c
        self.max_iter = max_iter

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_gpu = self.device.type == "cuda"

    def _abs_to_rel(self, foci_positions: List[Tuple[float, float]]) -> np.ndarray:
        """Convert absolute coordinates to relative coordinates (origin at image center) + z."""
        points = []
        for x_abs, y_abs in foci_positions:
            x_rel = x_abs - 128.0 / 2
            y_rel = y_abs - 128.0 / 2
            points.append([x_rel, y_rel, self.z])
        return np.array(points, dtype=np.float32)

    def infer(
        self,
        foci_positions_batch: List[List[Tuple[float, float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run GS-PAT on a batch of focus positions.

        Args:
            foci_positions_batch: List of focus position lists, each inner list is [(x, y), ...]

        Returns:
            phase: (B, 8, 8) numpy array, radians
            amplitude: (B, 8, 8) numpy array, [0, 1]
        """
        points_list = []
        p_list = []

        for foci in foci_positions_batch:
            points = self._abs_to_rel(foci)
            p = np.ones(len(foci), dtype=np.float32)
            points_list.append(points)
            p_list.append(p)

        if self.use_gpu:
            phases_t, amp_t = GS_PAT_GPU_batch(
                f=self.f,
                width=self.width,
                height=self.height,
                d=self.d,
                points_batch=points_list,
                p_batch=p_list,
                c=self.c,
                max_iter=self.max_iter,
                complicated_control=False,
                device=self.device,
            )
            phases = phases_t.cpu().numpy()
            amplitude = amp_t.cpu().numpy()
        else:
            phases_list = []
            amp_list = []
            for points, p in zip(points_list, p_list):
                ph, am = GS_PAT(
                    f=self.f,
                    width=self.width,
                    height=self.height,
                    d=self.d,
                    points=points,
                    p=p,
                    c=self.c,
                    max_iter=self.max_iter,
                    complicated_control=False,
                )
                phases_list.append(ph)
                amp_list.append(am)
            phases = np.stack(phases_list)
            amplitude = np.stack(amp_list)

        return phases, amplitude
