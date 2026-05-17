"""
GS-PAT PyTorch GPU batch implementation.
Same matrix construction and GS iteration as baseline.gs_pat.GS_PAT,
but parallelized on GPU with automatic bucketing for variable control-point counts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .control_utils import add_complicated_control_points


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    """Real float dtype corresponding to a complex dtype."""
    if dtype == torch.complex128:
        return torch.float64
    if getattr(torch, "complex32", None) is not None and dtype == torch.complex32:
        return torch.float16
    return torch.float32


def calculate_weight_matrix_torch(
    f: float,
    width: int,
    height: int,
    d: float,
    points: torch.Tensor,
    c: float = 343000.0,
    *,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Compute weight matrix (PyTorch), same semantics as data.weight_matrix.calculate_weight_matrix.

    Args:
        f: Frequency (Hz)
        width, height: Array grid dimensions
        d: Element spacing (mm)
        points: Target coordinates, shape (..., n, 3), each point (x, y, z)
        c: Speed of sound (mm/s)
        dtype: Output complex type, default complex64

    Returns:
        (..., n, width, height) complex tensor, same device as points.
    """
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (..., n, 3), got {tuple(points.shape)}")

    device = points.device
    real_dtype = _real_dtype_for_complex(dtype)

    pts = points.to(device=device, dtype=real_dtype)
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    a = torch.tensor(d / 2.0, device=device, dtype=real_dtype)
    k = torch.tensor(2.0 * math.pi * f / c, device=device, dtype=real_dtype)

    iw = torch.arange(width, device=device, dtype=real_dtype)
    ih = torch.arange(height, device=device, dtype=real_dtype)
    center_w = torch.tensor((width - 1) / 2.0, device=device, dtype=real_dtype)
    center_h = torch.tensor((height - 1) / 2.0, device=device, dtype=real_dtype)

    dd = torch.tensor(d, device=device, dtype=real_dtype)
    x0 = (iw[:, None] - center_w) * dd
    y0 = (ih[None, :] - center_h) * dd

    dx = x0 - x[..., None, None]
    dy = y0 - y[..., None, None]
    zz = z[..., None, None]
    r = torch.sqrt(dx * dx + dy * dy + zz * zz)
    horizontal_dist = torch.sqrt(dx * dx + dy * dy)
    sin_theta = horizontal_dist / torch.clamp_min(r, torch.finfo(real_dtype).tiny)

    ka_sin_theta = k * a * sin_theta
    eps = torch.tensor(1e-10, device=device, dtype=real_dtype)
    sinc_term = torch.where(
        torch.abs(ka_sin_theta) < eps,
        torch.ones_like(ka_sin_theta),
        torch.sin(ka_sin_theta) / ka_sin_theta,
    )

    kr = k * r
    phase = torch.complex(torch.cos(kr), torch.sin(kr))

    inv_r = 1.0 / torch.clamp_min(r, eps)
    weight = phase * (sinc_term * inv_r)
    return weight.to(dtype=dtype)


def _gs_pat_gpu_forward_fixed_n(
    f: float,
    width: int,
    height: int,
    d: float,
    points_b: torch.Tensor,
    p_b: torch.Tensor,
    c: float,
    max_iter: int,
    *,
    dtype_c: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU GS-PAT forward for fixed control-point count n (internal use).

    Args:
        points_b: (B, n, 3) float32, on target device
        p_b: (B, n) float32, target magnitudes
    """
    if points_b.ndim != 3 or points_b.shape[-1] != 3:
        raise ValueError(f"points_b must be (B, n, 3), got {tuple(points_b.shape)}")
    if p_b.ndim != 2 or p_b.shape != points_b.shape[:2]:
        raise ValueError(f"p_b must be (B, n) matching points_b, got {tuple(p_b.shape)}")

    device = points_b.device
    B, n, _ = points_b.shape
    m = width * height

    wm = calculate_weight_matrix_torch(f, width, height, d, points_b, c, dtype=dtype_c)
    G = wm.reshape(B, n, m)

    G_H = torch.conj_physical(G.transpose(-2, -1))
    s = torch.sum(torch.abs(G_H) ** 2, dim=-2)
    eps_s = torch.tensor(1e-30, device=device, dtype=s.dtype)
    F = G_H / torch.clamp_min(s.unsqueeze(-2), eps_s)
    R = torch.matmul(G, F)

    p_vec = p_b.to(dtype=torch.float32)
    p_cpx = torch.complex(p_vec, torch.zeros_like(p_vec)).to(dtype_c)

    if generator is None:
        phi = torch.rand(B, n, device=device, dtype=torch.float32)
    else:
        phi = torch.rand(B, n, device=device, dtype=torch.float32, generator=generator)

    two_pi_phi = (2.0 * math.pi) * phi
    p0 = p_cpx * torch.complex(torch.cos(two_pi_phi), torch.sin(two_pi_phi)).to(dtype_c)

    for _ in range(max_iter):
        p0_prime = torch.matmul(R, p0.unsqueeze(-1)).squeeze(-1)
        mag = torch.abs(p0_prime)
        mag = torch.clamp_min(mag, torch.tensor(1e-10, device=device, dtype=mag.dtype))
        p0 = p_cpx * (p0_prime / mag.to(dtype_c))

    p0_prime_final = torch.matmul(R, p0.unsqueeze(-1)).squeeze(-1)
    denom = torch.abs(p0_prime_final) ** 2 + 1e-10
    p_Omega_num = torch.complex(p_vec * p_vec, torch.zeros_like(p_vec)).to(dtype_c) * p0_prime_final
    p_Omega = p_Omega_num / denom.to(dtype_c)

    Q_flat = torch.matmul(F, p_Omega.unsqueeze(-1)).squeeze(-1)
    Q = Q_flat.reshape(B, width, height)

    phases = torch.angle(Q).to(torch.float32)
    amplitude = torch.clamp(torch.abs(Q).to(torch.float32), 0.0, 1.0)
    return phases, amplitude


def _normalize_points_p_numpy(
    points: np.ndarray,
    p: np.ndarray,
    complicated_control: bool,
    image_size: float,
    image_resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy preprocessing consistent with GS_PAT (including complicated_control)."""
    points = np.asarray(points)
    p = np.asarray(p)

    if complicated_control:
        if points.ndim == 1:
            z = float(points[2])
        else:
            z = float(points[0, 2])
        points, p = add_complicated_control_points(points, p, image_size, image_resolution, z)

    if points.ndim == 1:
        points = points.reshape(1, 3)

    if p.ndim == 1:
        p = p.reshape(-1, 1)
    elif p.ndim == 0:
        p = p.reshape(1, 1)

    num_points = points.shape[0]
    if points.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {points.shape}")
    if p.shape[0] != num_points:
        raise ValueError(f"p must have {num_points} values, got {p.shape[0]}")

    p = p.reshape(-1).astype(np.float32, copy=False)
    points = points.astype(np.float32, copy=False)
    return points, p


def GS_PAT_GPU_batch(
    f: float,
    width: int,
    height: int,
    d: float,
    points_batch: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
    p_batch: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
    c: float = 343000.0,
    max_iter: int = 100,
    complicated_control: bool = False,
    image_size: float = 128.0,
    image_resolution: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    dtype_c: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch GS-PAT-GPU.

    If complicated_control=True, calls add_complicated_control_points per sample on CPU.
    If control point counts n differ within the batch, samples are bucketed by n and computed separately.

    Args:
        points_batch: (B, n0, 3) array, or a list of length B (each (n0, 3))
        p_batch: (B, n0) or aligned intensities
        device: Defaults to CUDA if available

    Returns:
        phases, amplitude: (B, width, height) float32, on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    samples_pts: List[np.ndarray] = []
    samples_p: List[np.ndarray] = []

    if isinstance(points_batch, torch.Tensor):
        pb = points_batch.detach().cpu().numpy()
        if isinstance(p_batch, torch.Tensor):
            pv = p_batch.detach().cpu().numpy()
        else:
            pv = np.asarray(p_batch)
        if pb.ndim != 3:
            raise ValueError(f"points_batch tensor must be (B, n, 3), got {tuple(pb.shape)}")
        B = pb.shape[0]
        for i in range(B):
            pts_i, p_i = _normalize_points_p_numpy(
                pb[i], pv[i], complicated_control, image_size, image_resolution
            )
            samples_pts.append(pts_i)
            samples_p.append(p_i)
    elif isinstance(points_batch, np.ndarray):
        pb = points_batch
        pv = np.asarray(p_batch)
        if pb.ndim != 3:
            raise ValueError(f"points_batch ndarray must be (B, n, 3), got {pb.shape}")
        B = pb.shape[0]
        for i in range(B):
            pts_i, p_i = _normalize_points_p_numpy(
                pb[i], pv[i], complicated_control, image_size, image_resolution
            )
            samples_pts.append(pts_i)
            samples_p.append(p_i)
    else:
        pts_list = list(points_batch)
        if isinstance(p_batch, (np.ndarray, torch.Tensor)):
            pv_arr = p_batch.detach().cpu().numpy() if isinstance(p_batch, torch.Tensor) else np.asarray(p_batch)
            if len(pts_list) != len(pv_arr):
                raise ValueError("points_batch list length must match p_batch")
            for i, pts_i_raw in enumerate(pts_list):
                pts_i, p_i = _normalize_points_p_numpy(
                    np.asarray(pts_i_raw),
                    np.asarray(pv_arr[i]),
                    complicated_control,
                    image_size,
                    image_resolution,
                )
                samples_pts.append(pts_i)
                samples_p.append(p_i)
        else:
            pv_list = list(p_batch)
            if len(pts_list) != len(pv_list):
                raise ValueError("points_batch and p_batch sequences must have same length")
            for pts_i_raw, pv_i_raw in zip(pts_list, pv_list):
                pts_i, p_i = _normalize_points_p_numpy(
                    np.asarray(pts_i_raw),
                    np.asarray(pv_i_raw),
                    complicated_control,
                    image_size,
                    image_resolution,
                )
                samples_pts.append(pts_i)
                samples_p.append(p_i)

    B = len(samples_pts)
    if B == 0:
        raise ValueError("empty batch")

    out_phases = torch.empty(B, width, height, device=device, dtype=torch.float32)
    out_amp = torch.empty(B, width, height, device=device, dtype=torch.float32)

    buckets: Dict[int, List[int]] = defaultdict(list)
    for i in range(B):
        buckets[int(samples_p[i].shape[0])].append(i)

    for _n_ctrl, idxs in buckets.items():
        pts_stack = np.stack([samples_pts[j] for j in idxs], axis=0)
        p_stack = np.stack([samples_p[j] for j in idxs], axis=0)
        pts_t = torch.from_numpy(pts_stack).to(device=device, dtype=torch.float32)
        p_t = torch.from_numpy(p_stack).to(device=device, dtype=torch.float32)
        ph, am = _gs_pat_gpu_forward_fixed_n(
            f,
            width,
            height,
            d,
            pts_t,
            p_t,
            c,
            max_iter,
            dtype_c=dtype_c,
            generator=generator,
        )
        for k, j in enumerate(idxs):
            out_phases[j] = ph[k]
            out_amp[j] = am[k]

    return out_phases, out_amp


def GS_PAT_GPU(
    f: float,
    width: int,
    height: int,
    d: float,
    points: np.ndarray,
    p: np.ndarray,
    c: float = 343000.0,
    max_iter: int = 100,
    complicated_control: bool = False,
    image_size: float = 128.0,
    image_resolution: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    dtype_c: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sample GS-PAT-GPU, same interface as baseline.gs_pat.GS_PAT (returns NumPy)."""
    pts = np.asarray(points)
    pv = np.asarray(p)

    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError(f"1D points must have length 3, got {pts.shape}")
        pts_b = pts.reshape(1, 1, 3)
    elif pts.ndim == 2:
        if pts.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
        pts_b = pts.reshape(1, pts.shape[0], 3)
    else:
        raise ValueError(f"points must be (n, 3) or (3,), got {pts.shape}")

    if pv.ndim == 0:
        pv_b = np.array([[float(pv)]], dtype=np.float32)
    elif pv.ndim == 1:
        pv_b = pv.reshape(1, -1)
    elif pv.ndim == 2:
        pv_b = pv.reshape(1, -1)
    else:
        raise ValueError(f"p must be 0D/1D/2D, got {pv.shape}")

    phases_t, amp_t = GS_PAT_GPU_batch(
        f=f,
        width=width,
        height=height,
        d=d,
        points_batch=pts_b,
        p_batch=pv_b,
        c=c,
        max_iter=max_iter,
        complicated_control=complicated_control,
        image_size=image_size,
        image_resolution=image_resolution,
        device=device,
        dtype_c=dtype_c,
        generator=generator,
    )
    return phases_t[0].detach().cpu().numpy(), amp_t[0].detach().cpu().numpy()
