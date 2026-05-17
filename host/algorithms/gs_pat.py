"""
GS (Gerchberg-Saxton) algorithm implementation.
Computes transducer phase matrix from focus positions.
"""

import numpy as np
from .weight_matrix import calculate_weight_matrix
from .control_utils import add_complicated_control_points


def GS_PAT(f, width, height, d, points, p, c=343000, max_iter=100,
           complicated_control=False, image_size=128.0, image_resolution=128):
    """Calculate transducer phases using GS (Gerchberg-Saxton) algorithm for pattern generation

    Args:
        f: Frequency (Hz)
        width: Array width (number of elements)
        height: Array height (number of elements)
        d: Element spacing (mm)
        points: Target position coordinates, shape (n, 3)
                Each row is (x, y, z) with array center as origin, array on xoy plane
        p: Focus intensities for each target point, shape (n, 1) or (n,)
        c: Speed of sound (default 343000 mm/s)
        max_iter: Maximum number of iterations (default 100)
        complicated_control: If True, add control points at centers of blocks without foci (default False)
        image_size: Image physical size in mm, used for complicated_control (default 128.0)
        image_resolution: Image pixel resolution, used for complicated_control (default 128)

    Returns:
        phases: width x height numpy array containing the phase (in radians) for each transducer
        amplitude: width x height numpy array containing the amplitude (clipped to [0, 1]) for each transducer
    """
    points = np.asarray(points)
    p = np.asarray(p)

    if complicated_control:
        if points.ndim == 1:
            z = points[2]
        else:
            z = points[0, 2]
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
        raise ValueError(f"p must have {num_points} values (one for each point), got {p.shape[0]}")

    weight_matrix = calculate_weight_matrix(f, width, height, d, points, c)

    G = weight_matrix.reshape(num_points, width * height)

    G_H = np.conj(G.T)

    s = np.sum(np.abs(G_H)**2, axis=0)

    F = G_H / s[np.newaxis, :]

    R = G @ F

    p = p.flatten()

    phi = np.random.rand(num_points)
    p0 = p * np.exp(2 * np.pi * 1j * phi)

    for iteration in range(max_iter):
        p0_prime = R @ p0

        p1 = p * p0_prime / (np.abs(p0_prime) + 1e-10)

        p0 = p1

    p0_prime_final = R @ p0

    p0_prime_sq = np.abs(p0_prime_final)**2 + 1e-10
    p_Omega = (p**2 * p0_prime_final / p0_prime_sq)

    p_Omega_reshaped = p_Omega.reshape(num_points, 1)
    Q_flat = F @ p_Omega_reshaped
    Q_flat = Q_flat.flatten()

    Q = Q_flat.reshape(width, height)

    phases = np.angle(Q)
    amplitude = np.clip(np.abs(Q), 0, 1)

    return phases, amplitude
