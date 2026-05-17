"""
Weight matrix computation module.
Computes the acoustic propagation weight from each transducer element to each field point.
"""

import numpy as np


def calculate_weight_matrix(f, width, height, d, points, c=343000):
    """Calculate weight matrix: acoustic propagation from each transducer element to each field point.

    Args:
        f: Frequency (Hz)
        width: Array width (number of elements)
        height: Array height (number of elements)
        d: Element spacing (mm)
        points: Target position coordinates, shape (n, 3) or (3,)
                Each row is (x, y, z), origin at array center, array on xoy plane
        c: Speed of sound (default 343000 mm/s)

    Returns:
        weight_matrix: n x width x height complex numpy array (or width x height for single point)
    """
    points = np.asarray(points)

    if points.ndim == 1:
        points = points.reshape(1, 3)

    num_points = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    a = d / 2
    k = 2 * np.pi * f / c

    indices_w = np.arange(width)
    indices_h = np.arange(height)

    center_w = (width - 1) / 2
    center_h = (height - 1) / 2

    x0 = (indices_w[:, np.newaxis] - center_w) * d
    y0 = (indices_h[np.newaxis, :] - center_h) * d

    x = x[:, np.newaxis, np.newaxis]
    y = y[:, np.newaxis, np.newaxis]
    z = z[:, np.newaxis, np.newaxis]

    dx = x0[np.newaxis, :, :] - x
    dy = y0[np.newaxis, :, :] - y

    r = np.sqrt(dx**2 + dy**2 + z**2)

    horizontal_dist = np.sqrt(dx**2 + dy**2)
    sin_theta = horizontal_dist / r

    ka_sin_theta = k * a * sin_theta

    mask = np.abs(ka_sin_theta) < 1e-10
    sinc_term = np.zeros_like(ka_sin_theta, dtype=np.complex128)

    sinc_term[~mask] = np.sin(ka_sin_theta[~mask]) / ka_sin_theta[~mask]
    sinc_term[mask] = 1.0

    directional_term = sinc_term
    phase = np.exp(1j * k * r)

    weight_matrix = directional_term * phase / r

    return weight_matrix
