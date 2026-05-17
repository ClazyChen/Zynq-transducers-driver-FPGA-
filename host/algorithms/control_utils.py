"""
Complicated Control Utilities.
Helper functions for adding complicated control points.
"""

import numpy as np
from typing import List, Tuple


def add_complicated_control_points(
    points: np.ndarray,
    p: np.ndarray,
    image_size: float,
    image_resolution: int,
    z: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Add control points for blocks without foci (target intensity 0).

    Divides the 128x128 field into 4x4 blocks (32x32 pixels each).
    For blocks without any focus, adds the block center as a control point with target intensity 0.

    Args:
        points: Control point coordinates, shape (n, 3), each row (x, y, z) relative to image center
        p: Control point intensities, shape (n,)
        image_size: Image physical size (mm), default 128.0
        image_resolution: Image pixel resolution, default 128
        z: z coordinate (field plane distance from array)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Extended points and p arrays
    """
    points = np.asarray(points)
    p = np.asarray(p)

    if points.ndim == 1:
        points = points.reshape(1, 3)
    num_existing_points = points.shape[0]

    if p.ndim == 0:
        p = np.array([p])
    elif p.ndim == 2:
        p = p.flatten()
    p = p.flatten()

    if len(p) != num_existing_points:
        raise ValueError(f"points and p must have the same length, got {num_existing_points} and {len(p)}")

    num_blocks_per_dim = 4
    block_size_pixels = image_resolution // num_blocks_per_dim

    blocks_with_foci = np.zeros((num_blocks_per_dim, num_blocks_per_dim), dtype=bool)

    for i in range(num_existing_points):
        x_rel, y_rel, _ = points[i]

        x_abs = x_rel + image_size / 2
        y_abs = y_rel + image_size / 2

        x_pixel = (x_abs / image_size) * image_resolution
        y_pixel = (y_abs / image_size) * image_resolution

        block_i = int(x_pixel // block_size_pixels)
        block_j = int(y_pixel // block_size_pixels)

        block_i = max(0, min(num_blocks_per_dim - 1, block_i))
        block_j = max(0, min(num_blocks_per_dim - 1, block_j))

        blocks_with_foci[block_j, block_i] = True

    additional_points = []
    additional_intensities = []

    for block_j in range(num_blocks_per_dim):
        for block_i in range(num_blocks_per_dim):
            if not blocks_with_foci[block_j, block_i]:
                x_pixel_center = block_i * block_size_pixels + block_size_pixels / 2 - 0.5
                y_pixel_center = block_j * block_size_pixels + block_size_pixels / 2 - 0.5

                x_abs = (x_pixel_center / image_resolution) * image_size
                y_abs = (y_pixel_center / image_resolution) * image_size

                x_rel = x_abs - image_size / 2
                y_rel = y_abs - image_size / 2

                additional_points.append([x_rel, y_rel, z])
                additional_intensities.append(0.0)

    if len(additional_points) > 0:
        additional_points = np.array(additional_points)
        additional_intensities = np.array(additional_intensities)

        extended_points = np.vstack([points, additional_points])
        extended_p = np.hstack([p, additional_intensities])
    else:
        extended_points = points
        extended_p = p

    return extended_points, extended_p
