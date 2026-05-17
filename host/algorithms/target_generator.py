"""
Target field generation module.
Generates positive and negative excitation region masks from focus positions.
"""

import numpy as np
from typing import Tuple, List, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def generate_foci_positions(
    num_foci: int,
    center_region_size: float = 80.0,
    min_distance: float = 10.0,
    image_size: float = 128.0,
    max_attempts: int = 1000,
    rng: Optional[random.Random] = None
) -> List[Tuple[float, float]]:
    """Generate random focus positions within a central region with minimum distance constraints.

    Args:
        num_foci: Number of foci (1-3)
        center_region_size: Central region size (mm), default 80mm
        min_distance: Minimum distance between foci (mm), default 10mm
        image_size: Total image size (mm), default 128mm
        max_attempts: Max attempts to avoid infinite loop
        rng: Random number generator, uses global random if None

    Returns:
        List of focus positions, each (x, y) in mm, absolute coordinates
    """
    if num_foci < 1 or num_foci > 3:
        raise ValueError("Number of foci must be between 1 and 3")

    if rng is None:
        rng = random

    region_min_physical = (image_size - center_region_size) / 2
    region_max_physical = (image_size + center_region_size) / 2

    foci_positions = []

    for attempt in range(max_attempts):
        x = rng.uniform(region_min_physical, region_max_physical)
        y = rng.uniform(region_min_physical, region_max_physical)
        new_position = (x, y)

        valid = True
        for existing_pos in foci_positions:
            distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
            if distance < min_distance:
                valid = False
                break

        if valid:
            foci_positions.append(new_position)
            if len(foci_positions) == num_foci:
                return foci_positions

    if len(foci_positions) > 0:
        return foci_positions
    else:
        raise RuntimeError(f"Could not generate {num_foci} foci within {max_attempts} attempts")


class TargetGenerator:
    """Target field generator: produces positive and negative excitation region masks."""

    def __init__(
        self,
        image_size: float = 128.0,
        image_resolution: int = 128,
        center_region_size: float = 80.0,
        focus_radius: float = 4.68,
        min_foci_distance: float = 8.5,
        num_workers: Optional[int] = None
    ):
        self.image_size = image_size
        self.image_resolution = image_resolution
        self.center_region_size = center_region_size
        self.focus_radius = focus_radius
        self.min_foci_distance = min_foci_distance
        self.num_workers = num_workers

        x = np.linspace(-image_size/2, image_size/2, image_resolution)
        y = np.linspace(-image_size/2, image_size/2, image_resolution)
        self.X, self.Y = np.meshgrid(x, y)

        self._local = threading.local()

    def generate_positive_mask(
        self,
        foci_positions: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Generate positive excitation region mask.

        Args:
            foci_positions: List of focus positions, each (x, y) in mm, absolute coordinates

        Returns:
            2D array, positive mask (1 inside focus radius, 0 outside)
        """
        positive_mask = np.zeros((self.image_resolution, self.image_resolution))

        for x, y in foci_positions:
            center_x = x - self.image_size / 2
            center_y = y - self.image_size / 2

            distance = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)

            positive_mask = np.maximum(positive_mask, (distance <= self.focus_radius).astype(float))

        return positive_mask

    def generate_negative_mask(
        self,
        foci_positions: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Generate negative excitation region mask.

        Args:
            foci_positions: List of focus positions, each (x, y) in mm, absolute coordinates

        Returns:
            2D array, negative mask (0 inside focus radius, Gaussian decay to 1 outside)
        """
        if len(foci_positions) == 0:
            return np.ones((self.image_resolution, self.image_resolution))

        min_distance = np.full((self.image_resolution, self.image_resolution), np.inf)

        for x, y in foci_positions:
            center_x = x - self.image_size / 2
            center_y = y - self.image_size / 2

            distance = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)

            min_distance = np.minimum(min_distance, distance)

        excess_distance = np.maximum(0, min_distance - self.focus_radius)

        sigma = self.focus_radius

        negative_mask = 1.0 - np.exp(-(excess_distance**2) / (2 * sigma**2))

        negative_mask[min_distance <= self.focus_radius] = 0.0

        return negative_mask

    def generate_masks(
        self,
        foci_positions: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate positive and negative masks from focus positions.

        Args:
            foci_positions: List of focus positions, each (x, y) in mm, absolute coordinates

        Returns:
            (positive_mask, negative_mask)
        """
        positive_mask = self.generate_positive_mask(foci_positions)
        negative_mask = self.generate_negative_mask(foci_positions)

        return positive_mask, negative_mask

    def generate_sample(
        self,
        num_foci: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], int]:
        """Generate a single training sample.

        Args:
            num_foci: Number of foci, random 1-3 if None

        Returns:
            (positive_mask, negative_mask, foci_positions, num_foci)
        """
        if num_foci is None:
            num_foci = random.randint(1, 3)

        foci_positions = generate_foci_positions(
            num_foci=num_foci,
            center_region_size=self.center_region_size,
            min_distance=self.min_foci_distance,
            image_size=self.image_size
        )

        positive_mask, negative_mask = self.generate_masks(foci_positions)

        return positive_mask, negative_mask, foci_positions, num_foci

    def _generate_sample_worker(self, num_foci: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], int]:
        """Worker helper for multithreaded generation."""
        if not hasattr(self._local, 'rng'):
            self._local.rng = random.Random()
            import time
            self._local.rng.seed(int(time.time() * 1000000) + threading.get_ident())

        if num_foci is None:
            num_foci = self._local.rng.randint(1, 3)

        foci_positions = generate_foci_positions(
            num_foci=num_foci,
            center_region_size=self.center_region_size,
            min_distance=self.min_foci_distance,
            image_size=self.image_size,
            rng=self._local.rng
        )

        positive_mask, negative_mask = self.generate_masks(foci_positions)

        return positive_mask, negative_mask, foci_positions, num_foci

    def generate_batch(
        self,
        batch_size: int,
        use_multithreading: bool = True,
        num_foci: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[float, float]]], List[int]]:
        """Generate a batch of training data.

        Args:
            batch_size: Batch size
            use_multithreading: Whether to use multithreading
            num_foci: Fixed number of foci, random 1-3 if None

        Returns:
            (positive_masks, negative_masks, foci_positions_list, num_foci_list)
        """
        if use_multithreading and batch_size > 1:
            positive_masks = []
            negative_masks = []
            foci_positions_list = []
            num_foci_list = []

            if self.num_workers is None:
                import os
                num_workers = min(batch_size, os.cpu_count() or 4)
            else:
                num_workers = min(batch_size, self.num_workers)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._generate_sample_worker, num_foci) for _ in range(batch_size)]

                for future in as_completed(futures):
                    positive_mask, negative_mask, foci_positions, num_foci_result = future.result()
                    positive_masks.append(positive_mask)
                    negative_masks.append(negative_mask)
                    foci_positions_list.append(foci_positions)
                    num_foci_list.append(num_foci_result)

            positive_masks = np.array(positive_masks)
            negative_masks = np.array(negative_masks)
        else:
            positive_masks = []
            negative_masks = []
            foci_positions_list = []
            num_foci_list = []

            for _ in range(batch_size):
                positive_mask, negative_mask, foci_positions, num_foci_result = self.generate_sample(num_foci=num_foci)
                positive_masks.append(positive_mask)
                negative_masks.append(negative_mask)
                foci_positions_list.append(foci_positions)
                num_foci_list.append(num_foci_result)

            positive_masks = np.array(positive_masks)
            negative_masks = np.array(negative_masks)

        return positive_masks, negative_masks, foci_positions_list, num_foci_list
