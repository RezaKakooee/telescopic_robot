"""Pure geometry: Fibonacci sphere, sinusoidal path, quaternion math.

No simulator or config dependency — safe to import anywhere.
"""
from __future__ import annotations

import numpy as np

PATH_LENGTH = 6.0
PATH_AMPLITUDE = 0.9
PATH_WAVES = 1.5


def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately-uniform unit vectors on the sphere, shape (n, 3)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def path_xy(
    s: float,
    length: float = PATH_LENGTH,
    amplitude: float = PATH_AMPLITUDE,
    waves: float = PATH_WAVES,
) -> np.ndarray:
    """Sinusoid in the xy plane, parameterised by x = s."""
    f = 2 * np.pi * waves / length
    return np.array([s, amplitude * np.sin(f * s)])


def sample_path(
    n: int = 240,
    length: float = PATH_LENGTH,
    amplitude: float = PATH_AMPLITUDE,
    waves: float = PATH_WAVES,
) -> np.ndarray:
    """Discretise the path into n points along x ∈ [0, length]."""
    s = np.linspace(0, length, n)
    return np.stack([path_xy(si, length, amplitude, waves) for si in s], axis=0)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """wxyz unit quaternion → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
