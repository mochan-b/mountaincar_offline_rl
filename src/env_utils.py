"""
env_utils.py
------------
State-discretisation helpers for MountainCar-v0.

MountainCar has a continuous observation space:
  obs[0]  position  ∈ [-1.2,  0.6]
  obs[1]  velocity  ∈ [-0.07, 0.07]

We bin each dimension uniformly and map each episode observation to
an integer tuple (pos_bin, vel_bin).
"""

import numpy as np


def make_bins(n_bins: int, low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """Return a list of bin-edge arrays, one per observation dimension.

    Args:
        n_bins: Number of bins for each dimension.
        low:    Lower bounds of the observation space (shape: (n_dims,)).
        high:   Upper bounds of the observation space (shape: (n_dims,)).

    Returns:
        List of length n_dims, each element is a 1-D array of (n_bins - 1)
        interior cut-points used by np.digitize.
    """
    bins = []
    for lo, hi in zip(low, high):
        # n_bins + 1 edges → n_bins intervals; we keep only the interior edges
        edges = np.linspace(lo, hi, n_bins + 1)
        bins.append(edges[1:-1])  # interior cut-points only
    return bins


def discretise(obs: np.ndarray, bins: list[np.ndarray]) -> tuple:
    """Map a continuous observation to a discrete state tuple.

    Args:
        obs:  Continuous observation vector (shape: (n_dims,)).
        bins: Output of make_bins().

    Returns:
        Tuple of integer bin indices, one per dimension.
        Values are in [0, n_bins - 1].
    """
    return tuple(int(np.digitize(o, b)) for o, b in zip(obs, bins))
