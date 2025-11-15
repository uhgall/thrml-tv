"""
Utilities for Potts-model evaluation on TV interference data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def indices_from_station_ids(station_ids: np.ndarray) -> Dict[int, int]:
    """
    Map station identifiers to contiguous indices.
    """
    if station_ids.ndim != 1:
        raise ValueError("station_ids must be one-dimensional.")
    return {int(station_id): int(idx) for idx, station_id in enumerate(station_ids.tolist())}


def potts_energy(
    colors: jnp.ndarray,
    edges: jnp.ndarray,
    edge_weights: jnp.ndarray,
    edge_penalty: float,
) -> jnp.ndarray:
    """
    Compute the Potts-model energy for a given assignment.
    """
    if edges.size == 0:
        return jnp.array(0.0, dtype=jnp.float32)

    idx = jnp.arange(edges.shape[0], dtype=jnp.int32)
    u = edges[:, 0]
    v = edges[:, 1]
    penalties = edge_weights[idx, colors[u], colors[v]]
    return edge_penalty * jnp.sum(penalties)


def conflict_count(colors: jnp.ndarray, edges: jnp.ndarray, edge_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Count how many edges violate any constraint under the current coloring.
    """
    if edges.size == 0:
        return jnp.array(0, dtype=jnp.int32)

    idx = jnp.arange(edges.shape[0], dtype=jnp.int32)
    u = edges[:, 0]
    v = edges[:, 1]
    penalties = edge_weights[idx, colors[u], colors[v]]
    return jnp.sum(penalties > 0.0, dtype=jnp.int32)


def random_initial_colors(
    key: jax.Array,
    N: int,
    K: int,
    domain_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample an initial configuration that respects per-station channel domains.
    """
    domain_mask = jnp.asarray(domain_mask, dtype=bool)
    if domain_mask.shape != (N, K):
        raise ValueError(f"domain_mask must have shape ({N}, {K}); received {domain_mask.shape}")

    mask_np = np.asarray(domain_mask)
    keys = jax.random.split(key, N + 1)
    draws = []
    for i in range(N):
        allowed = np.flatnonzero(mask_np[i])
        if allowed.size == 0:
            raise ValueError(f"Station index {i} has no allowed channels.")
        choice = jax.random.choice(keys[i], jnp.asarray(allowed, dtype=jnp.int32))
        draws.append(jnp.asarray(choice, dtype=jnp.int32))
    return jnp.stack(draws, axis=0)

