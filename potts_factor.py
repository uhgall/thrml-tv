"""
Custom THRML factor for Potts-style pairwise categorical penalties.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.models.discrete_ebm import CategoricalEBMFactor
from thrml.pgm import CategoricalNode


class PottsGraphFactor(CategoricalEBMFactor):
    """
    Wrap a batch of pairwise Potts interactions for categorical nodes.
    """

    nodes: tuple[CategoricalNode, ...]
    edges: jnp.ndarray
    base_edge_weights: jnp.ndarray
    lam: float

    def __init__(
        self,
        nodes: Sequence[CategoricalNode],
        edges: jnp.ndarray,
        edge_weights: jnp.ndarray,
        lam: float,
    ):
        if not nodes:
            raise ValueError("nodes must contain at least one CategoricalNode.")

        edges_arr = np.asarray(edges, dtype=np.int32)
        if edges_arr.ndim != 2 or edges_arr.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2).")
        if edges_arr.shape[0] == 0:
            raise ValueError("PottsGraphFactor requires at least one edge.")

        weights_arr = np.asarray(edge_weights, dtype=np.float32)
        if weights_arr.ndim != 3:
            raise ValueError("edge_weights must have shape (E, K, K).")
        if weights_arr.shape[0] != edges_arr.shape[0]:
            raise ValueError("edge_weights leading dimension must match number of edges.")

        node_list = tuple(nodes)
        edges_jnp = jnp.asarray(edges_arr, dtype=jnp.int32)
        weights_jnp = jnp.asarray(weights_arr, dtype=jnp.float32)
        lam_value = float(lam)

        self.nodes = node_list
        self.edges = edges_jnp
        self.base_edge_weights = weights_jnp
        self.lam = lam_value

        scaled_weights = lam_value * weights_jnp

        left_nodes = [node_list[int(idx)] for idx in edges_arr[:, 0].tolist()]
        right_nodes = [node_list[int(idx)] for idx in edges_arr[:, 1].tolist()]

        node_groups = [Block(left_nodes), Block(right_nodes)]

        super().__init__(node_groups=node_groups, weights=scaled_weights)

