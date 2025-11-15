
"""
Model construction utilities leveraging THRML for the Potts interference graph.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode

from potts_factor import PottsGraphFactor


def build_potts_coloring_model(
    station_ids: np.ndarray,
    channel_values: np.ndarray,
    edges_np: np.ndarray,
    edge_weights_np: np.ndarray,
    lam: float,
    domain_mask_np: np.ndarray | None = None,
    domain_penalty: float = 10.0,
) -> tuple[FactorSamplingProgram, list[CategoricalNode]]:
    """
    Assemble THRML nodes, factors, and a block Gibbs sampler for the Potts model.
    """
    num_stations = int(station_ids.shape[0])
    num_channels = int(channel_values.shape[0])

    nodes = [CategoricalNode() for _ in range(num_stations)]

    edges = jnp.asarray(edges_np, dtype=jnp.int32)
    edge_weights = jnp.asarray(edge_weights_np, dtype=jnp.float32)

    factors = []
    if edges.shape[0] > 0:
        potts_factor = PottsGraphFactor(nodes=nodes, edges=edges, edge_weights=edge_weights, lam=lam)
        factors.append(potts_factor)

    if domain_mask_np is not None:
        domain_mask = np.asarray(domain_mask_np, dtype=bool)
        if domain_mask.shape != (num_stations, num_channels):
            raise ValueError(
                f"domain_mask must have shape ({num_stations}, {num_channels}); got {domain_mask.shape}"
            )
        domain_weights = jnp.asarray(np.where(domain_mask, 0.0, -float(domain_penalty)), dtype=jnp.float32)
        bias_factor = CategoricalEBMFactor([Block(nodes)], domain_weights)
        factors.append(bias_factor)

    free_blocks = [Block([node]) for node in nodes]
    gibbs_spec = BlockGibbsSpec(free_blocks, [])

    samplers = [CategoricalGibbsConditional(num_channels) for _ in free_blocks]

    sampling_program = FactorSamplingProgram(gibbs_spec, samplers, factors, [])

    return sampling_program, nodes


def colors_to_block_state(colors: jnp.ndarray, free_blocks: Sequence[Block]) -> list[jnp.ndarray]:
    """
    Convert a color vector into THRML block state representation.
    """
    colors = jnp.asarray(colors, dtype=jnp.uint8)
    expected = sum(len(block.nodes) for block in free_blocks)
    if colors.shape[0] != expected:
        raise ValueError(f"Expected {expected} colors; received {colors.shape[0]}")

    block_states: list[jnp.ndarray] = []
    offset = 0
    for block in free_blocks:
        block_len = len(block.nodes)
        block_states.append(colors[offset : offset + block_len])
        offset += block_len
    return block_states


def block_state_to_colors(block_state: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """
    Convert a THRML block state (list of arrays) back into a flat color vector.
    """
    if not block_state:
        return jnp.array([], dtype=jnp.int32)
    return jnp.concatenate([jnp.asarray(state, dtype=jnp.int32) for state in block_state], axis=0)

