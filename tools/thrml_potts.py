"""
Utilities to translate a ``TVGraph`` into THRML sampling primitives.

The helper exposed here follows the workflow outlined in
``raw_data/potts_model_approach.md``:

1. Instantiate a categorical node per station.
2. Build Potts-style pairwise factors that penalise monochromatic edges.
3. Add unary factors that mask out station-channel assignments outside each domain.
4. Assemble a THRML ``FactorizedEBM`` and ``FactorSamplingProgram`` that run block Gibbs.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SquareCategoricalEBMFactor,
)
from thrml.models.ebm import FactorizedEBM
from thrml.pgm import CategoricalNode

from .tv_graph import TVGraph


@dataclass(frozen=True)
class ThrmlPottsArtifacts:
    """
    Container with all THRML structures derived from a ``TVGraph``.

    Attributes
    ----------
    graph:
        The source ``TVGraph`` instance.
    nodes:
        Categorical THRML nodes, ordered by contiguous station index.
    station_ids:
        FCC identifiers aligned with ``nodes``.
    channel_values:
        Physical channel values aligned with contiguous channel indices.
    domain_mask:
        Boolean array (stations × channels) marking admissible assignments.
    edges:
        Tuple of (u, v) station indices used in the Potts factor.
    domain_factor:
        Unary categorical factor enforcing station-specific domains.
    potts_factor:
        Pairwise Potts factor (``None`` if the graph has no edges).
    factors:
        Tuple of factors (domain first, followed by optional Potts factor).
    ebm:
        Factorized energy-based model composed from ``factors``.
    node_block:
        Single block holding all categorical nodes; sampled jointly.
    gibbs_spec:
        Gibbs specification describing free/clamped blocks.
    sampler:
        Categorical Gibbs conditional with ``channel_count`` categories.
    program:
        THRML factor sampling program ready to simulate the model.
    beta:
        Inverse temperature multiplier applied to the Potts penalty.
    penalty:
        Base Potts penalty λ used for monochromatic edges.
    mask_penalty:
        Penalty applied to invalid domain assignments (pre-multiplied by β).
    """

    graph: TVGraph
    nodes: tuple[CategoricalNode, ...]
    station_ids: tuple[int, ...]
    channel_values: tuple[int, ...]
    domain_mask: jnp.ndarray
    edges: tuple[tuple[int, int], ...]
    domain_factor: CategoricalEBMFactor
    potts_factor: SquareCategoricalEBMFactor | None
    factors: tuple[CategoricalEBMFactor | SquareCategoricalEBMFactor, ...]
    ebm: FactorizedEBM
    node_block: Block
    gibbs_spec: BlockGibbsSpec
    sampler: CategoricalGibbsConditional
    program: FactorSamplingProgram
    beta: float
    penalty: float
    mask_penalty: float


def build_thrml_potts_program(
    tv_graph: TVGraph,
    *,
    penalty: float = 1.0,
    beta: float = 1.0,
    mask_penalty: float = jnp.inf,
    dtype: jnp.dtype = jnp.float32,
) -> ThrmlPottsArtifacts:
    """
    Translate a ``TVGraph`` into THRML sampling primitives implementing a Potts model.

    Parameters
    ----------
    tv_graph:
        Source interference graph (already loaded from FCC CSV files).
    penalty:
        Base Potts penalty λ applied when neighbouring stations share the same channel.
    beta:
        Inverse temperature multiplier. The effective penalty in the factor graph is β·λ.
    mask_penalty:
        Energy penalty applied when sampling leaves the per-station channel domain.
        Defaults to ``+∞`` to strictly forbid such assignments.
    dtype:
        Floating dtype for factor weights (defaults to ``float32``).

    Returns
    -------
    ThrmlPottsArtifacts
        Structured bundle with THRML nodes, factors, EBM, and sampling program.
    """

    if penalty < 0:
        raise ValueError("penalty must be non-negative.")
    if beta <= 0:
        raise ValueError("beta must be positive.")

    station_ids = tuple(tv_graph.station_ids_by_index)
    channel_values = tuple(int(ch) for ch in tv_graph.channel_values)

    nodes = tuple(CategoricalNode() for _ in station_ids)
    node_block = Block(nodes)

    domain_mask = jnp.asarray(tv_graph.domain_mask, dtype=bool)
    invalid_penalty = jnp.array(mask_penalty, dtype=dtype) * beta
    domain_weights = jnp.where(domain_mask, jnp.array(0.0, dtype=dtype), -invalid_penalty)
    domain_weights = domain_weights.astype(dtype)
    domain_factor = CategoricalEBMFactor([node_block], domain_weights)

    edges = _collect_edges(tv_graph)
    if edges:
        block_u = Block(tuple(nodes[u] for u, _ in edges))
        block_v = Block(tuple(nodes[v] for _, v in edges))

        diag = -beta * penalty * jnp.eye(tv_graph.channel_count, dtype=dtype)
        pair_weights = jnp.tile(diag[None, :, :], (len(edges), 1, 1))
        potts_factor: SquareCategoricalEBMFactor | None = SquareCategoricalEBMFactor(
            [block_u, block_v],
            pair_weights,
        )
        factors: tuple[CategoricalEBMFactor | SquareCategoricalEBMFactor, ...] = (
            domain_factor,
            potts_factor,
        )
    else:
        potts_factor = None
        factors = (domain_factor,)

    ebm = FactorizedEBM(list(factors))
    gibbs_spec = BlockGibbsSpec([node_block], [])
    sampler = CategoricalGibbsConditional(n_categories=tv_graph.channel_count)
    program = FactorSamplingProgram(gibbs_spec, [sampler], list(factors), [])

    return ThrmlPottsArtifacts(
        graph=tv_graph,
        nodes=nodes,
        station_ids=station_ids,
        channel_values=channel_values,
        domain_mask=domain_mask,
        edges=edges,
        domain_factor=domain_factor,
        potts_factor=potts_factor,
        factors=factors,
        ebm=ebm,
        node_block=node_block,
        gibbs_spec=gibbs_spec,
        sampler=sampler,
        program=program,
        beta=float(beta),
        penalty=float(penalty),
        mask_penalty=float(mask_penalty),
    )


def sample_uniform_initial_state(
    key: jax.Array,
    artifacts: ThrmlPottsArtifacts,
) -> list[jax.Array]:
    """
    Draw an initial colouring uniformly from each station's admissible channel set.

    Parameters
    ----------
    key:
        JAX PRNG key.
    artifacts:
        Bundle returned by :func:`build_thrml_potts_program`.

    Returns
    -------
    list[jax.Array]
        Block-organised state suitable for passing to THRML sampling routines.
    """

    domain_mask = artifacts.domain_mask
    n_stations, n_channels = domain_mask.shape
    channel_indices = jnp.arange(n_channels, dtype=jnp.int32)

    def _draw_one(k, mask_row):
        probs = jnp.where(mask_row, 1.0, 0.0)
        probs = probs / probs.sum()
        choice = jax.random.choice(k, channel_indices, p=probs)
        return choice.astype(jnp.uint8)

    keys = jax.random.split(key, n_stations)
    draws = jax.vmap(_draw_one)(keys, domain_mask)
    return [draws]


def _collect_edges(tv_graph: TVGraph) -> tuple[tuple[int, int], ...]:
    """
    Collapse FCC interference records into undirected station edges.
    """

    pairs: set[tuple[int, int]] = set()
    for station in tv_graph.stations_by_id.values():
        if station.station_index is None:
            continue
        u = station.station_index
        for interference in station.interferences:
            for v in interference.station_indices:
                if u == v:
                    continue
                pairs.add((min(u, v), max(u, v)))
    return tuple(sorted(pairs))


