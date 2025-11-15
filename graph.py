"""
Utilities for parsing TV interference CSVs and evaluating Potts energies.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from tools.tv_graph import TVGraph


def indices_from_station_ids(station_ids: np.ndarray) -> Dict[int, int]:
    """
    Map station identifiers to contiguous indices.
    """
    if station_ids.ndim != 1:
        raise ValueError("station_ids must be one-dimensional.")
    return {int(station_id): int(idx) for idx, station_id in enumerate(station_ids.tolist())}


def load_domain_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse Domain.csv and produce station/channel metadata.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Domain CSV not found: {path}")

    station_channels: Dict[int, set[int]] = {}
    channels_seen: set[int] = set()

    with path_obj.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().upper() != "DOMAIN":
                continue
            if len(row) < 3:
                raise ValueError(f"DOMAIN row must include a station ID and at least one channel: {row}")

            station_id = int(row[1])
            channel_values = [int(x) for x in row[2:] if x]
            if not channel_values:
                raise ValueError(f"Station {station_id} has no allowed channels in {path}")

            station_channels.setdefault(station_id, set()).update(channel_values)
            channels_seen.update(channel_values)

    if not station_channels:
        raise ValueError(f"No DOMAIN rows parsed from {path}")

    station_ids = np.array(sorted(station_channels.keys()), dtype=np.int32)
    channel_values = np.array(sorted(channels_seen), dtype=np.int32)

    station_index = indices_from_station_ids(station_ids)
    channel_index = {int(ch): int(idx) for idx, ch in enumerate(channel_values.tolist())}

    domain_mask = np.zeros((station_ids.shape[0], channel_values.shape[0]), dtype=bool)
    for station_id, channels in station_channels.items():
        i = station_index[station_id]
        for channel in channels:
            domain_mask[i, channel_index[channel]] = True

    return station_ids, channel_values, domain_mask


def load_interference_csv(
    path: str,
    station_ids: np.ndarray,
    channel_values: np.ndarray,
    penalty_value: float = 1.0,
    *,
    return_edge_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    Parse Interference_Paired.csv and build Potts-style edge penalties.

    Parameters
    ----------
    path:
        CSV file containing interference constraints.
    station_ids:
        Array of station IDs to ensure constraints only reference known stations.
    channel_values:
        Array of channel values to map textual channels onto indices.
    penalty_value:
        Base penalty used for any constraint that does not include an explicit weight.
    return_edge_metadata:
        If True, also return a list of constraint tags per edge for downstream visualization.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Interference CSV not found: {path}")

    station_index = indices_from_station_ids(station_ids)
    channel_index = {int(ch): int(idx) for idx, ch in enumerate(channel_values.tolist())}
    n_channels = int(channel_values.shape[0])

    edge_penalties: Dict[Tuple[int, int], np.ndarray] = {}
    edge_constraint_tags: Dict[Tuple[int, int], set[str]] = {}

    with path_obj.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            constraint_type = row[0].strip().upper()
            if constraint_type not in {"CO", "ADJ+1", "ADJ-1"}:
                continue
            if len(row) < 5:
                raise ValueError(f"Malformed interference row (requires >=5 columns): {row}")

            channel_a = int(row[1])
            channel_b = int(row[2])
            station_a = int(row[3])
            partners = [int(value) for value in row[4:] if value]

            if station_a not in station_index:
                raise KeyError(f"Station {station_a} not present in domain data.")
            if channel_a not in channel_index or channel_b not in channel_index:
                raise KeyError(f"Channels ({channel_a}, {channel_b}) missing from domain channel set.")

            idx_a = station_index[station_a]
            chan_idx_a = channel_index[channel_a]
            chan_idx_b = channel_index[channel_b]

            for station_b in partners:
                if station_b not in station_index:
                    raise KeyError(f"Station {station_b} not present in domain data.")

                idx_b = station_index[station_b]
                if idx_a == idx_b:
                    continue

                edge = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                chan_i, chan_j = (chan_idx_a, chan_idx_b) if edge[0] == idx_a else (chan_idx_b, chan_idx_a)

                penalties = edge_penalties.get(edge)
                if penalties is None:
                    penalties = np.zeros((n_channels, n_channels), dtype=np.float32)
                    edge_penalties[edge] = penalties

                penalties[chan_i, chan_j] = max(penalties[chan_i, chan_j], penalty_value)
                penalties[chan_j, chan_i] = max(penalties[chan_j, chan_i], penalty_value)  # TODO: confirm symmetry.

                tags = edge_constraint_tags.setdefault(edge, set())
                tags.add(constraint_type)

    if not edge_penalties:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, n_channels, n_channels), dtype=np.float32)

    edges_sorted = sorted(edge_penalties.keys())
    edges = np.array(edges_sorted, dtype=np.int32)
    edge_weight_tensors = np.stack([edge_penalties[edge] for edge in edges_sorted], axis=0)

    if return_edge_metadata:
        edge_tags = [sorted(edge_constraint_tags.get(edge, set())) for edge in edges_sorted]
        return edges, edge_weight_tensors, edge_tags

    return edges, edge_weight_tensors


@dataclass(frozen=True)
class PottsGraphData:
    """
    Container for Potts-model inputs derived from an in-memory ``TVGraph``.
    """

    station_ids: np.ndarray
    channel_values: np.ndarray
    domain_mask: np.ndarray
    edges: np.ndarray
    edge_weights: np.ndarray
    edge_tags: List[List[str]]


def potts_data_from_tv_graph(
    graph: "TVGraph",
    *,
    penalty_value: float = 1.0,
) -> PottsGraphData:
    """
    Convert a ``TVGraph`` instance into Potts-model arrays compatible with THRML.
    """

    station_items = sorted(graph.stations.items(), key=lambda item: item[0])
    station_ids = np.asarray([int(station_id) for station_id, _ in station_items], dtype=np.int32)
    station_index = {station_id: idx for idx, station_id in enumerate(station_ids.tolist())}

    channel_values = np.asarray(list(graph.channel_for_channel_id), dtype=np.int32)
    channel_index = {int(channel): idx for idx, channel in enumerate(channel_values.tolist())}

    num_stations = station_ids.shape[0]
    num_channels = channel_values.shape[0]

    domain_mask = np.zeros((num_stations, num_channels), dtype=bool)
    for station_id, station in station_items:
        row = station_index[int(station_id)]
        for channel in station.domain:
            if channel not in channel_index:
                raise KeyError(f"Channel {channel} not present in graph channel set.")
            domain_mask[row, channel_index[int(channel)]] = True

    edge_penalties: Dict[Tuple[int, int], np.ndarray] = {}
    edge_tags: Dict[Tuple[int, int], set[str]] = {}

    for station_id, station in station_items:
        idx_a = station_index[int(station_id)]
        for interference in station.interferences:
            subject_channel = int(interference.subject_channel)
            other_channel = int(interference.other_channel)
            if subject_channel not in channel_index or other_channel not in channel_index:
                raise KeyError(
                    f"Constraint channels ({subject_channel}, {other_channel}) missing from graph channel set."
                )

            chan_idx_a = channel_index[subject_channel]
            chan_idx_b = channel_index[other_channel]

            for partner in interference.stations:
                idx_b = station_index.get(int(partner.station_id))
                if idx_b is None or idx_a == idx_b:
                    continue

                edge = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                chan_i, chan_j = (chan_idx_a, chan_idx_b) if edge[0] == idx_a else (chan_idx_b, chan_idx_a)

                penalties = edge_penalties.get(edge)
                if penalties is None:
                    penalties = np.zeros((num_channels, num_channels), dtype=np.float32)
                    edge_penalties[edge] = penalties

                penalties[chan_i, chan_j] = max(penalties[chan_i, chan_j], float(penalty_value))
                penalties[chan_j, chan_i] = max(penalties[chan_j, chan_i], float(penalty_value))

                tags = edge_tags.setdefault(edge, set())
                tags.add(interference.constraint_type)

    if not edge_penalties:
        edges = np.zeros((0, 2), dtype=np.int32)
        edge_weight_tensors = np.zeros((0, num_channels, num_channels), dtype=np.float32)
        tags_list: List[List[str]] = []
    else:
        edges_sorted = sorted(edge_penalties.keys())
        edges = np.asarray(edges_sorted, dtype=np.int32)
        edge_weight_tensors = np.stack([edge_penalties[edge] for edge in edges_sorted], axis=0)
        tags_list = [sorted(edge_tags.get(edge, set())) for edge in edges_sorted]

    return PottsGraphData(
        station_ids=station_ids,
        channel_values=channel_values,
        domain_mask=domain_mask,
        edges=edges,
        edge_weights=edge_weight_tensors,
        edge_tags=tags_list,
    )


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

