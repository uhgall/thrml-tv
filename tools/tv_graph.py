"""
Core data structures for representing FCC interference graphs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Station:
    """
    Representation of a broadcast station and its constraint metadata.
    """

    station_id: int
    interferences: list["Interference"] = field(default_factory=list)
    domain: list[int] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    attrs: dict[str, str] = field(default_factory=dict)

    def neighbor_ids(self) -> set[int]:
        """
        IDs for stations that appear in any interference constraint with this station.
        """
        neighbor_ids: set[int] = set()
        for interference in self.interferences:
            for other in interference.stations:
                neighbor_ids.add(other.station_id)
        return neighbor_ids


@dataclass
class Interference:
    """
    Representation of a channel constraint between a station and its peers.
    """

    constraint_type: str
    subject_channel: int
    other_channel: int
    stations: list[Station] = field(default_factory=list)


@dataclass
class TVGraph:
    """
    Container for stations and channel mappings derived from FCC input files.
    """

    stations: dict[int, Station]
    channel_id_for_channel: dict[int, int]
    channel_for_channel_id: list[int]

    def station(self, station_id: int) -> Station:
        """
        Retrieve a station object by ID.
        """
        if station_id not in self.stations:
            raise KeyError(f"Station {station_id} not present in graph.")
        return self.stations[station_id]

    def ordered_station_items(self) -> List[tuple[int, Station]]:
        """
        Stations sorted by identifier to stabilise downstream array layouts.
        """
        return sorted(self.stations.items(), key=lambda item: item[0])

    def ordered_station_ids(self) -> np.ndarray:
        """
        Station identifiers sorted in ascending order.
        """
        return np.asarray([station_id for station_id, _ in self.ordered_station_items()], dtype=np.int32)

    def channel_values(self) -> np.ndarray:
        """
        Channel values in the order encoded by ``channel_for_channel_id``.
        """
        return np.asarray(self.channel_for_channel_id, dtype=np.int32)

    def build_channel_index(self) -> dict[int, int]:
        """
        Mapping from channel value to its contiguous index.
        """
        return {int(channel): idx for idx, channel in enumerate(self.channel_for_channel_id)}

    def station_index(self) -> dict[int, int]:
        """
        Mapping from station identifier to its contiguous index.
        """
        return {int(station_id): idx for idx, station_id in enumerate(self.ordered_station_ids().tolist())}

    def domain_mask(self) -> np.ndarray:
        """
        Boolean mask encoding station -> allowed channel assignments.
        """
        station_items = self.ordered_station_items()
        station_index = {station_id: idx for idx, (station_id, _) in enumerate(station_items)}
        channel_values = self.channel_values()
        channel_index = {int(channel): idx for idx, channel in enumerate(channel_values.tolist())}
        num_stations = len(station_items)
        num_channels = channel_values.shape[0]
        mask = np.zeros((num_stations, num_channels), dtype=bool)
        for station_id, station in station_items:
            row = station_index[int(station_id)]
            for channel in station.domain:
                if channel not in channel_index:
                    raise KeyError(f"Channel {channel} not present in graph channel set.")
                mask[row, channel_index[int(channel)]] = True
        return mask

    def edge_weight_tensors(self, *, penalty_value: float) -> tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """
        Compute edge list, edge weights tensor, and constraint tags.
        """
        station_items = self.ordered_station_items()
        station_index = {station_id: idx for idx, (station_id, _) in enumerate(station_items)}
        channel_values = self.channel_values()
        channel_index = {int(channel): idx for idx, channel in enumerate(channel_values.tolist())}
        num_channels = channel_values.shape[0]
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
        return edges, edge_weight_tensors, tags_list

    def make_subgraph(
        self,
        seed_station: Station | int,
        channel_count: int,
        station_count: int,
    ) -> "TVGraph":
        """
        Build a connected subgraph using breadth-first traversal from ``seed_station``.
        """
        if channel_count <= 0:
            raise ValueError("channel_count must be positive.")
        if station_count <= 0:
            raise ValueError("station_count must be positive.")
        if channel_count > len(self.channel_for_channel_id):
            raise ValueError(
                f"Requested channel_count={channel_count} exceeds "
                f"{len(self.channel_for_channel_id)} available channels."
            )

        seed_id = seed_station.station_id if isinstance(seed_station, Station) else int(seed_station)
        if seed_id not in self.stations:
            raise KeyError(f"Seed station {seed_id} not found.")

        included_order: list[int] = []
        visited: set[int] = set()
        frontier: deque[int] = deque()

        visited.add(seed_id)
        frontier.append(seed_id)

        while frontier and len(included_order) < station_count:
            current_id = frontier.popleft()
            included_order.append(current_id)
            current_station = self.stations[current_id]
            for interference in current_station.interferences:
                for other in interference.stations:
                    neighbor_id = other.station_id
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        frontier.append(neighbor_id)

        if len(included_order) < station_count:
            raise ValueError(
                f"Only {len(included_order)} stations reachable from seed {seed_id}; "
                f"cannot satisfy station_count={station_count}."
            )

        channel_subset = self.channel_for_channel_id[:channel_count]
        new_channel_id_for_channel = {channel: idx for idx, channel in enumerate(channel_subset)}
        included_ids = set(included_order)

        new_stations: dict[int, Station] = {}
        for station_id in included_order:
            old_station = self.stations[station_id]
            filtered_domain = [channel for channel in old_station.domain if channel in new_channel_id_for_channel]
            new_attrs = dict(old_station.attrs)
            new_station = Station(
                station_id=old_station.station_id,
                domain=filtered_domain,
                lat=old_station.lat,
                lon=old_station.lon,
                attrs=new_attrs,
            )
            new_stations[station_id] = new_station

        for station_id in included_order:
            old_station = self.stations[station_id]
            new_station = new_stations[station_id]
            for interference in old_station.interferences:
                if interference.subject_channel not in new_channel_id_for_channel:
                    continue
                if interference.other_channel not in new_channel_id_for_channel:
                    continue
                partner_objs = [
                    new_stations[partner.station_id]
                    for partner in interference.stations
                    if partner.station_id in included_ids
                ]
                if not partner_objs:
                    continue
                new_station.interferences.append(
                    Interference(
                        constraint_type=interference.constraint_type,
                        subject_channel=interference.subject_channel,
                        other_channel=interference.other_channel,
                        stations=partner_objs,
                    )
                )
            new_station.interferences.sort(key=lambda item: (item.subject_channel, item.other_channel))

        return TVGraph(
            stations=new_stations,
            channel_id_for_channel=new_channel_id_for_channel,
            channel_for_channel_id=list(channel_subset),
        )


__all__ = ["Station", "Interference", "TVGraph"]

