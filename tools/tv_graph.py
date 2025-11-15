"""
Core data structures for representing FCC interference graphs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


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
class tv_graph:
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

    def make_subgraph(
        self,
        seed_station: Station | int,
        channel_count: int,
        station_count: int,
    ) -> "tv_graph":
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

        return tv_graph(
            stations=new_stations,
            channel_id_for_channel=new_channel_id_for_channel,
            channel_for_channel_id=list(channel_subset),
        )


__all__ = ["Station", "Interference", "tv_graph"]

