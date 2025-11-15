"""
Core data structures for representing FCC interference graphs.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Station:
    """
    Representation of a broadcast station and its constraint metadata.
    """

    station_id: int
    station_index: int | None = None
    interferences: list["Interference"] = field(default_factory=list)
    domain: list[int] = field(default_factory=list)
    _domain_indices: list[int] = field(default_factory=list, repr=False)
    lat: float | None = None
    lon: float | None = None
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def domain_indices(self) -> list[int]:
        """
        Allowed channel indices for this station in contiguous index space.
        """
        return list(self._domain_indices)

    def set_domain_indices(self, indices: Iterable[int]) -> None:
        """
        Persist channel indices corresponding to ``domain`` values.
        """
        self._domain_indices = [int(value) for value in indices]

    def neighbor_indices(self) -> set[int]:
        """
        Contiguous indices for stations appearing with this station in any interference.
        """
        neighbor_indices: set[int] = set()
        for interference in self.interferences:
            neighbor_indices.update(interference.station_indices)
        return neighbor_indices


@dataclass
class Interference:
    """
    Representation of a channel constraint between a station and its peers.
    """

    constraint_type: str
    subject_channel: int
    other_channel: int
    subject_channel_index: int
    other_channel_index: int
    station_indices: list[int] = field(default_factory=list)


class TVGraph:
    """
    Container for stations, channel mappings, and interference data loaded from FCC inputs.
    """

    def __init__(
        self,
        dataset: str | Path,
    ) -> None:
        root = self._resolve_input_directory(dataset)
        self.dataset_root = root

        station_domains, all_channels = self._load_domain(root / "Domain.csv")

        channel_values = sorted(all_channels)
        self.channel_for_channel_id = [int(c) for c in channel_values]
        self.channel_id_for_channel = {c: i for i, c in enumerate(self.channel_for_channel_id)}
        self._channel_values = np.asarray(self.channel_for_channel_id, dtype=np.int32)

        params_header, params_by_station = self._load_parameters(root / "parameters.csv")
        self.parameters_header = list(params_header)

        # Create Station objects with associated parameters.
        stations: dict[int, Station] = {}
        for station_id, domain_values in station_domains.items():
            station_params = params_by_station.get(station_id, {})
            station = Station(
                station_id=station_id,
                domain=domain_values,
                lat=self._parse_float(station_params.get("Lat")),
                lon=self._parse_float(station_params.get("Lon")),
                attrs=dict(station_params),
            )
            station.set_domain_indices(self.channel_id_for_channel[ch] for ch in domain_values)
            stations[station_id] = station

        self.stations_by_id = {int(sid): st for sid, st in stations.items()}
        self._station_ids_by_index = sorted(stations.keys())
        self._station_index_for_station_id = {sid: idx for idx, sid in enumerate(self._station_ids_by_index)}

        # Assign indices and domain indices for each station
        for sid, st in stations.items():
            st.station_index = self._station_index_for_station_id[sid]

        self._domain_mask = np.zeros((self.station_count, self.channel_count), dtype=bool)
        for st in stations.values():
            if st.station_index is None:
                raise ValueError(f"Station {st.station_id} missing contiguous index assignment.")
            self._domain_mask[st.station_index, st.domain_indices] = True

        self._load_interferences(root / "Interference_Paired.csv")

    def save_sub_graph(
        self,
        seed_station: int = 87,
        station_limit: int = 5,
        channel_limit: int = 7,
    ) -> Path:
        """
        Persist a BFS-limited subset of the current graph to ``input/``.

        Parameters
        ----------
        seed_station:
            FCC station identifier to start the breadth-first traversal.
        station_limit:
            Maximum number of stations to include in the exported dataset.
        channel_limit:
            Maximum number of unique channels to retain across the subset.

        Returns
        -------
        Path
            Directory path created under ``input/`` containing ``Domain.csv``,
            ``parameters.csv``, and ``Interference_Paired.csv`` for the subset.
        """
        if station_limit <= 0:
            raise ValueError("station_limit must be positive.")
        if channel_limit <= 0:
            raise ValueError("channel_limit must be positive.")

        seed_station = int(seed_station)
        if seed_station not in self.stations_by_id:
            raise KeyError(f"Seed station {seed_station} not present in graph.")

        visited: list[int] = []
        seen: set[int] = set()
        queue: deque[int] = deque([seed_station])

        while queue and len(visited) < station_limit:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            visited.append(current)

            station = self.station(current)
            neighbors = {
                self.station_id_for_index(idx)
                for interference in station.interferences
                for idx in interference.station_indices
            }
            for neighbor in sorted(neighbors):
                if neighbor not in seen and neighbor not in queue:
                    queue.append(neighbor)

        if len(visited) < station_limit:
            raise ValueError(
                f"Only discovered {len(visited)} station(s) reachable from seed {seed_station}, "
                f"fewer than requested station_limit={station_limit}."
            )

        station_ids = visited
        station_set = set(station_ids)

        # Determine channel subset ensuring each station retains at least one channel.
        selected_channel_set: set[int] = set()

        for station_id in station_ids:
            station = self.station(station_id)
            if not station.domain:
                raise ValueError(f"Station {station_id} has an empty domain.")

            if any(channel in selected_channel_set for channel in station.domain):
                continue

            for channel in station.domain:
                if channel not in selected_channel_set:
                    selected_channel_set.add(channel)
                    break
            else:
                raise ValueError(f"Unable to allocate a channel for station {station_id}.")

        if len(selected_channel_set) > channel_limit:
            raise ValueError(
                f"Channel limit {channel_limit} is insufficient; "
                f"requires at least {len(selected_channel_set)} unique channels."
            )

        if len(selected_channel_set) < channel_limit:
            all_candidate_channels = sorted(
                {
                    channel
                    for station_id in station_ids
                    for channel in self.station(station_id).domain
                }
            )
            for channel in all_candidate_channels:
                if channel in selected_channel_set:
                    continue
                selected_channel_set.add(channel)
                if len(selected_channel_set) >= channel_limit:
                    break

        if len(selected_channel_set) < channel_limit:
            raise ValueError(
                f"Only {len(selected_channel_set)} unique channels available across the selected stations; "
                f"cannot satisfy channel_limit={channel_limit}."
            )

        if not selected_channel_set:
            raise ValueError("Unable to identify any channels for the sub-graph.")

        output_dir = self.dataset_root.parent / f"{self.dataset_root.name}-seed{seed_station}-st{station_limit}-ch{channel_limit}"
        output_dir.mkdir(parents=True, exist_ok=False)

        domain_path = output_dir / "Domain.csv"
        with domain_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for station_id in station_ids:
                station = self.station(station_id)
                domain_subset = [channel for channel in station.domain if channel in selected_channel_set]
                if not domain_subset:
                    raise ValueError(
                        f"Station {station_id} does not retain any channels within the selected subset."
                    )
                writer.writerow(["DOMAIN", station_id, *domain_subset])

        params_path = output_dir / "parameters.csv"
        with params_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.parameters_header)
            for station_id in station_ids:
                station = self.station(station_id)
                row = [station.attrs.get(column, "") for column in self.parameters_header]
                writer.writerow(row)

        interference_path = output_dir / "Interference_Paired.csv"
        with interference_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for station_id in station_ids:
                station = self.station(station_id)
                for interference in station.interferences:
                    if interference.subject_channel not in selected_channel_set:
                        continue
                    if interference.other_channel not in selected_channel_set:
                        continue

                    partners: list[int] = []
                    for idx in interference.station_indices:
                        partner_id = self.station_id_for_index(idx)
                        if partner_id in station_set:
                            partners.append(partner_id)
                    if not partners:
                        continue
                    partners.sort()

                    writer.writerow(
                        [
                            interference.constraint_type,
                            interference.subject_channel,
                            interference.other_channel,
                            station_id,
                            *partners,
                        ]
                    )

        return output_dir

    def station(self, station_id: int) -> Station:
        """
        Retrieve a station object by FCC identifier.
        """
        try:
            return self.stations_by_id[int(station_id)]
        except KeyError as exc:
            raise KeyError(f"Station {station_id} not present in graph.") from exc

    def station_by_index(self, station_index: int) -> Station:
        """
        Retrieve a station by contiguous station index.
        """
        station_id = self.station_id_for_index(station_index)
        return self.station(station_id)

    @property
    def station_ids_by_index(self) -> list[int]:
        """
        Station identifiers ordered by contiguous station index.
        """
        return list(self._station_ids_by_index)

    @property
    def station_indices(self) -> np.ndarray:
        """
        Contiguous station indices as a NumPy array.
        """
        return np.arange(self.station_count, dtype=np.int32)

    def station_index_for_id(self, station_id: int) -> int:
        """
        Look up the contiguous index for a given station identifier.
        """
        try:
            return self._station_index_for_station_id[int(station_id)]
        except KeyError as exc:
            raise KeyError(f"Station {station_id} not present in graph.") from exc

    def station_id_for_index(self, station_index: int) -> int:
        """
        Look up the FCC identifier for a given station index.
        """
        try:
            return self._station_ids_by_index[int(station_index)]
        except IndexError as exc:
            raise IndexError(f"Station index {station_index} out of range.") from exc

    @property
    def station_count(self) -> int:
        """
        Total number of stations in the graph.
        """
        return len(self._station_ids_by_index)

    @property
    def channel_indices(self) -> np.ndarray:
        """
        Contiguous channel indices as a NumPy array.
        """
        return np.arange(self.channel_count, dtype=np.int32)

    def channel_index_for_channel(self, channel: int) -> int:
        """
        Look up the contiguous index for a channel value.
        """
        try:
            return self.channel_id_for_channel[int(channel)]
        except KeyError as exc:
            raise KeyError(f"Channel {channel} not present in graph channel set.") from exc

    def channel_for_index(self, channel_index: int) -> int:
        """
        Look up the channel value for a contiguous channel index.
        """
        try:
            return self.channel_for_channel_id[int(channel_index)]
        except IndexError as exc:
            raise IndexError(f"Channel index {channel_index} out of range.") from exc

    @property
    def channel_values(self) -> np.ndarray:
        """
        Channel values ordered by channel index.
        """
        return self._channel_values

    @property
    def channel_count(self) -> int:
        """
        Total number of channel values represented in the graph.
        """
        return len(self.channel_for_channel_id)

    @property
    def domain_mask(self) -> np.ndarray:
        """
        Boolean mask encoding station index -> allowed channel indices.
        """
        return self._domain_mask


    def _load_interferences(self, path: Path) -> None:

        if not path.exists():
            raise FileNotFoundError(f"Interference CSV not found: {path}")

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 5:
                    continue
                constraint_type = row[0].strip().upper()
                if not constraint_type:
                    raise ValueError(f"Invalid constraint type in row: {row}")

                subject_channel = self._parse_int(row[1])
                other_channel = self._parse_int(row[2])
                subject_station_id = self._parse_int(row[3])

                if subject_station_id not in self.stations_by_id:
                    raise KeyError(f"Subject station {subject_station_id} not found in stations.")
                partner_indices: list[int] = []
                for raw_partner in row[4:]:
                    partner_id = self._parse_int(raw_partner)
                    if partner_id == subject_station_id:
                        raise ValueError(f"Partner cannot be the same as the subject station ({subject_station_id}).")
                    partner_indices.append(self.station_index_for_id(partner_id))
                partner_indices.sort()
                subject_channel_index = self.channel_index_for_channel(subject_channel)
                other_channel_index = self.channel_index_for_channel(other_channel)
                self.stations_by_id[subject_station_id].interferences.append(
                    Interference(
                        constraint_type=constraint_type,
                        subject_channel=subject_channel,
                        other_channel=other_channel,
                        subject_channel_index=subject_channel_index,
                        other_channel_index=other_channel_index,
                        station_indices=partner_indices,
                    )
                )

    @staticmethod
    def _resolve_input_directory(path: str | Path) -> Path:
        base = Path("input").resolve()
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            parts = candidate.parts
            if parts and parts[0] == "input":
                resolved = (Path.cwd() / candidate).resolve()
            else:
                resolved = (base / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Input directory must be inside {base}, received {resolved}") from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Input directory not found: {resolved}")
        return resolved

    @staticmethod
    def _load_domain(path: Path) -> tuple[dict[int, list[int]], set[int]]:
        
        if not path.exists():
            raise FileNotFoundError(f"Domain CSV not found: {path}")

        station_domains: dict[int, list[int]] = {}
        channels_seen: set[int] = set()

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].strip().upper() != "DOMAIN":
                    continue
                if len(row) < 3:
                    raise ValueError(f"Invalid DOMAIN row: {row}")

                try:
                    station_id = TVGraph._parse_int(row[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid station identifier in DOMAIN row: {row}") from exc

                channels_raw = [value for value in row[2:] if value]
                channel_values = [TVGraph._parse_int(value) for value in channels_raw]
                station_domains[station_id] = channel_values
                channels_seen.update(channel_values)

        return station_domains, list(channels_seen)

    @staticmethod
    def _load_parameters(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
        if not path.exists():
            raise FileNotFoundError(f"parameters.csv not found: {path}")

        header: list[str] | None = None
        params_by_station: dict[int, dict[str, str]] = {}

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if header is None:
                    if row[0].strip() == "FacID":
                        header = [value.strip() for value in row]
                    continue
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                values = {header[i]: row[i].strip() for i in range(len(header))}
                fac_id_raw = values.get("FacID")
                if not fac_id_raw:
                    continue
                try:
                    station_id = int(fac_id_raw)
                except ValueError:
                    continue
                params_by_station[station_id] = values

        if header is None:
            raise ValueError(f"Unable to locate header row in {path}")

        return header, params_by_station

    @staticmethod
    def _unique_preserving_order(values: Iterable[int]) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    @staticmethod
    def _parse_int(value: str) -> int:
        return int(value.strip())

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None


__all__ = ["Station", "Interference", "TVGraph"]

