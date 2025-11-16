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
    post_attrs: dict[str, str] = field(default_factory=dict)
    new_channel: int | None = None

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


@dataclass(frozen=True)
class DomainViolation:
    """
    A post-auction channel assignment that is not permitted by the station domain.
    """

    station_id: int
    new_channel: int
    domain: tuple[int, ...]


@dataclass(frozen=True)
class InterferenceViolation:
    """
    A pair of post-auction channel assignments that violates a directional constraint.
    """

    constraint_type: str
    subject_station_id: int
    subject_channel: int
    other_station_id: int
    other_channel: int


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

        (
            post_params_header,
            post_params_by_station,
        ) = self._load_parameters(root / "post_auction_parameters.csv")
        self.post_parameters_header = list(post_params_header)

        # Create Station objects with associated parameters.
        stations: dict[int, Station] = {}
        for station_id, domain_values in station_domains.items():
            station_params = params_by_station.get(station_id, {})
            post_params = post_params_by_station.get(station_id, {})
            station = Station(
                station_id=station_id,
                domain=domain_values,
                lat=self._parse_float(station_params.get("Lat")),
                lon=self._parse_float(station_params.get("Lon")),
                attrs=dict(station_params),
                post_attrs=dict(post_params),
                new_channel=self._parse_optional_int(post_params.get("Ch")),
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
        new_channel_only: bool = False,
        remove_top_channel: bool = False,
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
        new_channel_only:
            When ``True``, ignore BFS parameters and emit a subgraph consisting of
            every station that has a post-auction channel assignment. Domains are
            intersected with the set of observed post-auction channels and the
            resulting files are filtered accordingly.
        remove_top_channel:
            When ``True`` and ``new_channel_only`` is set, exclude the domain channel with
            the highest contiguous index from the exported graph while leaving post-auction
            assignments untouched.

        Returns
        -------
        Path
            Directory path created under ``input/`` containing ``Domain.csv``,
            ``parameters.csv``, ``post_auction_parameters.csv``, and
            ``Interference_Paired.csv`` for the subset.
        """
        if remove_top_channel and not new_channel_only:
            raise ValueError("remove_top_channel requires new_channel_only=True.")

        removed_channel_value: int | None = None
        domain_channel_set: set[int]

        if new_channel_only:
            station_ids = [
                station_id
                for station_id in self.station_ids_by_index
                if self.stations_by_id[station_id].new_channel is not None
            ]
            if not station_ids:
                raise ValueError("No stations contain a post-auction channel assignment.")
            selected_channel_set = {
                self.stations_by_id[station_id].new_channel
                for station_id in station_ids
                if self.stations_by_id[station_id].new_channel is not None
            }
            if not selected_channel_set:
                raise ValueError("No usable post-auction channels located in dataset.")
            station_ids = list(station_ids)
            domain_channel_set = set(selected_channel_set)

            if remove_top_channel:
                max_channel_index: int | None = None
                for station_id in station_ids:
                    station = self.station(station_id)
                    for channel in station.domain:
                        if channel not in domain_channel_set:
                            continue
                        channel_index = self.channel_index_for_channel(channel)
                        if max_channel_index is None or channel_index > max_channel_index:
                            max_channel_index = channel_index
                            removed_channel_value = channel
                if removed_channel_value is None:
                    raise ValueError("Unable to locate a domain channel eligible for removal.")
                domain_channel_set.discard(removed_channel_value)

            output_dir = (
                self.dataset_root.parent
                / f"{self.dataset_root.name}-post-auction-{len(station_ids)}st-{len(domain_channel_set)}ch"
            )
        else:
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

            channel_sequence: list[int] = []
            channel_set: set[int] = set()

            # Always retain post-auction assignments when present.
            for station_id in station_ids:
                station = self.station(station_id)
                if not station.domain:
                    raise ValueError(f"Station {station_id} has an empty domain.")

                new_channel = station.new_channel
                if new_channel is None:
                    continue
                if new_channel not in station.domain:
                    raise ValueError(
                        f"Station {station_id} post-auction channel {new_channel} not present in its domain."
                    )
                if new_channel in channel_set:
                    continue
                channel_set.add(new_channel)
                channel_sequence.append(new_channel)

            if len(channel_set) > channel_limit:
                raise ValueError(
                    f"Post-auction assignments require {len(channel_set)} unique channels, "
                    f"exceeding channel_limit={channel_limit}."
                )

            for station_id in station_ids:
                if len(channel_set) >= channel_limit:
                    break
                station = self.station(station_id)
                for channel in station.domain:
                    if channel in channel_set:
                        continue
                    channel_set.add(channel)
                    channel_sequence.append(channel)
                    if len(channel_set) >= channel_limit:
                        break

            if len(channel_set) < channel_limit:
                total_unique = len(
                    {
                        channel
                        for station_id in station_ids
                        for channel in self.station(station_id).domain
                    }
                )
                raise ValueError(
                    f"Only {total_unique} unique channels available across the selected stations; "
                    f"cannot satisfy channel_limit={channel_limit}."
                )

            domain_channel_set = set(channel_sequence[:channel_limit])

            output_dir = (
                self.dataset_root.parent
                / f"{self.dataset_root.name}-seed{seed_station}-st{station_limit}-ch{channel_limit}"
            )

        station_set = set(station_ids)

        if new_channel_only and not domain_channel_set and not remove_top_channel:
            raise ValueError("Post-auction subgraph requires at least one valid channel.")

        output_dir.mkdir(parents=True, exist_ok=False)

        domain_path = output_dir / "Domain.csv"
        with domain_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for station_id in station_ids:
                station = self.station(station_id)
                domain_subset = [channel for channel in station.domain if channel in domain_channel_set]
                if not domain_subset and not remove_top_channel:
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

        post_params_path = output_dir / "post_auction_parameters.csv"
        with post_params_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.post_parameters_header)
            for station_id in station_ids:
                station = self.station(station_id)
                row: list[str] = []
                for column in self.post_parameters_header:
                    if column == "FacID":
                        value = station.post_attrs.get(column) or str(station.station_id)
                    else:
                        value = station.post_attrs.get(column, "")
                    row.append(value)
                writer.writerow(row)

        interference_path = output_dir / "Interference_Paired.csv"
        with interference_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            for station_id in station_ids:
                station = self.station(station_id)
                for interference in station.interferences:
                    if interference.subject_channel not in domain_channel_set:
                        continue
                    if interference.other_channel not in domain_channel_set:
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

    def assignment_violations(self) -> tuple[list[DomainViolation], list[InterferenceViolation]]:
        """
        Identify post-auction assignments that violate domain or interference constraints.

        Returns
        -------
        tuple[list[DomainViolation], list[InterferenceViolation]]
            A tuple containing all domain violations and all interference violations.
        """
        domain_violations: list[DomainViolation] = []
        interference_violations: list[InterferenceViolation] = []

        for station in self.stations_by_id.values():
            new_channel = station.new_channel
            if new_channel is None:
                continue
            if new_channel not in station.domain:
                domain_violations.append(
                    DomainViolation(
                        station_id=station.station_id,
                        new_channel=new_channel,
                        domain=tuple(station.domain),
                    )
                )

        assigned_channels: dict[int, int] = {}
        for station_id, station in self.stations_by_id.items():
            if station.new_channel is not None:
                assigned_channels[station_id] = station.new_channel

        for station in self.stations_by_id.values():
            subject_channel = station.new_channel
            if subject_channel is None:
                continue
            for interference in station.interferences:
                if subject_channel != interference.subject_channel:
                    continue
                for partner_index in interference.station_indices:
                    partner_id = self.station_id_for_index(partner_index)
                    partner_channel = assigned_channels.get(partner_id)
                    if partner_channel is None:
                        continue
                    if partner_channel == interference.other_channel:
                        interference_violations.append(
                            InterferenceViolation(
                                constraint_type=interference.constraint_type,
                                subject_station_id=station.station_id,
                                subject_channel=subject_channel,
                                other_station_id=partner_id,
                                other_channel=partner_channel,
                            )
                        )

        return domain_violations, interference_violations


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
                    first_cell = TVGraph._strip_bom(row[0].strip())
                    if first_cell == "FacID":
                        header = [TVGraph._strip_bom(value.strip()) for value in row]
                    continue
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                values = {
                    header[i]: TVGraph._strip_bom(row[i].strip()) for i in range(len(header))
                }
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

    @staticmethod
    def _parse_optional_int(value: str | None) -> int | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _strip_bom(value: str) -> str:
        if value.startswith("\ufeff"):
            return value.lstrip("\ufeff")
        return value


__all__ = ["Station", "Interference", "DomainViolation", "InterferenceViolation", "TVGraph"]

