#!/usr/bin/env python3
"""
Utilities for building smaller FCC interference datasets.

This module parses `Domain.csv`, `Interference_Paired.csv`, and `parameters.csv`
files from an FCC input directory into a structured `tv_graph`. It provides
helpers to generate connected subgraphs with limited station and channel counts,
and it can emit trimmed CSV files plus per-station interference counts.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tv_graph import Interference, Station, tv_graph


def load_tv_graph(input_dir: str | Path) -> tuple[tv_graph, list[str]]:
    """
    Parse FCC CSV inputs under ``input_dir`` into a ``tv_graph`` instance.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")

    domain_path = root / "Domain.csv"
    interference_path = root / "Interference_Paired.csv"
    parameters_path = root / "parameters.csv"

    station_domains, all_channels = _load_domain(domain_path)
    params_header, params_by_station = _load_parameters(parameters_path)

    stations: dict[int, Station] = {}
    for station_id, domain_values in station_domains.items():
        params = params_by_station.get(station_id, {})
        lat = _parse_float(params.get("Lat"))
        lon = _parse_float(params.get("Lon"))
        stations[station_id] = Station(
            station_id=station_id,
            domain=domain_values,
            lat=lat,
            lon=lon,
            attrs=dict(params),
        )

    channel_for_channel_id = sorted(all_channels)
    channel_id_for_channel = {channel: idx for idx, channel in enumerate(channel_for_channel_id)}

    _populate_interferences(interference_path, stations)

    graph = tv_graph(
        stations=stations,
        channel_id_for_channel=channel_id_for_channel,
        channel_for_channel_id=channel_for_channel_id,
    )
    return graph, params_header


def save_interference_counts(graph: tv_graph, output_path: str | Path) -> None:
    """
    Persist a CSV containing per-station interference counts sorted by constraint volume.
    """
    rows: list[tuple[int, int]] = []
    for station in graph.stations.values():
        constraint_count = sum(len(interference.stations) for interference in station.interferences)
        rows.append((station.station_id, constraint_count))

    rows.sort(key=lambda item: (-item[1], item[0]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "constraint_count"])
        writer.writerows((station_id, count) for station_id, count in rows)


def save_domain_csv(graph: tv_graph, output_path: str | Path) -> None:
    """
    Write a DOMAIN file representing the station-channel domains for ``graph``.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for station in sorted(graph.stations.values(), key=lambda item: item.station_id):
            if not station.domain:
                raise ValueError(
                    f"Station {station.station_id} has no remaining allowed channels after trimming."
                )
            row = ["DOMAIN", str(station.station_id)]
            row.extend(str(channel) for channel in station.domain)
            writer.writerow(row)


def save_interference_csv(graph: tv_graph, output_path: str | Path) -> None:
    """
    Write an Interference_Paired-style CSV for ``graph``.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for station in sorted(graph.stations.values(), key=lambda item: item.station_id):
            for interference in station.interferences:
                constraint_type = interference.constraint_type
                row = [
                    constraint_type,
                    str(interference.subject_channel),
                    str(interference.other_channel),
                    str(station.station_id),
                ]
                partner_ids = [str(partner.station_id) for partner in interference.stations]
                if not partner_ids:
                    continue
                row.extend(partner_ids)
                writer.writerow(row)


def save_parameters_csv(
    graph: tv_graph,
    header: list[str],
    output_path: str | Path,
) -> None:
    """
    Write a trimmed parameters.csv limited to the stations in ``graph``.
    """
    if not header:
        raise ValueError("parameters header is required to write parameters.csv.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    stations_sorted = sorted(graph.stations.values(), key=lambda item: item.station_id)

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for station in stations_sorted:
            row: list[str] = []
            for column in header:
                if column == "FacID":
                    row.append(str(station.station_id))
                else:
                    row.append(station.attrs.get(column, ""))
            writer.writerow(row)


def save_subgraph_files(
    graph: tv_graph,
    parameters_header: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """
    Persist the subgraph as a set of FCC-compatible CSV files plus statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_path = output_dir / "Domain.csv"
    interference_path = output_dir / "Interference_Paired.csv"
    parameters_path = output_dir / "parameters.csv"
    counts_path = output_dir / "interference_counts.csv"

    save_domain_csv(graph, domain_path)
    save_interference_csv(graph, interference_path)
    save_parameters_csv(graph, parameters_header, parameters_path)
    save_interference_counts(graph, counts_path)

    return {
        "domain": domain_path,
        "interference": interference_path,
        "parameters": parameters_path,
        "counts": counts_path,
    }


def _load_domain(path: Path) -> tuple[dict[int, list[int]], set[int]]:
    """
    Load Domain.csv into per-station domain lists and global channel set.
    """
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
                station_id = int(row[1])
            except ValueError as exc:
                raise ValueError(f"Invalid station identifier in DOMAIN row: {row}") from exc

            channels_raw = [value for value in row[2:] if value]
            channel_values = [_parse_int(value) for value in channels_raw]
            station_domains[station_id] = _unique_preserving_order(channel_values)
            channels_seen.update(channel_values)

    if not station_domains:
        raise ValueError(f"No DOMAIN rows parsed from {path}")

    return station_domains, channels_seen


def _load_parameters(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    """
    Load parameters.csv and return a mapping from station ID to attribute dict.
    """
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
            # Pad short rows with empty strings to align with header length.
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


def _populate_interferences(path: Path, stations: dict[int, Station]) -> None:
    """
    Populate `Station.interferences` based on Interference_Paired.csv content.
    """
    if not path.exists():
        raise FileNotFoundError(f"Interference CSV not found: {path}")

    grouped_constraints: dict[tuple[int, str, int, int], set[int]] = defaultdict(set)

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 5:
                continue
            constraint_type = row[0].strip().upper()
            if not constraint_type:
                continue
            try:
                subject_channel = _parse_int(row[1])
                other_channel = _parse_int(row[2])
                subject_station = _parse_int(row[3])
            except ValueError:
                continue
            if subject_station not in stations:
                continue

            partners: list[int] = []
            for raw_partner in row[4:]:
                if not raw_partner:
                    continue
                try:
                    partner_id = _parse_int(raw_partner)
                except ValueError:
                    continue
                if partner_id == subject_station:
                    continue
                if partner_id in stations:
                    partners.append(partner_id)

            if not partners:
                continue

            key = (subject_station, constraint_type, subject_channel, other_channel)
            grouped_constraints[key].update(partners)

    for (station_id, constraint_type, subject_channel, other_channel), partner_ids in grouped_constraints.items():
        station = stations.get(station_id)
        if station is None:
            continue
        partner_objects = [stations[partner_id] for partner_id in sorted(partner_ids)]
        station.interferences.append(
            Interference(
                constraint_type=constraint_type,
                subject_channel=subject_channel,
                other_channel=other_channel,
                stations=partner_objects,
            )
        )

    for station in stations.values():
        station.interferences.sort(key=lambda item: (item.subject_channel, item.other_channel))


def _unique_preserving_order(values: Iterable[int]) -> list[int]:
    """
    Return a list of unique integers preserving their first-seen order.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _parse_int(value: str) -> int:
    """
    Convert a string to an integer, raising ValueError on failure.
    """
    return int(value.strip())


def _parse_float(value: str | None) -> float | None:
    """
    Convert a string to a float, returning None for falsy inputs or parse failures.
    """
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct an argument parser for the CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Generate connected subgraphs from FCC interference datasets.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input/fcc"),
        help="Directory containing Domain.csv, Interference_Paired.csv, and parameters.csv.",
    )
    parser.add_argument(
        "--seed-station",
        type=int,
        required=True,
        help="Station ID to start the breadth-first traversal from.",
    )
    parser.add_argument(
        "--station-count",
        type=int,
        required=True,
        help="Number of stations to include in the generated subgraph.",
    )
    parser.add_argument(
        "--channel-count",
        type=int,
        required=True,
        help="Number of channels to retain (prefix of the global channel list).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory (relative paths are placed under ./output).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for generating subgraphs and exporting statistics.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    graph, params_header = load_tv_graph(args.input_dir)
    subgraph = graph.make_subgraph(args.seed_station, args.channel_count, args.station_count)

    output_dir = args.output_path
    if not output_dir.is_absolute():
        output_dir = Path("output") / output_dir

    paths = save_subgraph_files(subgraph, params_header, output_dir)

    print(
        f"Generated subgraph with {len(subgraph.stations)} stations and "
        f"{len(subgraph.channel_for_channel_id)} channels.",
    )
    print(f"Domain written to: {paths['domain']}")
    print(f"Interference written to: {paths['interference']}")
    print(f"Parameters written to: {paths['parameters']}")
    print(f"Constraint counts written to: {paths['counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

