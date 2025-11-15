#!/usr/bin/env python3
"""
Utilities for building smaller FCC interference datasets.

This module parses `Domain.csv`, `Interference_Paired.csv`, and `parameters.csv`
files from an FCC input directory into a structured `TVGraph`. It provides
helpers to generate connected subgraphs with limited station and channel counts,
and it can emit trimmed CSV files plus per-station interference counts.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:  # pragma: no cover - package vs script execution
    from .tv_graph import TVGraph
except ImportError:  # pragma: no cover
    from tv_graph import TVGraph


def save_interference_counts(graph: TVGraph, output_path: str | Path) -> None:
    """
    Persist a CSV containing per-station interference counts sorted by constraint volume.
    """
    rows: list[tuple[int, int]] = []
    for station in graph.stations_by_id.values():
        constraint_count = sum(len(interference.station_indices) for interference in station.interferences)
        rows.append((station.station_id, constraint_count))

    rows.sort(key=lambda item: (-item[1], item[0]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_id", "constraint_count"])
        writer.writerows((station_id, count) for station_id, count in rows)


def save_domain_csv(graph: TVGraph, output_path: str | Path) -> None:
    """
    Write a DOMAIN file representing the station-channel domains for ``graph``.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for station in sorted(graph.stations_by_id.values(), key=lambda item: item.station_id):
            if not station.domain:
                raise ValueError(
                    f"Station {station.station_id} has no remaining allowed channels after trimming."
                )
            row = ["DOMAIN", str(station.station_id)]
            row.extend(str(channel) for channel in station.domain)
            writer.writerow(row)


def save_interference_csv(graph: TVGraph, output_path: str | Path) -> None:
    """
    Write an Interference_Paired-style CSV for ``graph``.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for station in sorted(graph.stations_by_id.values(), key=lambda item: item.station_id):
            for interference in station.interferences:
                constraint_type = interference.constraint_type
                row = [
                    constraint_type,
                    str(interference.subject_channel),
                    str(interference.other_channel),
                    str(station.station_id),
                ]
                partner_ids = [
                    str(graph.station_id_for_index(index)) for index in interference.station_indices
                ]
                if not partner_ids:
                    continue
                row.extend(partner_ids)
                writer.writerow(row)


def save_parameters_csv(
    graph: TVGraph,
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

    stations_sorted = sorted(graph.stations_by_id.values(), key=lambda item: item.station_id)

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
    graph: TVGraph,
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


def _sanitize_label(text: str) -> str:
    """
    Collapse an arbitrary string into a filesystem-friendly label.
    """
    normalized = text.replace("/", "-").replace("\\", "-")
    normalized = re.sub(r"[^A-Za-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-") or "dataset"


def _resolve_output_dir(
    input_dir: Path,
    seed_station: int,
    station_count: int,
    channel_count: int,
) -> Path:
    """
    Determine an output subdirectory inside ./input/ that encodes configuration parameters.
    """
    base_root = Path("input")
    base_root.mkdir(parents=True, exist_ok=True)

    input_dir_abs = input_dir.resolve()
    base_root_abs = base_root.resolve()

    try:
        relative_segment = input_dir_abs.relative_to(base_root_abs)
        source_label = _sanitize_label(str(relative_segment))
    except ValueError:
        source_label = _sanitize_label(input_dir_abs.stem or input_dir_abs.name)

    descriptor = _sanitize_label(
        f"{source_label}_seed{seed_station}_st{station_count}_ch{channel_count}"
    )
    if not descriptor:
        descriptor = "subset"

    candidate = base_root / descriptor
    suffix = 2
    while candidate.exists():
        candidate = base_root / f"{descriptor}-{suffix}"
        suffix += 1
    return candidate


def _build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct an argument parser for the CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Generate connected subgraphs from FCC interference datasets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("default"),
        help="Subdirectory under ./input/ containing Domain.csv, Interference_Paired.csv, and parameters.csv (default: input/default).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for generating subgraphs and exporting statistics.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    graph = TVGraph(args.input)
    subgraph = graph.make_subgraph(args.seed_station, args.channel_count, args.station_count)

    dataset_root = graph.dataset_root if graph.dataset_root is not None else Path(args.input).resolve()

    output_dir = _resolve_output_dir(
        dataset_root,
        args.seed_station,
        args.station_count,
        args.channel_count,
    )

    paths = save_subgraph_files(subgraph, subgraph.parameters_header, output_dir)

    print(
        f"Generated subgraph with {len(subgraph.stations_by_id)} stations and "
        f"{len(subgraph.channel_for_channel_id)} channels.",
    )
    print(f"Domain written to: {paths['domain']}")
    print(f"Interference written to: {paths['interference']}")
    print(f"Parameters written to: {paths['parameters']}")
    print(f"Constraint counts written to: {paths['counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

