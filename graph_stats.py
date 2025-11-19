#!/usr/bin/env python3
"""
Inspect FCC interference datasets using in-memory TVGraph structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from lib.tv_graph import DomainViolation, InterferenceViolation, TVGraph


def _iter_constraints(graph: TVGraph) -> Iterable[tuple[int, int, int, int, str]]:
    """
    Yield directional constraints as (station_a, channel_a, station_b, channel_b, type).
    """
    for station in graph.stations_by_id.values():
        for interference in station.interferences:
            for partner_index in interference.station_indices:
                partner_id = graph.station_id_for_index(partner_index)
                yield (
                    station.station_id,
                    interference.subject_channel,
                    partner_id,
                    interference.other_channel,
                    interference.constraint_type,
                )


def _build_constraint_index(graph: TVGraph) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """
    Map each (station, subject_channel) to a set of partner tuples (peer_station, other_channel).
    """
    index: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for station in graph.stations_by_id.values():
        for interference in station.interferences:
            key = (station.station_id, interference.subject_channel)
            partners = index.setdefault(key, set())
            for partner_index in interference.station_indices:
                partner_id = graph.station_id_for_index(partner_index)
                partners.add((partner_id, interference.other_channel))
    return index


def find_asymmetric_constraints(graph: TVGraph) -> list[tuple[int, int, int, int, str]]:
    """
    Identify directional constraints missing a reverse counterpart.
    """
    index = _build_constraint_index(graph)
    missing: list[tuple[int, int, int, int, str]] = []

    for station_a, channel_a, station_b, channel_b, constraint_type in _iter_constraints(graph):
        reverse_key = (station_b, channel_b)
        reverse_tuple = (station_a, channel_a)
        if reverse_key not in index or reverse_tuple not in index[reverse_key]:
            missing.append((station_a, channel_a, station_b, channel_b, constraint_type))

    return missing


def compute_graph_stats(graph: TVGraph) -> dict[str, object]:
    """
    Calculate descriptive statistics for a TVGraph instance.
    """
    stations = list(graph.stations_by_id.values())
    station_count = len(stations)
    channels = graph.channel_for_channel_id
    channel_count = len(channels)

    domain_lengths = [len(station.domain) for station in stations]
    total_domain = sum(domain_lengths)
    max_domain = max(domain_lengths) if domain_lengths else 0
    min_domain = min(domain_lengths) if domain_lengths else 0

    constraint_counts: list[int] = [
        sum(len(interference.station_indices) for interference in station.interferences)
        for station in stations
    ]
    total_directional_constraints = sum(constraint_counts)
    max_constraints = max(constraint_counts) if constraint_counts else 0
    min_constraints = min(constraint_counts) if constraint_counts else 0

    constraint_counts_by_type: Counter[str] = Counter()
    for station_a, channel_a, station_b, channel_b, constraint_type in _iter_constraints(graph):
        constraint_counts_by_type[constraint_type] += 1

    constraint_histogram = Counter(constraint_counts)

    top_by_constraints = sorted(
        [
            (station.station_id, count)
            for station, count in zip(stations, constraint_counts)
        ],
        key=lambda item: (-item[1], item[0]),
    )

    return {
        "station_count": station_count,
        "channel_count": channel_count,
        "domain_sum": total_domain,
        "domain_avg": total_domain / station_count if station_count else 0.0,
        "domain_min": min_domain,
        "domain_max": max_domain,
        "constraint_total": total_directional_constraints,
        "constraint_avg": total_directional_constraints / station_count if station_count else 0.0,
        "constraint_min": min_constraints,
        "constraint_max": max_constraints,
        "constraint_histogram": constraint_histogram,
        "constraint_counts_by_type": constraint_counts_by_type,
        "top_by_constraints": top_by_constraints,
    }


def render_stats(
    graph: TVGraph,
    stats: dict[str, object],
    *,
    top_k: int,
    domain_violations: list[DomainViolation],
    interference_violations: list[InterferenceViolation],
) -> None:
    """
    Print formatted statistics and symmetry checks.
    """
    print("TV Graph Statistics")
    print("===================")
    print(f"Stations             : {stats['station_count']:,}")
    print(f"Channels             : {stats['channel_count']:,}")
    print(f"Domain entries       : {stats['domain_sum']:,}")
    print(f"Domain size (avg/min/max): {stats['domain_avg']:.2f} / {stats['domain_min']} / {stats['domain_max']}")
    print(f"Total directional constraints: {stats['constraint_total']:,}")
    print(
        "Constraints per station (avg/min/max): "
        f"{stats['constraint_avg']:.2f} / {stats['constraint_min']} / {stats['constraint_max']}"
    )

    domain_violation_count = len(domain_violations)
    interference_violation_count = len(interference_violations)
    print(f"Domain violations (new_channel outside domain): {domain_violation_count:,}")
    print(f"Interference violations (new_channel conflicts): {interference_violation_count:,}")

    if domain_violations:
        print("\nStations with new_channel outside domain:")
        for violation in sorted(domain_violations, key=lambda item: item.station_id):
            print(f"  {violation.station_id:<8} new_channel {violation.new_channel}")
    else:
        print("\n✅ All new_channel assignments fall within station domains.")

    type_counts: Counter[str] = stats["constraint_counts_by_type"]  # type: ignore[index]
    if type_counts:
        print("\nDirectional constraints by type:")
        for constraint_type, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {constraint_type:<8} {count:,}")

    missing = find_asymmetric_constraints(graph)
    if missing:
        print(f"\n⚠️  Found {len(missing):,} directional constraints lacking a reverse entry.")
        missing_by_type = Counter(item[4] for item in missing)
        for station_a, channel_a, station_b, channel_b, constraint_type in missing[:top_k]:
            print(
                f"  {station_a} ch {channel_a} → {station_b} ch {channel_b} "
                f"[{constraint_type}] (missing reverse)"
            )
        if len(missing) > top_k:
            print(f"  ... and {len(missing) - top_k:,} more.")
        print("\nMissing constraints by type:")
        for constraint_type, count in sorted(missing_by_type.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {constraint_type:<8} {count:,}")
    else:
        print("\n✅ All parsed constraints are symmetric.")

    if top_k > 0:
        print(f"\nTop {top_k} stations by directional constraint count:")
        for station_id, count in stats["top_by_constraints"][:top_k]:
            print(f"  {station_id:<8} {count:>8}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise FCC interference datasets and verify constraint symmetry.",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("default"),
        help="Subdirectory under ./input/ containing Domain.csv, Interference_Paired.csv, and parameters.csv (default: input/default).",
    )
    parser.add_argument(
        "-t", "--top",
        type=int,
        default=10,
        help="Number of stations to display in top constraints listing.",
    )
    parser.add_argument(
        "-s", "--strict",
        action="store_true",
        help="Return exit code 1 when asymmetries are detected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    graph = TVGraph(args.input)
    stats = compute_graph_stats(graph)
    domain_violations, interference_violations = graph.assignment_violations()
    render_stats(
        graph,
        stats,
        top_k=args.top,
        domain_violations=domain_violations,
        interference_violations=interference_violations,
    )

    if args.strict:
        missing = find_asymmetric_constraints(graph)
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

