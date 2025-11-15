#!/usr/bin/env python3
"""
CLI utility to export a limited subgraph from an FCC dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - allow running as script or module
    from .tv_graph import TVGraph
except ImportError:  # pragma: no cover
    from tv_graph import TVGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a BFS-limited FCC subgraph into a new input directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fcc"),
        help="Subdirectory under ./input/ to load (default: fcc).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=87,
        help="Seed station ID to start the breadth-first traversal (default: 87).",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=5,
        help="Maximum number of stations to include (default: 5).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=7,
        help="Maximum number of unique channels to retain (default: 7).",
    )
    parser.add_argument(
        "--new-channel-only",
        action="store_true",
        help="Export every station with a post-auction channel assignment instead of running BFS.",
    )
    parser.add_argument(
        "--remove-top-channel",
        action="store_true",
        help="With --new-channel-only, drop the highest indexed domain channel from the export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    graph = TVGraph(args.input)
    output_dir = graph.save_sub_graph(
        seed_station=args.seed,
        station_limit=args.stations,
        channel_limit=args.channels,
        new_channel_only=args.new_channel_only,
        remove_top_channel=args.remove_top_channel,
    )
    print(f"Subgraph saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


