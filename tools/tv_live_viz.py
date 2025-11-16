"""
Matplotlib-based live visualisation for THRML TV sampling runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

try:  # pragma: no cover - allow running as script or module import
    from .tv_graph import TVGraph
except ImportError:  # pragma: no cover
    from tv_graph import TVGraph


@dataclass
class MatplotlibSamplerViz:
    """
    Render live sampler state using a Matplotlib scatter plot with optional edges.
    """

    graph: TVGraph
    marker_size: float = 36.0
    show_edges: bool = True
    palette_name: str = "tab20"
    positions: np.ndarray = field(init=False, repr=False)
    edges: np.ndarray = field(init=False, repr=False)
    edge_lookup: Dict[Tuple[int, int], int] = field(init=False, repr=False)
    figure: plt.Figure = field(init=False, repr=False)
    ax: plt.Axes = field(init=False, repr=False)
    scatter: object = field(init=False, repr=False)
    edge_collection: LineCollection | None = field(init=False, default=None, repr=False)
    title: object = field(init=False, repr=False)
    labels: List[object] = field(init=False, repr=False)
    _palette: np.ndarray = field(init=False, repr=False)
    _domain_ok_edge: np.ndarray = field(init=False, repr=False)
    _domain_violation_edge: np.ndarray = field(init=False, repr=False)
    _edge_ok: np.ndarray = field(init=False, repr=False)
    _edge_violation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = self._compute_positions(self.graph)
        self.edges, self.edge_lookup = self._extract_edges(self.graph)

        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(9, 7))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        palette = plt.cm.get_cmap(self.palette_name)
        self._palette = np.asarray([palette(i) for i in range(palette.N)], dtype=float)

        station_count = self.graph.station_count
        default_assignment = np.zeros(station_count, dtype=np.int32)

        self.scatter = self.ax.scatter(
            self.positions[:, 1],
            self.positions[:, 0],
            s=self.marker_size,
            c=self._palette[default_assignment % len(self._palette)],
            edgecolors=np.tile(np.array([0.1, 0.1, 0.1, 0.8]), (station_count, 1)),
            linewidths=0.6,
        )

        if self.show_edges and self.edges.size:
            segments = self.positions[self.edges][:, :, [1, 0]]
            self.edge_collection = LineCollection(
                segments,
                colors=np.tile(np.array([0.7, 0.7, 0.7, 0.4]), (len(self.edges), 1)),
                linewidths=0.6,
            )
            self.ax.add_collection(self.edge_collection)
        else:
            self.edge_collection = None

        self.title = self.ax.set_title("Sampler initialised")
        self.labels = self._add_city_labels()
        self._set_axis_bounds()
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

        self._domain_ok_edge = np.array([0.1, 0.1, 0.1, 0.8])
        self._domain_violation_edge = np.array([0.9, 0.05, 0.05, 1.0])
        self._edge_ok = np.array([0.7, 0.7, 0.7, 0.4])
        self._edge_violation = np.array([0.95, 0.2, 0.2, 0.9])

    def compute_violation_masks(self, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute domain and interference violation masks for a contiguous assignment.
        """

        domain_matrix = self.graph.domain_mask
        index = np.arange(len(assignment), dtype=np.int32)
        domain_ok = domain_matrix[index, assignment]
        domain_violation_mask = ~domain_ok

        edge_violation_mask = np.zeros(len(self.edges), dtype=bool)
        if len(self.edges) == 0:
            return domain_violation_mask, edge_violation_mask

        for station in self.graph.stations_by_id.values():
            if station.station_index is None:
                continue
            a_idx = station.station_index
            assigned_a_idx = int(assignment[a_idx])
            for interference in station.interferences:
                if assigned_a_idx != interference.subject_channel_index:
                    continue
                for partner_idx in interference.station_indices:
                    assigned_b_idx = int(assignment[partner_idx])
                    if assigned_b_idx != interference.other_channel_index:
                        continue
                    key = (min(a_idx, partner_idx), max(a_idx, partner_idx))
                    edge_id = self.edge_lookup.get(key)
                    if edge_id is not None:
                        edge_violation_mask[edge_id] = True

        return domain_violation_mask, edge_violation_mask

    def update(
        self,
        step: int,
        assignment: np.ndarray,
        domain_violation_mask: np.ndarray,
        edge_violation_mask: np.ndarray,
    ) -> None:
        """
        Apply visual updates for the latest sampler state.
        """

        palette_len = len(self._palette)
        colours = self._palette[assignment % palette_len]
        self.scatter.set_facecolor(colours)

        station_edge_colours = np.tile(self._domain_ok_edge, (len(domain_violation_mask), 1))
        station_edge_colours[domain_violation_mask] = self._domain_violation_edge
        self.scatter.set_edgecolor(station_edge_colours)

        if self.edge_collection is not None and len(edge_violation_mask):
            edge_colours = np.tile(self._edge_ok, (len(edge_violation_mask), 1))
            edge_colours[edge_violation_mask] = self._edge_violation
            self.edge_collection.set_color(edge_colours)

        for idx, label in enumerate(self.labels):
            colour = self._domain_violation_edge if domain_violation_mask[idx] else (0.1, 0.1, 0.1, 0.9)
            label.set_color(colour)

        self.title.set_text(
            f"Step {step:,} · domain violations {int(domain_violation_mask.sum())} · "
            f"edge violations {int(edge_violation_mask.sum())}"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def block(self) -> None:
        """
        Block the main thread until the Matplotlib window is dismissed.
        """

        plt.ioff()
        try:
            plt.show(block=True)
        finally:
            plt.ion()

    @staticmethod
    def _compute_positions(graph: TVGraph) -> np.ndarray:
        """
        Return station latitude/longitude pairs with a small fallback jitter when missing.
        """

        lats: list[float] = []
        lons: list[float] = []
        for station in graph.stations_by_id.values():
            if station.lat is not None and station.lon is not None:
                lats.append(float(station.lat))
                lons.append(float(station.lon))

        lat0 = float(np.mean(lats)) if lats else 0.0
        lon0 = float(np.mean(lons)) if lons else 0.0

        coords: list[tuple[float, float]] = []
        station_count = graph.station_count
        for idx in range(station_count):
            station = graph.station_by_index(idx)
            lat = station.lat
            lon = station.lon
            if lat is None or lon is None:
                angle = 2.0 * math.pi * idx / max(station_count, 1)
                radius = 0.5 + 0.1 * (idx // max(station_count, 1) + 1)
                lat = lat0 + radius * math.sin(angle)
                lon = lon0 + radius * math.cos(angle)
            coords.append((float(lat), float(lon)))

        return np.asarray(coords, dtype=float)

    @staticmethod
    def _extract_edges(graph: TVGraph) -> tuple[np.ndarray, Dict[Tuple[int, int], int]]:
        """
        Deduplicate interference edges into undirected pairs.
        """

        edge_set: set[tuple[int, int]] = set()
        for station in graph.stations_by_id.values():
            if station.station_index is None:
                continue
            a_idx = station.station_index
            for interference in station.interferences:
                for partner_idx in interference.station_indices:
                    pair = (min(a_idx, partner_idx), max(a_idx, partner_idx))
                    edge_set.add(pair)

        if not edge_set:
            return np.zeros((0, 2), dtype=np.int32), {}

        sorted_edges = sorted(edge_set)
        edge_lookup = {edge: idx for idx, edge in enumerate(sorted_edges)}
        return np.asarray(sorted_edges, dtype=np.int32), edge_lookup

    def _set_axis_bounds(self) -> None:
        """
        Set Matplotlib axis limits based on latitude and longitude ranges.
        """

        latitudes = self.positions[:, 0]
        longitudes = self.positions[:, 1]
        if latitudes.size == 0 or longitudes.size == 0:
            return

        lat_min, lat_max = float(latitudes.min()), float(latitudes.max())
        lon_min, lon_max = float(longitudes.min()), float(longitudes.max())

        lat_pad = max((lat_max - lat_min) * 0.05, 0.1)
        lon_pad = max((lon_max - lon_min) * 0.05, 0.1)

        self.ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
        self.ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)

    def _add_city_labels(self) -> List[object]:
        """
        Attach the station city name (or station ID) next to each point.
        """

        labels: List[object] = []
        for idx, (lat, lon) in enumerate(self.positions):
            station = self.graph.station_by_index(idx)
            city = (station.attrs.get("City") if station.attrs else None) or str(station.station_id)
            label = self.ax.text(
                lon + 0.05,
                lat + 0.05,
                city,
                fontsize=8,
                color=(0.1, 0.1, 0.1, 0.9),
                ha="left",
                va="bottom",
            )
            labels.append(label)
        return labels


__all__ = ["MatplotlibSamplerViz"]


