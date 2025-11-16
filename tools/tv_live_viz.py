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

    Use ←/→ (or h/l) to step through recorded updates, Home/End to jump to the
    first or most recent state.
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
    energy_ax: plt.Axes = field(init=False, repr=False)
    scatter: object = field(init=False, repr=False)
    edge_collection: LineCollection | None = field(init=False, default=None, repr=False)
    title: object = field(init=False, repr=False)
    labels: List[object] = field(init=False, repr=False)
    energy_line: object = field(init=False, repr=False)
    energy_marker: object = field(init=False, repr=False)
    history_steps: List[int] = field(init=False, repr=False)
    history_assignments: List[np.ndarray] = field(init=False, repr=False)
    history_domain_masks: List[np.ndarray] = field(init=False, repr=False)
    history_edge_masks: List[np.ndarray] = field(init=False, repr=False)
    history_energies: List[float] = field(init=False, repr=False)
    current_index: int = field(init=False, repr=False)
    _palette: np.ndarray = field(init=False, repr=False)
    _domain_ok_edge: np.ndarray = field(init=False, repr=False)
    _domain_violation_edge: np.ndarray = field(init=False, repr=False)
    _edge_ok: np.ndarray = field(init=False, repr=False)
    _edge_violation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = self._compute_positions(self.graph)
        self.edges, self.edge_lookup = self._extract_edges(self.graph)

        plt.ion()
        self.figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 9),
            gridspec_kw={"height_ratios": [3.0, 1.2]},
        )
        self.ax, self.energy_ax = axes
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

        self.energy_ax.set_title("Energy trajectory")
        self.energy_ax.set_xlabel("Step")
        self.energy_ax.set_ylabel("Energy (log scale)")
        self.energy_ax.set_yscale("log")
        self.energy_ax.grid(True, alpha=0.3)
        (self.energy_line,) = self.energy_ax.plot([], [], color="tab:blue", linewidth=1.5)
        (self.energy_marker,) = self.energy_ax.plot([], [], marker="o", color="tab:red", markersize=6)

        self.history_steps = []
        self.history_assignments = []
        self.history_domain_masks = []
        self.history_edge_masks = []
        self.history_energies = []
        self.current_index = -1

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)

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
        energy: float,
    ) -> None:
        """
        Record a new sampler state and refresh the visualisation.
        """

        self.record_state(step, assignment, domain_violation_mask, edge_violation_mask, energy)

    def record_state(
        self,
        step: int,
        assignment: np.ndarray,
        domain_violation_mask: np.ndarray,
        edge_violation_mask: np.ndarray,
        energy: float,
    ) -> None:
        """
        Store the provided sampler state in the history and redraw the current view.
        """

        step_value = int(step)
        assignment_arr = np.asarray(assignment, dtype=np.int32).copy()
        domain_mask_arr = np.asarray(domain_violation_mask, dtype=bool).copy()
        edge_mask_arr = np.asarray(edge_violation_mask, dtype=bool).copy()
        energy_value = float(energy)

        if self.history_steps and self.history_steps[-1] == step_value:
            self.history_assignments[-1] = assignment_arr
            self.history_domain_masks[-1] = domain_mask_arr
            self.history_edge_masks[-1] = edge_mask_arr
            self.history_energies[-1] = energy_value
        else:
            self.history_steps.append(step_value)
            self.history_assignments.append(assignment_arr)
            self.history_domain_masks.append(domain_mask_arr)
            self.history_edge_masks.append(edge_mask_arr)
            self.history_energies.append(energy_value)

        new_index = len(self.history_steps) - 1
        self._update_energy_plot()
        self._set_current_index(new_index, force=True)

    def _apply_state(self) -> None:
        if self.current_index < 0 or not self.history_steps:
            return

        assignment = self.history_assignments[self.current_index]
        domain_mask = self.history_domain_masks[self.current_index]
        edge_mask = self.history_edge_masks[self.current_index]
        step_value = self.history_steps[self.current_index]
        energy_value = self.history_energies[self.current_index]

        palette_len = len(self._palette)
        colours = self._palette[assignment % palette_len]
        self.scatter.set_facecolor(colours)

        station_edge_colours = np.tile(self._domain_ok_edge, (len(domain_mask), 1))
        station_edge_colours[domain_mask] = self._domain_violation_edge
        self.scatter.set_edgecolor(station_edge_colours)

        incident_edge_violation = np.zeros(len(domain_mask), dtype=bool)

        if self.show_edges and len(self.edges):
            segments = self.positions[self.edges][:, :, [1, 0]]
            violated_segments = segments[edge_mask]
            ok_segments = segments[~edge_mask]

            if ok_segments.size:
                ok_collection = LineCollection(
                    ok_segments,
                    colors=np.tile(self._edge_ok, (len(ok_segments), 1)),
                    linewidths=0.6,
                    zorder=1,
                )
            else:
                ok_collection = None

            if violated_segments.size:
                violation_collection = LineCollection(
                    violated_segments,
                    colors=np.tile(self._edge_violation, (len(violated_segments), 1)),
                    linewidths=1.3,
                    zorder=2,
                )
            else:
                violation_collection = None

            # remove previous edge collections (anything that's a LineCollection and not the scatter itself)
            for collection in [
                coll for coll in self.ax.collections if isinstance(coll, LineCollection)
            ]:
                collection.remove()

            if ok_collection is not None:
                self.ax.add_collection(ok_collection)
            if violation_collection is not None:
                self.ax.add_collection(violation_collection)

            if violated_segments.size:
                violated_nodes = self.edges[edge_mask].reshape(-1)
                incident_edge_violation[violated_nodes] = True

        for idx, label in enumerate(self.labels):
            if domain_mask[idx]:
                colour = self._domain_violation_edge
            elif incident_edge_violation[idx]:
                colour = self._edge_violation
            else:
                colour = (0.1, 0.1, 0.1, 0.0)
            label.set_color(colour)

        if self.history_steps:
            current_step = self.history_steps[self.current_index]
            current_energy = self.history_energies[self.current_index]
            self.energy_marker.set_data([current_step], [current_energy])

        self.title.set_text(
            f"Step {step_value:,} · energy {energy_value:.2f} · "
            f"domain violations {int(domain_mask.sum())} · edge violations {int(edge_mask.sum())}"
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def _update_energy_plot(self) -> None:
        if not self.history_steps:
            self.energy_line.set_data([], [])
            self.energy_marker.set_data([], [])
            return

        steps = np.asarray(self.history_steps, dtype=float)
        energies = np.asarray(self.history_energies, dtype=float)
        self.energy_line.set_data(steps, energies)

        x_min = float(steps.min())
        x_max = float(steps.max())
        x_pad = max((x_max - x_min) * 0.05, 1.0) if x_max != x_min else 1.0
        self.energy_ax.set_xlim(x_min - x_pad, x_max + x_pad)

        y_min = float(energies.min())
        y_max = float(energies.max())
        span = y_max - y_min
        y_min = max(energies.min(), 1e-6)
        y_max = max(energies.max(), 1e-6)
        self.energy_ax.set_ylim(y_min * 0.9, y_max * 1.1)

    def _set_current_index(self, index: int, *, force: bool = False) -> None:
        if not self.history_steps:
            return
        index = max(0, min(index, len(self.history_steps) - 1))
        if not force and index == self.current_index:
            return
        self.current_index = index
        self._apply_state()

    def _on_key_press(self, event) -> None:
        if not self.history_steps:
            return

        if event.key in ("left", "h"):
            self._set_current_index(self.current_index - 1)
        elif event.key in ("right", "l"):
            self._set_current_index(self.current_index + 1)
        elif event.key in ("home",):
            self._set_current_index(0)
        elif event.key in ("end",):
            self._set_current_index(len(self.history_steps) - 1)

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


