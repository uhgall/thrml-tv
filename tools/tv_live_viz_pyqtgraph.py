"""
PyQtGraph-based live visualisation for THRML TV sampling runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui, QtWidgets

try:  # pragma: no cover
    from .tv_graph import TVGraph
except ImportError:  # pragma: no cover
    from tv_graph import TVGraph


@dataclass
class PyQtGraphSamplerViz:
    """
    Render THRML sampler state using a PyQtGraph GUI.

    Keyboard controls mirror the Matplotlib backend:
      - ←/→ (or h/l) step through recorded frames
      - Home/End jump to first/last frame
    """

    graph: TVGraph
    marker_size: float = 12.0
    show_edges: bool = True
    palette_name: str = "tab20"
    positions: np.ndarray = field(init=False, repr=False)
    edges: np.ndarray = field(init=False, repr=False)
    edge_lookup: Dict[Tuple[int, int], int] = field(init=False, repr=False)

    app: QtGui.QGuiApplication = field(init=False, repr=False)
    window: QtGui.QMainWindow = field(init=False, repr=False)
    layout: pg.GraphicsLayoutWidget = field(init=False, repr=False)
    map_plot: pg.PlotItem = field(init=False, repr=False)
    energy_plot: pg.PlotItem = field(init=False, repr=False)
    scatter: pg.ScatterPlotItem = field(init=False, repr=False)
    ok_edge_item: pg.PlotDataItem = field(init=False, repr=False)
    violation_edge_item: pg.PlotDataItem = field(init=False, repr=False)
    labels: List[pg.TextItem] = field(init=False, repr=False)
    energy_curve: pg.PlotDataItem = field(init=False, repr=False)
    energy_marker: pg.ScatterPlotItem = field(init=False, repr=False)

    history_steps: List[int] = field(init=False, repr=False)
    history_assignments: List[np.ndarray] = field(init=False, repr=False)
    history_domain_masks: List[np.ndarray] = field(init=False, repr=False)
    history_edge_masks: List[np.ndarray] = field(init=False, repr=False)
    history_energies: List[float] = field(init=False, repr=False)
    current_index: int = field(init=False, repr=False)

    _palette: np.ndarray = field(init=False, repr=False)
    _domain_ok_pen: QtGui.QPen = field(init=False, repr=False)
    _domain_violation_pen: QtGui.QPen = field(init=False, repr=False)
    _edge_ok_pen: QtGui.QPen = field(init=False, repr=False)
    _edge_violation_pen: QtGui.QPen = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = self._compute_positions(self.graph)
        self.edges, self.edge_lookup = self._extract_edges(self.graph)
        self._init_palette()

        self.app = pg.mkQApp("THRML PyQtGraph Visualiser")
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("THRML TV Graph (PyQtGraph)")
        self.window.resize(1100, 900)
        self.window.keyPressEvent = self._on_key_press  # type: ignore[assignment]

        self.layout = pg.GraphicsLayoutWidget()
        self.window.setCentralWidget(self.layout)

        self.map_plot = self.layout.addPlot(row=0, col=0)
        self.map_plot.setLabel("left", "Latitude")
        self.map_plot.setLabel("bottom", "Longitude")
        self.map_plot.setAspectLocked(True, ratio=1.0)
        self.map_plot.addLegend(offset=(10, 10))

        self.energy_plot = self.layout.addPlot(row=1, col=0)
        self.energy_plot.setLabel("left", "Energy (log scale)")
        self.energy_plot.setLabel("bottom", "Step")
        self.energy_plot.setLogMode(y=True)
        self.energy_plot.showGrid(x=True, y=True, alpha=0.3)

        station_count = self.graph.station_count
        self.scatter = pg.ScatterPlotItem(pxMode=False, size=self.marker_size)
        self._apply_scatter_data(
            assignment=np.zeros(station_count, dtype=np.int32),
            domain_mask=np.zeros(station_count, dtype=bool),
        )
        self.map_plot.addItem(self.scatter)

        self.ok_edge_item = self.map_plot.plot([], [], pen=self._edge_ok_pen)
        self.violation_edge_item = self.map_plot.plot([], [], pen=self._edge_violation_pen)

        self.labels = self._create_labels()
        self._set_axis_bounds()

        self.energy_curve = self.energy_plot.plot([], [], pen=pg.mkPen("c", width=2))
        self.energy_marker = pg.ScatterPlotItem(brush=pg.mkBrush("r"), pen=pg.mkPen("r"), size=12)
        self.energy_plot.addItem(self.energy_marker)

        self.history_steps = []
        self.history_assignments = []
        self.history_domain_masks = []
        self.history_edge_masks = []
        self.history_energies = []
        self.current_index = -1

        self.window.show()
        self.app.processEvents()

    def update(
        self,
        step: int,
        assignment: np.ndarray,
        domain_violation_mask: np.ndarray,
        edge_violation_mask: np.ndarray,
        energy: float,
    ) -> None:
        """
        Record sampler state and refresh the visualisation.
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
        step_value = int(step)
        assignment_arr = np.asarray(assignment, dtype=np.int32).copy()
        domain_mask_arr = np.asarray(domain_violation_mask, dtype=bool).copy()
        edge_mask_arr = np.asarray(edge_violation_mask, dtype=bool).copy()
        energy_value = float(max(energy, 1e-6))

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
        self.app.processEvents()

    def _set_current_index(self, index: int, *, force: bool = False) -> None:
        if not self.history_steps:
            return
        index = max(0, min(index, len(self.history_steps) - 1))
        if not force and index == self.current_index:
            return
        self.current_index = index
        self._apply_state()

    def _apply_state(self) -> None:
        if self.current_index < 0:
            return

        assignment = self.history_assignments[self.current_index]
        domain_mask = self.history_domain_masks[self.current_index]
        edge_mask = self.history_edge_masks[self.current_index]
        step_value = self.history_steps[self.current_index]
        energy_value = self.history_energies[self.current_index]

        self._apply_scatter_data(assignment, domain_mask)

        incident_edge_violation = np.zeros(len(domain_mask), dtype=bool)
        if self.show_edges and len(self.edges):
            segments = self.positions[self.edges][:, :, [1, 0]]
            violated_segments = segments[edge_mask]
            ok_segments = segments[~edge_mask]

            if ok_segments.size:
                ok_x, ok_y = self._flatten_segments(ok_segments)
            else:
                ok_x = ok_y = np.array([])

            if violated_segments.size:
                vio_x, vio_y = self._flatten_segments(violated_segments)
                violated_nodes = self.edges[edge_mask].reshape(-1)
                incident_edge_violation[violated_nodes] = True
            else:
                vio_x = vio_y = np.array([])

            self.ok_edge_item.setData(ok_x, ok_y)
            self.violation_edge_item.setData(vio_x, vio_y)

        for idx, label in enumerate(self.labels):
            if domain_mask[idx]:
                colour = QtGui.QColor(230, 50, 50, 255)
            elif incident_edge_violation[idx]:
                colour = QtGui.QColor(230, 60, 60, 255)
            else:
                colour = QtGui.QColor(20, 20, 20, 0)
            label.setColor(colour)

        steps = np.asarray(self.history_steps, dtype=float)
        energies = np.asarray(self.history_energies, dtype=float)
        self.energy_marker.setData([step_value], [energy_value])
        self.energy_plot.setXRange(steps.min(), steps.max(), padding=0.05)
        ymin = max(energies.min(), 1e-6)
        ymax = max(energies.max(), 1e-6)
        self.energy_plot.setYRange(ymin * 0.9, ymax * 1.1)

        self.window.setWindowTitle(
            f"THRML TV Graph (PyQtGraph) – Step {step_value:,} · "
            f"Energy {energy_value:.2f} · Domain {int(domain_mask.sum())} · Edges {int(edge_mask.sum())}"
        )

    def _update_energy_plot(self) -> None:
        if not self.history_steps:
            self.energy_curve.setData([])
            self.energy_marker.setData([], [])
            return

        steps = np.asarray(self.history_steps, dtype=float)
        energies = np.asarray(self.history_energies, dtype=float)
        self.energy_curve.setData(steps, energies)

    def _apply_scatter_data(self, assignment: np.ndarray, domain_mask: np.ndarray) -> None:
        palette_len = len(self._palette)
        brushes = [self._palette[int(idx) % palette_len] for idx in assignment]
        pens = [self._domain_violation_pen if domain_mask[i] else self._domain_ok_pen for i in range(len(domain_mask))]
        self.scatter.setData(
            x=self.positions[:, 1],
            y=self.positions[:, 0],
            brush=brushes,
            pen=pens,
        )

    def compute_violation_masks(self, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def block(self) -> None:
        """
        Enter the Qt event loop until the window is closed.
        """

        self.app.exec()

    def _on_key_press(self, event) -> None:  # type: ignore[override]
        if not self.history_steps:
            return
        key = event.key()
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_H):
            self._set_current_index(self.current_index - 1)
        elif key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_L):
            self._set_current_index(self.current_index + 1)
        elif key == QtCore.Qt.Key_Home:
            self._set_current_index(0, force=True)
        elif key == QtCore.Qt.Key_End:
            self._set_current_index(len(self.history_steps) - 1, force=True)

    def _init_palette(self) -> None:
        colours = [pg.intColor(i, hues=20, values=1, maxValue=255) for i in range(20)]
        self._palette = np.array([pg.mkBrush(c) for c in colours], dtype=object)

        self._domain_ok_pen = pg.mkPen(color=(25, 25, 25, 200), width=1.0)
        self._domain_violation_pen = pg.mkPen(color=(230, 50, 50, 255), width=2.0)
        self._edge_ok_pen = pg.mkPen(color=(180, 180, 180, 120), width=1.0)
        self._edge_violation_pen = pg.mkPen(color=(235, 50, 50, 220), width=2.0)

    def _set_axis_bounds(self) -> None:
        lats = self.positions[:, 0]
        lons = self.positions[:, 1]
        if not lats.size or not lons.size:
            return
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_pad = max((lat_max - lat_min) * 0.05, 0.1)
        lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
        self.map_plot.setYRange(lat_min - lat_pad, lat_max + lat_pad, padding=0.0)
        self.map_plot.setXRange(lon_min - lon_pad, lon_max + lon_pad, padding=0.0)

    def _create_labels(self) -> List[pg.TextItem]:
        labels: List[pg.TextItem] = []
        for idx, (lat, lon) in enumerate(self.positions):
            station = self.graph.station_by_index(idx)
            city = (station.attrs.get("City") if station.attrs else None) or str(station.station_id)
            label = pg.TextItem(
                text=city,
                color=(20, 20, 20, 0),
                anchor=(0, 1),
                fill=None,
            )
            label.setPos(lon + 0.05, lat + 0.05)
            self.map_plot.addItem(label)
            labels.append(label)
        return labels

    @staticmethod
    def _compute_positions(graph: TVGraph) -> np.ndarray:
        lats: List[float] = []
        lons: List[float] = []
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

    @staticmethod
    def _flatten_segments(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert an array of shape (n, 2, 2) into x/y arrays with NaN separators.
        """

        if segments.size == 0:
            return np.array([]), np.array([])
        xs = segments[:, :, 0]
        ys = segments[:, :, 1]
        x_flat = np.column_stack([xs[:, 0], xs[:, 1], np.full(len(xs), np.nan)]).ravel()
        y_flat = np.column_stack([ys[:, 0], ys[:, 1], np.full(len(ys), np.nan)]).ravel()
        return x_flat, y_flat

__all__ = ["PyQtGraphSamplerViz"]


