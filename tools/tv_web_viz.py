from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, TextIO

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:  # pragma: no cover - allow running as script or module import
    from .tv_graph import TVGraph
except ImportError:  # pragma: no cover
    from tv_graph import TVGraph


@dataclass
class WebVizConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    history_dir: Path = Path("runs")
    auto_open: bool = True
    block: bool = True
    static_dir: Path | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        self.history_dir = Path(self.history_dir)
        if self.static_dir is not None:
            self.static_dir = Path(self.static_dir)


class WebVizController:
    """
    Launch a FastAPI + D3 visualisation server for live sampler updates.
    """

    def __init__(self, graph: TVGraph, config: WebVizConfig) -> None:
        self.graph = graph
        self.config = config

        self._static_dir = (config.static_dir or Path(__file__).with_name("web_viz_static")).resolve()
        if not self._static_dir.exists():
            raise FileNotFoundError(f"Static assets for web visualiser not found at {self._static_dir}.")

        self._positions = self._compute_positions(graph)
        self._edges, self._edge_lookup = self._extract_edges(graph)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._run_label = config.run_name or f"run-{timestamp}"

        self._history_dir = config.history_dir.resolve()
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self._history_dir / f"{self._run_label}.ndjson"

        self._graph_payload = self._build_graph_payload()

        self._app = self._create_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        self._loop_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task[None] | None = None

        self._history_fp: TextIO | None = None
        self._history_lock = threading.Lock()

    @property
    def url(self) -> str:
        """
        Return a browser-friendly URL for the running server.
        """

        host = self.config.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{self.config.port}/"

    @property
    def history_path(self) -> Path:
        return self._history_path

    def start(self) -> None:
        """
        Run the FastAPI server in a background thread.
        """

        if self._server is not None:
            raise RuntimeError("Web visualiser already running.")

        self._history_fp = self._history_path.open("w", encoding="utf-8", buffering=1)

        config = uvicorn.Config(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        def _run_server() -> None:
            self._server.run()

        self._thread = threading.Thread(target=_run_server, name="web-viz-server", daemon=True)
        self._thread.start()

        if not self._loop_ready.wait(timeout=10):
            raise RuntimeError("Web visualiser failed to start within timeout.")

        if self.config.auto_open:
            self._open_browser()

    def stop(self) -> None:
        """
        Shut down the server and flush history to disk.
        """

        if self._server is None:
            return

        if self._loop and self._queue:
            future = asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)
            with suppress(Exception):
                future.result(timeout=2)

        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

        self._server = None
        self._thread = None
        self._loop = None
        self._queue = None
        self._loop_ready.clear()

        if self._history_fp is not None:
            self._history_fp.close()
            self._history_fp = None

    def is_running(self) -> bool:
        return self._server is not None and not self._server.should_exit

    def block_until_interrupt(self) -> None:
        """
        Keep the main thread alive until the user interrupts or the server stops.
        """

        if not self.config.block:
            return
        try:
            while self.is_running():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def record_state(self, step: int, assignment: np.ndarray, energy: float) -> None:
        """
        Append a sampler state to disk and broadcast to connected clients.
        """

        if self._history_fp is None:
            return

        assignment_arr = np.asarray(assignment, dtype=np.int32)
        domain_mask, edge_mask = self._compute_violation_masks(assignment_arr)
        payload = {
            "type": "state",
            "run": self._run_label,
            "step": int(step),
            "assignment": assignment_arr.tolist(),
            "domain_violation_mask": domain_mask.tolist(),
            "edge_violation_mask": edge_mask.tolist(),
            "domain_violation_count": int(np.count_nonzero(domain_mask)),
            "edge_violation_count": int(np.count_nonzero(edge_mask)),
            "energy": float(energy),
            "timestamp": time.time(),
        }

        line = json.dumps(payload, separators=(",", ":"))
        with self._history_lock:
            self._history_fp.write(line + "\n")
            self._history_fp.flush()

        if self._loop and self._queue:
            future = asyncio.run_coroutine_threadsafe(self._queue.put(payload), self._loop)
            with suppress(Exception):
                future.result(timeout=2)

    def _open_browser(self) -> None:
        try:
            webbrowser.open(self.url)
        except Exception:
            pass

    def _create_app(self) -> FastAPI:
        app = FastAPI()
        app.mount("/static", StaticFiles(directory=str(self._static_dir)), name="static")

        @app.on_event("startup")
        async def on_startup() -> None:
            self._loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            self._clients = set()
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
            self._loop_ready.set()

        @app.on_event("shutdown")
        async def on_shutdown() -> None:
            if self._broadcast_task is not None:
                self._broadcast_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._broadcast_task
                self._broadcast_task = None
            self._loop_ready.clear()

        @app.get("/", response_class=FileResponse)
        async def index() -> FileResponse:
            return FileResponse(self._static_dir / "index.html")

        @app.get("/graph")
        async def graph_endpoint() -> JSONResponse:
            return JSONResponse(self._graph_payload)

        @app.get("/history")
        async def history_endpoint() -> StreamingResponse:
            if not self._history_path.exists():
                return StreamingResponse(iter(()), media_type="application/x-ndjson")

            def iter_history() -> Any:
                with self._history_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        yield line

            return StreamingResponse(iter_history(), media_type="application/x-ndjson")

        @app.websocket("/ws/state")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._clients.add(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)

        return app

    async def _broadcast_loop(self) -> None:
        if self._queue is None:
            return

        while True:
            message = await self._queue.get()
            if message is None:
                break

            dead: list[WebSocket] = []
            for client in list(self._clients):
                try:
                    await client.send_json(message)
                except WebSocketDisconnect:
                    dead.append(client)
                except RuntimeError:
                    dead.append(client)

            for client in dead:
                self._clients.discard(client)

            self._queue.task_done()

    def _build_graph_payload(self) -> Dict[str, Any]:
        stations_payload: List[Dict[str, Any]] = []
        for idx, (lat, lon) in enumerate(self._positions):
            station = self.graph.station_by_index(idx)
            city = (station.attrs.get("City") if station.attrs else None) or str(station.station_id)
            stations_payload.append(
                {
                    "index": idx,
                    "station_id": station.station_id,
                    "lat": float(lat),
                    "lon": float(lon),
                    "city": city,
                }
            )

        if self._positions.size:
            lat_range = [float(self._positions[:, 0].min()), float(self._positions[:, 0].max())]
            lon_range = [float(self._positions[:, 1].min()), float(self._positions[:, 1].max())]
        else:
            lat_range = [0.0, 0.0]
            lon_range = [0.0, 0.0]

        return {
            "run_name": self._run_label,
            "station_count": self.graph.station_count,
            "channel_count": self.graph.channel_count,
            "stations": stations_payload,
            "edges": self._edges.tolist(),
            "lat_range": lat_range,
            "lon_range": lon_range,
        }

    def _compute_violation_masks(self, assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        domain_matrix = self.graph.domain_mask
        index = np.arange(len(assignment), dtype=np.int32)
        domain_ok = domain_matrix[index, assignment]
        domain_violation_mask = np.logical_not(domain_ok)

        edge_violation_mask = np.zeros(len(self._edges), dtype=bool)
        if self._edges.size == 0:
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
                    edge_id = self._edge_lookup.get(key)
                    if edge_id is not None:
                        edge_violation_mask[edge_id] = True

        return domain_violation_mask, edge_violation_mask

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

        coords: List[Tuple[float, float]] = []
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
    def _extract_edges(graph: TVGraph) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
        edge_set: set[Tuple[int, int]] = set()
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


__all__ = ["WebVizConfig", "WebVizController"]


