# Web-Based Visualisation for THRML TV

Vibe coded browser-based dashboard built on FastAPI, WebSockets, and D3. It streams sampler updates in real time, keeps the entire history on disk (NDJSON), and exposes controls for replaying runs.

## Requirements

- Python dependencies: `fastapi`, `uvicorn`, `jinja2` (installed automatically via FastAPI), and their transitive deps.
- Front-end assets live in `lib/web_viz_static/`; no build step is required.

Install the Python pieces (once per environment):

```bash
pip install fastapi uvicorn
```

## Running the sampler with the web UI

Pass `--web-viz` to `solver.py`:

```bash
./solver.py \
    --input fcc \
    --samples 2000 \
    --web-viz \
    --web-viz-port 8765 \
    --web-viz-history-dir runs \
    --web-viz-every 1
```

Flags of note:

| Flag | Purpose |
| --- | --- |
| `--web-viz` | Enable the FastAPI server + WebSocket broadcaster. |
| `--web-viz-host` | Bind address (default `127.0.0.1`; use `0.0.0.0` to expose on LAN). |
| `--web-viz-port` | TCP port for the dashboard. |
| `--web-viz-history-dir` | Directory for NDJSON history files (`run-<timestamp>.ndjson`). |
| `--web-viz-every` | Number of Gibbs sweeps between pushes (default 1 = every iteration). |
| `--web-viz-no-open` | Disable automatic browser launch. |
| `--web-viz-no-block` | Return to the shell immediately after sampling (server keeps writing during the run but shuts down once sampling ends). |
| `--web-viz-run-name` | Custom label for the run (used in the dashboard and filenames). |

When `--web-viz` is active, the script starts the server in a background thread before sampling begins. The server streams updates while sampling runs and remains available afterward until you press `Ctrl+C` (or until `--web-viz-no-block` is set).

Warmup defaults to `0`, so omit `--warmup` unless you explicitly want burn-in sweeps.

## What the dashboard shows

- **Scatter plot (map view):** stations coloured by assignment, with domain violations (red outline) and interference violations (amber outline). Violating edges are rendered in red; labels highlight stations involved in any conflict.
- **Energy chart (log scale):** cumulative energy trajectory on a logarithmic axis, including a marker for the currently selected sample (energies clamp to a small positive floor to keep the log defined).
- **Event log:** rolling console of server/client status messages (startup, websocket reconnects, etc.) to help debugging.
- **Playback controls:**
  - Buttons: play/pause, first/last, step forward/back.
  - Keyboard: `←/→` step, `Home/End` jump, `Space` play/pause.
  - Slider: scrub through the recorded history.

The UI fetches `/graph` once on load (station metadata + edges), drains `/history` (NDJSON) for recorded states, and subscribes to `/ws/state` for streaming updates.

## History files

- Each run writes a single NDJSON file under `--web-viz-history-dir`, e.g. `runs/run-20250115-223000.ndjson`.
- Each line contains:

```json
{
  "type": "state",
  "run": "run-20250115-223000",
  "step": 42,
  "assignment": [...],
  "domain_violation_mask": [...],
  "edge_violation_mask": [...],
  "domain_violation_count": 3,
  "edge_violation_count": 1,
  "energy": 1234.56,
  "timestamp": 1736990994.312
}
```

- Files are append-only and never truncated; use them to replay a run by reloading the dashboard (it will fetch the history before reconnecting to the stream).

## Troubleshooting

- If the browser fails to connect, ensure `fastapi`/`uvicorn` are installed and that the chosen port is free.
- When binding to `0.0.0.0`, the auto-open step still targets `http://127.0.0.1:<port>/`—adjust manually if you need remote access.
- NDJSON files grow with the number of iterations; rotate or compress them afterward if you need to manage disk usage.


