# THRML Example: Graph Coloring

This repository demonstrates how [Extropic](https://extropic.com)'s thermodynamic computing technology can tackle challenging graph coloring problems, focusing on a real-world example: a TV station channel packing problem.

Here is an [interactive demo showing the solver at work.](https://uhgall.github.io/thrml-tv/doc/demo/)

This project was one of the winners at the THRML-HACK hackathon on Nov 16, 2025. Feel free to do whatever you want with this code - MIT License.

## Example problem: FCC Incentive Auction

The FCC Incentive Auction (2016–2017) was a government-run two-sided auction, in which:

- **TV stations** sold their broadcast rights.
- **Mobile carriers** purchased the newly released spectrum.

The primary engineering challenge was **channel repacking**: The auction algorithm required checking, during the bidding phase, whether subsets of TV stations could be packed into a given number of channels without violating interference constraints. This step is essentially an **enormous graph coloring problem**:

- **Stations** = nodes
- **Channels** = colors
- **Interference** = edges
- **Allowed channels** = per-node domain

The original FCC auction system solved the problem by reducing it to SAT and running it on standard SAT solvers. Even on the significant compute cluster allocated for the auction, due to the problem size (2990 nodes × 49 colors × 2.7 million edge constraints), optimal solutions could not be guaranteed. Even small improvements in coloring directly corresponded to significant public savings.

## Solution using THRML

We formulate repacking as an energy minimization problem:

- **Unary factors**: penalize assignments outside each station’s allowed channel list.
- **Pairwise factors**: penalize interfering stations assigned to the same channel.

The software implementation supports CPU and GPU execution via [Extropic's THRML SDK](https://extropic.com/). 
It runs ok with up to about 400 nodes (on my Macbook Pro, without GPU support I think). 
GPU did not help much (thanks, @Dtrimcev, for working on that during the hackathon!)

Once Extropic's hardware becomes available, it should be possible to run it directly on the hardware, and it should run a lot faster, and hopefully handle all 2000+ stations.

For details of how it was all implemented, see [doc/overview.md](doc/overview.md).

## Source Code 

- `lib/tv_graph.py` parses the FCC inputs and builds the in-memory graph representation used everywhere else.
- `lib/tv_web_viz.py` (with static assets in `lib/web_viz_static/`) drives the live FastAPI/D3 visualiser for sampler runs.
- `solver.py`, `graph_stats.py`, and `save_sub_graph.py` are CLI entry points you can run from the project root.

## Offline replays

Whenever you run `solver.py` with the web visualiser enabled (the default), the sampler stream is captured to `runs/<run-label>.ndjson` and, when the run finishes, a static replay bundle is exported to `output/<dataset>-<run-label>/`. Each bundle contains:

- `index.html`, `styles.css`, `app.js`, and `d3.v7.min.js` — the self-contained UI assets.
- `data.js` — embedded graph metadata and the entire sampler history (no network calls required).
- `history.ndjson` and `run.log` — the raw event log for auditing.

You can open the bundle directly from the filesystem (double-click `index.html`) without running a backend server. If you want to change where bundles are written, pass `--web-viz-output-dir /path/to/output` when invoking `solver.py`.




